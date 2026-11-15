package rtvi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type Params struct {
	About               map[string]any
	Capabilities        *Capabilities
	DisableAutoBotReady bool
	OnBotStarted        func(p *Processor)
	OnClientReady       func(p *Processor)
	OnClientMessage     func(p *Processor, msg ClientMessage)
}

type Processor struct {
	*processors.BaseProcessor
	params Params
	log    *logger.Logger

	clientVersion atomic.Uint64
	clientReady   atomic.Bool
	botReady      atomic.Bool
	llmSkipTTS    atomic.Bool

	mu            sync.Mutex
	clientReadyID string
	botStarted    bool
	flushID       uint64
	flushCh       chan struct{}
	observed      atomic.Bool
}

const flushTimeout = 300 * time.Millisecond

func New(params Params) *Processor {
	p := &Processor{params: params, log: logger.WithPrefix("RTVIProcessor")}
	p.BaseProcessor = processors.NewBaseProcessor("RTVIProcessor", p)
	return p
}

func packVersion(v [3]int) uint64 {
	return uint64(v[0])<<42 | uint64(v[1]&0x1fffff)<<21 | uint64(v[2]&0x1fffff)
}

func (p *Processor) ClientVersion() [3]int {
	v := p.clientVersion.Load()
	return [3]int{int(v >> 42), int(v >> 21 & 0x1fffff), int(v & 0x1fffff)}
}

func (p *Processor) clientAtLeast(major, minor, patch int) bool {
	return p.clientVersion.Load() >= packVersion([3]int{major, minor, patch})
}

func (p *Processor) isLegacyClient() bool {
	return p.ClientVersion()[0] == LegacySupportedMajor
}

func (p *Processor) IsClientReady() bool { return p.clientReady.Load() }

func (p *Processor) IsBotReady() bool { return p.botReady.Load() }

func (p *Processor) NewObserver(params *ObserverParams) *Observer {
	return NewObserver(p, params)
}

func (p *Processor) interruptAndFlush(ctx context.Context) {
	var ch chan struct{}
	first := true
	_ = p.BroadcastFrame(ctx, func() frames.Frame {
		f := frames.NewInterruptionFrame()
		if first && p.observed.Load() {
			first = false
			ch = make(chan struct{})
			p.mu.Lock()
			p.flushID, p.flushCh = f.ID(), ch
			p.mu.Unlock()
		}
		return f
	})
	if ch == nil {
		return
	}
	t := time.NewTimer(flushTimeout)
	select {
	case <-ch:
	case <-t.C:
	case <-ctx.Done():
	}
	t.Stop()
	p.mu.Lock()
	if p.flushCh == ch {
		p.flushID, p.flushCh = 0, nil
	}
	p.mu.Unlock()
}

func (p *Processor) interruptionReachedOutput(id uint64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.flushCh != nil && p.flushID == id {
		close(p.flushCh)
		p.flushID, p.flushCh = 0, nil
	}
}

func (p *Processor) SetBotReady(about map[string]any) {
	p.botReady.Store(true)
	if len(about) == 0 {
		about = p.params.About
	}
	if len(about) == 0 {
		about = map[string]any{"library": "strawgo-ai", "library_version": libraryVersion()}
	}
	version := ProtocolVersion
	if v := p.ClientVersion(); v[0] == LegacySupportedMajor {
		version = fmt.Sprintf("%d.%d.%d", v[0], v[1], v[2])
	}
	p.mu.Lock()
	id := p.clientReadyID
	p.mu.Unlock()
	p.Send(NewResponse(TypeBotReady, id, BotReadyData{Version: version, About: about, Capabilities: p.params.Capabilities}))
}

func libraryVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	const module = "github.com/square-key-labs/strawgo-ai"
	if info.Main.Path == module {
		return info.Main.Version
	}
	for _, d := range info.Deps {
		if d.Path == module {
			return d.Version
		}
	}
	return "unknown"
}

func (p *Processor) Send(msg Message) {
	_ = p.PushFrame(frames.NewOutputTransportMessageUrgentFrame(msg), frames.Downstream)
}

func (p *Processor) SendServerMessage(data any) {
	p.Send(NewMessage(TypeServerMessage, data))
}

func (p *Processor) SendServerResponse(msg ClientMessage, data any) {
	p.Send(NewResponse(TypeServerResponse, msg.MsgID, RawServerResponseData{T: msg.Type, D: data}))
}

func (p *Processor) SendErrorResponse(msg ClientMessage, err string) {
	p.sendErrorResponse(msg.MsgID, err)
}

func (p *Processor) sendErrorResponse(id, err string) {
	p.Send(NewResponse(TypeErrorResponse, id, ErrorResponseData{Error: err}))
}

func (p *Processor) SendError(err string) {
	p.Send(NewMessage(TypeError, ErrorData{Error: err}))
}

func (p *Processor) InterruptBot() {
	_ = p.BroadcastInterruption(p.Context())
}

func (p *Processor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		err := p.PushFrame(frame, direction)
		p.mu.Lock()
		first := !p.botStarted
		p.botStarted = true
		p.mu.Unlock()
		if first && p.params.OnBotStarted != nil {
			p.params.OnBotStarted(p)
		}
		return err
	case *frames.ErrorFrame:
		msg := "unknown error"
		if f.Error != nil {
			msg = f.Error.Error()
		}
		p.Send(NewMessage(TypeError, ErrorData{Error: msg, Fatal: f.Fatal}))
	case *frames.InputTransportMessageFrame:
		if env, ok := parseClientMessage(f.Message); ok {
			p.handleMessage(ctx, env)
			return nil
		}
	case *frames.LLMConfigureOutputFrame:
		p.llmSkipTTS.Store(f.SkipTTS)
	}
	return p.PushFrame(frame, direction)
}

func parseClientMessage(raw any) (clientEnvelope, bool) {
	var env clientEnvelope
	switch m := raw.(type) {
	case map[string]any:
		env.Label, _ = m["label"].(string)
		env.Type, _ = m["type"].(string)
		switch id := m["id"].(type) {
		case string:
			env.ID = id
		case float64:
			env.ID = strconv.FormatFloat(id, 'f', -1, 64)
		}
		env.Data, _ = m["data"].(map[string]any)
	case string:
		if json.Unmarshal([]byte(m), &env) != nil {
			return env, false
		}
	case []byte:
		if json.Unmarshal(m, &env) != nil {
			return env, false
		}
	case json.RawMessage:
		if json.Unmarshal(m, &env) != nil {
			return env, false
		}
	default:
		return env, false
	}
	return env, env.Label == MessageLabel
}

func decodeData(data map[string]any, v any) error {
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

func (p *Processor) handleMessage(ctx context.Context, env clientEnvelope) {
	defer func() {
		if r := recover(); r != nil {
			p.sendErrorResponse(env.ID, fmt.Sprintf("Exception processing message: %v", r))
		}
	}()
	var err error
	switch env.Type {
	case "client-ready":
		p.handleClientReady(env)
	case "disconnect-bot":
		_ = p.PushFrame(frames.NewEndTaskFrame("rtvi disconnect-bot"), frames.Upstream)
	case "client-message":
		err = p.handleClientMessage(env)
	case "llm-function-call-result":
		err = p.handleFunctionCallResult(env)
	case "send-text":
		err = p.handleSendText(ctx, env)
	case "send-file":
		err = p.handleSendFile(ctx, env)
	case "raw-audio", "raw-audio-batch":
		p.handleAudio(env)
	case "dtmf":
		err = p.handleDTMF(env)
	default:
		p.log.Warn("unsupported RTVI message type: %s", env.Type)
		p.sendErrorResponse(env.ID, "Unsupported type "+env.Type)
		return
	}
	if err != nil {
		p.log.Warn("invalid message: %v", err)
		p.sendErrorResponse(env.ID, "Invalid message: "+err.Error())
	}
}

func (p *Processor) handleClientReady(env clientEnvelope) {
	versionErr := ""
	version, _ := env.Data["version"].(string)
	about := AboutClientData{Library: "unknown"}
	if raw, ok := env.Data["about"].(map[string]any); ok {
		if decodeData(raw, &about) != nil {
			about = AboutClientData{Library: "unknown"}
		}
	}
	if version == "" {
		versionErr = "Client version unknown."
	} else {
		parts := strings.Split(version, ".")
		var v [3]int
		ok := len(parts) == 3
		for i := 0; ok && i < 3; i++ {
			n, err := strconv.Atoi(parts[i])
			if err != nil || n < 0 {
				ok = false
				break
			}
			v[i] = n
		}
		if !ok {
			versionErr = fmt.Sprintf("Invalid client version format (%s).", version)
		} else {
			p.clientVersion.Store(packVersion(v))
			serverMajor, _ := strconv.Atoi(strings.SplitN(ProtocolVersion, ".", 2)[0])
			if v[0] != serverMajor && v[0] != LegacySupportedMajor {
				versionErr = fmt.Sprintf("RTVI version %s is not compatible with server protocol %s.", version, ProtocolVersion)
			}
		}
	}
	if versionErr != "" {
		versionErr += " Compatibility issues may occur."
		p.log.Warn("%s", versionErr)
		p.sendErrorResponse(env.ID, versionErr)
	}
	p.log.Debug("client ready: version=%s library=%s", version, about.Library)
	_ = p.PushFrame(frames.NewInputTransportStartAudioStreamingFrame(), frames.Downstream)
	p.mu.Lock()
	p.clientReadyID = env.ID
	p.mu.Unlock()
	p.clientReady.Store(true)
	if p.params.OnClientReady != nil {
		p.params.OnClientReady(p)
	}
	if !p.params.DisableAutoBotReady {
		p.SetBotReady(nil)
	}
}

func (p *Processor) handleClientMessage(env clientEnvelope) error {
	t, ok := env.Data["t"].(string)
	if !ok {
		return errors.New("client-message requires field 't'")
	}
	d := env.Data["d"]
	_ = p.PushFrame(NewClientMessageFrame(env.ID, t, d), frames.Downstream)
	if p.params.OnClientMessage != nil {
		p.params.OnClientMessage(p, ClientMessage{MsgID: env.ID, Type: t, Data: d})
	}
	return nil
}

func (p *Processor) handleFunctionCallResult(env clientEnvelope) error {
	var data struct {
		FunctionName *string        `json:"function_name"`
		ToolCallID   *string        `json:"tool_call_id"`
		Arguments    map[string]any `json:"arguments"`
		Result       any            `json:"result"`
	}
	if err := decodeData(env.Data, &data); err != nil {
		return err
	}
	if data.FunctionName == nil || data.ToolCallID == nil || data.Arguments == nil || data.Result == nil {
		return errors.New("llm-function-call-result requires function_name, tool_call_id, arguments and result")
	}
	f := frames.NewFunctionCallResultFrame(*data.ToolCallID, *data.FunctionName, data.Result, nil)
	f.Arguments = data.Arguments
	return p.PushFrame(f, frames.Downstream)
}

type sendOptions struct {
	RunImmediately *bool          `json:"run_immediately"`
	AudioResponse  *bool          `json:"audio_response"`
	CustomOptions  map[string]any `json:"custom_options"`
}

func (o *sendOptions) run() bool   { return o == nil || o.RunImmediately == nil || *o.RunImmediately }
func (o *sendOptions) audio() bool { return o == nil || o.AudioResponse == nil || *o.AudioResponse }

func (p *Processor) withSkipTTS(audio bool, push func()) {
	cur := p.llmSkipTTS.Load()
	should := !audio
	toggle := cur != should
	if toggle {
		_ = p.PushFrame(frames.NewLLMConfigureOutputFrame(should), frames.Downstream)
	}
	push()
	if toggle {
		_ = p.PushFrame(frames.NewLLMConfigureOutputFrame(cur), frames.Downstream)
	}
}

func (p *Processor) handleSendText(ctx context.Context, env clientEnvelope) error {
	var data struct {
		Content *string      `json:"content"`
		Options *sendOptions `json:"options"`
	}
	if err := decodeData(env.Data, &data); err != nil {
		return err
	}
	if data.Content == nil {
		return errors.New("send-text requires field 'content'")
	}
	run := data.Options.run()
	if run {
		p.interruptAndFlush(ctx)
	}
	p.withSkipTTS(data.Options.audio(), func() {
		_ = p.PushFrame(frames.NewLLMMessagesAppendFrame([]llm.Message{llm.User(*data.Content)}, run), frames.Downstream)
		_ = p.PushFrame(NewSendTextFrame(env.ID, *data.Content, ""), frames.Downstream)
	})
	return nil
}

func (p *Processor) handleSendFile(ctx context.Context, env clientEnvelope) error {
	var data struct {
		Content string `json:"content"`
		File    *struct {
			Format string `json:"format"`
			Name   string `json:"name"`
			Source struct {
				Type   string `json:"type"`
				Bytes  string `json:"bytes"`
				URL    string `json:"url"`
				Width  int    `json:"width"`
				Height int    `json:"height"`
			} `json:"source"`
		} `json:"file"`
		Options *sendOptions `json:"options"`
	}
	if err := decodeData(env.Data, &data); err != nil {
		return err
	}
	if data.File == nil || data.File.Format == "" {
		return errors.New("send-file requires field 'file' with a format")
	}
	file := data.File
	run := data.Options.run()
	appendToContext := true
	var out frames.Frame
	switch file.Source.Type {
	case "bytes":
		src := file.Source.Bytes
		if !strings.HasPrefix(src, "data:") {
			src = "data:" + file.Format + ";base64," + src
		}
		if strings.HasPrefix(file.Format, "image/") {
			_, b64, _ := strings.Cut(src, "base64,")
			img, err := base64.StdEncoding.DecodeString(b64)
			if err != nil {
				return err
			}
			f := frames.NewUserImageRawFrame("", img, file.Source.Width, file.Source.Height, file.Format)
			f.Text = data.Content
			f.AppendToContext = &appendToContext
			f.RunLLM = &run
			out = f
		} else {
			out = p.fileFrame(src, frames.FileSourceBytes, file.Name, file.Format, data.Content, data.Options, &appendToContext, run)
		}
	case "url":
		out = p.fileFrame(file.Source.URL, frames.FileSourceURL, file.Name, file.Format, data.Content, data.Options, &appendToContext, run)
	default:
		p.sendErrorResponse(env.ID, "Unsupported file source type: "+file.Source.Type)
		return nil
	}
	if run {
		p.interruptAndFlush(ctx)
	}
	p.withSkipTTS(data.Options.audio(), func() { _ = p.PushFrame(out, frames.Downstream) })
	return nil
}

func (p *Processor) fileFrame(src string, typ frames.FileSourceType, name, format, text string, opts *sendOptions, appendToContext *bool, run bool) frames.Frame {
	f := frames.NewUserFileRawFrame("", src, typ, name, format)
	f.Text = text
	f.AppendToContext = appendToContext
	f.RunLLM = &run
	if opts != nil {
		f.CustomOptions = opts.CustomOptions
	}
	return f
}

func (p *Processor) handleAudio(env clientEnvelope) {
	rate, _ := env.Data["sampleRate"].(float64)
	channels, _ := env.Data["numChannels"].(float64)
	if rate <= 0 || channels <= 0 {
		p.log.Error("error processing audio buffer: missing sampleRate or numChannels")
		return
	}
	var batch []any
	if b, ok := env.Data["base64AudioBatch"].([]any); ok && len(b) > 0 {
		batch = b
	} else {
		batch = []any{env.Data["base64Audio"]}
	}
	for _, item := range batch {
		s, _ := item.(string)
		if s == "" {
			continue
		}
		pcm, err := base64.StdEncoding.DecodeString(s)
		if err != nil {
			p.log.Error("error processing audio buffer: %v", err)
			return
		}
		f := frames.NewAudioFrame(pcm, int(rate), int(channels))
		f.SetMetadata("codec", "linear16")
		_ = p.PushFrame(f, frames.Downstream)
	}
}

func (p *Processor) handleDTMF(env clientEnvelope) error {
	list, _ := env.Data["buttons"].([]any)
	if len(list) == 0 {
		return errors.New("dtmf requires at least one button")
	}
	buttons := make([]frames.KeypadEntry, 0, len(list))
	for _, b := range list {
		s, _ := b.(string)
		k := frames.KeypadEntry(s)
		if !k.Valid() {
			return fmt.Errorf("invalid keypad entry %q", s)
		}
		buttons = append(buttons, k)
	}
	for _, k := range buttons {
		_ = p.PushFrame(frames.NewInputDTMFFrame(k), frames.Downstream)
	}
	return nil
}
