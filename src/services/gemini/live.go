package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services/realtime"
)

const (
	GeminiLiveURL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

	DefaultLiveModel            = "gemini-3.8-live"
	DefaultLiveInputSampleRate  = 16000
	DefaultLiveOutputSampleRate = 24000
	DefaultLiveOutputChannels   = 1
)

type LiveConfig struct {
	APIKey            string
	Model             string
	SystemInstruction string
	Voice             string
	Endpoint          string
	InputMIMEType     string
	OutputSampleRate  int
	OutputChannels    int
	DisableVAD        bool
	Tools             llm.Tools
	FunctionTimeout   time.Duration
	Backoff           realtime.Backoff
	Dialer            *websocket.Dialer

	ContextWindowCompression bool
	CompressionTriggerTokens int
}

type LiveService struct {
	*processors.BaseProcessor

	cfg    LiveConfig
	conv   *Provider
	socket *realtime.Socket
	out    *realtime.Output
	fns    *realtime.Functions

	mu                sync.Mutex
	systemInstruction string
	tools             llm.Tools
	llmCtx            *llm.Context
	contextSent       bool
	resumeHandle      string
	suppress          bool
	userText          strings.Builder
	openCalls         map[string]string
	usage             *llm.Usage

	inRate   int
	inCodec  string
	inPrefix string
}

func NewLiveService(config LiveConfig) *LiveService {
	if config.Model == "" {
		config.Model = DefaultLiveModel
	}
	if config.Endpoint == "" {
		config.Endpoint = GeminiLiveURL
	}
	if config.OutputSampleRate == 0 {
		config.OutputSampleRate = DefaultLiveOutputSampleRate
	}
	if config.OutputChannels == 0 {
		config.OutputChannels = DefaultLiveOutputChannels
	}
	if config.Dialer == nil {
		config.Dialer = websocket.DefaultDialer
	}
	s := &LiveService{
		cfg:               config,
		conv:              NewProvider(ProviderConfig{Name: "GeminiLive"}),
		systemInstruction: config.SystemInstruction,
		tools:             config.Tools,
		openCalls:         make(map[string]string),
	}
	s.BaseProcessor = processors.NewBaseProcessor("GeminiLive", s)
	s.out = realtime.NewOutput(s, config.Model, config.OutputChannels)
	s.fns = realtime.NewFunctions(s)
	s.fns.SetTimeout(config.FunctionTimeout)
	s.socket = realtime.NewSocket(realtime.SocketConfig{
		Name:         "GeminiLive",
		Dial:         s.dial,
		OnOpen:       s.onOpen,
		OnMessage:    s.onMessage,
		OnDisconnect: s.onDisconnect,
		OnFatal: func(err error) {
			_ = s.PushError("Gemini Live connection lost", err, true)
		},
		Backoff: config.Backoff,
	})
	return s
}

func NewGeminiLiveService(apiKey string, model string) *LiveService {
	return NewLiveService(LiveConfig{APIKey: apiKey, Model: model})
}

func (s *LiveService) RegisterFunction(name string, handler llm.FunctionHandler, opts ...llm.FunctionOptions) {
	s.fns.Register(name, handler, opts...)
}

func (s *LiveService) UnregisterFunction(name string) { s.fns.Unregister(name) }

func (s *LiveService) HasFunction(name string) bool { return s.fns.Has(name) }

func (s *LiveService) UpdateSystemInstruction(_ context.Context, instruction string) error {
	s.mu.Lock()
	changed := s.systemInstruction != instruction
	s.systemInstruction = instruction
	s.mu.Unlock()
	if changed {
		s.restart()
	}
	return nil
}

func (s *LiveService) restart() {
	s.mu.Lock()
	s.resumeHandle = ""
	s.contextSent = false
	s.mu.Unlock()
	s.out.Interrupt()
	s.socket.Reconnect()
}

func (s *LiveService) Stop() error {
	s.shutdown()
	err := s.BaseProcessor.Stop()
	s.fns.Wait()
	return err
}

func (s *LiveService) shutdown() {
	s.fns.Cancel(false)
	s.socket.Stop()
	s.out.Interrupt()
}

func (s *LiveService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		s.socket.Start(s.Context())
	case *frames.EndFrame, *frames.CancelFrame:
		s.shutdown()
	case *frames.InterruptionFrame:
		s.fns.Cancel(true)
		s.mu.Lock()
		if s.out.Responding() {
			s.suppress = true
		}
		s.mu.Unlock()
		s.out.Interrupt()
	case *frames.UserStartedSpeakingFrame:
		if s.cfg.DisableVAD {
			s.send(map[string]any{"realtimeInput": map[string]any{"activityStart": map[string]any{}}})
		}
	case *frames.UserStoppedSpeakingFrame:
		s.mu.Lock()
		s.suppress = false
		s.mu.Unlock()
		if s.cfg.DisableVAD {
			s.send(map[string]any{"realtimeInput": map[string]any{"activityEnd": map[string]any{}}})
		}
		s.out.MarkRequest()
	case *frames.AudioFrame:
		return s.sendAudio(f)
	case *frames.LLMContextFrame:
		if f.Context != nil {
			s.handleContext(f.Context)
		}
		return nil
	case *frames.LLMMessagesAppendFrame:
		s.appendMessages(f.Messages, f.RunLLM)
	case *frames.LLMSetToolsFrame:
		s.mu.Lock()
		s.tools = f.Tools
		s.mu.Unlock()
		s.restart()
	case *frames.LLMUpdateSettingsFrame:
		if f.Service == nil || f.Service.Name() == s.Name() {
			if f.Delta.SystemInstruction != "" {
				_ = s.UpdateSystemInstruction(ctx, f.Delta.SystemInstruction)
			}
			return nil
		}
	case *frames.FunctionCallResultFrame:
		s.deliverResult(f.ToolCallID, f.FunctionName, realtime.ResultString(f.Result, nil))
	}
	return s.PushFrame(frame, direction)
}

func (s *LiveService) send(v any) {
	if err := s.socket.WriteJSON(v); err != nil && err != realtime.ErrNotConnected {
		logger.Warn("[GeminiLive] send failed: %v", err)
	}
}

func (s *LiveService) dial(ctx context.Context) (*websocket.Conn, error) {
	h := http.Header{}
	if s.cfg.APIKey != "" {
		h.Set("x-goog-api-key", s.cfg.APIKey)
	}
	conn, _, err := s.cfg.Dialer.DialContext(ctx, s.cfg.Endpoint, h)
	if err != nil {
		return nil, fmt.Errorf("gemini live dial: %w", err)
	}
	return conn, nil
}

func (s *LiveService) onOpen(_ context.Context, w realtime.Writer) error {
	return w.WriteJSON(s.setupMessage())
}

func (s *LiveService) onDisconnect(error) {
	s.out.Interrupt()
	s.mu.Lock()
	if s.resumeHandle == "" {
		s.contextSent = false
	}
	s.mu.Unlock()
}

func (s *LiveService) setupMessage() map[string]any {
	s.mu.Lock()
	system, tools, handle := s.systemInstruction, s.tools, s.resumeHandle
	s.mu.Unlock()

	gen := map[string]any{"responseModalities": []string{"AUDIO"}}
	if s.cfg.Voice != "" {
		gen["speechConfig"] = map[string]any{
			"voiceConfig": map[string]any{"prebuiltVoiceConfig": map[string]any{"voiceName": s.cfg.Voice}},
		}
	}
	setup := map[string]any{
		"model":                    modelPath(s.cfg.Model),
		"generationConfig":         gen,
		"inputAudioTranscription":  map[string]any{},
		"outputAudioTranscription": map[string]any{},
		"historyConfig":            map[string]any{"initialHistoryInClientContent": true},
	}
	if system != "" {
		setup["systemInstruction"] = map[string]any{"parts": []map[string]any{{"text": system}}}
	}
	if t := ConvertTools(tools); len(t) > 0 {
		if s.declaresBlockingTools() {
			markBlocking(t)
		}
		setup["tools"] = t
	}
	if s.cfg.ContextWindowCompression {
		cwc := map[string]any{"slidingWindow": map[string]any{}}
		if s.cfg.CompressionTriggerTokens > 0 {
			cwc["triggerTokens"] = s.cfg.CompressionTriggerTokens
		}
		setup["contextWindowCompression"] = cwc
	}
	if s.cfg.DisableVAD {
		setup["realtimeInputConfig"] = map[string]any{"automaticActivityDetection": map[string]any{"disabled": true}}
	}
	resumption := map[string]any{}
	if handle != "" {
		resumption["handle"] = handle
	}
	setup["sessionResumption"] = resumption
	return map[string]any{"setup": setup}
}

var geminiVersionPattern = regexp.MustCompile(`gemini(?:-[a-z]+)*-(\d+)(?:\.(\d+))?`)

func (s *LiveService) modelVersion() (major, minor int, ok bool) {
	m := geminiVersionPattern.FindStringSubmatch(s.cfg.Model)
	if m == nil {
		return 0, 0, false
	}
	major, _ = strconv.Atoi(m[1])
	if m[2] != "" {
		minor, _ = strconv.Atoi(m[2])
	}
	return major, minor, true
}

func (s *LiveService) isGemini3() bool {
	major, _, ok := s.modelVersion()
	return ok && major == 3
}

func (s *LiveService) declaresBlockingTools() bool {
	major, minor, ok := s.modelVersion()
	atLeast38 := ok && (major > 3 || major == 3 && minor >= 8)
	return atLeast38 && !strings.Contains(s.cfg.Model, "thinking")
}

func markBlocking(tools []map[string]any) {
	for _, tool := range tools {
		decls, _ := tool["functionDeclarations"].([]map[string]any)
		for _, d := range decls {
			d["behavior"] = "BLOCKING"
		}
	}
}

func (s *LiveService) sendTurn(turns []Content, complete bool) {
	s.send(map[string]any{"clientContent": map[string]any{"turns": turns, "turnComplete": complete}})
	if complete && s.isGemini3() {
		s.send(map[string]any{"realtimeInput": map[string]any{"text": " "}})
	}
}

func (s *LiveService) sendAudio(f *frames.AudioFrame) error {
	if len(f.Data) == 0 {
		return nil
	}
	prefix := s.audioPrefix(f)
	err := s.socket.WriteBase64(prefix, f.Data, `"}}}`)
	if err != nil && err != realtime.ErrNotConnected {
		logger.Debug("[GeminiLive] audio send failed: %v", err)
	}
	return nil
}

func (s *LiveService) audioPrefix(f *frames.AudioFrame) string {
	rate := f.SampleRate
	if rate == 0 {
		rate = DefaultLiveInputSampleRate
	}
	codec := "linear16"
	if meta := f.Metadata(); meta != nil {
		if raw, ok := meta["codec"].(string); ok {
			switch strings.ToLower(strings.TrimSpace(raw)) {
			case "mulaw", "ulaw", "pcmu":
				codec = "mulaw"
			case "alaw", "pcma":
				codec = "alaw"
			}
		}
	}
	if s.inPrefix != "" && rate == s.inRate && codec == s.inCodec {
		return s.inPrefix
	}
	mime := s.cfg.InputMIMEType
	if mime == "" {
		if codec == "linear16" {
			mime = "audio/pcm;rate=" + strconv.Itoa(rate)
		} else {
			mime = "audio/pcm;encoding=" + codec + ";rate=" + strconv.Itoa(rate)
		}
	}
	s.inRate, s.inCodec = rate, codec
	s.inPrefix = `{"realtimeInput":{"audio":{"mimeType":"` + mime + `","data":"`
	return s.inPrefix
}

func (s *LiveService) handleContext(c *llm.Context) {
	snapshot := c.Snapshot()
	system, _ := s.conv.ConvertMessages(snapshot.Messages, "")
	s.mu.Lock()
	first := s.llmCtx == nil
	s.llmCtx = c
	reconnect := false
	if first && s.systemInstruction == "" && system != "" {
		s.systemInstruction = system
		reconnect = true
	}
	if first && !snapshot.Tools.Empty() && s.tools.Empty() {
		s.tools = snapshot.Tools
		reconnect = true
	}
	s.mu.Unlock()

	if first {
		if reconnect && s.socket.Ready() {
			s.restart()
			return
		}
		if s.socket.Ready() {
			s.sendContext()
		}
		return
	}
	for _, m := range snapshot.Messages {
		for _, p := range m.Parts {
			if r, ok := p.(llm.ToolResultPart); ok {
				s.deliverResult(r.ToolCallID, r.Name, r.Content)
			}
		}
	}
}

func (s *LiveService) sendContext() {
	s.mu.Lock()
	c := s.llmCtx
	if c == nil || s.contextSent {
		s.mu.Unlock()
		return
	}
	s.contextSent = true
	s.mu.Unlock()

	_, contents := s.conv.ConvertMessages(c.Snapshot().Messages, "")
	if len(contents) == 0 {
		return
	}
	complete := contents[len(contents)-1].Role == "user"
	if complete {
		s.out.MarkRequest()
	}
	s.sendTurn(contents, complete)
}

func (s *LiveService) appendMessages(msgs []llm.Message, run bool) {
	if !s.socket.Ready() {
		return
	}
	_, contents := s.conv.ConvertMessages(msgs, "")
	if len(contents) == 0 {
		return
	}
	if run {
		s.out.MarkRequest()
	}
	s.sendTurn(contents, run)
}

func (s *LiveService) deliverResult(id, name, content string) {
	s.mu.Lock()
	known, open := s.openCalls[id]
	if open {
		delete(s.openCalls, id)
	}
	s.mu.Unlock()
	if !open {
		return
	}
	if name == "" {
		name = known
	}
	s.send(map[string]any{"toolResponse": map[string]any{"functionResponses": []map[string]any{{
		"id":       id,
		"name":     name,
		"response": FunctionResponse(content, false),
	}}}})
}

type liveInbound struct {
	SetupComplete *json.RawMessage `json:"setupComplete"`
	ServerContent *struct {
		ModelTurn *struct {
			Parts []struct {
				Text       string `json:"text"`
				Thought    bool   `json:"thought"`
				InlineData *struct {
					MimeType string `json:"mimeType"`
					Data     string `json:"data"`
				} `json:"inlineData"`
			} `json:"parts"`
		} `json:"modelTurn"`
		InputTranscription  *liveTranscription `json:"inputTranscription"`
		OutputTranscription *liveTranscription `json:"outputTranscription"`
		TurnComplete        bool               `json:"turnComplete"`
		Interrupted         bool               `json:"interrupted"`
	} `json:"serverContent"`
	ToolCall *struct {
		FunctionCalls []struct {
			ID   string          `json:"id"`
			Name string          `json:"name"`
			Args json.RawMessage `json:"args"`
		} `json:"functionCalls"`
	} `json:"toolCall"`
	ToolCallCancellation *struct {
		IDs []string `json:"ids"`
	} `json:"toolCallCancellation"`
	UsageMetadata           *usageMetadata `json:"usageMetadata"`
	GoAway                  *struct{}      `json:"goAway"`
	SessionResumptionUpdate *struct {
		NewHandle string `json:"newHandle"`
		Resumable bool   `json:"resumable"`
	} `json:"sessionResumptionUpdate"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

type liveTranscription struct {
	Text string `json:"text"`
}

func (s *LiveService) onMessage(data []byte) {
	var msg liveInbound
	if err := json.Unmarshal(data, &msg); err != nil {
		logger.Warn("[GeminiLive] invalid message: %v", err)
		return
	}
	if msg.Error != nil {
		_ = s.PushError("Gemini Live API error", fmt.Errorf("%s", msg.Error.Message), false)
		return
	}
	if msg.SetupComplete != nil {
		s.socket.MarkReady()
		s.sendContext()
	}
	if u := msg.SessionResumptionUpdate; u != nil {
		s.mu.Lock()
		if u.Resumable && u.NewHandle != "" {
			s.resumeHandle = u.NewHandle
		}
		s.mu.Unlock()
	}
	if msg.UsageMetadata != nil {
		u := msg.UsageMetadata.toUsage()
		s.mu.Lock()
		s.usage = &u
		s.mu.Unlock()
	}
	if sc := msg.ServerContent; sc != nil {
		s.handleServerContent(sc.InputTranscription, sc.OutputTranscription, sc.Interrupted, sc.TurnComplete, func() {
			if sc.ModelTurn == nil {
				return
			}
			for _, part := range sc.ModelTurn.Parts {
				if part.InlineData != nil && part.InlineData.Data != "" {
					audio, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
					if err != nil {
						logger.Warn("[GeminiLive] invalid audio chunk: %v", err)
						continue
					}
					codec, rate := parseOutputAudioMIMEType(part.InlineData.MimeType)
					if rate == 0 {
						rate = s.cfg.OutputSampleRate
					}
					if codec == "" {
						codec = "linear16"
					}
					s.out.Audio(audio, rate, codec)
				} else if part.Text != "" && !part.Thought {
					s.out.Text(part.Text, false)
				}
			}
		})
	}
	if tc := msg.ToolCall; tc != nil && len(tc.FunctionCalls) > 0 {
		s.flushUser()
		calls := make([]llm.ToolCall, 0, len(tc.FunctionCalls))
		s.mu.Lock()
		c := s.llmCtx
		for _, fc := range tc.FunctionCalls {
			args := []byte(fc.Args)
			if len(args) == 0 || string(args) == "null" {
				args = []byte("{}")
			}
			s.openCalls[fc.ID] = fc.Name
			calls = append(calls, llm.ToolCall{ID: fc.ID, Name: fc.Name, Arguments: args})
		}
		s.mu.Unlock()
		s.fns.Dispatch(c, calls, func(o realtime.Outcome) {
			if o.Cancelled {
				s.mu.Lock()
				delete(s.openCalls, o.Call.ID)
				s.mu.Unlock()
				return
			}
			s.deliverResult(o.Call.ID, o.Call.Name, realtime.ResultString(o.Value, o.Err))
		})
	}
	if cc := msg.ToolCallCancellation; cc != nil {
		s.mu.Lock()
		for _, id := range cc.IDs {
			delete(s.openCalls, id)
		}
		s.mu.Unlock()
		s.fns.CancelIDs(cc.IDs)
	}
	if msg.GoAway != nil {
		s.mu.Lock()
		resumable := s.resumeHandle != ""
		s.mu.Unlock()
		if resumable {
			s.socket.Reconnect()
		}
	}
}

func (s *LiveService) handleServerContent(in, out *liveTranscription, interrupted, complete bool, model func()) {
	if in != nil && in.Text != "" {
		s.mu.Lock()
		s.userText.WriteString(in.Text)
		s.mu.Unlock()
	}
	s.mu.Lock()
	suppressed := s.suppress
	s.mu.Unlock()
	if !suppressed {
		hasOutput := out != nil && out.Text != ""
		if hasOutput {
			s.flushUser()
		}
		model()
		if hasOutput {
			s.out.Text(out.Text, true)
		}
	}
	if interrupted {
		s.flushUser()
		wasSpeaking := s.out.Speaking()
		s.out.Interrupt()
		s.mu.Lock()
		s.suppress = false
		s.mu.Unlock()
		if wasSpeaking && s.InterruptionsAllowed() {
			_ = s.BroadcastInterruption(s.Context())
		}
	}
	if complete {
		s.flushUser()
		s.mu.Lock()
		u := s.usage
		s.usage = nil
		s.suppress = false
		s.mu.Unlock()
		if !interrupted {
			s.out.End(u)
		}
	}
}

func (s *LiveService) flushUser() {
	s.mu.Lock()
	text := strings.TrimSpace(s.userText.String())
	s.userText.Reset()
	s.mu.Unlock()
	if text != "" {
		_ = s.PushFrame(frames.NewTranscriptionFrame(text, true), frames.Upstream)
	}
}

func parseOutputAudioMIMEType(mimeType string) (codec string, sampleRate int) {
	mimeType = strings.ToLower(strings.TrimSpace(mimeType))
	if mimeType == "" {
		return "", 0
	}
	switch {
	case strings.Contains(mimeType, "mulaw"), strings.Contains(mimeType, "mu-law"), strings.Contains(mimeType, "pcmu"):
		codec = "mulaw"
	case strings.Contains(mimeType, "alaw"), strings.Contains(mimeType, "pcma"):
		codec = "alaw"
	case strings.Contains(mimeType, "opus"):
		codec = "opus"
	case strings.Contains(mimeType, "mp3"):
		codec = "mp3"
	case strings.Contains(mimeType, "pcm"), strings.Contains(mimeType, "linear16"):
		codec = "linear16"
	}
	for _, part := range strings.Split(mimeType, ";") {
		part = strings.TrimSpace(part)
		for _, prefix := range []string{"rate=", "sample_rate="} {
			if value, ok := strings.CutPrefix(part, prefix); ok {
				if parsed, err := strconv.Atoi(strings.TrimSpace(value)); err == nil {
					sampleRate = parsed
				}
			}
		}
	}
	return codec, sampleRate
}
