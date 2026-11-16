package openai_realtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
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
	OpenAIRealtimeURL         = "wss://api.openai.com/v1/realtime"
	DefaultRealtimeModel      = "gpt-realtime-2.1"
	DefaultTranscriptionModel = RealtimeWhisperModel
	RealtimeWhisperModel      = "gpt-realtime-whisper"

	TurnDetectionServerVAD   = "server_vad"
	TurnDetectionSemanticVAD = "semantic_vad"

	DefaultInputSampleRate  = 16000
	DefaultOutputSampleRate = 24000
	DefaultOutputChannels   = 1
)

type LLMConfig struct {
	APIKey             string
	Model              string
	SystemInstruction  string
	Voice              string
	Tools              llm.Tools
	ToolChoice         llm.ToolChoice
	Endpoint           string
	InputAudioFormat   string
	OutputAudioFormat  string
	OutputSampleRate   int
	OutputChannels     int
	TranscriptionModel string
	DisableServerVAD   bool
	FunctionTimeout    time.Duration
	Backoff            realtime.Backoff
	Dialer             *websocket.Dialer

	TurnDetection            *TurnDetection
	NoiseReduction           string
	TranscriptionLanguage    string
	TranscriptionPrompt      string
	Speed                    float64
	ReasoningEffort          string
	Truncation               string
	TruncationRetentionRatio float64
}

type TurnDetection struct {
	Type              string
	Threshold         float64
	PrefixPaddingMs   int
	SilenceDurationMs int
	Eagerness         string
	CreateResponse    *bool
	InterruptResponse *bool
}

type audioItem struct {
	id           string
	contentIndex int
	start        time.Time
	bytes        int
}

type LLMService struct {
	*processors.BaseProcessor

	cfg    LLMConfig
	socket *realtime.Socket
	out    *realtime.Output
	fns    *realtime.Functions

	mu                sync.Mutex
	instructions      string
	tools             llm.Tools
	toolChoice        llm.ToolChoice
	llmCtx            *llm.Context
	conversationSetup bool
	runWhenReady      bool
	responseActive    bool
	responseOwed      bool
	resultsAwaiting   bool
	openCalls         map[string]string
	callNames         map[string]string
	current           *audioItem

	resampleBuf []byte
}

func NewLLMService(config LLMConfig) *LLMService {
	if config.Model == "" {
		config.Model = DefaultRealtimeModel
	}
	if config.Endpoint == "" {
		config.Endpoint = OpenAIRealtimeURL
	}
	config.InputAudioFormat = normalizeAudioFormat(config.InputAudioFormat)
	config.OutputAudioFormat = normalizeAudioFormat(config.OutputAudioFormat)
	if config.OutputSampleRate == 0 {
		config.OutputSampleRate = DefaultOutputSampleRate
		if config.OutputAudioFormat != "pcm16" {
			config.OutputSampleRate = 8000
		}
	}
	if config.OutputChannels == 0 {
		config.OutputChannels = DefaultOutputChannels
	}
	if config.TranscriptionModel == "" {
		config.TranscriptionModel = DefaultTranscriptionModel
	}
	if config.Dialer == nil {
		config.Dialer = websocket.DefaultDialer
	}
	s := &LLMService{
		cfg:          config,
		instructions: config.SystemInstruction,
		tools:        config.Tools,
		toolChoice:   config.ToolChoice,
		openCalls:    make(map[string]string),
		callNames:    make(map[string]string),
	}
	s.BaseProcessor = processors.NewBaseProcessor("OpenAIRealtime", s)
	s.out = realtime.NewOutput(s, config.Model, config.OutputChannels)
	s.fns = realtime.NewFunctions(s)
	s.fns.SetTimeout(config.FunctionTimeout)
	s.socket = realtime.NewSocket(realtime.SocketConfig{
		Name:         "OpenAIRealtime",
		Dial:         s.dial,
		OnOpen:       func(_ context.Context, w realtime.Writer) error { return w.WriteJSON(s.sessionUpdate()) },
		ReadyOnOpen:  true,
		OnMessage:    s.onMessage,
		OnDisconnect: s.onDisconnect,
		OnFatal: func(err error) {
			_ = s.PushError("OpenAI Realtime connection lost", err, true)
		},
		Backoff: config.Backoff,
	})
	return s
}

func NewOpenAIRealtimeLLMService(apiKey, model string) *LLMService {
	return NewLLMService(LLMConfig{APIKey: apiKey, Model: model})
}

func (s *LLMService) RegisterFunction(name string, handler llm.FunctionHandler, opts ...llm.FunctionOptions) {
	s.fns.Register(name, handler, opts...)
}

func (s *LLMService) UnregisterFunction(name string) { s.fns.Unregister(name) }

func (s *LLMService) HasFunction(name string) bool { return s.fns.Has(name) }

func (s *LLMService) Stop() error {
	s.shutdown()
	err := s.BaseProcessor.Stop()
	s.fns.Wait()
	return err
}

func (s *LLMService) shutdown() {
	s.fns.Cancel(false)
	s.socket.Stop()
	s.out.Interrupt()
}

func (s *LLMService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		s.socket.Start(s.Context())
		_ = s.PushFrame(frames.NewLLMServiceMetadataFrame(s.Name(), true), frames.Upstream)
	case *frames.ServiceSwitcherRequestMetadataFrame:
		if f.Service != nil && f.Service.Name() == s.Name() {
			_ = s.PushFrame(frames.NewLLMServiceMetadataFrame(s.Name(), true), frames.Upstream)
		}
	case *frames.EndFrame, *frames.CancelFrame:
		s.shutdown()
	case *frames.InterruptionFrame:
		s.handleInterruption()
	case *frames.UserStoppedSpeakingFrame:
		if s.cfg.DisableServerVAD {
			s.send(map[string]any{"type": "input_audio_buffer.commit"})
			s.createResponse()
		}
	case *frames.BotStoppedSpeakingFrame:
		s.mu.Lock()
		s.current = nil
		s.mu.Unlock()
	case *frames.AudioFrame:
		s.sendAudio(f)
		return nil
	case *frames.LLMContextFrame:
		if f.Context != nil {
			s.handleContext(f.Context, direction)
		}
		return nil
	case *frames.LLMMessagesAppendFrame:
		if items := messageItems(f.Messages); len(items) > 0 && s.socket.Ready() {
			for _, it := range items {
				s.send(map[string]any{"type": "conversation.item.create", "item": it})
			}
			if f.RunLLM {
				s.createResponse()
			}
		}
	case *frames.LLMSetToolsFrame:
		s.mu.Lock()
		s.tools = f.Tools
		s.mu.Unlock()
		s.send(s.sessionUpdate())
	case *frames.LLMSetToolChoiceFrame:
		s.mu.Lock()
		s.toolChoice = f.ToolChoice
		s.mu.Unlock()
		s.send(s.sessionUpdate())
	case *frames.LLMUpdateSettingsFrame:
		if f.Service == nil || f.Service.Name() == s.Name() {
			if f.Delta.SystemInstruction != "" {
				s.mu.Lock()
				s.instructions = f.Delta.SystemInstruction
				s.mu.Unlock()
				s.send(s.sessionUpdate())
			}
			return nil
		}
	case *frames.FunctionCallResultFrame:
		s.deliverResult(f.ToolCallID, realtime.ResultString(f.Result, nil), f.RunLLM)
	}
	return s.PushFrame(frame, direction)
}

func (s *LLMService) send(v any) {
	if err := s.socket.WriteJSON(v); err != nil && err != realtime.ErrNotConnected {
		logger.Warn("[OpenAIRealtime] send failed: %v", err)
	}
}

func (s *LLMService) dial(ctx context.Context) (*websocket.Conn, error) {
	u, err := url.Parse(s.cfg.Endpoint)
	if err != nil {
		return nil, fmt.Errorf("invalid OpenAI Realtime endpoint: %w", err)
	}
	q := u.Query()
	if q.Get("model") == "" {
		q.Set("model", s.cfg.Model)
	}
	u.RawQuery = q.Encode()
	h := http.Header{}
	if s.cfg.APIKey != "" {
		h.Set("Authorization", "Bearer "+s.cfg.APIKey)
	}
	conn, _, err := s.cfg.Dialer.DialContext(ctx, u.String(), h)
	if err != nil {
		return nil, fmt.Errorf("openai realtime dial: %w", err)
	}
	return conn, nil
}

func (s *LLMService) onDisconnect(error) {
	s.out.Interrupt()
	s.mu.Lock()
	s.conversationSetup = false
	s.responseActive = false
	s.responseOwed = false
	s.current = nil
	s.openCalls = make(map[string]string)
	s.mu.Unlock()
}

func (s *LLMService) sessionUpdate() map[string]any {
	s.mu.Lock()
	instructions, tools, choice := s.instructions, s.tools, s.toolChoice
	if instructions == "" && s.llmCtx != nil {
		instructions = leadingSystem(s.llmCtx.Snapshot().Messages)
	}
	if tools.Empty() && s.llmCtx != nil {
		tools = s.llmCtx.Tools()
	}
	s.mu.Unlock()

	input := map[string]any{
		"format":        audioFormatObject(s.cfg.InputAudioFormat),
		"transcription": s.transcriptionConfig(),
	}
	if s.cfg.DisableServerVAD {
		input["turn_detection"] = nil
	} else if td := s.cfg.TurnDetection; td != nil {
		input["turn_detection"] = td.wire()
	}
	if s.cfg.NoiseReduction != "" {
		input["noise_reduction"] = map[string]any{"type": s.cfg.NoiseReduction}
	}
	output := map[string]any{"format": audioFormatObject(s.cfg.OutputAudioFormat)}
	if s.cfg.Voice != "" {
		output["voice"] = s.cfg.Voice
	}
	if s.cfg.Speed != 0 {
		output["speed"] = s.cfg.Speed
	}
	session := map[string]any{
		"type":  "realtime",
		"model": s.cfg.Model,
		"audio": map[string]any{"input": input, "output": output},
	}
	if instructions != "" {
		session["instructions"] = instructions
	}
	if s.cfg.ReasoningEffort != "" && strings.Contains(s.cfg.Model, "gpt-realtime-2") {
		session["reasoning"] = map[string]any{"effort": s.cfg.ReasoningEffort}
	}
	if s.cfg.TruncationRetentionRatio > 0 {
		session["truncation"] = map[string]any{"type": "retention_ratio", "retention_ratio": s.cfg.TruncationRetentionRatio}
	} else if s.cfg.Truncation != "" {
		session["truncation"] = s.cfg.Truncation
	}
	if len(tools.Functions) > 0 {
		list := make([]map[string]any, 0, len(tools.Functions))
		for _, f := range tools.Functions {
			list = append(list, map[string]any{
				"type":        "function",
				"name":        f.Name,
				"description": f.Description,
				"parameters":  f.JSONSchema(),
			})
		}
		session["tools"] = list
		switch choice.Mode {
		case llm.ToolChoiceAuto, llm.ToolChoiceNone, llm.ToolChoiceRequired:
			session["tool_choice"] = string(choice.Mode)
		case llm.ToolChoiceFunction:
			session["tool_choice"] = map[string]any{"type": "function", "name": choice.Name}
		}
	}
	return map[string]any{"type": "session.update", "session": session}
}

func (s *LLMService) transcriptionConfig() map[string]any {
	t := map[string]any{"model": s.cfg.TranscriptionModel}
	if s.cfg.TranscriptionLanguage != "" {
		t["language"] = s.cfg.TranscriptionLanguage
	}
	if s.cfg.TranscriptionPrompt != "" && s.cfg.TranscriptionModel != RealtimeWhisperModel {
		t["prompt"] = s.cfg.TranscriptionPrompt
	}
	return t
}

func (td *TurnDetection) wire() map[string]any {
	if td.Type == TurnDetectionSemanticVAD {
		out := map[string]any{"type": TurnDetectionSemanticVAD}
		if td.Eagerness != "" {
			out["eagerness"] = td.Eagerness
		}
		if td.CreateResponse != nil {
			out["create_response"] = *td.CreateResponse
		}
		if td.InterruptResponse != nil {
			out["interrupt_response"] = *td.InterruptResponse
		}
		return out
	}
	out := map[string]any{"type": TurnDetectionServerVAD}
	if td.Threshold != 0 {
		out["threshold"] = td.Threshold
	}
	if td.PrefixPaddingMs != 0 {
		out["prefix_padding_ms"] = td.PrefixPaddingMs
	}
	if td.SilenceDurationMs != 0 {
		out["silence_duration_ms"] = td.SilenceDurationMs
	}
	return out
}

func normalizeAudioFormat(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "", "pcm16", "audio/pcm":
		return "pcm16"
	case "g711_ulaw", "audio/pcmu":
		return "g711_ulaw"
	case "g711_alaw", "audio/pcma":
		return "g711_alaw"
	}
	return strings.TrimSpace(format)
}

func audioFormatObject(format string) map[string]any {
	switch format {
	case "pcm16":
		return map[string]any{"type": "audio/pcm", "rate": DefaultRealtimeRateHz}
	case "g711_ulaw":
		return map[string]any{"type": "audio/pcmu"}
	case "g711_alaw":
		return map[string]any{"type": "audio/pcma"}
	}
	return map[string]any{"type": format}
}

func (s *LLMService) sendAudio(f *frames.AudioFrame) {
	if len(f.Data) == 0 || !s.socket.Ready() {
		return
	}
	data := f.Data
	if s.cfg.InputAudioFormat == "pcm16" && f.SampleRate != 0 && f.SampleRate != DefaultRealtimeRateHz {
		s.resampleBuf = realtime.AppendResamplePCM16(s.resampleBuf[:0], f.Data, f.SampleRate, DefaultRealtimeRateHz)
		data = s.resampleBuf
	}
	if err := s.socket.WriteBase64(`{"type":"input_audio_buffer.append","audio":"`, data, `"}`); err != nil && err != realtime.ErrNotConnected {
		logger.Debug("[OpenAIRealtime] audio send failed: %v", err)
	}
}

func (s *LLMService) handleContext(c *llm.Context, direction frames.FrameDirection) {
	s.mu.Lock()
	first := s.llmCtx == nil
	s.llmCtx = c
	awaiting := s.resultsAwaiting
	s.mu.Unlock()

	if first {
		if s.instructions == "" || s.tools.Empty() {
			s.send(s.sessionUpdate())
		}
		s.createResponse()
		return
	}
	for _, m := range c.Snapshot().Messages {
		for _, p := range m.Parts {
			if r, ok := p.(llm.ToolResultPart); ok {
				no := false
				s.deliverResult(r.ToolCallID, r.Content, &no)
			}
		}
	}
	s.mu.Lock()
	awaiting = awaiting || s.resultsAwaiting
	s.mu.Unlock()
	if direction == frames.Upstream && awaiting {
		s.createResponse()
	}
}

func (s *LLMService) createResponse() {
	if !s.socket.Ready() {
		s.mu.Lock()
		s.runWhenReady = true
		s.mu.Unlock()
		return
	}
	s.mu.Lock()
	if s.responseActive {
		s.responseOwed = true
		s.mu.Unlock()
		return
	}
	var items []map[string]any
	if !s.conversationSetup && s.llmCtx != nil {
		items = messageItems(s.llmCtx.Snapshot().Messages)
	}
	s.conversationSetup = true
	s.responseActive = true
	s.resultsAwaiting = false
	s.mu.Unlock()

	for _, it := range items {
		s.send(map[string]any{"type": "conversation.item.create", "item": it})
	}
	s.out.MarkRequest()
	s.out.BeginResponse()
	s.send(map[string]any{"type": "response.create", "response": map[string]any{"output_modalities": []string{"audio"}}})
}

func (s *LLMService) handleInterruption() {
	s.fns.Cancel(true)
	s.mu.Lock()
	active := s.responseActive
	s.responseOwed = false
	item := s.current
	s.current = nil
	s.mu.Unlock()
	if active {
		s.send(map[string]any{"type": "response.cancel"})
	}
	s.truncate(item)
	s.out.Interrupt()
}

func (s *LLMService) truncate(item *audioItem) {
	if item == nil || item.id == "" {
		return
	}
	bytesPerMs := s.cfg.OutputSampleRate * 2 / 1000
	if s.cfg.OutputAudioFormat != "pcm16" {
		bytesPerMs = s.cfg.OutputSampleRate / 1000
	}
	if bytesPerMs <= 0 {
		bytesPerMs = 1
	}
	audioMs := item.bytes / bytesPerMs
	elapsed := int(time.Since(item.start) / time.Millisecond)
	if elapsed < audioMs {
		audioMs = elapsed
	}
	s.send(map[string]any{
		"type":          "conversation.item.truncate",
		"item_id":       item.id,
		"content_index": item.contentIndex,
		"audio_end_ms":  audioMs,
	})
}

func (s *LLMService) deliverResult(id, output string, runLLM *bool) {
	s.mu.Lock()
	_, open := s.openCalls[id]
	if !open {
		s.mu.Unlock()
		return
	}
	delete(s.openCalls, id)
	s.resultsAwaiting = true
	remaining := len(s.openCalls)
	s.mu.Unlock()

	s.send(map[string]any{"type": "conversation.item.create", "item": map[string]any{
		"type":    "function_call_output",
		"call_id": id,
		"output":  output,
	}})
	if remaining == 0 && (runLLM == nil || *runLLM) {
		s.createResponse()
	}
}

type serverEvent struct {
	Type         string `json:"type"`
	Delta        string `json:"delta"`
	Transcript   string `json:"transcript"`
	ItemID       string `json:"item_id"`
	ContentIndex int    `json:"content_index"`
	CallID       string `json:"call_id"`
	Name         string `json:"name"`
	Arguments    string `json:"arguments"`
	Item         *struct {
		Type   string `json:"type"`
		CallID string `json:"call_id"`
		Name   string `json:"name"`
	} `json:"item"`
	Response *struct {
		Status        string          `json:"status"`
		StatusDetails json.RawMessage `json:"status_details"`
		Usage         *struct {
			InputTokens       int `json:"input_tokens"`
			OutputTokens      int `json:"output_tokens"`
			TotalTokens       int `json:"total_tokens"`
			InputTokenDetails *struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"input_token_details"`
		} `json:"usage"`
	} `json:"response"`
	Error *struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func (s *LLMService) onMessage(data []byte) {
	var ev serverEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		logger.Warn("[OpenAIRealtime] invalid event: %v", err)
		return
	}
	switch ev.Type {
	case "session.updated":
		s.mu.Lock()
		run := s.runWhenReady
		s.runWhenReady = false
		s.mu.Unlock()
		if run {
			s.createResponse()
		}
	case "response.created":
		s.mu.Lock()
		s.responseActive = true
		s.resultsAwaiting = false
		s.mu.Unlock()
		s.out.BeginResponse()
	case "response.output_audio.delta":
		if ev.Delta == "" {
			return
		}
		audio, err := base64.StdEncoding.DecodeString(ev.Delta)
		if err != nil {
			logger.Warn("[OpenAIRealtime] invalid audio delta: %v", err)
			return
		}
		s.mu.Lock()
		if s.current == nil || s.current.id != ev.ItemID {
			s.current = &audioItem{id: ev.ItemID, contentIndex: ev.ContentIndex, start: time.Now()}
		}
		s.current.bytes += len(audio)
		s.mu.Unlock()
		s.out.Audio(audio, s.cfg.OutputSampleRate, codecName(s.cfg.OutputAudioFormat))
	case "response.output_audio_transcript.delta":
		s.out.Text(ev.Delta, true)
	case "response.output_text.delta":
		s.out.Text(ev.Delta, false)
	case "conversation.item.added", "response.output_item.added":
		if ev.Item != nil && ev.Item.Type == "function_call" && ev.Item.CallID != "" {
			s.mu.Lock()
			s.callNames[ev.Item.CallID] = ev.Item.Name
			s.mu.Unlock()
		}
	case "conversation.item.input_audio_transcription.delta":
		if ev.Delta != "" {
			_ = s.PushFrame(frames.NewInterimTranscriptionFrame(ev.Delta, ""), frames.Upstream)
		}
	case "conversation.item.input_audio_transcription.completed":
		if t := strings.TrimSpace(ev.Transcript); t != "" {
			_ = s.PushFrame(frames.NewTranscriptionFrame(t, true), frames.Upstream)
		}
	case "input_audio_buffer.speech_started":
		s.mu.Lock()
		item := s.current
		s.current = nil
		s.responseOwed = false
		s.mu.Unlock()
		s.truncate(item)
		wasSpeaking := s.out.Speaking()
		s.out.Interrupt()
		s.broadcast(func() frames.Frame { return frames.NewUserStartedSpeakingFrame() })
		if wasSpeaking && s.InterruptionsAllowed() {
			_ = s.BroadcastInterruption(s.Context())
		}
	case "input_audio_buffer.speech_stopped":
		s.out.MarkRequest()
		s.broadcast(func() frames.Frame { return frames.NewUserStoppedSpeakingFrame() })
	case "response.function_call_arguments.done":
		s.handleFunctionCall(ev)
	case "response.done":
		s.handleResponseDone(ev)
	case "error":
		if ev.Error == nil {
			return
		}
		switch ev.Error.Code {
		case "response_cancel_not_active", "conversation_already_has_active_response":
			logger.Debug("[OpenAIRealtime] %s", ev.Error.Message)
			return
		}
		_ = s.PushError("OpenAI Realtime API error", fmt.Errorf("%s: %s", ev.Error.Code, ev.Error.Message), false)
	}
}

func (s *LLMService) broadcast(build func() frames.Frame) {
	if err := s.BroadcastFrame(nil, build); err != nil {
		logger.Debug("[OpenAIRealtime] broadcast failed: %v", err)
	}
}

func (s *LLMService) handleFunctionCall(ev serverEvent) {
	if ev.CallID == "" {
		return
	}
	s.mu.Lock()
	name := ev.Name
	if name == "" {
		name = s.callNames[ev.CallID]
	}
	delete(s.callNames, ev.CallID)
	if _, dup := s.openCalls[ev.CallID]; dup {
		s.mu.Unlock()
		return
	}
	s.openCalls[ev.CallID] = name
	c := s.llmCtx
	s.mu.Unlock()

	args := []byte(ev.Arguments)
	if strings.TrimSpace(ev.Arguments) == "" {
		args = []byte("{}")
	}
	s.fns.Dispatch(c, []llm.ToolCall{{ID: ev.CallID, Name: name, Arguments: args}}, func(o realtime.Outcome) {
		if o.Cancelled {
			s.deliverResult(o.Call.ID, llm.ToolResultInterrupted, boolPtr(false))
			return
		}
		s.deliverResult(o.Call.ID, realtime.ResultString(o.Value, o.Err), o.RunLLM)
	})
}

func (s *LLMService) handleResponseDone(ev serverEvent) {
	var usage *llm.Usage
	status := ""
	var details json.RawMessage
	if r := ev.Response; r != nil {
		status, details = r.Status, r.StatusDetails
		if u := r.Usage; u != nil {
			usage = &llm.Usage{PromptTokens: u.InputTokens, CompletionTokens: u.OutputTokens, TotalTokens: u.TotalTokens}
			if u.InputTokenDetails != nil {
				usage.CacheReadTokens = u.InputTokenDetails.CachedTokens
			}
		}
	}
	s.out.End(usage)
	s.mu.Lock()
	s.responseActive = false
	owed := s.responseOwed && s.resultsAwaiting
	s.responseOwed = false
	s.mu.Unlock()
	if status == "failed" {
		_ = s.PushError("OpenAI Realtime response failed", fmt.Errorf("%s", string(details)), false)
	}
	if owed {
		s.createResponse()
	}
}

func boolPtr(b bool) *bool { return &b }

func codecName(format string) string {
	switch format {
	case "g711_ulaw":
		return "mulaw"
	case "g711_alaw":
		return "alaw"
	}
	return "linear16"
}

func leadingSystem(msgs []llm.Message) string {
	var parts []string
	for _, m := range msgs {
		if m.Role != llm.RoleSystem && m.Role != llm.RoleDeveloper {
			break
		}
		if t := strings.TrimSpace(m.Text()); t != "" {
			parts = append(parts, t)
		}
	}
	return strings.Join(parts, "\n\n")
}

func messageItems(msgs []llm.Message) []map[string]any {
	var items []map[string]any
	leading := true
	for _, m := range msgs {
		if leading && (m.Role == llm.RoleSystem || m.Role == llm.RoleDeveloper) {
			continue
		}
		leading = false
		for _, c := range m.ToolCalls() {
			args := string(c.Arguments)
			if args == "" {
				args = "{}"
			}
			items = append(items, map[string]any{"type": "function_call", "call_id": c.ID, "name": c.Name, "arguments": args})
		}
		if r, ok := m.ToolResult(); ok {
			items = append(items, map[string]any{"type": "function_call_output", "call_id": r.ToolCallID, "output": r.Content})
			continue
		}
		text := m.Text()
		if strings.TrimSpace(text) == "" {
			continue
		}
		role, kind := "user", "input_text"
		switch m.Role {
		case llm.RoleAssistant:
			role, kind = "assistant", "output_text"
		case llm.RoleSystem, llm.RoleDeveloper:
			role = "system"
		}
		items = append(items, map[string]any{
			"type":    "message",
			"role":    role,
			"content": []map[string]any{{"type": kind, "text": text}},
		})
	}
	return items
}
