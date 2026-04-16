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
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

const (
	OpenAIRealtimeURL    = "wss://api.openai.com/v1/realtime"
	DefaultRealtimeModel = "gpt-4o-realtime-preview"

	DefaultInputSampleRate  = 16000
	DefaultOutputSampleRate = 24000
	DefaultOutputChannels   = 1
)

type LLMConfig struct {
	APIKey            string
	Model             string
	SystemInstruction string
	Voice             string
	Tools             []services.Tool
	ToolChoice        interface{}

	Endpoint string

	InputAudioFormat  string
	OutputAudioFormat string
	OutputSampleRate  int
	OutputChannels    int

	Dialer *websocket.Dialer
}

type LLMService struct {
	*processors.BaseProcessor

	apiKey            string
	model             string
	systemInstruction string
	voice             string
	tools             []services.Tool
	toolChoice        interface{}
	endpoint          string
	inputAudioFormat  string
	outputAudioFormat string
	outputSampleRate  int
	outputChannels    int
	dialer            *websocket.Dialer

	ctx    context.Context
	cancel context.CancelFunc

	conn      *websocket.Conn
	connMu    sync.RWMutex
	connectMu sync.Mutex
	writeMu   sync.Mutex
	readWG    sync.WaitGroup

	stateMu              sync.Mutex
	isSpeaking           bool
	currentContextID     string
	functionCallBuilders map[string]*functionCallBuilder
}

type functionCallBuilder struct {
	name string
	args strings.Builder
}

func NewLLMService(config LLMConfig) *LLMService {
	model := config.Model
	if model == "" {
		model = DefaultRealtimeModel
	}

	endpoint := config.Endpoint
	if endpoint == "" {
		endpoint = OpenAIRealtimeURL
	}

	inputAudioFormat := strings.TrimSpace(config.InputAudioFormat)
	if inputAudioFormat == "" {
		inputAudioFormat = "pcm16"
	}

	outputAudioFormat := strings.TrimSpace(config.OutputAudioFormat)
	if outputAudioFormat == "" {
		outputAudioFormat = "pcm16"
	}

	outputSampleRate := config.OutputSampleRate
	if outputSampleRate == 0 {
		outputSampleRate = DefaultOutputSampleRate
	}

	outputChannels := config.OutputChannels
	if outputChannels == 0 {
		outputChannels = DefaultOutputChannels
	}

	dialer := config.Dialer
	if dialer == nil {
		dialer = websocket.DefaultDialer
	}

	s := &LLMService{
		apiKey:            config.APIKey,
		model:             model,
		systemInstruction: config.SystemInstruction,
		voice:             config.Voice,
		tools:             config.Tools,
		toolChoice:        config.ToolChoice,
		endpoint:          endpoint,
		inputAudioFormat:  inputAudioFormat,
		outputAudioFormat: outputAudioFormat,
		outputSampleRate:  outputSampleRate,
		outputChannels:    outputChannels,
		dialer:            dialer,

		functionCallBuilders: make(map[string]*functionCallBuilder),
	}
	s.BaseProcessor = processors.NewBaseProcessor("OpenAIRealtime", s)
	return s
}

func NewOpenAIRealtimeLLMService(apiKey, model string) *LLMService {
	return NewLLMService(LLMConfig{APIKey: apiKey, Model: model})
}

func (s *LLMService) SetModel(model string) {
	if strings.TrimSpace(model) == "" {
		return
	}
	s.model = model
}

func (s *LLMService) SetSystemPrompt(prompt string) {
	s.systemInstruction = prompt
}

func (s *LLMService) SetTemperature(temp float64) {
	_ = temp
}

func (s *LLMService) AddMessage(role, content string) {
	_ = role
	_ = content
}

func (s *LLMService) ClearContext() {}

func (s *LLMService) Initialize(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}

	s.stateMu.Lock()
	if s.ctx == nil {
		s.ctx, s.cancel = context.WithCancel(ctx)
	}
	rtCtx := s.ctx
	s.stateMu.Unlock()

	return s.connect(rtCtx)
}

func (s *LLMService) Cleanup() error {
	s.stateMu.Lock()
	if s.cancel != nil {
		s.cancel()
		s.cancel = nil
	}
	s.ctx = nil
	s.isSpeaking = false
	s.currentContextID = ""
	s.functionCallBuilders = make(map[string]*functionCallBuilder)
	s.stateMu.Unlock()

	s.disconnect()
	s.readWG.Wait()
	return nil
}

func (s *LLMService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		s.HandleStartFrame(f)
		if err := s.Initialize(ctx); err != nil {
			s.pushError(err)
		}
		return s.PushFrame(frame, direction)

	case *frames.EndFrame:
		if err := s.Cleanup(); err != nil {
			logger.Error("[OpenAIRealtime] Cleanup failed: %v", err)
		}
		return s.PushFrame(frame, direction)

	case *frames.CancelFrame:
		if err := s.Cleanup(); err != nil {
			logger.Error("[OpenAIRealtime] Cleanup failed: %v", err)
		}
		return s.PushFrame(frame, direction)

	case *frames.InterruptionFrame:
		if err := s.handleInterruption(ctx); err != nil {
			s.pushError(err)
		}
		return s.PushFrame(frame, direction)

	case *frames.AudioFrame:
		if err := s.handleAudioFrame(ctx, f); err != nil {
			s.pushError(err)
			return err
		}
		return nil

	default:
		return s.PushFrame(frame, direction)
	}
}

func (s *LLMService) connect(ctx context.Context) error {
	s.connectMu.Lock()
	defer s.connectMu.Unlock()

	if s.getConn() != nil {
		return nil
	}

	wsURL, err := s.websocketURL()
	if err != nil {
		return err
	}

	headers := http.Header{}
	if s.apiKey != "" {
		headers.Set("Authorization", "Bearer "+s.apiKey)
	}
	headers.Set("OpenAI-Beta", "realtime=v1")

	conn, _, err := s.dialer.DialContext(ctx, wsURL, headers)
	if err != nil {
		return fmt.Errorf("failed to connect to OpenAI Realtime: %w", err)
	}

	s.connMu.Lock()
	s.conn = conn
	s.connMu.Unlock()

	if err := s.writeJSON(s.sessionUpdateMessage()); err != nil {
		s.disconnect()
		return fmt.Errorf("failed to send OpenAI Realtime session.update: %w", err)
	}

	s.readWG.Add(1)
	go s.readLoop(conn)

	logger.Info("[OpenAIRealtime] Connected (model=%s)", s.model)
	return nil
}

func (s *LLMService) disconnect() {
	s.connectMu.Lock()
	defer s.connectMu.Unlock()

	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	s.connMu.Lock()
	conn := s.conn
	s.conn = nil
	s.connMu.Unlock()

	if conn == nil {
		return
	}

	_ = conn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseNormalClosure, "shutdown"),
		time.Now().Add(100*time.Millisecond),
	)
	_ = conn.Close()
}

func (s *LLMService) websocketURL() (string, error) {
	u, err := url.Parse(s.endpoint)
	if err != nil {
		return "", fmt.Errorf("invalid OpenAI Realtime endpoint: %w", err)
	}

	q := u.Query()
	if q.Get("model") == "" && s.model != "" {
		q.Set("model", s.model)
	}
	u.RawQuery = q.Encode()

	return u.String(), nil
}

func (s *LLMService) sessionUpdateMessage() map[string]interface{} {
	session := map[string]interface{}{
		"model":               s.model,
		"modalities":          []string{"audio", "text"},
		"input_audio_format":  s.inputAudioFormat,
		"output_audio_format": s.outputAudioFormat,
	}

	if s.systemInstruction != "" {
		session["instructions"] = s.systemInstruction
	}

	if s.voice != "" {
		session["voice"] = s.voice
	}

	if len(s.tools) > 0 {
		session["tools"] = toolsToSessionTools(s.tools)
		if s.toolChoice != nil {
			session["tool_choice"] = s.toolChoice
		}
	}

	return map[string]interface{}{
		"type":    "session.update",
		"session": session,
	}
}

func toolsToSessionTools(tools []services.Tool) []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(tools))
	for _, tool := range tools {
		if strings.TrimSpace(tool.Type) != "function" {
			continue
		}
		result = append(result, map[string]interface{}{
			"type":        "function",
			"name":        tool.Function.Name,
			"description": tool.Function.Description,
			"parameters":  tool.Function.Parameters,
		})
	}
	return result
}

func (s *LLMService) handleAudioFrame(ctx context.Context, frame *frames.AudioFrame) error {
	if frame == nil || len(frame.Data) == 0 {
		return nil
	}

	if s.getConn() == nil {
		if err := s.Initialize(ctx); err != nil {
			return err
		}
	}

	payload := map[string]interface{}{
		"type":  "input_audio_buffer.append",
		"audio": base64.StdEncoding.EncodeToString(frame.Data),
	}

	if err := s.writeJSON(payload); err != nil {
		return fmt.Errorf("failed to send input audio: %w", err)
	}

	return nil
}

func (s *LLMService) handleInterruption(ctx context.Context) error {
	s.stopSpeaking()

	if s.getConn() == nil {
		return s.BroadcastInterruption(ctx)
	}

	if err := s.writeJSON(map[string]interface{}{"type": "response.cancel"}); err != nil {
		return fmt.Errorf("failed to send response.cancel: %w", err)
	}

	logger.Debug("[OpenAIRealtime] Sent response.cancel")
	return s.BroadcastInterruption(ctx)
}

func (s *LLMService) readLoop(conn *websocket.Conn) {
	defer s.readWG.Done()
	defer s.clearConnection(conn)

	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			if s.isExpectedReadClose(err) {
				logger.Debug("[OpenAIRealtime] Connection closed")
				return
			}
			s.pushError(fmt.Errorf("OpenAI Realtime read error: %w", err))
			return
		}

		if err := s.handleServerMessage(message); err != nil {
			logger.Warn("[OpenAIRealtime] Failed to process message: %v", err)
		}
	}
}

func (s *LLMService) handleServerMessage(message []byte) error {
	var event realtimeLLMEvent
	if err := json.Unmarshal(message, &event); err != nil {
		return fmt.Errorf("invalid OpenAI Realtime payload: %w", err)
	}

	if event.Type == "" {
		return nil
	}

	switch event.Type {
	case "conversation.item.input_audio_transcription.completed":
		if strings.TrimSpace(event.Transcript) != "" {
			s.pushFrameSafe(frames.NewTranscriptionFrame(event.Transcript, true), frames.Downstream)
		}

	case "response.text.delta":
		if strings.TrimSpace(event.Delta) != "" {
			s.pushFrameSafe(frames.NewLLMTextFrame(event.Delta), frames.Downstream)
		}

	case "response.audio_transcript.delta":
		if strings.TrimSpace(event.Delta) != "" {
			s.pushFrameSafe(frames.NewLLMTextFrame(event.Delta), frames.Downstream)
		}

	case "response.audio.delta":
		if strings.TrimSpace(event.Delta) == "" {
			return nil
		}
		audioData, err := base64.StdEncoding.DecodeString(event.Delta)
		if err != nil {
			return fmt.Errorf("invalid base64 response.audio.delta: %w", err)
		}
		s.emitTTSAudio(audioData)

	case "response.done":
		s.stopSpeaking()

	case "response.function_call_arguments.delta":
		s.pushFunctionCallFrame(event, false)

	case "response.function_call_arguments.done":
		s.pushFunctionCallFrame(event, true)

	case "error":
		if event.Error == nil || strings.TrimSpace(event.Error.Message) == "" {
			s.pushError(fmt.Errorf("OpenAI Realtime API error"))
			return nil
		}
		s.pushError(fmt.Errorf("OpenAI Realtime API error: %s", event.Error.Message))
	}

	return nil
}

func (s *LLMService) emitTTSAudio(audio []byte) {
	if len(audio) == 0 {
		return
	}

	contextID, firstChunk := s.beginSpeech()
	if firstChunk {
		started := frames.NewTTSStartedFrameWithContext(contextID)
		started.SetMetadata("context_id", contextID)
		s.pushFrameSafe(started, frames.Upstream)
		s.pushFrameSafe(started, frames.Downstream)
	}

	audioFrame := frames.NewTTSAudioFrame(audio, s.outputSampleRate, s.outputChannels)
	audioFrame.ContextID = contextID
	audioFrame.SetMetadata("context_id", contextID)
	audioFrame.SetMetadata("codec", s.outputAudioFormat)
	s.pushFrameSafe(audioFrame, frames.Downstream)
}

func (s *LLMService) beginSpeech() (contextID string, firstChunk bool) {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()

	if s.currentContextID == "" {
		s.currentContextID = services.GenerateContextID()
	}

	firstChunk = !s.isSpeaking
	if firstChunk {
		s.isSpeaking = true
	}

	return s.currentContextID, firstChunk
}

func (s *LLMService) stopSpeaking() {
	s.stateMu.Lock()
	if !s.isSpeaking {
		s.currentContextID = ""
		s.stateMu.Unlock()
		return
	}

	contextID := s.currentContextID
	s.isSpeaking = false
	s.currentContextID = ""
	s.stateMu.Unlock()

	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = contextID
	stopped.SetMetadata("context_id", contextID)
	s.pushFrameSafe(stopped, frames.Upstream)
	s.pushFrameSafe(stopped, frames.Downstream)
}

func (s *LLMService) pushFunctionCallFrame(event realtimeLLMEvent, done bool) {
	toolCallID := strings.TrimSpace(event.CallID)
	if toolCallID == "" {
		toolCallID = strings.TrimSpace(event.ItemID)
	}
	if toolCallID == "" {
		return
	}

	name := strings.TrimSpace(event.Name)
	if name == "" {
		name = strings.TrimSpace(event.FunctionName)
	}

	var argsJSON string
	s.stateMu.Lock()
	b, ok := s.functionCallBuilders[toolCallID]
	if !ok {
		b = &functionCallBuilder{name: name}
		s.functionCallBuilders[toolCallID] = b
	}
	if b.name == "" && name != "" {
		b.name = name
	}
	if event.Delta != "" {
		b.args.WriteString(event.Delta)
	}
	if event.Arguments != "" {
		b.args.Reset()
		b.args.WriteString(event.Arguments)
	}
	argsJSON = b.args.String()
	if done {
		delete(s.functionCallBuilders, toolCallID)
	}
	s.stateMu.Unlock()

	args := map[string]interface{}{}
	if strings.TrimSpace(argsJSON) != "" {
		if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
			args = map[string]interface{}{}
		}
	}

	fnName := name
	if fnName == "" {
		fnName = "function"
	}

	s.pushFrameSafe(frames.NewFunctionCallInProgressFrame(toolCallID, fnName, args, true), frames.Downstream)
}

func (s *LLMService) writeJSON(payload interface{}) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	s.connMu.RLock()
	conn := s.conn
	s.connMu.RUnlock()

	if conn == nil {
		return fmt.Errorf("OpenAI Realtime websocket is not connected")
	}

	return conn.WriteJSON(payload)
}

func (s *LLMService) getConn() *websocket.Conn {
	s.connMu.RLock()
	defer s.connMu.RUnlock()
	return s.conn
}

func (s *LLMService) clearConnection(conn *websocket.Conn) {
	s.connMu.Lock()
	if s.conn == conn {
		s.conn = nil
	}
	s.connMu.Unlock()
}

func (s *LLMService) isExpectedReadClose(err error) bool {
	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		return true
	}

	if strings.Contains(err.Error(), "use of closed network connection") {
		return true
	}

	s.stateMu.Lock()
	ctx := s.ctx
	s.stateMu.Unlock()

	if ctx != nil {
		select {
		case <-ctx.Done():
			return true
		default:
		}
	}

	return false
}

func (s *LLMService) pushFrameSafe(frame frames.Frame, direction frames.FrameDirection) {
	if err := s.PushFrame(frame, direction); err != nil {
		logger.Error("[OpenAIRealtime] Failed to push %s frame (%s): %v", frame.Name(), direction.String(), err)
	}
}

func (s *LLMService) pushError(err error) {
	if err == nil {
		return
	}

	logger.Error("[OpenAIRealtime] %v", err)
	s.pushFrameSafe(frames.NewErrorFrame(err), frames.Upstream)
}

type realtimeLLMEvent struct {
	Type         string                `json:"type"`
	Delta        string                `json:"delta,omitempty"`
	Transcript   string                `json:"transcript,omitempty"`
	ItemID       string                `json:"item_id,omitempty"`
	CallID       string                `json:"call_id,omitempty"`
	Name         string                `json:"name,omitempty"`
	FunctionName string                `json:"function_name,omitempty"`
	Arguments    string                `json:"arguments,omitempty"`
	Error        *realtimeLLMErrorBody `json:"error,omitempty"`
}

type realtimeLLMErrorBody struct {
	Message string `json:"message,omitempty"`
}
