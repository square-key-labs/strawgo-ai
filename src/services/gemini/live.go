package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
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
	GeminiLiveURL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"

	DefaultLiveModel            = "gemini-2.0-flash-exp"
	DefaultLiveInputSampleRate  = 16000
	DefaultLiveOutputSampleRate = 24000
	DefaultLiveOutputChannels   = 1
)

type LiveConfig struct {
	APIKey            string
	Model             string
	SystemInstruction string
	Voice             string

	Endpoint string

	InputMIMEType string

	OutputSampleRate int
	OutputChannels   int

	// DisableVAD disables Gemini's server-side VAD so that an external VAD
	// (e.g. Silero) drives turn detection. When true, setupMessage emits
	// realtimeInputConfig.voiceActivityDetection.disabled = true.
	DisableVAD bool

	Dialer *websocket.Dialer
}

type LiveService struct {
	*processors.BaseProcessor

	apiKey            string
	model             string
	systemInstruction string
	voice             string
	endpoint          string
	inputMIMEType     string
	outputSampleRate  int
	outputChannels    int
	disableVAD        bool
	dialer            *websocket.Dialer

	ctx    context.Context
	cancel context.CancelFunc

	conn      *websocket.Conn
	connMu    sync.RWMutex
	connectMu sync.Mutex
	writeMu   sync.Mutex
	readWG    sync.WaitGroup

	stateMu          sync.Mutex
	isSpeaking       bool
	currentContextID string
	suppressAudio    bool
}

func NewLiveService(config LiveConfig) *LiveService {
	model := config.Model
	if model == "" {
		model = DefaultLiveModel
	}

	endpoint := config.Endpoint
	if endpoint == "" {
		endpoint = GeminiLiveURL
	}

	outputSampleRate := config.OutputSampleRate
	if outputSampleRate == 0 {
		outputSampleRate = DefaultLiveOutputSampleRate
	}

	outputChannels := config.OutputChannels
	if outputChannels == 0 {
		outputChannels = DefaultLiveOutputChannels
	}

	dialer := config.Dialer
	if dialer == nil {
		dialer = websocket.DefaultDialer
	}

	service := &LiveService{
		apiKey:            config.APIKey,
		model:             model,
		systemInstruction: config.SystemInstruction,
		voice:             config.Voice,
		endpoint:          endpoint,
		inputMIMEType:     config.InputMIMEType,
		outputSampleRate:  outputSampleRate,
		outputChannels:    outputChannels,
		disableVAD:        config.DisableVAD,
		dialer:            dialer,
	}

	service.BaseProcessor = processors.NewBaseProcessor("GeminiLive", service)
	return service
}

func NewGeminiLiveService(apiKey string, model string) *LiveService {
	return NewLiveService(LiveConfig{
		APIKey: apiKey,
		Model:  model,
	})
}

func (s *LiveService) Initialize(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}

	s.stateMu.Lock()
	if s.ctx == nil {
		s.ctx, s.cancel = context.WithCancel(ctx)
	}
	liveCtx := s.ctx
	s.stateMu.Unlock()

	return s.connect(liveCtx)
}

func (s *LiveService) Cleanup() error {
	s.stateMu.Lock()
	if s.cancel != nil {
		s.cancel()
		s.cancel = nil
	}
	s.ctx = nil
	s.suppressAudio = false
	s.stateMu.Unlock()

	s.stopSpeaking()
	s.disconnect()
	s.readWG.Wait()

	return nil
}

func (s *LiveService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		s.HandleStartFrame(f)

		if err := s.Initialize(ctx); err != nil {
			s.pushError(err)
		}

		return s.PushFrame(frame, direction)

	case *frames.EndFrame:
		if err := s.Cleanup(); err != nil {
			logger.Error("[GeminiLive] Cleanup failed: %v", err)
		}
		return s.PushFrame(frame, direction)

	case *frames.CancelFrame:
		if err := s.Cleanup(); err != nil {
			logger.Error("[GeminiLive] Cleanup failed: %v", err)
		}
		return s.PushFrame(frame, direction)

	case *frames.InterruptionFrame:
		if err := s.handleInterruption(); err != nil {
			s.pushError(err)
		}
		return s.PushFrame(frame, direction)

	case *frames.UserStartedSpeakingFrame:
		if err := s.handleInterruption(); err != nil {
			s.pushError(err)
		}
		return s.PushFrame(frame, direction)

	case *frames.UserStoppedSpeakingFrame:
		if err := s.handleUserTurnComplete(); err != nil {
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

func (s *LiveService) connect(ctx context.Context) error {
	s.connectMu.Lock()
	defer s.connectMu.Unlock()

	if s.getConn() != nil {
		return nil
	}

	wsURL, err := s.websocketURL()
	if err != nil {
		return err
	}

	conn, _, err := s.dialer.DialContext(ctx, wsURL, nil)
	if err != nil {
		return fmt.Errorf("failed to connect to Gemini Live: %w", err)
	}

	s.connMu.Lock()
	s.conn = conn
	s.connMu.Unlock()

	if err := s.writeJSON(s.setupMessage()); err != nil {
		s.disconnect()
		return fmt.Errorf("failed to send Gemini Live setup: %w", err)
	}

	s.readWG.Add(1)
	go s.readLoop(conn)

	logger.Info("[GeminiLive] Connected (model=%s)", s.model)
	return nil
}

func (s *LiveService) disconnect() {
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

func (s *LiveService) reconnect(ctx context.Context) error {
	logger.Warn("[GeminiLive] Reconnecting websocket")
	s.disconnect()
	return s.Initialize(ctx)
}

func (s *LiveService) websocketURL() (string, error) {
	u, err := url.Parse(s.endpoint)
	if err != nil {
		return "", fmt.Errorf("invalid Gemini Live endpoint: %w", err)
	}

	if s.apiKey != "" {
		q := u.Query()
		q.Set("key", s.apiKey)
		u.RawQuery = q.Encode()
	}

	return u.String(), nil
}

func (s *LiveService) setupMessage() map[string]interface{} {
	setup := map[string]interface{}{
		"model": modelPath(s.model),
		"generationConfig": map[string]interface{}{
			"responseModalities": []string{"AUDIO"},
		},
		"inputAudioTranscription":  map[string]interface{}{},
		"outputAudioTranscription": map[string]interface{}{},
	}

	if s.systemInstruction != "" {
		setup["systemInstruction"] = map[string]interface{}{
			"parts": []map[string]string{{"text": s.systemInstruction}},
		}
	}

	if s.voice != "" {
		generationConfig := setup["generationConfig"].(map[string]interface{})
		generationConfig["speechConfig"] = map[string]interface{}{
			"voiceConfig": map[string]interface{}{
				"prebuiltVoiceConfig": map[string]string{
					"voiceName": s.voice,
				},
			},
		}
	}

	// When external VAD drives turn detection, disable Gemini's server-side VAD
	// to prevent double-VAD conflicts (premature turn completions, duplicate interruptions).
	if s.disableVAD {
		setup["realtimeInputConfig"] = map[string]interface{}{
			"voiceActivityDetection": map[string]interface{}{
				"disabled": true,
			},
		}
	}

	return map[string]interface{}{"setup": setup}
}

func (s *LiveService) handleAudioFrame(ctx context.Context, frame *frames.AudioFrame) error {
	if frame == nil || len(frame.Data) == 0 {
		return nil
	}

	if s.getConn() == nil {
		if err := s.Initialize(ctx); err != nil {
			return err
		}
	}

	payload := map[string]interface{}{
		"realtimeInput": map[string]interface{}{
			"mediaChunks": []map[string]interface{}{
				{
					"mimeType": s.inputAudioMIMEType(frame),
					"data":     base64.StdEncoding.EncodeToString(frame.Data),
				},
			},
		},
	}

	if err := s.writeJSON(payload); err != nil {
		if reconnectErr := s.reconnect(ctx); reconnectErr != nil {
			return fmt.Errorf("failed to send input audio: %w", err)
		}
		if retryErr := s.writeJSON(payload); retryErr != nil {
			return fmt.Errorf("failed to send input audio after reconnect: %w", retryErr)
		}
	}

	return nil
}

func (s *LiveService) handleInterruption() error {
	s.setAudioSuppression(true)
	s.stopSpeaking()

	if s.getConn() == nil {
		return nil
	}

	if err := s.sendTurnCompleteSignal(); err != nil {
		return fmt.Errorf("failed to send interruption signal: %w", err)
	}

	logger.Debug("[GeminiLive] Interruption signal sent")
	return nil
}

func (s *LiveService) handleUserTurnComplete() error {
	s.setAudioSuppression(false)

	if s.getConn() == nil {
		return nil
	}

	if err := s.sendTurnCompleteSignal(); err != nil {
		return fmt.Errorf("failed to send turn-complete signal: %w", err)
	}

	return nil
}

func (s *LiveService) sendTurnCompleteSignal() error {
	return s.writeJSON(map[string]interface{}{
		"clientContent": map[string]interface{}{
			"turnComplete": true,
		},
	})
}

func (s *LiveService) readLoop(conn *websocket.Conn) {
	defer s.readWG.Done()
	defer s.clearConnection(conn)

	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			if s.isExpectedReadClose(err) {
				logger.Debug("[GeminiLive] Connection closed")
				return
			}

			s.pushError(fmt.Errorf("Gemini Live read error: %w", err))
			return
		}

		if err := s.handleServerMessage(message); err != nil {
			logger.Warn("[GeminiLive] Failed to process message: %v", err)
		}
	}
}

func (s *LiveService) handleServerMessage(message []byte) error {
	var payload liveInboundMessage
	if err := json.Unmarshal(message, &payload); err != nil {
		return fmt.Errorf("invalid Gemini Live payload: %w", err)
	}

	if payload.Error != nil {
		errMessage := payload.Error.Message
		if errMessage == "" {
			errMessage = "unknown Gemini Live error"
		}
		s.pushError(fmt.Errorf("Gemini Live API error: %s", errMessage))
		return nil
	}

	if payload.ServerContent != nil {
		s.handleServerContent(payload.ServerContent)
	}

	return nil
}

func (s *LiveService) handleServerContent(content *liveServerContent) {
	if content == nil {
		return
	}

	if text, final := content.inputTranscript(); text != "" {
		frame := frames.NewTranscriptionFrame(text, final)
		s.pushFrameSafe(frame, frames.Downstream)
	}

	if text := content.outputTranscript(); text != "" {
		s.setAudioSuppression(false)
		s.pushFrameSafe(frames.NewLLMTextFrame(text), frames.Downstream)
	}

	if content.ModelTurn != nil {
		for _, part := range content.ModelTurn.Parts {
			if part.Text != "" {
				s.setAudioSuppression(false)
				s.pushFrameSafe(frames.NewLLMTextFrame(part.Text), frames.Downstream)
			}

			if part.InlineData == nil || part.InlineData.Data == "" {
				continue
			}

			audioData, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
			if err != nil {
				logger.Warn("[GeminiLive] Invalid base64 audio chunk: %v", err)
				continue
			}

			s.emitTTSAudio(audioData, part.InlineData.MimeType)
		}
	}

	if content.Interrupted {
		s.setAudioSuppression(true)
		s.stopSpeaking()
		return
	}

	if content.TurnComplete {
		s.setAudioSuppression(false)
		s.stopSpeaking()
	}
}

func (s *LiveService) emitTTSAudio(audio []byte, mimeType string) {
	if len(audio) == 0 {
		return
	}

	contextID, firstChunk, suppressed := s.beginSpeech()
	if suppressed {
		return
	}

	if firstChunk {
		startFrame := frames.NewTTSStartedFrameWithContext(contextID)
		startFrame.SetMetadata("context_id", contextID)
		s.pushFrameSafe(startFrame, frames.Upstream)
		s.pushFrameSafe(startFrame, frames.Downstream)
	}

	codec, sampleRate := parseOutputAudioMIMEType(mimeType)
	if sampleRate == 0 {
		sampleRate = s.outputSampleRate
	}
	if codec == "" {
		codec = "linear16"
	}

	audioFrame := frames.NewTTSAudioFrame(audio, sampleRate, s.outputChannels)
	audioFrame.ContextID = contextID
	audioFrame.SetMetadata("context_id", contextID)
	audioFrame.SetMetadata("codec", codec)

	s.pushFrameSafe(audioFrame, frames.Downstream)
}

func (s *LiveService) beginSpeech() (contextID string, firstChunk bool, suppressed bool) {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()

	if s.suppressAudio {
		return "", false, true
	}

	if s.currentContextID == "" {
		s.currentContextID = services.GenerateContextID()
	}

	firstChunk = !s.isSpeaking
	if firstChunk {
		s.isSpeaking = true
	}

	return s.currentContextID, firstChunk, false
}

func (s *LiveService) stopSpeaking() {
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

	stopFrame := frames.NewTTSStoppedFrame()
	stopFrame.ContextID = contextID
	stopFrame.SetMetadata("context_id", contextID)

	s.pushFrameSafe(stopFrame, frames.Upstream)
	s.pushFrameSafe(stopFrame, frames.Downstream)
}

func (s *LiveService) setAudioSuppression(suppressed bool) {
	s.stateMu.Lock()
	s.suppressAudio = suppressed
	s.stateMu.Unlock()
}

func (s *LiveService) inputAudioMIMEType(frame *frames.AudioFrame) string {
	if s.inputMIMEType != "" {
		return s.inputMIMEType
	}

	sampleRate := frame.SampleRate
	if sampleRate == 0 {
		sampleRate = DefaultLiveInputSampleRate
	}

	codec := "linear16"
	if meta := frame.Metadata(); meta != nil {
		if rawCodec, ok := meta["codec"].(string); ok {
			switch strings.ToLower(strings.TrimSpace(rawCodec)) {
			case "mulaw", "ulaw", "pcmu":
				codec = "mulaw"
			case "alaw", "pcma":
				codec = "alaw"
			}
		}
	}

	if codec == "linear16" {
		return fmt.Sprintf("audio/pcm;rate=%d", sampleRate)
	}

	return fmt.Sprintf("audio/pcm;encoding=%s;rate=%d", codec, sampleRate)
}

func (s *LiveService) writeJSON(payload interface{}) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	s.connMu.RLock()
	conn := s.conn
	s.connMu.RUnlock()

	if conn == nil {
		return fmt.Errorf("Gemini Live websocket is not connected")
	}

	return conn.WriteJSON(payload)
}

func (s *LiveService) getConn() *websocket.Conn {
	s.connMu.RLock()
	defer s.connMu.RUnlock()
	return s.conn
}

func (s *LiveService) clearConnection(conn *websocket.Conn) {
	s.connMu.Lock()
	if s.conn == conn {
		s.conn = nil
	}
	s.connMu.Unlock()
}

func (s *LiveService) isExpectedReadClose(err error) bool {
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

func (s *LiveService) pushFrameSafe(frame frames.Frame, direction frames.FrameDirection) {
	if err := s.PushFrame(frame, direction); err != nil {
		logger.Error("[GeminiLive] Failed to push %s frame (%s): %v", frame.Name(), direction.String(), err)
	}
}

func (s *LiveService) pushError(err error) {
	if err == nil {
		return
	}

	logger.Error("[GeminiLive] %v", err)
	s.pushFrameSafe(frames.NewErrorFrame(err), frames.Upstream)
}

func modelPath(model string) string {
	if strings.HasPrefix(model, "models/") {
		return model
	}
	return "models/" + model
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
		if part == "" {
			continue
		}

		if value, ok := strings.CutPrefix(part, "rate="); ok {
			if parsed, err := strconv.Atoi(strings.TrimSpace(value)); err == nil {
				sampleRate = parsed
			}
			continue
		}

		if value, ok := strings.CutPrefix(part, "sample_rate="); ok {
			if parsed, err := strconv.Atoi(strings.TrimSpace(value)); err == nil {
				sampleRate = parsed
			}
		}
	}

	return codec, sampleRate
}

type liveInboundMessage struct {
	ServerContent *liveServerContent `json:"serverContent,omitempty"`
	Error         *liveErrorPayload  `json:"error,omitempty"`
}

type liveErrorPayload struct {
	Message string `json:"message,omitempty"`
}

type liveServerContent struct {
	ModelTurn           *liveModelTurn     `json:"modelTurn,omitempty"`
	InputTranscription  *liveTranscription `json:"inputTranscription,omitempty"`
	OutputTranscription *liveTranscription `json:"outputTranscription,omitempty"`
	TurnComplete        bool               `json:"turnComplete,omitempty"`
	Interrupted         bool               `json:"interrupted,omitempty"`
}

type liveModelTurn struct {
	Parts []livePart `json:"parts,omitempty"`
}

type livePart struct {
	Text       string          `json:"text,omitempty"`
	InlineData *liveInlineData `json:"inlineData,omitempty"`
}

type liveInlineData struct {
	MimeType string `json:"mimeType,omitempty"`
	Data     string `json:"data,omitempty"`
}

type liveTranscription struct {
	Text       string `json:"text,omitempty"`
	Transcript string `json:"transcript,omitempty"`
	IsFinal    *bool  `json:"isFinal,omitempty"`
	Final      *bool  `json:"final,omitempty"`
}

func (c *liveServerContent) inputTranscript() (text string, final bool) {
	if c == nil || c.InputTranscription == nil {
		return "", false
	}

	text = c.InputTranscription.text()
	if text == "" {
		return "", false
	}

	return text, c.InputTranscription.isFinal()
}

func (c *liveServerContent) outputTranscript() string {
	if c == nil || c.OutputTranscription == nil {
		return ""
	}

	return c.OutputTranscription.text()
}

func (t *liveTranscription) text() string {
	if t == nil {
		return ""
	}

	if t.Text != "" {
		return t.Text
	}

	return t.Transcript
}

func (t *liveTranscription) isFinal() bool {
	if t == nil {
		return true
	}

	if t.IsFinal != nil {
		return *t.IsFinal
	}

	if t.Final != nil {
		return *t.Final
	}

	return true
}
