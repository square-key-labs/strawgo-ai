package openai_realtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	DefaultSTTEndpoint    = "wss://api.openai.com/v1/realtime?intent=transcription"
	DefaultSTTModel       = "gpt-4o-transcribe"
	DefaultInputRateHz    = 16000
	DefaultRealtimeRateHz = 24000
)

type STTConfig struct {
	APIKey        string
	Model         string
	Endpoint      string
	TurnDetection string
	Dialer        *websocket.Dialer
}

type STTService struct {
	*processors.BaseProcessor

	apiKey        string
	model         string
	endpoint      string
	turnDetection string
	dialer        *websocket.Dialer

	stateMu sync.Mutex
	ctx     context.Context
	cancel  context.CancelFunc

	conn      *websocket.Conn
	connMu    sync.RWMutex
	connectMu sync.Mutex
	writeMu   sync.Mutex
	readWG    sync.WaitGroup
}

func NewSTTService(config STTConfig) *STTService {
	model := strings.TrimSpace(config.Model)
	if model == "" {
		model = DefaultSTTModel
	}

	endpoint := strings.TrimSpace(config.Endpoint)
	if endpoint == "" {
		endpoint = DefaultSTTEndpoint
	}

	dialer := config.Dialer
	if dialer == nil {
		dialer = websocket.DefaultDialer
	}

	s := &STTService{
		apiKey:        config.APIKey,
		model:         model,
		endpoint:      endpoint,
		turnDetection: strings.TrimSpace(config.TurnDetection),
		dialer:        dialer,
	}

	s.BaseProcessor = processors.NewBaseProcessor("OpenAIRealtimeSTT", s)
	return s
}

func (s *STTService) SetLanguage(lang string) {}

func (s *STTService) SetModel(model string) {
	if strings.TrimSpace(model) != "" {
		s.model = strings.TrimSpace(model)
	}
}

func (s *STTService) Initialize(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}

	s.stateMu.Lock()
	if s.ctx == nil {
		s.ctx, s.cancel = context.WithCancel(ctx)
	}
	realtimeCtx := s.ctx
	s.stateMu.Unlock()

	return s.connect(realtimeCtx)
}

func (s *STTService) Cleanup() error {
	s.stateMu.Lock()
	if s.cancel != nil {
		s.cancel()
		s.cancel = nil
	}
	s.ctx = nil
	s.stateMu.Unlock()

	s.disconnect()
	s.readWG.Wait()
	return nil
}

func (s *STTService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		s.HandleStartFrame(f)
		return s.PushFrame(frame, direction)

	case *frames.EndFrame:
		if err := s.Cleanup(); err != nil {
			logger.Error("[OpenAIRealtimeSTT] Cleanup failed: %v", err)
		}
		return s.PushFrame(frame, direction)

	case *frames.CancelFrame:
		if err := s.Cleanup(); err != nil {
			logger.Error("[OpenAIRealtimeSTT] Cleanup failed: %v", err)
		}
		return s.PushFrame(frame, direction)

	case *frames.UserStoppedSpeakingFrame:
		if !s.useServerVAD() {
			if err := s.sendCommit(ctx); err != nil {
				s.pushError(err)
			}
		}
		return s.PushFrame(frame, direction)

	case *frames.AudioFrame:
		if err := s.handleAudioFrame(ctx, f); err != nil {
			s.pushError(err)
			return err
		}
		return s.PushFrame(frame, direction)

	default:
		return s.PushFrame(frame, direction)
	}
}

func (s *STTService) connect(ctx context.Context) error {
	s.connectMu.Lock()
	defer s.connectMu.Unlock()

	if s.getConn() != nil {
		return nil
	}

	header := http.Header{}
	if strings.TrimSpace(s.apiKey) != "" {
		header.Set("Authorization", fmt.Sprintf("Bearer %s", s.apiKey))
	}
	header.Set("OpenAI-Beta", "realtime=v1")

	conn, _, err := s.dialer.DialContext(ctx, s.endpoint, header)
	if err != nil {
		return fmt.Errorf("failed to connect to OpenAI Realtime STT: %w", err)
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

	logger.Info("[OpenAIRealtimeSTT] Connected (model=%s turn_detection=%s)", s.model, s.turnDetectionMode())
	return nil
}

func (s *STTService) disconnect() {
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

func (s *STTService) reconnect(ctx context.Context) error {
	logger.Warn("[OpenAIRealtimeSTT] Reconnecting websocket")
	s.disconnect()
	return s.Initialize(ctx)
}

func (s *STTService) handleAudioFrame(ctx context.Context, frame *frames.AudioFrame) error {
	if frame == nil || len(frame.Data) == 0 {
		return nil
	}

	if s.getConn() == nil {
		if err := s.Initialize(ctx); err != nil {
			return err
		}
	}

	resampled, err := s.toRealtimeSampleRate(frame)
	if err != nil {
		return err
	}

	payload := map[string]interface{}{
		"type":  "input_audio_buffer.append",
		"audio": base64.StdEncoding.EncodeToString(resampled),
	}

	if err := s.writeJSON(payload); err != nil {
		if reconnectErr := s.reconnect(ctx); reconnectErr != nil {
			return fmt.Errorf("failed to send audio append: %w", err)
		}
		if retryErr := s.writeJSON(payload); retryErr != nil {
			return fmt.Errorf("failed to send audio append after reconnect: %w", retryErr)
		}
	}

	return nil
}

func (s *STTService) toRealtimeSampleRate(frame *frames.AudioFrame) ([]byte, error) {
	inputRate := frame.SampleRate
	if inputRate == 0 {
		inputRate = DefaultInputRateHz
	}

	if inputRate == DefaultRealtimeRateHz {
		return frame.Data, nil
	}

	pcm, err := audio.BytesToPCM(frame.Data)
	if err != nil {
		return nil, fmt.Errorf("invalid PCM audio payload: %w", err)
	}

	resampled := audio.Resample(pcm, inputRate, DefaultRealtimeRateHz)
	return audio.PCMToBytes(resampled), nil
}

func (s *STTService) sendCommit(ctx context.Context) error {
	if s.getConn() == nil {
		if err := s.Initialize(ctx); err != nil {
			return err
		}
	}

	payload := map[string]interface{}{
		"type": "input_audio_buffer.commit",
	}

	if err := s.writeJSON(payload); err != nil {
		if reconnectErr := s.reconnect(ctx); reconnectErr != nil {
			return fmt.Errorf("failed to send commit: %w", err)
		}
		if retryErr := s.writeJSON(payload); retryErr != nil {
			return fmt.Errorf("failed to send commit after reconnect: %w", retryErr)
		}
	}

	return nil
}

func (s *STTService) sessionUpdateMessage() map[string]interface{} {
	var turnDetection interface{}
	if s.useServerVAD() {
		turnDetection = nil
	} else {
		turnDetection = false
	}

	return map[string]interface{}{
		"type": "session.update",
		"session": map[string]interface{}{
			"input_audio_format": "pcm16",
			"input_audio_transcription": map[string]interface{}{
				"model": s.model,
			},
			"turn_detection": turnDetection,
		},
	}
}

func (s *STTService) readLoop(conn *websocket.Conn) {
	defer s.readWG.Done()
	defer s.clearConnection(conn)

	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			if s.isExpectedReadClose(err) {
				logger.Debug("[OpenAIRealtimeSTT] Connection closed")
				return
			}

			ctx := s.getContext()
			if ctx != nil {
				if reconnectErr := s.reconnect(ctx); reconnectErr != nil {
					s.pushError(fmt.Errorf("OpenAI Realtime STT read error: %w", err))
				}
			}
			return
		}

		if err := s.handleServerMessage(message); err != nil {
			logger.Warn("[OpenAIRealtimeSTT] Failed to process message: %v", err)
		}
	}
}

func (s *STTService) handleServerMessage(message []byte) error {
	var event sttRealtimeEvent
	if err := json.Unmarshal(message, &event); err != nil {
		return fmt.Errorf("invalid OpenAI Realtime STT event payload: %w", err)
	}

	switch event.Type {
	case "input_audio_buffer.speech_started":
		return s.PushFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)

	case "input_audio_buffer.speech_stopped":
		if err := s.BroadcastInterruption(s.getContext()); err != nil {
			return fmt.Errorf("failed to broadcast interruption: %w", err)
		}
		return nil

	case "conversation.item.input_audio_transcription.completed":
		transcript := event.extractTranscript()
		if transcript == "" {
			return nil
		}
		return s.PushFrame(frames.NewTranscriptionFrame(transcript, true), frames.Downstream)

	default:
		return nil
	}
}

func (s *STTService) useServerVAD() bool {
	mode := strings.ToLower(strings.TrimSpace(s.turnDetection))
	switch mode {
	case "", "server", "null":
		return true
	case "local", "false":
		return false
	default:
		return true
	}
}

func (s *STTService) turnDetectionMode() string {
	if s.useServerVAD() {
		return "server"
	}
	return "local"
}

func (s *STTService) writeJSON(payload interface{}) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	s.connMu.RLock()
	conn := s.conn
	s.connMu.RUnlock()

	if conn == nil {
		return fmt.Errorf("OpenAI Realtime STT websocket is not connected")
	}

	return conn.WriteJSON(payload)
}

func (s *STTService) getConn() *websocket.Conn {
	s.connMu.RLock()
	defer s.connMu.RUnlock()
	return s.conn
}

func (s *STTService) clearConnection(conn *websocket.Conn) {
	s.connMu.Lock()
	if s.conn == conn {
		s.conn = nil
	}
	s.connMu.Unlock()
}

func (s *STTService) getContext() context.Context {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	return s.ctx
}

func (s *STTService) isExpectedReadClose(err error) bool {
	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		return true
	}

	if strings.Contains(err.Error(), "use of closed network connection") {
		return true
	}

	ctx := s.getContext()
	if ctx != nil {
		select {
		case <-ctx.Done():
			return true
		default:
		}
	}

	return false
}

func (s *STTService) pushError(err error) {
	if err == nil {
		return
	}

	logger.Error("[OpenAIRealtimeSTT] %v", err)
	if pushErr := s.PushFrame(frames.NewErrorFrame(err), frames.Upstream); pushErr != nil {
		logger.Error("[OpenAIRealtimeSTT] Failed to push ErrorFrame upstream: %v", pushErr)
	}
}

type sttRealtimeEvent struct {
	Type       string                `json:"type"`
	Transcript string                `json:"transcript,omitempty"`
	Item       *sttRealtimeEventItem `json:"item,omitempty"`
}

type sttRealtimeEventItem struct {
	Transcript string `json:"transcript,omitempty"`
	Content    []struct {
		Transcript string `json:"transcript,omitempty"`
		Text       string `json:"text,omitempty"`
	} `json:"content,omitempty"`
}

func (e sttRealtimeEvent) extractTranscript() string {
	if strings.TrimSpace(e.Transcript) != "" {
		return strings.TrimSpace(e.Transcript)
	}

	if e.Item == nil {
		return ""
	}

	if strings.TrimSpace(e.Item.Transcript) != "" {
		return strings.TrimSpace(e.Item.Transcript)
	}

	for _, content := range e.Item.Content {
		if strings.TrimSpace(content.Transcript) != "" {
			return strings.TrimSpace(content.Transcript)
		}
		if strings.TrimSpace(content.Text) != "" {
			return strings.TrimSpace(content.Text)
		}
	}

	return ""
}
