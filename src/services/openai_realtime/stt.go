package openai_realtime

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services/realtime"
)

const (
	DefaultSTTEndpoint    = "wss://api.openai.com/v1/realtime?intent=transcription"
	DefaultSTTModel       = RealtimeWhisperModel
	DefaultInputRateHz    = 16000
	DefaultRealtimeRateHz = 24000
)

type STTConfig struct {
	APIKey        string
	Model         string
	Endpoint      string
	TurnDetection string
	Backoff       realtime.Backoff
	Dialer        *websocket.Dialer

	Language       string
	Prompt         string
	NoiseReduction string
}

type STTService struct {
	*processors.BaseProcessor

	apiKey         string
	endpoint       string
	turnDetection  string
	prompt         string
	noiseReduction string
	dialer         *websocket.Dialer
	socket         *realtime.Socket

	mu       sync.Mutex
	model    string
	language string

	buf []byte
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
		apiKey:         config.APIKey,
		model:          model,
		endpoint:       endpoint,
		turnDetection:  strings.TrimSpace(config.TurnDetection),
		prompt:         config.Prompt,
		noiseReduction: config.NoiseReduction,
		language:       languageCode(config.Language),
		dialer:         dialer,
	}
	s.BaseProcessor = processors.NewBaseProcessor("OpenAIRealtimeSTT", s)
	s.socket = realtime.NewSocket(realtime.SocketConfig{
		Name:        "OpenAIRealtimeSTT",
		Dial:        s.dial,
		OnOpen:      func(_ context.Context, w realtime.Writer) error { return w.WriteJSON(s.sessionUpdateMessage()) },
		ReadyOnOpen: true,
		OnMessage:   s.onMessage,
		OnFatal: func(err error) {
			_ = s.PushError("OpenAI Realtime STT connection lost", err, true)
		},
		Backoff: config.Backoff,
	})
	return s
}

func (s *STTService) SetLanguage(language string) {
	language = languageCode(language)
	s.mu.Lock()
	changed := s.language != language
	s.language = language
	s.mu.Unlock()
	if changed && s.socket.Ready() {
		_ = s.socket.WriteJSON(s.sessionUpdateMessage())
	}
}

func languageCode(language string) string {
	code, _, _ := strings.Cut(strings.TrimSpace(language), "-")
	return strings.ToLower(code)
}

func (s *STTService) SetModel(model string) {
	model = strings.TrimSpace(model)
	if model == "" {
		return
	}
	s.mu.Lock()
	changed := s.model != model
	s.model = model
	s.mu.Unlock()
	if changed && s.socket.Ready() {
		_ = s.socket.WriteJSON(s.sessionUpdateMessage())
	}
}

func (s *STTService) Initialize(ctx context.Context) error {
	if ctx == nil {
		ctx = s.Context()
	}
	s.socket.Start(ctx)
	return nil
}

func (s *STTService) Cleanup() error {
	s.socket.Stop()
	return nil
}

func (s *STTService) Stop() error {
	s.socket.Stop()
	return s.BaseProcessor.Stop()
}

func (s *STTService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		s.socket.Start(s.Context())
	case *frames.EndFrame, *frames.CancelFrame:
		s.socket.Stop()
	case *frames.UserStoppedSpeakingFrame:
		if !s.useServerVAD() {
			if err := s.socket.WriteJSON(map[string]any{"type": "input_audio_buffer.commit"}); err != nil {
				logger.Debug("[OpenAIRealtimeSTT] commit failed: %v", err)
			}
		}
	case *frames.AudioFrame:
		s.sendAudio(f)
	}
	return s.PushFrame(frame, direction)
}

func (s *STTService) dial(ctx context.Context) (*websocket.Conn, error) {
	h := http.Header{}
	if strings.TrimSpace(s.apiKey) != "" {
		h.Set("Authorization", "Bearer "+s.apiKey)
	}
	conn, _, err := s.dialer.DialContext(ctx, s.endpoint, h)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to OpenAI Realtime STT: %w", err)
	}
	return conn, nil
}

func (s *STTService) sendAudio(f *frames.AudioFrame) {
	if len(f.Data) == 0 {
		return
	}
	if !s.socket.Running() {
		s.socket.Start(s.Context())
	}
	rate := f.SampleRate
	if rate == 0 {
		rate = DefaultInputRateHz
	}
	s.buf = realtime.AppendResamplePCM16(s.buf[:0], f.Data, rate, DefaultRealtimeRateHz)
	if err := s.socket.WriteBase64(`{"type":"input_audio_buffer.append","audio":"`, s.buf, `"}`); err != nil && err != realtime.ErrNotConnected {
		logger.Debug("[OpenAIRealtimeSTT] audio send failed: %v", err)
	}
}

func (s *STTService) sessionUpdateMessage() map[string]any {
	s.mu.Lock()
	model, language := s.model, s.language
	s.mu.Unlock()
	transcription := map[string]any{"model": model}
	if language != "" {
		transcription["language"] = language
	}
	if s.prompt != "" && model != RealtimeWhisperModel {
		transcription["prompt"] = s.prompt
	}
	input := map[string]any{
		"format":        audioFormatObject("pcm16"),
		"transcription": transcription,
	}
	if s.useServerVAD() {
		input["turn_detection"] = map[string]any{"type": TurnDetectionServerVAD}
	} else {
		input["turn_detection"] = nil
	}
	if s.noiseReduction != "" {
		input["noise_reduction"] = map[string]any{"type": s.noiseReduction}
	}
	return map[string]any{"type": "session.update", "session": map[string]any{
		"type":  "transcription",
		"audio": map[string]any{"input": input},
	}}
}

func (s *STTService) onMessage(data []byte) {
	var event sttRealtimeEvent
	if err := json.Unmarshal(data, &event); err != nil {
		logger.Warn("[OpenAIRealtimeSTT] invalid event: %v", err)
		return
	}
	switch event.Type {
	case "input_audio_buffer.speech_started":
		_ = s.PushFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
		if s.InterruptionsAllowed() {
			_ = s.BroadcastInterruption(s.Context())
		}
	case "input_audio_buffer.speech_stopped":
		_ = s.PushFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	case "conversation.item.input_audio_transcription.delta":
		if event.Delta != "" {
			_ = s.PushFrame(frames.NewInterimTranscriptionFrame(event.Delta, ""), frames.Downstream)
		}
	case "conversation.item.input_audio_transcription.completed":
		if transcript := event.extractTranscript(); transcript != "" {
			_ = s.PushFrame(frames.NewTranscriptionFrame(transcript, true), frames.Downstream)
		}
	case "error":
		if event.Error != nil {
			_ = s.PushError("OpenAI Realtime STT error", fmt.Errorf("%s", event.Error.Message), false)
		}
	}
}

func (s *STTService) useServerVAD() bool {
	switch strings.ToLower(s.turnDetection) {
	case "local", "false":
		return false
	}
	return true
}

type sttRealtimeEvent struct {
	Type       string                `json:"type"`
	Delta      string                `json:"delta,omitempty"`
	Transcript string                `json:"transcript,omitempty"`
	Item       *sttRealtimeEventItem `json:"item,omitempty"`
	Error      *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type sttRealtimeEventItem struct {
	Transcript string `json:"transcript,omitempty"`
	Content    []struct {
		Transcript string `json:"transcript,omitempty"`
		Text       string `json:"text,omitempty"`
	} `json:"content,omitempty"`
}

func (e sttRealtimeEvent) extractTranscript() string {
	if t := strings.TrimSpace(e.Transcript); t != "" {
		return t
	}
	if e.Item == nil {
		return ""
	}
	if t := strings.TrimSpace(e.Item.Transcript); t != "" {
		return t
	}
	for _, content := range e.Item.Content {
		if t := strings.TrimSpace(content.Transcript); t != "" {
			return t
		}
		if t := strings.TrimSpace(content.Text); t != "" {
			return t
		}
	}
	return ""
}
