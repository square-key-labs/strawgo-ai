package assemblyai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

const (
	DefaultSampleRate = 16000
	DefaultModel      = "u3-rt-pro"
	DefaultBaseURL    = "wss://streaming.assemblyai.com/v3/ws"
	minChunk          = 50 * time.Millisecond
)

var (
	terminateMessage     = []byte(`{"type":"Terminate"}`)
	forceEndpointMessage = []byte(`{"type":"ForceEndpoint"}`)
)

type STTConfig struct {
	APIKey             string
	Language           string
	Model              string
	SampleRate         int
	MinTurnSilence     int
	MaxTurnSilence     int
	DisableFormatTurns bool
	DisableVADEndpoint bool
	KeytermsPrompt     []string
	BaseURL            string
	OnEndOfTurn        func(transcript string)
}

type STTService struct {
	*stt.Base
	apiKey         string
	baseURL        string
	minTurnSilence int
	maxTurnSilence int
	formatTurns    bool
	vadEndpoint    bool
	keyterms       []string
	onEndOfTurn    func(string)

	session atomic.Pointer[reconnect.Session]
	buf     []byte
	forced  atomic.Bool
}

func NewSTTService(config STTConfig) *STTService {
	model := config.Model
	if model == "" {
		model = DefaultModel
	}
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	s := &STTService{
		apiKey:         config.APIKey,
		baseURL:        baseURL,
		minTurnSilence: config.MinTurnSilence,
		maxTurnSilence: config.MaxTurnSilence,
		formatTurns:    !config.DisableFormatTurns,
		vadEndpoint:    !config.DisableVADEndpoint,
		keyterms:       config.KeytermsPrompt,
		onEndOfTurn:    config.OnEndOfTurn,
	}
	s.Base = stt.NewStreaming("AssemblyAISTT", s, stt.Options{
		Provider:   "assemblyai",
		Settings:   frames.STTSettings{Model: model, Language: config.Language},
		SampleRate: config.SampleRate,
		TTFSP99:    400 * time.Millisecond,
	})
	return s
}

func (s *STTService) endpoint() (string, http.Header) {
	settings := s.Settings()
	params := url.Values{}
	params.Set("sample_rate", strconv.Itoa(s.SampleRate()))
	params.Set("encoding", "pcm_s16le")
	if settings.Model != "" {
		params.Set("speech_model", settings.Model)
	}
	if settings.Language != "" {
		params.Set("language_code", settings.Language)
	}
	params.Set("format_turns", strconv.FormatBool(s.formatTurns))
	if s.minTurnSilence > 0 {
		params.Set("min_turn_silence", strconv.Itoa(s.minTurnSilence))
	}
	if s.maxTurnSilence > 0 {
		params.Set("max_turn_silence", strconv.Itoa(s.maxTurnSilence))
	}
	if len(s.keyterms) > 0 {
		if data, err := json.Marshal(s.keyterms); err == nil {
			params.Set("keyterms_prompt", string(data))
		}
	}
	header := http.Header{}
	header.Set("Authorization", s.apiKey)
	return s.baseURL + "?" + params.Encode(), header
}

func (s *STTService) Connect(ctx context.Context, sink stt.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:         "AssemblyAISTT",
		Endpoint:     s.endpoint,
		OnMessage:    func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:      sink.Error,
		CloseMessage: &reconnect.Message{Type: websocket.TextMessage, Data: terminateMessage},
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *STTService) SendAudio(audio []byte) {
	session := s.session.Load()
	if session == nil {
		return
	}
	chunk := int(minChunk.Seconds()*float64(s.SampleRate())) * 2
	if len(s.buf) == 0 && len(audio) >= chunk {
		session.SendDroppable(websocket.BinaryMessage, audio)
		return
	}
	s.buf = append(s.buf, audio...)
	if len(s.buf) >= chunk {
		session.SendDroppable(websocket.BinaryMessage, s.buf)
		s.buf = make([]byte, 0, 2*chunk)
	}
}

func (s *STTService) Finalize() {
	if !s.vadEndpoint {
		return
	}
	session := s.session.Load()
	if session == nil {
		return
	}
	if len(s.buf) > 0 {
		session.SendDroppable(websocket.BinaryMessage, s.buf)
		s.buf = nil
	}
	s.forced.Store(true)
	session.Send(websocket.TextMessage, forceEndpointMessage)
}

func (s *STTService) Disconnect(graceful bool) {
	session := s.session.Load()
	if session == nil {
		return
	}
	if graceful && len(s.buf) > 0 {
		session.SendDroppable(websocket.BinaryMessage, s.buf)
		s.buf = nil
	}
	session.Close(graceful)
}

func (s *STTService) ApplySettings(frames.STTSettings) error {
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
	return nil
}

type turnMessage struct {
	Type            string `json:"type"`
	Transcript      string `json:"transcript"`
	EndOfTurn       bool   `json:"end_of_turn"`
	TurnIsFormatted bool   `json:"turn_is_formatted"`
	LanguageCode    string `json:"language_code"`
	Error           string `json:"error"`
}

func (s *STTService) onMessage(sink stt.Sink, data []byte) {
	var msg turnMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return
	}
	switch msg.Type {
	case "Turn":
		if msg.Transcript == "" {
			return
		}
		final := msg.EndOfTurn && (!s.formatTurns || msg.TurnIsFormatted)
		if msg.EndOfTurn && !final {
			return
		}
		t := stt.Transcript{Text: msg.Transcript, Final: final, Language: msg.LanguageCode}
		if final {
			t.Finalized = s.forced.Swap(false)
		}
		sink.Transcript(t)
		if final && s.onEndOfTurn != nil {
			s.onEndOfTurn(msg.Transcript)
		}
	case "Error":
		sink.Error(&providerError{msg.Error}, false)
	}
}

type providerError struct{ msg string }

func (e *providerError) Error() string { return "assemblyai: " + e.msg }
