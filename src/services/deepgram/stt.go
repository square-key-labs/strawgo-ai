package deepgram

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

const DeepgramSTTURL = "wss://api.deepgram.com/v1/listen"

var (
	finalizeMessage  = []byte(`{"type":"Finalize"}`)
	keepaliveMessage = []byte(`{"type":"KeepAlive"}`)
	closeStream      = []byte(`{"type":"CloseStream"}`)
)

type STTConfig struct {
	APIKey            string
	Language          string
	Model             string
	Encoding          string
	SampleRate        int
	KeepaliveInterval time.Duration
	SmartFormat       bool
	BaseURL           string
}

type STTService struct {
	*stt.Base
	apiKey    string
	encoding  string
	baseURL   string
	keepalive time.Duration
	smart     bool
	session   atomic.Pointer[reconnect.Session]
}

func NewSTTService(config STTConfig) *STTService {
	encoding := normalizeDeepgramEncoding(config.Encoding)
	if encoding == "" {
		encoding = "linear16"
	}
	keepalive := config.KeepaliveInterval
	if keepalive <= 0 {
		keepalive = 5 * time.Second
	}
	sampleRate := config.SampleRate
	bytesPerSample := 2
	if encoding == "mulaw" || encoding == "alaw" {
		bytesPerSample = 1
		if sampleRate == 0 {
			sampleRate = 8000
		}
	}
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DeepgramSTTURL
	}
	s := &STTService{
		apiKey:    config.APIKey,
		encoding:  encoding,
		baseURL:   baseURL,
		keepalive: keepalive,
		smart:     config.SmartFormat,
	}
	s.Base = stt.NewStreaming("DeepgramSTT", s, stt.Options{
		Provider:       "deepgram",
		Settings:       frames.STTSettings{Model: config.Model, Language: config.Language},
		SampleRate:     sampleRate,
		BytesPerSample: bytesPerSample,
		TTFSP99:        300 * time.Millisecond,
	})
	return s
}

func normalizeDeepgramEncoding(encoding string) string {
	switch encoding {
	case "ulaw", "PCMU":
		return "mulaw"
	case "PCMA":
		return "alaw"
	case "pcm", "PCM":
		return "linear16"
	default:
		return encoding
	}
}

func (s *STTService) endpoint() (string, http.Header) {
	settings := s.Settings()
	params := url.Values{}
	if settings.Language != "" {
		params.Set("language", settings.Language)
	}
	if settings.Model != "" {
		params.Set("model", settings.Model)
	}
	params.Set("encoding", s.encoding)
	params.Set("sample_rate", strconv.Itoa(s.SampleRate()))
	params.Set("channels", "1")
	params.Set("interim_results", "true")
	if s.smart {
		params.Set("smart_format", "true")
	}
	header := http.Header{}
	header.Set("Authorization", "Token "+s.apiKey)
	return s.baseURL + "?" + params.Encode(), header
}

func (s *STTService) Connect(ctx context.Context, sink stt.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:          "DeepgramSTT",
		Endpoint:      s.endpoint,
		OnMessage:     func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:       sink.Error,
		CloseMessage:  &reconnect.Message{Type: websocket.TextMessage, Data: closeStream},
		KeepaliveIdle: s.keepalive,
		Keepalive: func() (reconnect.Message, bool) {
			return reconnect.Message{Type: websocket.TextMessage, Data: keepaliveMessage}, true
		},
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *STTService) SendAudio(audio []byte) {
	if session := s.session.Load(); session != nil {
		session.SendDroppable(websocket.BinaryMessage, audio)
	}
}

func (s *STTService) Finalize() {
	if session := s.session.Load(); session != nil {
		session.Send(websocket.TextMessage, finalizeMessage)
	}
}

func (s *STTService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func (s *STTService) ApplySettings(frames.STTSettings) error {
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
	return nil
}

type sttResponse struct {
	Type         string `json:"type"`
	IsFinal      bool   `json:"is_final"`
	FromFinalize bool   `json:"from_finalize"`
	Channel      struct {
		Alternatives []struct {
			Transcript string   `json:"transcript"`
			Languages  []string `json:"languages"`
		} `json:"alternatives"`
	} `json:"channel"`
}

func (s *STTService) onMessage(sink stt.Sink, data []byte) {
	var resp sttResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return
	}
	if resp.Type != "" && resp.Type != "Results" {
		return
	}
	if len(resp.Channel.Alternatives) == 0 {
		return
	}
	alt := resp.Channel.Alternatives[0]
	t := stt.Transcript{Text: alt.Transcript, Final: resp.IsFinal, Finalized: resp.FromFinalize}
	if len(alt.Languages) > 0 {
		t.Language = alt.Languages[0]
	}
	sink.Transcript(t)
}
