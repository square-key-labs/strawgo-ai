package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DeepgramTTSURL     = "wss://api.deepgram.com/v1/speak"
	DefaultTTSModel    = "aura-asteria-en"
	DefaultTTSEncoding = "linear16"
)

var (
	flushMessage = []byte(`{"type":"Flush"}`)
	clearMessage = []byte(`{"type":"Clear"}`)
	closeMessage = []byte(`{"type":"Close"}`)
)

type TTSConfig struct {
	APIKey     string
	Model      string
	Encoding   string
	SampleRate int
	BaseURL    string
}

type TTSService struct {
	*tts.Base
	apiKey   string
	encoding string
	baseURL  string
	session  atomic.Pointer[reconnect.Session]

	mu       sync.Mutex
	order    []string
	clearing int
}

func NewTTSService(config TTSConfig) *TTSService {
	model := config.Model
	if model == "" {
		model = DefaultTTSModel
	}
	encoding := config.Encoding
	if encoding == "" {
		encoding = DefaultTTSEncoding
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
		baseURL = DeepgramTTSURL
	}
	s := &TTSService{apiKey: config.APIKey, encoding: encoding, baseURL: baseURL}
	s.Base = tts.NewStreaming("DeepgramTTS", s, tts.Options{
		Settings:       frames.TTSSettings{Model: model, Voice: model},
		SampleRate:     sampleRate,
		Codec:          encodingToCodec(encoding),
		BytesPerSample: bytesPerSample,
	})
	return s
}

func encodingToCodec(encoding string) string {
	switch encoding {
	case "mulaw", "alaw":
		return encoding
	default:
		return "linear16"
	}
}

func (s *TTSService) endpoint() (string, http.Header) {
	settings := s.Settings()
	model := settings.Voice
	if model == "" {
		model = settings.Model
	}
	params := url.Values{}
	params.Set("model", model)
	params.Set("encoding", s.encoding)
	params.Set("sample_rate", strconv.Itoa(s.SampleRate()))
	header := http.Header{}
	header.Set("Authorization", "Token "+s.apiKey)
	return s.baseURL + "?" + params.Encode(), header
}

func (s *TTSService) Connect(ctx context.Context, sink tts.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:         "DeepgramTTS",
		Endpoint:     s.endpoint,
		OnDisconnect: func(error) { s.resetStream() },
		OnMessage:    func(mt int, data []byte) { s.onMessage(sink, mt, data) },
		OnError:      sink.Error,
		CloseMessage: &reconnect.Message{Type: websocket.TextMessage, Data: closeMessage},
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *TTSService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func (s *TTSService) resetStream() {
	s.mu.Lock()
	s.order = s.order[:0]
	s.clearing = 0
	s.mu.Unlock()
}

func (s *TTSService) Speak(contextID, text string) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("deepgram TTS not connected")
	}
	s.mu.Lock()
	if n := len(s.order); n == 0 || s.order[n-1] != contextID {
		s.order = append(s.order, contextID)
	}
	s.mu.Unlock()
	return session.SendJSON(struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}{Type: "Speak", Text: text})
}

func (s *TTSService) Flush(string) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("deepgram TTS not connected")
	}
	if !session.Send(websocket.TextMessage, flushMessage) {
		return reconnect.ErrQueueFull
	}
	return nil
}

func (s *TTSService) Cancel(string) error {
	session := s.session.Load()
	if session == nil {
		return nil
	}
	s.mu.Lock()
	s.order = s.order[:0]
	s.clearing++
	s.mu.Unlock()
	session.Clear()
	session.Send(websocket.TextMessage, clearMessage)
	return nil
}

func (s *TTSService) ApplySettings(frames.TTSSettings) error {
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
	return nil
}

func (s *TTSService) currentContext() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.clearing > 0 || len(s.order) == 0 {
		return ""
	}
	return s.order[0]
}

func (s *TTSService) onMessage(sink tts.Sink, mt int, data []byte) {
	if mt == websocket.BinaryMessage {
		if id := s.currentContext(); id != "" {
			sink.Audio(id, data)
		}
		return
	}
	var msg struct {
		Type     string `json:"type"`
		ErrMsg   string `json:"err_msg"`
		Error    string `json:"error"`
		Category string `json:"category"`
	}
	if err := json.Unmarshal(data, &msg); err != nil {
		return
	}
	switch msg.Type {
	case "Flushed":
		s.mu.Lock()
		var done string
		if s.clearing == 0 && len(s.order) > 0 {
			done = s.order[0]
			s.order = append(s.order[:0], s.order[1:]...)
		}
		s.mu.Unlock()
		if done != "" {
			sink.Done(done)
		}
	case "Cleared":
		s.mu.Lock()
		if s.clearing > 0 {
			s.clearing--
		}
		s.mu.Unlock()
	case "Error", "Warning":
		text := msg.ErrMsg
		if text == "" {
			text = msg.Error
		}
		if text == "" {
			text = msg.Category
		}
		sink.Error(errors.New("deepgram: "+text), false)
	}
}
