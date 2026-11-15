package fish

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DefaultURL     = "wss://api.fish.audio/v1/tts/live"
	DefaultModel   = "s2.1-pro"
	DefaultLatency = "balanced"
)

var (
	flushMessage = []byte{0x81, 0xa5, 'e', 'v', 'e', 'n', 't', 0xa5, 'f', 'l', 'u', 's', 'h'}
	stopMessage  = []byte{0x81, 0xa5, 'e', 'v', 'e', 'n', 't', 0xa4, 's', 't', 'o', 'p'}
)

type TTSConfig struct {
	APIKey      string
	ReferenceID string
	Model       string
	Language    string
	Latency     string
	Speed       float64
	Volume      int
	Params      map[string]any
	SampleRate  int
	URL         string
}

type TTSService struct {
	*tts.Base
	apiKey  string
	url     string
	session atomic.Pointer[reconnect.Session]

	mu      sync.Mutex
	current string
	stale   bool
}

func NewTTSService(config TTSConfig) *TTSService {
	settings := frames.TTSSettings{Model: config.Model, Voice: config.ReferenceID, Language: config.Language}
	if settings.Model == "" {
		settings.Model = DefaultModel
	}
	speed := config.Speed
	if speed <= 0 {
		speed = 1
	}
	settings.Speed = &speed
	settings.Extra = map[string]any{"volume": config.Volume, "normalize": true}
	settings.Extra["latency"] = DefaultLatency
	if config.Latency != "" {
		settings.Extra["latency"] = config.Latency
	}
	for k, v := range config.Params {
		settings.Extra[k] = v
	}
	u := config.URL
	if u == "" {
		u = DefaultURL
	}
	s := &TTSService{apiKey: config.APIKey, url: u}
	s.Base = tts.NewStreaming("FishTTS", s, tts.Options{Settings: settings, SampleRate: config.SampleRate})
	return s
}

func (s *TTSService) endpoint() (string, http.Header) {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.apiKey)
	if model := s.Settings().Model; model != "" {
		header.Set("model", model)
	}
	return s.url, header
}

func (s *TTSService) startRequest() map[string]any {
	settings := s.Settings()
	prosody := map[string]any{"speed": 1.0, "volume": 0}
	if settings.Speed != nil {
		prosody["speed"] = *settings.Speed
	}
	req := map[string]any{
		"text":         "",
		"sample_rate":  s.SampleRate(),
		"format":       "pcm",
		"reference_id": settings.Voice,
		"prosody":      prosody,
	}
	for k, v := range settings.Extra {
		switch k {
		case "volume", "normalize_loudness":
			prosody[k] = v
		default:
			req[k] = v
		}
	}
	if settings.Language != "" {
		req["language"] = settings.Language
	}
	return map[string]any{"event": "start", "request": req}
}

func (s *TTSService) Connect(ctx context.Context, sink tts.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:         "FishTTS",
		Endpoint:     s.endpoint,
		OnOpen:       s.onOpen,
		OnMessage:    func(mt int, data []byte) { s.onMessage(sink, mt, data) },
		OnError:      sink.Error,
		CloseMessage: &reconnect.Message{Type: websocket.BinaryMessage, Data: stopMessage},
		PingInterval: 20 * time.Second,
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *TTSService) onOpen(w reconnect.Writer) error {
	msg, err := pack(nil, s.startRequest())
	if err != nil {
		return err
	}
	if err := w.WriteMessage(websocket.BinaryMessage, msg); err != nil {
		return err
	}
	s.mu.Lock()
	s.stale = false
	s.mu.Unlock()
	return nil
}

func (s *TTSService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func (s *TTSService) send(data []byte) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("fish not connected")
	}
	if !session.Send(websocket.BinaryMessage, data) {
		return reconnect.ErrQueueFull
	}
	return nil
}

func (s *TTSService) Speak(contextID, text string) error {
	msg, err := pack(nil, map[string]any{"event": "text", "text": text})
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.current = contextID
	s.mu.Unlock()
	if err := s.send(msg); err != nil {
		return err
	}
	return s.send(flushMessage)
}

func (s *TTSService) Flush(string) error {
	return s.send(flushMessage)
}

func (s *TTSService) Cancel(string) error {
	session := s.session.Load()
	if session == nil {
		return nil
	}
	s.mu.Lock()
	s.current = ""
	s.stale = true
	s.mu.Unlock()
	session.Clear()
	session.Reconnect()
	return nil
}

func (s *TTSService) ApplySettings(frames.TTSSettings) error {
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
	return nil
}

func (s *TTSService) onMessage(sink tts.Sink, mt int, data []byte) {
	if mt != websocket.BinaryMessage {
		return
	}
	v, err := unpack(data)
	if err != nil {
		sink.Error(err, false)
		return
	}
	msg, ok := v.(map[string]any)
	if !ok {
		return
	}
	switch msg["event"] {
	case "audio":
		audio, _ := msg["audio"].([]byte)
		if len(audio) == 0 {
			return
		}
		s.mu.Lock()
		id := s.current
		if s.stale {
			id = ""
		}
		s.mu.Unlock()
		if id != "" {
			sink.Audio(id, audio)
		}
	case "finish":
		if msg["reason"] == "error" {
			sink.Error(errors.New("fish: server error during synthesis"), false)
		}
	}
}
