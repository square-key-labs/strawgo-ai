package neuphonic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DefaultURL      = "wss://api.neuphonic.com"
	DefaultHTTPURL  = "https://api.neuphonic.com"
	DefaultLanguage = "en"
	DefaultEncoding = "pcm_linear"
)

var (
	stopMessage      = []byte(`{"text":"<STOP>"}`)
	keepaliveMessage = []byte(`{"text":""}`)
)

var languages = map[string]bool{
	"de": true, "en": true, "es": true, "nl": true, "ar": true, "fr": true, "pt": true, "ru": true, "hi": true, "zh": true,
}

func Language(code string) string {
	base := strings.ToLower(code)
	if i := strings.IndexAny(base, "-_"); i > 0 {
		base = base[:i]
	}
	if languages[base] {
		return base
	}
	return code
}

type TTSConfig struct {
	APIKey       string
	VoiceID      string
	Language     string
	Speed        float64
	Temperature  float64
	IPAKey       string
	SampleRate   int
	StreamTokens bool
	URL          string
	HTTPClient   *http.Client
}

func defaultSettings(config TTSConfig) frames.TTSSettings {
	settings := frames.TTSSettings{Voice: config.VoiceID, Language: config.Language}
	if settings.Language == "" {
		settings.Language = DefaultLanguage
	}
	speed := config.Speed
	if speed <= 0 {
		speed = 1
	}
	settings.Speed = &speed
	extra := map[string]any{}
	if config.Temperature > 0 {
		extra["temperature"] = config.Temperature
	}
	if config.IPAKey != "" {
		extra["ipa_key"] = config.IPAKey
	}
	if len(extra) > 0 {
		settings.Extra = extra
	}
	return settings
}

func fields(settings frames.TTSSettings, sampleRate int) map[string]any {
	m := map[string]any{
		"lang_code":     Language(settings.Language),
		"encoding":      DefaultEncoding,
		"sampling_rate": sampleRate,
	}
	if settings.Speed != nil {
		m["speed"] = *settings.Speed
	}
	if settings.Voice != "" {
		m["voice_id"] = settings.Voice
	}
	for k, v := range settings.Extra {
		if v != nil {
			m[k] = v
		}
	}
	return m
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
	u := strings.TrimRight(config.URL, "/")
	if u == "" {
		u = DefaultURL
	}
	s := &TTSService{apiKey: config.APIKey, url: u}
	aggregation := tts.AggregateSentences
	if config.StreamTokens {
		aggregation = tts.AggregateTokens
	}
	s.Base = tts.NewStreaming("NeuphonicTTS", s, tts.Options{
		Settings:    defaultSettings(config),
		SampleRate:  config.SampleRate,
		Aggregation: aggregation,
	})
	return s
}

func (s *TTSService) endpoint() (string, http.Header) {
	settings := s.Settings()
	q := url.Values{}
	for k, v := range fields(settings, s.SampleRate()) {
		switch t := v.(type) {
		case float64:
			q.Set(k, strconv.FormatFloat(t, 'f', -1, 64))
		case int:
			q.Set(k, strconv.Itoa(t))
		case string:
			q.Set(k, t)
		default:
			b, _ := json.Marshal(t)
			q.Set(k, string(b))
		}
	}
	header := http.Header{}
	header.Set("x-api-key", s.apiKey)
	return s.url + "/speak/" + url.PathEscape(Language(settings.Language)) + "?" + q.Encode(), header
}

func (s *TTSService) Connect(ctx context.Context, sink tts.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:     "NeuphonicTTS",
		Endpoint: s.endpoint,
		OnOpen: func(reconnect.Writer) error {
			s.mu.Lock()
			s.stale = false
			s.mu.Unlock()
			return nil
		},
		OnMessage:     func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:       sink.Error,
		KeepaliveIdle: 10 * time.Second,
		Keepalive: func() (reconnect.Message, bool) {
			return reconnect.Message{Type: websocket.TextMessage, Data: keepaliveMessage}, true
		},
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

func (s *TTSService) Speak(contextID, text string) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("neuphonic not connected")
	}
	s.mu.Lock()
	s.current = contextID
	s.mu.Unlock()
	return session.SendJSON(struct {
		Text string `json:"text"`
	}{Text: text + " <STOP>"})
}

func (s *TTSService) Flush(string) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("neuphonic not connected")
	}
	if !session.Send(websocket.TextMessage, stopMessage) {
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

func (s *TTSService) target() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stale {
		return ""
	}
	return s.current
}

type message struct {
	Data *struct {
		Audio string `json:"audio"`
	} `json:"data"`
	Errors     any `json:"errors"`
	StatusCode int `json:"status_code"`
}

func decode(data []byte) ([]byte, error) {
	var msg message
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, nil
	}
	if msg.Errors != nil {
		b, _ := json.Marshal(msg.Errors)
		return nil, errors.New("neuphonic API error " + strconv.Itoa(msg.StatusCode) + ": " + string(b))
	}
	if msg.Data == nil || msg.Data.Audio == "" {
		return nil, nil
	}
	return base64.StdEncoding.DecodeString(msg.Data.Audio)
}

func (s *TTSService) onMessage(sink tts.Sink, data []byte) {
	audio, err := decode(data)
	if err != nil {
		sink.Error(err, false)
		return
	}
	if len(audio) == 0 {
		return
	}
	if id := s.target(); id != "" {
		sink.Audio(id, audio)
	}
}
