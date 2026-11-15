package rime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
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
	DefaultURL     = "wss://users-ws.rime.ai/ws3"
	DefaultHTTPURL = "https://users.rime.ai/v1/rime-tts"
	DefaultModel   = "coda"
)

var (
	flushMessage = []byte(`{"operation":"flush"}`)
	clearMessage = []byte(`{"operation":"clear"}`)
	eosMessage   = []byte(`{"operation":"eos"}`)
)

var languages = map[string]string{
	"ar": "ara", "de": "ger", "en": "eng", "es": "spa", "fr": "fra",
	"hi": "hin", "it": "ita", "ja": "jpn", "pt": "por",
}

func Language(code string) string {
	base := strings.ToLower(code)
	if i := strings.IndexAny(base, "-_"); i > 0 {
		base = base[:i]
	}
	if mapped, ok := languages[base]; ok {
		return mapped
	}
	return code
}

type TTSConfig struct {
	APIKey       string
	VoiceID      string
	Model        string
	Language     string
	Speed        float64
	SampleRate   int
	Params       map[string]any
	StreamTokens bool
	URL          string
	HTTPClient   *http.Client
}

type TTSService struct {
	*tts.Base
	apiKey  string
	url     string
	session atomic.Pointer[reconnect.Session]

	mu     sync.Mutex
	offset map[string]float64
}

func defaultSettings(config TTSConfig) frames.TTSSettings {
	settings := frames.TTSSettings{Model: config.Model, Voice: config.VoiceID, Language: config.Language}
	if settings.Model == "" {
		settings.Model = DefaultModel
	}
	if config.Speed > 0 {
		speed := config.Speed
		settings.Speed = &speed
	}
	if len(config.Params) > 0 {
		settings.Extra = make(map[string]any, len(config.Params))
		for k, v := range config.Params {
			settings.Extra[k] = v
		}
	}
	return settings
}

func NewTTSService(config TTSConfig) *TTSService {
	u := config.URL
	if u == "" {
		u = DefaultURL
	}
	s := &TTSService{apiKey: config.APIKey, url: u, offset: make(map[string]float64)}
	aggregation := tts.AggregateSentences
	if config.StreamTokens {
		aggregation = tts.AggregateTokens
	}
	s.Base = tts.NewStreaming("RimeTTS", s, tts.Options{
		Settings:       defaultSettings(config),
		SampleRate:     config.SampleRate,
		Aggregation:    aggregation,
		WordTimestamps: true,
	})
	return s
}

func params(settings frames.TTSSettings, sampleRate int) map[string]any {
	p := make(map[string]any, 6+len(settings.Extra))
	p["speaker"] = settings.Voice
	p["modelId"] = settings.Model
	p["audioFormat"] = "pcm"
	p["samplingRate"] = sampleRate
	if settings.Language != "" {
		p["lang"] = Language(settings.Language)
	}
	if settings.Speed != nil && *settings.Speed > 0 {
		p["speedAlpha"] = 1 / *settings.Speed
	}
	for k, v := range settings.Extra {
		p[k] = v
	}
	return p
}

func (s *TTSService) endpoint() (string, http.Header) {
	q := url.Values{}
	for k, v := range params(s.Settings(), s.SampleRate()) {
		if v == nil {
			continue
		}
		switch t := v.(type) {
		case float64:
			q.Set(k, strconv.FormatFloat(t, 'f', -1, 64))
		default:
			q.Set(k, fmt.Sprint(t))
		}
	}
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.apiKey)
	return s.url + "?" + q.Encode(), header
}

func (s *TTSService) Connect(ctx context.Context, sink tts.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:         "RimeTTS",
		Endpoint:     s.endpoint,
		OnDisconnect: func(error) { s.resetOffsets() },
		OnMessage:    func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:      sink.Error,
		CloseMessage: &reconnect.Message{Type: websocket.TextMessage, Data: eosMessage},
		PingInterval: 20 * time.Second,
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

func (s *TTSService) resetOffsets() {
	s.mu.Lock()
	clear(s.offset)
	s.mu.Unlock()
}

func (s *TTSService) sendRaw(data []byte) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("rime not connected")
	}
	if !session.Send(websocket.TextMessage, data) {
		return reconnect.ErrQueueFull
	}
	return nil
}

func (s *TTSService) Speak(contextID, text string) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("rime not connected")
	}
	if !strings.HasSuffix(text, " ") {
		text += " "
	}
	return session.SendJSON(struct {
		Text      string `json:"text"`
		ContextID string `json:"contextId"`
	}{Text: text, ContextID: contextID})
}

func (s *TTSService) Flush(string) error {
	return s.sendRaw(flushMessage)
}

func (s *TTSService) Cancel(contextID string) error {
	s.mu.Lock()
	delete(s.offset, contextID)
	s.mu.Unlock()
	session := s.session.Load()
	if session == nil {
		return nil
	}
	session.Clear()
	return s.sendRaw(clearMessage)
}

func (s *TTSService) ApplySettings(frames.TTSSettings) error {
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
	return nil
}

type timestamps struct {
	Words []string  `json:"words"`
	Start []float64 `json:"start"`
	End   []float64 `json:"end"`
}

type response struct {
	Type           string      `json:"type"`
	ContextID      string      `json:"contextId"`
	Data           string      `json:"data"`
	Message        string      `json:"message"`
	WordTimestamps *timestamps `json:"word_timestamps"`
}

func (s *TTSService) onMessage(sink tts.Sink, data []byte) {
	var resp response
	if err := json.Unmarshal(data, &resp); err != nil {
		return
	}
	switch resp.Type {
	case "chunk":
		if resp.Data == "" {
			return
		}
		audio, err := base64.StdEncoding.DecodeString(resp.Data)
		if err != nil {
			sink.Error(err, false)
			return
		}
		sink.Audio(resp.ContextID, audio)
	case "timestamps":
		if words := s.wordTimes(resp.ContextID, resp.WordTimestamps); len(words) > 0 {
			sink.Words(resp.ContextID, words)
		}
	case "done":
		s.mu.Lock()
		delete(s.offset, resp.ContextID)
		s.mu.Unlock()
		sink.Done(resp.ContextID)
	case "error":
		sink.Error(errors.New("rime: "+resp.Message), false)
		if resp.ContextID != "" {
			sink.Done(resp.ContextID)
		}
	}
}

func (s *TTSService) wordTimes(contextID string, ts *timestamps) []tts.Word {
	if ts == nil || len(ts.Words) == 0 || len(ts.Start) < len(ts.Words) {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	offset := s.offset[contextID]
	words := make([]tts.Word, 0, len(ts.Words))
	for i, w := range ts.Words {
		if strings.TrimSpace(w) == "" {
			continue
		}
		if strings.Trim(w, ",.!?") == "" && len(words) > 0 {
			words[len(words)-1].Text += w
			continue
		}
		words = append(words, tts.Word{Text: w, Start: time.Duration((offset + ts.Start[i]) * float64(time.Second))})
	}
	if n := len(ts.End); n > 0 && len(words) > 0 {
		s.offset[contextID] = offset + ts.End[n-1]
	}
	return words
}
