package sarvam

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
	DefaultTTSURL      = "wss://api.sarvam.ai/text-to-speech/ws"
	DefaultTTSHTTPURL  = "https://api.sarvam.ai"
	DefaultTTSModel    = "bulbul:v3"
	DefaultTTSVoice    = "shubh"
	DefaultTTSLanguage = "en-IN"
)

var (
	ttsFlushMessage = []byte(`{"type":"flush"}`)
	ttsPingMessage  = []byte(`{"type":"ping"}`)
)

type ttsModelConfig struct {
	pitch, loudness, temperature bool
	speaker                      string
}

var ttsModels = map[string]ttsModelConfig{
	"bulbul:v2":       {pitch: true, loudness: true, speaker: "anushka"},
	"bulbul:v3-beta":  {temperature: true, speaker: "shubh"},
	"bulbul:v3":       {temperature: true, speaker: "shubh"},
	"bulbul:v4-flash": {pitch: true, loudness: true, speaker: "shubh_en_narration_gentle"},
}

var ttsLanguages = map[string]string{
	"bn": "bn-IN", "en": "en-IN", "gu": "gu-IN", "hi": "hi-IN", "kn": "kn-IN",
	"ml": "ml-IN", "mr": "mr-IN", "od": "od-IN", "or": "od-IN", "pa": "pa-IN", "ta": "ta-IN", "te": "te-IN",
}

func ttsLanguage(code string) string {
	if strings.ContainsAny(code, "-_") {
		return strings.ReplaceAll(code, "_", "-")
	}
	if mapped, ok := ttsLanguages[strings.ToLower(code)]; ok {
		return mapped
	}
	return code
}

type TTSConfig struct {
	APIKey       string
	Model        string
	VoiceID      string
	Language     string
	Pace         float64
	SampleRate   int
	Params       map[string]any
	StreamTokens bool
	URL          string
	HTTPClient   *http.Client
}

func ttsSettings(config TTSConfig) frames.TTSSettings {
	settings := frames.TTSSettings{Model: config.Model, Voice: config.VoiceID, Language: config.Language}
	if settings.Model == "" {
		settings.Model = DefaultTTSModel
	}
	if settings.Voice == "" {
		settings.Voice = ttsModels[settings.Model].speaker
		if settings.Voice == "" {
			settings.Voice = DefaultTTSVoice
		}
	}
	if settings.Language == "" {
		settings.Language = DefaultTTSLanguage
	}
	if config.Pace > 0 {
		pace := config.Pace
		settings.Speed = &pace
	}
	if len(config.Params) > 0 {
		settings.Extra = make(map[string]any, len(config.Params))
		for k, v := range config.Params {
			settings.Extra[k] = v
		}
	}
	return settings
}

func ttsValidate(settings frames.TTSSettings) error {
	if _, ok := ttsModels[settings.Model]; !ok {
		return fmt.Errorf("sarvam TTS model %q is not supported", settings.Model)
	}
	return nil
}

func ttsFields(settings frames.TTSSettings, sampleRate int, ws bool) map[string]any {
	model := ttsModels[settings.Model]
	m := map[string]any{
		"target_language_code": ttsLanguage(settings.Language),
		"speaker":              settings.Voice,
		"model":                settings.Model,
		"enable_preprocessing": false,
		"pace":                 1.0,
	}
	if ws {
		m["speech_sample_rate"] = strconv.Itoa(sampleRate)
		m["min_buffer_size"] = 50
		m["max_chunk_length"] = 150
		m["output_audio_codec"] = "linear16"
		m["output_audio_bitrate"] = "128k"
	} else {
		m["speech_sample_rate"] = sampleRate
	}
	if settings.Speed != nil && *settings.Speed > 0 {
		m["pace"] = *settings.Speed
	}
	if model.pitch {
		m["pitch"] = 0.0
	}
	if model.loudness {
		m["loudness"] = 1.0
	}
	if model.temperature {
		m["temperature"] = 0.6
	}
	for k, v := range settings.Extra {
		if (k == "pitch" && !model.pitch) || (k == "loudness" && !model.loudness) || (k == "temperature" && !model.temperature) {
			continue
		}
		m[k] = v
	}
	if settings.Model != "bulbul:v2" {
		m["enable_preprocessing"] = true
	}
	return m
}

type TTSService struct {
	*tts.Base
	apiKey  string
	url     string
	session atomic.Pointer[reconnect.Session]

	mu       sync.Mutex
	model    string
	order    []string
	stale    bool
	sentConf string
}

func NewTTSService(config TTSConfig) *TTSService {
	u := config.URL
	if u == "" {
		u = DefaultTTSURL
	}
	settings := ttsSettings(config)
	s := &TTSService{apiKey: config.APIKey, url: u, model: settings.Model}
	aggregation := tts.AggregateSentences
	if config.StreamTokens {
		aggregation = tts.AggregateTokens
	}
	s.Base = tts.NewStreaming("SarvamTTS", s, tts.Options{
		Settings:    settings,
		SampleRate:  config.SampleRate,
		Aggregation: aggregation,
	})
	return s
}

func (s *TTSService) endpoint() (string, http.Header) {
	q := url.Values{}
	q.Set("model", s.Settings().Model)
	q.Set("send_completion_event", "true")
	header := http.Header{}
	header.Set("api-subscription-key", s.apiKey)
	return s.url + "?" + q.Encode(), header
}

func (s *TTSService) configMessage() ([]byte, error) {
	return json.Marshal(map[string]any{"type": "config", "data": ttsFields(s.Settings(), s.SampleRate(), true)})
}

func (s *TTSService) Connect(ctx context.Context, sink tts.Sink) error {
	if err := ttsValidate(s.Settings()); err != nil {
		return err
	}
	session := reconnect.New(reconnect.Config{
		Name:          "SarvamTTS",
		Endpoint:      s.endpoint,
		OnOpen:        s.onOpen,
		OnDisconnect:  func(error) { s.reset(true) },
		OnMessage:     func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:       sink.Error,
		KeepaliveIdle: 20 * time.Second,
		Keepalive: func() (reconnect.Message, bool) {
			return reconnect.Message{Type: websocket.TextMessage, Data: ttsPingMessage}, true
		},
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *TTSService) onOpen(w reconnect.Writer) error {
	msg, err := s.configMessage()
	if err != nil {
		return err
	}
	if err := w.WriteMessage(websocket.TextMessage, msg); err != nil {
		return err
	}
	s.mu.Lock()
	s.stale = false
	s.sentConf = string(msg)
	s.mu.Unlock()
	return nil
}

func (s *TTSService) reset(stale bool) {
	s.mu.Lock()
	s.order = s.order[:0]
	s.stale = stale
	s.mu.Unlock()
}

func (s *TTSService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func (s *TTSService) Speak(contextID, text string) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("sarvam TTS not connected")
	}
	s.mu.Lock()
	if n := len(s.order); n == 0 || s.order[n-1] != contextID {
		s.order = append(s.order, contextID)
	}
	s.mu.Unlock()
	return session.SendJSON(map[string]any{"type": "text", "data": map[string]string{"text": text}})
}

func (s *TTSService) Flush(string) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("sarvam TTS not connected")
	}
	if !session.Send(websocket.TextMessage, ttsFlushMessage) {
		return reconnect.ErrQueueFull
	}
	return nil
}

func (s *TTSService) Cancel(string) error {
	session := s.session.Load()
	if session == nil {
		return nil
	}
	s.reset(true)
	session.Clear()
	session.Reconnect()
	return nil
}

func (s *TTSService) ApplySettings(settings frames.TTSSettings) error {
	if err := ttsValidate(settings); err != nil {
		return err
	}
	session := s.session.Load()
	if session == nil {
		return nil
	}
	s.mu.Lock()
	modelChanged := s.model != settings.Model
	s.model = settings.Model
	s.mu.Unlock()
	if modelChanged {
		session.Reconnect()
		return nil
	}
	msg, err := s.configMessage()
	if err != nil {
		return err
	}
	s.mu.Lock()
	same := s.sentConf == string(msg)
	s.sentConf = string(msg)
	s.mu.Unlock()
	if same {
		return nil
	}
	if !session.Send(websocket.TextMessage, msg) {
		return reconnect.ErrQueueFull
	}
	return nil
}

type ttsMessage struct {
	Type string `json:"type"`
	Data struct {
		Audio     string `json:"audio"`
		EventType string `json:"event_type"`
		Message   string `json:"message"`
		RequestID string `json:"request_id"`
	} `json:"data"`
}

func (s *TTSService) current() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stale || len(s.order) == 0 {
		return ""
	}
	return s.order[0]
}

func (s *TTSService) onMessage(sink tts.Sink, data []byte) {
	var msg ttsMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return
	}
	switch msg.Type {
	case "audio":
		id := s.current()
		if id == "" || msg.Data.Audio == "" {
			return
		}
		audio, err := base64.StdEncoding.DecodeString(msg.Data.Audio)
		if err != nil {
			sink.Error(err, false)
			return
		}
		sink.Audio(id, audio)
	case "event":
		if msg.Data.EventType != "final" {
			return
		}
		s.mu.Lock()
		var done string
		if !s.stale && len(s.order) > 0 {
			done = s.order[0]
			s.order = append(s.order[:0], s.order[1:]...)
		}
		s.mu.Unlock()
		if done != "" {
			sink.Done(done)
		}
	case "error":
		text := "sarvam TTS: " + msg.Data.Message
		if msg.Data.RequestID != "" {
			text += " (request ID: " + msg.Data.RequestID + ")"
		}
		sink.Error(errors.New(text), false)
	}
}

type HTTPTTSService struct {
	*tts.Base
	apiKey  string
	baseURL string
	client  *http.Client
}

func NewHTTPTTSService(config TTSConfig) *HTTPTTSService {
	baseURL := strings.TrimRight(config.URL, "/")
	if baseURL == "" {
		baseURL = DefaultTTSHTTPURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &HTTPTTSService{apiKey: config.APIKey, baseURL: baseURL, client: client}
	s.Base = tts.NewHTTP("SarvamHTTPTTS", s, tts.Options{Settings: ttsSettings(config), SampleRate: config.SampleRate})
	return s
}

func (s *HTTPTTSService) ApplySettings(settings frames.TTSSettings) error {
	return ttsValidate(settings)
}

func (s *HTTPTTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	if err := ttsValidate(req.Settings); err != nil {
		return nil, err
	}
	body := ttsFields(req.Settings, req.SampleRate, false)
	body["text"] = req.Text
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+"/text-to-speech", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("api-subscription-key", s.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("sarvam TTS HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var out struct {
		Audios []string `json:"audios"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Audios) == 0 || out.Audios[0] == "" {
		return nil, errors.New("sarvam TTS: no audio data received")
	}
	audio := tts.SkipWAVHeader(base64.NewDecoder(base64.StdEncoding, strings.NewReader(out.Audios[0])))
	return io.NopCloser(audio), nil
}
