package elevenlabs

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
	DefaultWebsocketURL = "wss://api.elevenlabs.io"
	DefaultHTTPURL      = "https://api.elevenlabs.io"
)

type VoiceSettings struct {
	Stability       float64 `json:"stability,omitempty"`
	SimilarityBoost float64 `json:"similarity_boost,omitempty"`
	Style           float64 `json:"style,omitempty"`
	UseSpeakerBoost bool    `json:"use_speaker_boost,omitempty"`
	Speed           float64 `json:"speed,omitempty"`
}

type TTSConfig struct {
	APIKey             string
	VoiceID            string
	Model              string
	OutputFormat       string
	UseStreaming       bool
	VoiceSettings      *VoiceSettings
	Language           string
	AggregateSentences bool
	StreamTokens       bool
	WebsocketURL       string
	HTTPURL            string
	HTTPClient         *http.Client
}

var multilingualModels = map[string]bool{
	"eleven_flash_v2_5": true,
	"eleven_turbo_v2_5": true,
}

type wordState struct {
	partial      string
	partialStart float64
	offset       float64
}

type TTSService struct {
	*tts.Base
	apiKey        string
	outputFormat  string
	voiceSettings *VoiceSettings
	wsURL         string
	httpURL       string
	client        *http.Client
	session       atomic.Pointer[reconnect.Session]

	mu       sync.Mutex
	initSent map[string]bool
	open     string
	words    map[string]*wordState
}

func NewTTSService(config TTSConfig) *TTSService {
	voiceSettings := config.VoiceSettings
	if voiceSettings == nil {
		voiceSettings = &VoiceSettings{Stability: 0.5, SimilarityBoost: 0.75}
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &TTSService{
		apiKey:        config.APIKey,
		outputFormat:  config.OutputFormat,
		voiceSettings: voiceSettings,
		wsURL:         orDefault(config.WebsocketURL, DefaultWebsocketURL),
		httpURL:       orDefault(config.HTTPURL, DefaultHTTPURL),
		client:        client,
		initSent:      make(map[string]bool),
		words:         make(map[string]*wordState),
	}
	sampleRate, codec, bytesPerSample := parseOutputFormat(config.OutputFormat)
	aggregation := tts.AggregateSentences
	if config.StreamTokens && config.UseStreaming {
		aggregation = tts.AggregateTokens
	}
	opts := tts.Options{
		Settings:       frames.TTSSettings{Model: config.Model, Voice: config.VoiceID, Language: config.Language},
		SampleRate:     sampleRate,
		Codec:          codec,
		BytesPerSample: bytesPerSample,
		Aggregation:    aggregation,
		WordTimestamps: config.UseStreaming,
	}
	if config.UseStreaming {
		s.Base = tts.NewStreaming("ElevenLabsTTS", s, opts)
	} else {
		s.Base = tts.NewHTTP("ElevenLabsTTS", s, opts)
	}
	return s
}

func orDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

func parseOutputFormat(format string) (int, string, int) {
	if format == "" {
		return 0, "linear16", 2
	}
	parts := strings.Split(format, "_")
	rate := 0
	if len(parts) > 1 {
		rate, _ = strconv.Atoi(parts[1])
	}
	switch parts[0] {
	case "ulaw":
		return rate, "mulaw", 1
	case "alaw":
		return rate, "alaw", 1
	case "mp3":
		return rate, "mp3", 2
	case "opus":
		return rate, "opus", 2
	default:
		return rate, "linear16", 2
	}
}

func (s *TTSService) format() string {
	if s.outputFormat != "" {
		return s.outputFormat
	}
	return "pcm_" + strconv.Itoa(s.SampleRate())
}

func (s *TTSService) voiceSettingsMap(settings frames.TTSSettings) map[string]any {
	vs := s.voiceSettings
	m := map[string]any{}
	if vs != nil {
		if vs.Stability != 0 {
			m["stability"] = vs.Stability
		}
		if vs.SimilarityBoost != 0 {
			m["similarity_boost"] = vs.SimilarityBoost
		}
		if vs.Style != 0 {
			m["style"] = vs.Style
		}
		if vs.UseSpeakerBoost {
			m["use_speaker_boost"] = true
		}
		if vs.Speed != 0 {
			m["speed"] = vs.Speed
		}
	}
	if settings.Speed != nil {
		m["speed"] = *settings.Speed
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

func (s *TTSService) endpoint() (string, http.Header) {
	settings := s.Settings()
	params := url.Values{}
	params.Set("model_id", settings.Model)
	params.Set("output_format", s.format())
	params.Set("auto_mode", "true")
	params.Set("inactivity_timeout", "180")
	if settings.Language != "" && multilingualModels[settings.Model] {
		params.Set("language_code", settings.Language)
	}
	header := http.Header{}
	header.Set("xi-api-key", s.apiKey)
	return fmt.Sprintf("%s/v1/text-to-speech/%s/multi-stream-input?%s", s.wsURL, url.PathEscape(settings.Voice), params.Encode()), header
}

func (s *TTSService) Connect(ctx context.Context, sink tts.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:     "ElevenLabsTTS",
		Endpoint: s.endpoint,
		OnDisconnect: func(error) {
			s.mu.Lock()
			clear(s.initSent)
			clear(s.words)
			s.open = ""
			s.mu.Unlock()
		},
		OnMessage:     func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:       sink.Error,
		CloseMessage:  &reconnect.Message{Type: websocket.TextMessage, Data: []byte(`{"close_socket":true}`)},
		KeepaliveIdle: 10 * time.Second,
		Keepalive:     s.keepalive,
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *TTSService) keepalive() (reconnect.Message, bool) {
	s.mu.Lock()
	id := s.open
	s.mu.Unlock()
	msg := map[string]string{"text": ""}
	if id != "" {
		msg["context_id"] = id
	}
	data, err := json.Marshal(msg)
	if err != nil {
		return reconnect.Message{}, false
	}
	return reconnect.Message{Type: websocket.TextMessage, Data: data}, true
}

func (s *TTSService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func (s *TTSService) send(v any) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("elevenlabs not connected")
	}
	return session.SendJSON(v)
}

func (s *TTSService) Speak(contextID, text string) error {
	s.mu.Lock()
	needInit := !s.initSent[contextID]
	s.initSent[contextID] = true
	s.open = contextID
	s.mu.Unlock()
	if needInit {
		init := map[string]any{"text": " ", "context_id": contextID}
		if vs := s.voiceSettingsMap(s.Settings()); vs != nil {
			init["voice_settings"] = vs
		}
		if err := s.send(init); err != nil {
			return err
		}
	}
	if !strings.HasSuffix(text, " ") {
		text += " "
	}
	return s.send(map[string]string{"text": text, "context_id": contextID})
}

func (s *TTSService) Flush(contextID string) error {
	if err := s.send(map[string]any{"context_id": contextID, "flush": true}); err != nil {
		return err
	}
	return s.closeContext(contextID)
}

func (s *TTSService) Cancel(contextID string) error {
	s.mu.Lock()
	delete(s.words, contextID)
	s.mu.Unlock()
	return s.closeContext(contextID)
}

func (s *TTSService) closeContext(contextID string) error {
	s.mu.Lock()
	delete(s.initSent, contextID)
	if s.open == contextID {
		s.open = ""
	}
	s.mu.Unlock()
	return s.send(map[string]any{"context_id": contextID, "close_context": true})
}

func (s *TTSService) ApplySettings(frames.TTSSettings) error {
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
	return nil
}

type alignment struct {
	Chars            []string  `json:"chars"`
	CharStartTimesMs []float64 `json:"charStartTimesMs"`
	CharDurationsMs  []float64 `json:"charDurationsMs"`
}

type wsResponse struct {
	Audio     string     `json:"audio"`
	IsFinal   bool       `json:"isFinal"`
	ContextID string     `json:"contextId"`
	Alignment *alignment `json:"alignment"`
	Error     string     `json:"error"`
	Message   string     `json:"message"`
}

func (s *TTSService) onMessage(sink tts.Sink, data []byte) {
	var resp wsResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return
	}
	if resp.Error != "" {
		sink.Error(fmt.Errorf("elevenlabs: %s %s", resp.Error, resp.Message), false)
	}
	if resp.Audio != "" {
		audio, err := base64.StdEncoding.DecodeString(resp.Audio)
		if err != nil {
			sink.Error(err, false)
		} else {
			sink.Audio(resp.ContextID, audio)
		}
	}
	if resp.Alignment != nil {
		if words := s.wordTimes(resp.ContextID, resp.Alignment); len(words) > 0 {
			sink.Words(resp.ContextID, words)
		}
	}
	if resp.IsFinal {
		s.mu.Lock()
		state := s.words[resp.ContextID]
		delete(s.words, resp.ContextID)
		s.mu.Unlock()
		if state != nil && state.partial != "" {
			sink.Words(resp.ContextID, []tts.Word{{Text: state.partial, Start: seconds(state.partialStart)}})
		}
		sink.Done(resp.ContextID)
	}
}

func seconds(v float64) time.Duration {
	return time.Duration(v * float64(time.Second))
}

func (s *TTSService) wordTimes(contextID string, a *alignment) []tts.Word {
	if len(a.Chars) != len(a.CharStartTimesMs) {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state := s.words[contextID]
	if state == nil {
		state = &wordState{}
		s.words[contextID] = state
	}
	var words []tts.Word
	current, start := state.partial, state.partialStart
	for i, ch := range a.Chars {
		if ch == " " {
			if current != "" {
				words = append(words, tts.Word{Text: current, Start: seconds(start)})
				current = ""
			}
			continue
		}
		if current == "" {
			start = state.offset + a.CharStartTimesMs[i]/1000
		}
		current += ch
	}
	state.partial, state.partialStart = current, start
	if n := len(a.CharStartTimesMs); n > 0 && len(a.CharDurationsMs) == n {
		state.offset += (a.CharStartTimesMs[n-1] + a.CharDurationsMs[n-1]) / 1000
	}
	return words
}

func (s *TTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	body := map[string]any{"text": req.Text, "model_id": req.Settings.Model}
	if vs := s.voiceSettingsMap(req.Settings); vs != nil {
		body["voice_settings"] = vs
	}
	if req.Settings.Language != "" && multilingualModels[req.Settings.Model] {
		body["language_code"] = req.Settings.Language
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	endpoint := fmt.Sprintf("%s/v1/text-to-speech/%s/stream?output_format=%s",
		s.httpURL, url.PathEscape(req.Settings.Voice), url.QueryEscape(s.format()))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("xi-api-key", s.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("elevenlabs HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return resp.Body, nil
}
