package inworld

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DefaultURL      = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"
	DefaultHTTPURL  = "https://api.inworld.ai/tts/v1/voice:stream"
	DefaultModel    = "inworld-tts-2"
	DefaultVoice    = "Ashley"
	DefaultEncoding = "PCM"
)

var (
	breakTag  = regexp.MustCompile(`(?i)<break\b[^>]*>`)
	languages = map[string]string{
		"ar": "ar-SA", "de": "de-DE", "en": "en-US", "es": "es-ES", "fr": "fr-FR", "he": "he-IL",
		"hi": "hi-IN", "it": "it-IT", "ja": "ja-JP", "ko": "ko-KR", "nl": "nl-NL", "pl": "pl-PL",
		"pt": "pt-BR", "ru": "ru-RU", "zh": "zh-CN",
	}
)

func Language(code string) string {
	if strings.ContainsAny(code, "-_") {
		return code
	}
	if mapped, ok := languages[strings.ToLower(code)]; ok {
		return mapped
	}
	return code
}

type TTSConfig struct {
	APIKey                 string
	VoiceID                string
	Model                  string
	Language               string
	Speed                  float64
	SampleRate             int
	Encoding               string
	Params                 map[string]any
	AutoMode               *bool
	ApplyTextNormalization string
	MaxBufferDelayMs       int
	BufferCharThreshold    int
	StreamTokens           bool
	URL                    string
	HTTPClient             *http.Client
}

type contextState struct {
	offset   float64
	genEnd   float64
	text     strings.Builder
	sawWords bool
}

type TTSService struct {
	*tts.Base
	apiKey       string
	url          string
	encoding     string
	autoMode     bool
	normalize    string
	maxDelay     int
	charLimit    int
	streamTokens bool
	session      atomic.Pointer[reconnect.Session]

	mu       sync.Mutex
	contexts map[string]*contextState
	open     string
}

func defaultSettings(config TTSConfig) frames.TTSSettings {
	settings := frames.TTSSettings{Model: config.Model, Voice: config.VoiceID, Language: config.Language}
	if settings.Model == "" {
		settings.Model = DefaultModel
	}
	if settings.Voice == "" {
		settings.Voice = DefaultVoice
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
	encoding := config.Encoding
	if encoding == "" {
		encoding = DefaultEncoding
	}
	autoMode := !config.StreamTokens
	if config.AutoMode != nil {
		autoMode = *config.AutoMode
	}
	maxDelay := config.MaxBufferDelayMs
	if maxDelay <= 0 {
		maxDelay = 3000
	}
	s := &TTSService{
		apiKey:       config.APIKey,
		url:          u,
		encoding:     encoding,
		autoMode:     autoMode,
		normalize:    config.ApplyTextNormalization,
		maxDelay:     maxDelay,
		charLimit:    config.BufferCharThreshold,
		streamTokens: config.StreamTokens,
		contexts:     make(map[string]*contextState),
	}
	aggregation := tts.AggregateSentences
	if config.StreamTokens {
		aggregation = tts.AggregateTokens
	}
	s.Base = tts.NewStreaming("InworldTTS", s, tts.Options{
		Settings:       defaultSettings(config),
		SampleRate:     config.SampleRate,
		Aggregation:    aggregation,
		WordTimestamps: true,
	})
	return s
}

func audioConfig(settings frames.TTSSettings, encoding string, sampleRate int) map[string]any {
	cfg := map[string]any{"audioEncoding": encoding, "sampleRateHertz": sampleRate}
	if settings.Speed != nil && *settings.Speed > 0 {
		cfg["speakingRate"] = *settings.Speed
	}
	return cfg
}

func (s *TTSService) endpoint() (string, http.Header) {
	header := http.Header{}
	header.Set("Authorization", "Basic "+s.apiKey)
	header.Set("X-Request-Id", uuid.NewString())
	return s.url, header
}

func (s *TTSService) Connect(ctx context.Context, sink tts.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:     "InworldTTS",
		Endpoint: s.endpoint,
		OnDisconnect: func(error) {
			s.mu.Lock()
			clear(s.contexts)
			s.open = ""
			s.mu.Unlock()
		},
		OnMessage:     func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:       sink.Error,
		KeepaliveIdle: 60 * time.Second,
		Keepalive:     s.keepalive,
		PingInterval:  20 * time.Second,
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

func (s *TTSService) keepalive() (reconnect.Message, bool) {
	s.mu.Lock()
	id := s.open
	s.mu.Unlock()
	msg := map[string]any{"send_text": map[string]string{"text": ""}}
	if id != "" {
		msg["contextId"] = id
	}
	data, err := json.Marshal(msg)
	if err != nil {
		return reconnect.Message{}, false
	}
	return reconnect.Message{Type: websocket.TextMessage, Data: data}, true
}

func (s *TTSService) send(v any) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("inworld not connected")
	}
	return session.SendJSON(v)
}

func (s *TTSService) createConfig() map[string]any {
	settings := s.Settings()
	cfg := map[string]any{
		"voiceId":     settings.Voice,
		"modelId":     settings.Model,
		"audioConfig": audioConfig(settings, s.encoding, s.SampleRate()),
		"autoMode":    s.autoMode,
	}
	if settings.Language != "" {
		cfg["language"] = Language(settings.Language)
	}
	if s.normalize != "" {
		cfg["applyTextNormalization"] = s.normalize
	}
	sentenceBoundary := s.autoMode && s.streamTokens && strings.HasPrefix(settings.Model, "inworld-tts-2")
	if sentenceBoundary {
		cfg["autoModeStrategy"] = "SENTENCE_BOUNDARY"
	}
	cfg["timestampTransportStrategy"] = "ASYNC"
	cfg["maxBufferDelayMs"] = s.maxDelay
	if s.charLimit > 0 || !sentenceBoundary {
		limit := s.charLimit
		if limit <= 0 {
			limit = 250
		}
		cfg["bufferCharThreshold"] = limit
	}
	cfg["timestampType"] = "WORD"
	for k, v := range settings.Extra {
		cfg[k] = v
	}
	return cfg
}

func (s *TTSService) Speak(contextID, text string) error {
	if !strings.HasSuffix(text, " ") {
		text += " "
	}
	s.mu.Lock()
	state := s.contexts[contextID]
	created := state == nil
	if created {
		state = &contextState{}
		s.contexts[contextID] = state
	}
	state.text.WriteString(text)
	s.open = contextID
	s.mu.Unlock()
	if created {
		if err := s.send(map[string]any{"create": s.createConfig(), "contextId": contextID}); err != nil {
			return err
		}
	}
	return s.send(map[string]any{"send_text": map[string]string{"text": text}, "contextId": contextID})
}

func (s *TTSService) Flush(contextID string) error {
	if err := s.send(map[string]any{"flush_context": struct{}{}, "contextId": contextID}); err != nil {
		return err
	}
	return s.send(map[string]any{"close_context": struct{}{}, "contextId": contextID})
}

func (s *TTSService) Cancel(contextID string) error {
	s.mu.Lock()
	delete(s.contexts, contextID)
	if s.open == contextID {
		s.open = ""
	}
	s.mu.Unlock()
	return s.send(map[string]any{"cancel_context": struct{}{}, "contextId": contextID})
}

func (s *TTSService) ApplySettings(frames.TTSSettings) error {
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
	return nil
}

type wordAlignment struct {
	Words []string  `json:"words"`
	Start []float64 `json:"wordStartTimeSeconds"`
	End   []float64 `json:"wordEndTimeSeconds"`
}

type timestampInfo struct {
	WordAlignment *wordAlignment `json:"wordAlignment"`
}

type wsMessage struct {
	Result *struct {
		ContextID    string `json:"contextId"`
		ContextIDAlt string `json:"context_id"`
		Status       *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"status"`
		AudioChunk *struct {
			AudioContent  string         `json:"audioContent"`
			TimestampInfo *timestampInfo `json:"timestampInfo"`
		} `json:"audioChunk"`
		FlushCompleted json.RawMessage `json:"flushCompleted"`
		ContextClosed  json.RawMessage `json:"contextClosed"`
	} `json:"result"`
	Error json.RawMessage `json:"error"`
}

func decodeAudio(b64 string) ([]byte, error) {
	audio, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, err
	}
	if len(audio) > 44 && bytes.HasPrefix(audio, []byte("RIFF")) {
		audio = audio[44:]
	}
	return audio, nil
}

func wordTimes(a *wordAlignment, offset float64) ([]tts.Word, float64) {
	if a == nil || len(a.Words) == 0 || len(a.Start) != len(a.Words) {
		return nil, 0
	}
	words := make([]tts.Word, 0, len(a.Words))
	for i, w := range a.Words {
		w = breakTag.ReplaceAllString(w, "")
		if strings.TrimSpace(w) == "" {
			continue
		}
		words = append(words, tts.Word{Text: w, Start: time.Duration((offset + a.Start[i]) * float64(time.Second))})
	}
	end := 0.0
	if n := len(a.End); n > 0 {
		end = a.End[n-1]
	}
	return words, end
}

func benign(msg string) bool {
	lower := strings.ToLower(msg)
	return strings.Contains(lower, "not found") || strings.Contains(lower, "context_id is required") || strings.Contains(lower, "no open context")
}

func (s *TTSService) onMessage(sink tts.Sink, data []byte) {
	var msg wsMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return
	}
	if len(msg.Error) > 0 && string(msg.Error) != "null" {
		sink.Error(fmt.Errorf("inworld: %s", msg.Error), false)
		return
	}
	r := msg.Result
	if r == nil {
		return
	}
	id := r.ContextID
	if id == "" {
		id = r.ContextIDAlt
	}
	if r.Status != nil && r.Status.Code != 0 {
		if !benign(r.Status.Message) {
			sink.Error(fmt.Errorf("inworld: %s", r.Status.Message), false)
		}
		return
	}
	if id == "" {
		return
	}
	if chunk := r.AudioChunk; chunk != nil {
		if chunk.AudioContent != "" {
			audio, err := decodeAudio(chunk.AudioContent)
			if err != nil {
				sink.Error(err, false)
			} else {
				sink.Audio(id, audio)
			}
		}
		if chunk.TimestampInfo != nil {
			s.mu.Lock()
			state := s.contexts[id]
			var words []tts.Word
			if state != nil {
				var end float64
				words, end = wordTimes(chunk.TimestampInfo.WordAlignment, state.offset)
				if end > 0 {
					state.genEnd = state.offset + end
				}
				if len(words) > 0 {
					state.sawWords = true
				}
			}
			s.mu.Unlock()
			if len(words) > 0 {
				sink.Words(id, words)
			}
		}
	}
	if len(r.FlushCompleted) > 0 {
		s.mu.Lock()
		if state := s.contexts[id]; state != nil {
			state.offset = state.genEnd
		}
		s.mu.Unlock()
	}
	if len(r.ContextClosed) > 0 {
		s.mu.Lock()
		state := s.contexts[id]
		delete(s.contexts, id)
		if s.open == id {
			s.open = ""
		}
		s.mu.Unlock()
		if state != nil && !state.sawWords {
			if text := strings.TrimSpace(state.text.String()); text != "" {
				sink.Words(id, []tts.Word{{Text: text}})
			}
		}
		sink.Done(id)
	}
}
