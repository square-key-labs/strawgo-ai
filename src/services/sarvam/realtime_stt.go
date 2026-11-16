package sarvam

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

const (
	sarvamRealtimeSTTURL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"

	DefaultRealtimeModel = "saaras:v3-realtime"

	defaultRealtimeLanguage   = "en-IN"
	defaultRealtimeStreamType = "balanced"
	realtimeChunkMS           = 50
	rejectedSettingsCloseCode = 4000
)

var supportedRealtimeModels = map[string]bool{
	"saaras:v3-realtime": true,
	"saaras:v4":          true,
}

var supportedRealtimeSampleRates = map[int]bool{8000: true, 16000: true}

var realtimeLanguageCodes = map[string]string{
	"as":  "as-IN",
	"bn":  "bn-IN",
	"brx": "brx-IN",
	"doi": "doi-IN",
	"en":  "en-IN",
	"gu":  "gu-IN",
	"hi":  "hi-IN",
	"kn":  "kn-IN",
	"kok": "kok-IN",
	"ks":  "ks-IN",
	"mai": "mai-IN",
	"ml":  "ml-IN",
	"mni": "mni-IN",
	"mr":  "mr-IN",
	"ne":  "ne-IN",
	"or":  "or-IN",
	"pa":  "pa-IN",
	"sa":  "sa-IN",
	"sat": "sat-IN",
	"sd":  "sd-IN",
	"ta":  "ta-IN",
	"te":  "te-IN",
	"ur":  "ur-IN",
}

type RealtimeSTTConfig struct {
	APIKey              string
	Model               string
	Language            string
	StreamType          string
	Mode                string
	Prompt              string
	Threshold           *float64
	SilenceDurationMS   *int
	MinSpeechDurationMS *int
	PrefixPaddingMS     *int
	ReturnTimestamps    bool
	Keyterms            []string
	SampleRate          int
	KeepaliveInterval   time.Duration
	BaseURL             string
}

type RealtimeSTTService struct {
	*stt.Base
	cfg     RealtimeSTTConfig
	session atomic.Pointer[reconnect.Session]

	mu           sync.Mutex
	audioBuf     []byte
	speechActive bool
	pendingText  string
	pendingLang  string
	appliedModel string
}

func NewRealtimeSTTService(config RealtimeSTTConfig) (*RealtimeSTTService, error) {
	if config.Model == "" {
		config.Model = DefaultRealtimeModel
	}
	if err := validateRealtimeModel(config.Model); err != nil {
		return nil, err
	}
	if config.SampleRate != 0 && !supportedRealtimeSampleRates[config.SampleRate] {
		return nil, fmt.Errorf("sarvam realtime: unsupported sample rate %d, allowed values: 8000, 16000", config.SampleRate)
	}
	if config.Language == "" {
		config.Language = defaultRealtimeLanguage
	}
	if config.StreamType == "" {
		config.StreamType = defaultRealtimeStreamType
	}
	if config.Mode == "" {
		config.Mode = defaultMode
	}
	s := &RealtimeSTTService{cfg: config, appliedModel: config.Model}
	s.Base = stt.NewStreaming("SarvamRealtimeSTT", s, stt.Options{
		Provider:   "sarvam",
		Settings:   frames.STTSettings{Model: config.Model, Language: mapRealtimeLanguage(config.Language)},
		SampleRate: config.SampleRate,
	})
	return s, nil
}

func validateRealtimeModel(model string) error {
	if supportedRealtimeModels[model] {
		return nil
	}
	allowed := make([]string, 0, len(supportedRealtimeModels))
	for m := range supportedRealtimeModels {
		allowed = append(allowed, m)
	}
	sort.Strings(allowed)
	return fmt.Errorf("sarvam realtime: unsupported model %q, allowed values: %s", model, strings.Join(allowed, ", "))
}

func mapRealtimeLanguage(lang string) string {
	if code, ok := realtimeLanguageCodes[lang]; ok {
		return code
	}
	return lang
}

func (s *RealtimeSTTService) SetLanguage(lang string) {
	s.Base.SetLanguage(mapRealtimeLanguage(lang))
}

func (s *RealtimeSTTService) languageCode() string {
	if lang := mapRealtimeLanguage(s.Settings().Language); lang != "" {
		return lang
	}
	return "auto"
}

func (s *RealtimeSTTService) endpoint() (string, http.Header) {
	params := url.Values{}
	params.Set("language_code", s.languageCode())
	params.Set("stream_type", s.cfg.StreamType)
	params.Set("endpointing", "vad")
	params.Set("encoding", "linear16")
	params.Set("sample_rate", strconv.Itoa(s.SampleRate()))
	params.Set("model", s.Settings().Model)
	params.Set("mode", s.cfg.Mode)
	params.Set("return_timestamps", strconv.FormatBool(s.cfg.ReturnTimestamps))
	if s.cfg.Keyterms != nil {
		keyterms, _ := json.Marshal(s.cfg.Keyterms)
		params.Set("keyterms", string(keyterms))
	}
	if s.cfg.Prompt != "" {
		params.Set("prompt", s.cfg.Prompt)
	}
	if s.cfg.PrefixPaddingMS != nil {
		params.Set("prefix_padding_ms", strconv.Itoa(*s.cfg.PrefixPaddingMS))
	}
	if s.cfg.Threshold != nil {
		params.Set("threshold", strconv.FormatFloat(*s.cfg.Threshold, 'f', -1, 64))
	}
	if s.cfg.SilenceDurationMS != nil {
		params.Set("silence_duration_ms", strconv.Itoa(*s.cfg.SilenceDurationMS))
	}
	if s.cfg.MinSpeechDurationMS != nil {
		params.Set("min_speech_duration_ms", strconv.Itoa(*s.cfg.MinSpeechDurationMS))
	}
	base := s.cfg.BaseURL
	if base == "" {
		base = sarvamRealtimeSTTURL
	}
	header := http.Header{}
	header.Set("API-SUBSCRIPTION-KEY", s.cfg.APIKey)
	return base + "?" + params.Encode(), header
}

func (s *RealtimeSTTService) Connect(ctx context.Context, sink stt.Sink) error {
	if rate := s.SampleRate(); !supportedRealtimeSampleRates[rate] {
		return fmt.Errorf("sarvam realtime: unsupported sample rate %d, allowed values: 8000, 16000", rate)
	}
	cfg := reconnect.Config{
		Name:         "SarvamRealtimeSTT",
		Endpoint:     s.endpoint,
		OnMessage:    func(_ int, data []byte) { s.onMessage(sink, data) },
		OnDisconnect: func(err error) { s.onDisconnect(sink, err) },
		OnError:      sink.Error,
		CloseMessage: &reconnect.Message{Type: websocket.TextMessage, Data: []byte(`{"event":"end"}`)},
	}
	if s.cfg.KeepaliveInterval > 0 {
		cfg.KeepaliveIdle = s.cfg.KeepaliveInterval
		cfg.Keepalive = func() (reconnect.Message, bool) {
			return reconnect.Message{Type: websocket.TextMessage, Data: []byte(`{"event":"ping"}`)}, true
		}
	}
	session := reconnect.New(cfg)
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func marshalRealtimeAudio(raw []byte) []byte {
	const prefix = `{"event":"audio_input","audio":"`
	buf := make([]byte, 0, len(prefix)+base64.StdEncoding.EncodedLen(len(raw))+2)
	buf = append(buf, prefix...)
	buf = base64.StdEncoding.AppendEncode(buf, raw)
	return append(buf, `"}`...)
}

func (s *RealtimeSTTService) SendAudio(audio []byte) {
	session := s.session.Load()
	if session == nil {
		return
	}
	chunk := s.SampleRate() * 2 * realtimeChunkMS / 1000
	s.mu.Lock()
	s.audioBuf = append(s.audioBuf, audio...)
	var out [][]byte
	for len(s.audioBuf) >= chunk {
		out = append(out, marshalRealtimeAudio(s.audioBuf[:chunk]))
		s.audioBuf = s.audioBuf[chunk:]
	}
	if len(s.audioBuf) == 0 {
		s.audioBuf = nil
	}
	s.mu.Unlock()
	for _, msg := range out {
		session.SendDroppable(websocket.TextMessage, msg)
	}
}

func (s *RealtimeSTTService) Disconnect(graceful bool) {
	session := s.session.Load()
	if session == nil {
		return
	}
	s.mu.Lock()
	rest := s.audioBuf
	s.audioBuf = nil
	s.mu.Unlock()
	if graceful && len(rest) > 0 {
		session.Send(websocket.TextMessage, marshalRealtimeAudio(rest))
	}
	session.Close(graceful)
}

func (s *RealtimeSTTService) ApplySettings(settings frames.STTSettings) error {
	if err := validateRealtimeModel(settings.Model); err != nil {
		return err
	}
	session := s.session.Load()
	if session == nil {
		return nil
	}
	s.mu.Lock()
	modelChanged := settings.Model != s.appliedModel
	s.appliedModel = settings.Model
	s.mu.Unlock()
	if modelChanged {
		session.Reconnect()
		return nil
	}
	return session.SendJSON(map[string]string{"event": "config.update", "language_code": s.languageCode()})
}

type realtimeMsg struct {
	Event     string `json:"event"`
	Text      string `json:"text"`
	Language  string `json:"language"`
	RequestID string `json:"request_id"`
}

func (s *RealtimeSTTService) onMessage(sink stt.Sink, raw []byte) {
	var msg realtimeMsg
	if err := json.Unmarshal(raw, &msg); err != nil {
		return
	}
	switch msg.Event {
	case "vad.speech_start":
		s.mu.Lock()
		started := !s.speechActive
		s.speechActive = true
		s.mu.Unlock()
		if started {
			sink.Push(frames.NewUserStartedSpeakingFrame(), frames.Upstream)
		}
	case "vad.speech_end", "session.end":
		s.completeUtterance(sink)
	case "transcript.partial":
		text := strings.TrimSpace(msg.Text)
		if text == "" {
			return
		}
		lang := s.frameLanguage(msg.Language)
		s.mu.Lock()
		s.pendingText, s.pendingLang = text, lang
		s.mu.Unlock()
		sink.Transcript(stt.Transcript{Text: text, Language: lang})
	case "transcript.final":
		s.mu.Lock()
		s.pendingText, s.pendingLang = "", ""
		s.mu.Unlock()
		if text := strings.TrimSpace(msg.Text); text != "" {
			sink.Transcript(stt.Transcript{Text: text, Final: true, Finalized: true, Language: s.frameLanguage(msg.Language)})
		}
	case "error":
		sink.Error(&sarvamError{"realtime: " + string(raw)}, false)
	}
}

func (s *RealtimeSTTService) completeUtterance(sink stt.Sink) {
	s.mu.Lock()
	active := s.speechActive
	s.speechActive = false
	s.mu.Unlock()
	if active {
		sink.Push(frames.NewUserStoppedSpeakingFrame(), frames.Upstream)
	}
}

func (s *RealtimeSTTService) onDisconnect(sink stt.Sink, err error) {
	s.completeUtterance(sink)
	s.mu.Lock()
	text, lang := s.pendingText, s.pendingLang
	s.pendingText, s.pendingLang = "", ""
	s.audioBuf = nil
	s.mu.Unlock()
	if text != "" {
		sink.Transcript(stt.Transcript{Text: text, Final: true, Language: lang})
	}
	var closeErr *websocket.CloseError
	if errors.As(err, &closeErr) && closeErr.Code == rejectedSettingsCloseCode {
		sink.Error(&sarvamError{"realtime: connection rejected: " + closeErr.Text}, true)
		if session := s.session.Load(); session != nil {
			go session.Close(false)
		}
	}
}

func (s *RealtimeSTTService) frameLanguage(raw string) string {
	configured := s.languageCode()
	code := raw
	if code == "" {
		code = configured
	}
	if code != "auto" && !strings.Contains(code, "-") {
		if configured != "auto" && strings.HasPrefix(configured, code+"-") {
			code = configured
		} else {
			code = mapRealtimeLanguage(code)
		}
	}
	if code == "auto" {
		return ""
	}
	return code
}
