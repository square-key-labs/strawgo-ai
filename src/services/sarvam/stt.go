package sarvam

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

const (
	sarvamSTTURL          = "wss://api.sarvam.ai/speech-to-text/ws"
	sarvamSTTTranslateURL = "wss://api.sarvam.ai/speech-to-text-translate/ws"

	defaultModel    = "saaras:v3"
	defaultEncoding = "wav"
	defaultMode     = "transcribe"

	SarvamTTFSP99 = 500 * time.Millisecond
)

var flushMessage = []byte(`{"type":"flush"}`)

type STTConfig struct {
	APIKey             string
	Model              string
	Language           string
	Mode               string
	Prompt             string
	Encoding           string
	SampleRate         int
	VADSignals         *bool
	HighVADSensitivity *bool
	KeepaliveInterval  time.Duration
	BaseURL            string
}

type STTService struct {
	*stt.Base
	apiKey             string
	mode               string
	prompt             string
	encoding           string
	vadSignals         bool
	highVADSensitivity *bool
	vadSignalsSet      bool
	keepalive          time.Duration
	baseURL            string
	session            atomic.Pointer[reconnect.Session]
}

func NewSTTService(config STTConfig) *STTService {
	model := config.Model
	if model == "" {
		model = defaultModel
	}
	encoding := config.Encoding
	if encoding == "" {
		encoding = defaultEncoding
	}
	mode := config.Mode
	if mode == "" {
		mode = defaultMode
	}
	s := &STTService{
		apiKey:             config.APIKey,
		mode:               mode,
		prompt:             config.Prompt,
		encoding:           encoding,
		highVADSensitivity: config.HighVADSensitivity,
		vadSignalsSet:      config.VADSignals != nil,
		vadSignals:         config.VADSignals != nil && *config.VADSignals,
		keepalive:          config.KeepaliveInterval,
		baseURL:            config.BaseURL,
	}
	s.Base = stt.NewStreaming("SarvamSTT", s, stt.Options{
		Provider:   "sarvam",
		Settings:   frames.STTSettings{Model: model, Language: mapLanguage(config.Language)},
		SampleRate: config.SampleRate,
		TTFSP99:    SarvamTTFSP99,
	})
	return s
}

func mapLanguage(lang string) string {
	if code, ok := languageCodes[lang]; ok {
		return code
	}
	return lang
}

func (s *STTService) SetLanguage(lang string) {
	s.Base.SetLanguage(mapLanguage(lang))
}

func (s *STTService) endpoint() (string, http.Header) {
	settings := s.Settings()
	params := url.Values{}
	params.Set("model", settings.Model)
	params.Set("sample_rate", strconv.Itoa(s.SampleRate()))
	if settings.Language != "" {
		params.Set("language-code", mapLanguage(settings.Language))
	}
	if s.mode != "" && settings.Model == "saaras:v3" {
		params.Set("mode", s.mode)
	}
	if s.prompt != "" && settings.Model == "saaras:v2.5" {
		params.Set("prompt", s.prompt)
	}
	if s.vadSignalsSet {
		params.Set("vad_signals", strconv.FormatBool(s.vadSignals))
	}
	if s.highVADSensitivity != nil {
		params.Set("high_vad_sensitivity", strconv.FormatBool(*s.highVADSensitivity))
	}
	if !s.vadSignals {
		params.Set("flush_signal", "true")
	}
	params.Set("input_audio_codec", s.encoding)
	base := s.baseURL
	if base == "" {
		base = sarvamSTTURL
		if settings.Model == "saaras:v2.5" {
			base = sarvamSTTTranslateURL
		}
	}
	header := http.Header{}
	header.Set("api-subscription-key", s.apiKey)
	return base + "?" + params.Encode(), header
}

func (s *STTService) Connect(ctx context.Context, sink stt.Sink) error {
	cfg := reconnect.Config{
		Name:      "SarvamSTT",
		Endpoint:  s.endpoint,
		OnMessage: func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:   sink.Error,
	}
	if s.keepalive > 0 {
		silence := s.marshalAudio(make([]byte, s.SampleRate()/10*2))
		cfg.KeepaliveIdle = s.keepalive
		cfg.Keepalive = func() (reconnect.Message, bool) {
			return reconnect.Message{Type: websocket.TextMessage, Data: silence}, true
		}
	}
	session := reconnect.New(cfg)
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *STTService) marshalAudio(raw []byte) []byte {
	const prefix = `{"audio":{"data":"`
	suffix := `","sample_rate":` + strconv.Itoa(s.SampleRate()) + `,"encoding":"audio/wav"}}`
	buf := make([]byte, 0, len(prefix)+base64.StdEncoding.EncodedLen(len(raw))+len(suffix))
	buf = append(buf, prefix...)
	buf = base64.StdEncoding.AppendEncode(buf, raw)
	return append(buf, suffix...)
}

func (s *STTService) SendAudio(audio []byte) {
	if session := s.session.Load(); session != nil {
		session.SendDroppable(websocket.TextMessage, s.marshalAudio(audio))
	}
}

func (s *STTService) Finalize() {
	if s.vadSignals {
		return
	}
	if session := s.session.Load(); session != nil {
		session.Send(websocket.TextMessage, flushMessage)
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

type sarvamMsg struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

type sarvamEventData struct {
	SignalType string `json:"signal_type"`
}

type sarvamTranscriptData struct {
	Transcript   string `json:"transcript"`
	LanguageCode string `json:"language_code"`
}

func (s *STTService) onMessage(sink stt.Sink, raw []byte) {
	var msg sarvamMsg
	if err := json.Unmarshal(raw, &msg); err != nil {
		return
	}
	switch msg.Type {
	case "events":
		if !s.vadSignals {
			return
		}
		var event sarvamEventData
		if err := json.Unmarshal(msg.Data, &event); err != nil {
			return
		}
		switch event.SignalType {
		case "START_SPEECH":
			sink.Push(frames.NewUserStartedSpeakingFrame(), frames.Upstream)
		case "END_SPEECH":
			sink.Push(frames.NewUserStoppedSpeakingFrame(), frames.Upstream)
		}
	case "data":
		var td sarvamTranscriptData
		if err := json.Unmarshal(msg.Data, &td); err != nil {
			return
		}
		if t := strings.TrimSpace(td.Transcript); t != "" {
			sink.Transcript(stt.Transcript{Text: t, Final: true, Language: td.LanguageCode})
		}
	case "error":
		sink.Error(&sarvamError{string(msg.Data)}, false)
	}
}

type sarvamError struct{ msg string }

func (e *sarvamError) Error() string { return "sarvam: " + e.msg }

var languageCodes = map[string]string{
	"en": "en-IN",
	"hi": "hi-IN",
	"bn": "bn-IN",
	"gu": "gu-IN",
	"kn": "kn-IN",
	"ml": "ml-IN",
	"mr": "mr-IN",
	"ta": "ta-IN",
	"te": "te-IN",
	"pa": "pa-IN",
	"or": "od-IN",
	"as": "as-IN",
}
