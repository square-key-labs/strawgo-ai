package cartesia

import (
	"context"
	"encoding/json"
	"errors"
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
	DefaultSTTHost       = "api.cartesia.ai"
	DefaultSTTModel      = "ink-whisper"
	DefaultSTTVersion    = "2026-03-01"
	STTExtraKeyterm      = "keyterm"
	STTExtraMinVolume    = "min_volume"
	STTExtraMaxSilence   = "max_silence_duration_secs"
	sttMaxKeyterms       = 100
	sttMaxKeytermChars   = 1200
	sttKeepaliveIdle     = 120 * time.Second
	sttFinalizationModel = "ink-whisper"
)

var (
	sttFinalize        = []byte("finalize")
	sttDone            = []byte("done")
	sttKeytermPrefixes = []string{"ink-2", "ink-preview"}
)

type STTConfig struct {
	APIKey                 string
	BaseURL                string
	Model                  string
	Language               string
	Encoding               string
	SampleRate             int
	Keyterms               []string
	MinVolume              *float64
	MaxSilenceDurationSecs *float64
	ExtraHeaders           http.Header
}

type STTService struct {
	*stt.Base
	apiKey   string
	scheme   string
	host     string
	encoding string
	headers  http.Header
	session  atomic.Pointer[reconnect.Session]
}

func NewSTTService(config STTConfig) *STTService {
	model := config.Model
	if model == "" {
		model = DefaultSTTModel
	}
	language := config.Language
	if language == "" {
		language = "en"
	}
	encoding := config.Encoding
	if encoding == "" {
		encoding = "pcm_s16le"
	}
	scheme, host := "wss", DefaultSTTHost
	if config.BaseURL != "" {
		raw := config.BaseURL
		if !strings.Contains(raw, "//") {
			raw = "//" + raw
		}
		if u, err := url.Parse(raw); err == nil {
			if u.Host != "" {
				host = u.Host
			}
			if u.Scheme != "" {
				scheme = u.Scheme
			}
		}
	}
	extra := map[string]any{}
	if len(config.Keyterms) > 0 {
		extra[STTExtraKeyterm] = config.Keyterms
	}
	if config.MinVolume != nil {
		extra[STTExtraMinVolume] = *config.MinVolume
	}
	if config.MaxSilenceDurationSecs != nil {
		extra[STTExtraMaxSilence] = *config.MaxSilenceDurationSecs
	}
	s := &STTService{
		apiKey:   config.APIKey,
		scheme:   scheme,
		host:     host,
		encoding: encoding,
		headers:  config.ExtraHeaders,
	}
	s.Base = stt.NewStreaming("CartesiaSTT", s, stt.Options{
		Provider:   "cartesia",
		Settings:   frames.STTSettings{Model: model, Language: language, Extra: extra},
		SampleRate: config.SampleRate,
		TTFSP99:    810 * time.Millisecond,
	})
	return s
}

func sttKeyterms(v any) []string {
	var terms []string
	switch x := v.(type) {
	case []string:
		terms = x
	case []any:
		for _, e := range x {
			if str, ok := e.(string); ok {
				terms = append(terms, str)
			}
		}
	}
	out := make([]string, 0, len(terms))
	total := 0
	for _, t := range terms {
		t = strings.TrimSpace(t)
		if t == "" {
			continue
		}
		if len(out) >= sttMaxKeyterms || total+len(t) > sttMaxKeytermChars {
			break
		}
		out = append(out, t)
		total += len(t)
	}
	return out
}

func sttFloat(v any) (string, bool) {
	switch x := v.(type) {
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64), true
	case float32:
		return strconv.FormatFloat(float64(x), 'f', -1, 32), true
	case int:
		return strconv.Itoa(x), true
	}
	return "", false
}

func (s *STTService) endpoint() (string, http.Header) {
	settings := s.Settings()
	q := url.Values{}
	q.Set("model", settings.Model)
	q.Set("language", settings.Language)
	q.Set("encoding", s.encoding)
	q.Set("sample_rate", strconv.Itoa(s.SampleRate()))
	for _, p := range sttKeytermPrefixes {
		if strings.HasPrefix(settings.Model, p) {
			for _, t := range sttKeyterms(settings.Extra[STTExtraKeyterm]) {
				q.Add("keyterm", t)
			}
			break
		}
	}
	if strings.HasPrefix(settings.Model, sttFinalizationModel) {
		for _, k := range []string{STTExtraMinVolume, STTExtraMaxSilence} {
			if v, ok := sttFloat(settings.Extra[k]); ok {
				q.Set(k, v)
			}
		}
	}
	header := http.Header{}
	header.Set("Cartesia-Version", DefaultSTTVersion)
	header.Set("X-API-Key", s.apiKey)
	for k, v := range s.headers {
		header[k] = v
	}
	u := url.URL{Scheme: s.scheme, Host: s.host, Path: "/stt/websocket", RawQuery: q.Encode()}
	return u.String(), header
}

func (s *STTService) Connect(ctx context.Context, sink stt.Sink) error {
	silence := make([]byte, s.SampleRate()/10*2)
	session := reconnect.New(reconnect.Config{
		Name:          "CartesiaSTT",
		Endpoint:      s.endpoint,
		OnMessage:     func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:       sink.Error,
		CloseMessage:  &reconnect.Message{Type: websocket.TextMessage, Data: sttDone},
		KeepaliveIdle: sttKeepaliveIdle,
		Keepalive: func() (reconnect.Message, bool) {
			return reconnect.Message{Type: websocket.BinaryMessage, Data: silence}, true
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
	if session := s.session.Load(); session != nil && session.Connected() {
		session.Send(websocket.TextMessage, sttFinalize)
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
	Type     string  `json:"type"`
	Text     *string `json:"text"`
	IsFinal  bool    `json:"is_final"`
	Language string  `json:"language"`
	Message  string  `json:"message"`
}

func (s *STTService) onMessage(sink stt.Sink, data []byte) {
	var resp sttResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return
	}
	switch resp.Type {
	case "transcript":
		if resp.Text == nil || *resp.Text == "" {
			return
		}
		sink.Transcript(stt.Transcript{Text: *resp.Text, Final: resp.IsFinal, Language: resp.Language})
	case "error":
		msg := resp.Message
		if msg == "" {
			msg = "unknown error"
		}
		sink.Error(errors.New("cartesia stt: "+msg), false)
	}
}
