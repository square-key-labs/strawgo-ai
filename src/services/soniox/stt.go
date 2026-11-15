package soniox

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

const (
	DefaultURL     = "wss://stt-rt.soniox.com/transcribe-websocket"
	DefaultModel   = "stt-rt-v6"
	endToken       = "<end>"
	finalizedToken = "<fin>"
	keepaliveIdle  = 5 * time.Second
)

const (
	ExtraLanguageHints          = "language_hints"
	ExtraLanguageHintsStrict    = "language_hints_strict"
	ExtraContext                = "context"
	ExtraSpeakerDiarization     = "enable_speaker_diarization"
	ExtraLanguageIdentification = "enable_language_identification"
	ExtraMaxEndpointDelayMs     = "max_endpoint_delay_ms"
	ExtraEndpointSensitivity    = "endpoint_sensitivity"
	ExtraEndpointLatencyLevel   = "endpoint_latency_adjustment_level"
	ExtraClientReferenceID      = "client_reference_id"
)

var (
	keepaliveMessage = []byte(`{"type": "keepalive"}`)
	finalizeMessage  = []byte(`{"type": "finalize"}`)
	stopRecording    = []byte{}
)

type ContextGeneralItem struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

type ContextTranslationTerm struct {
	Source string `json:"source"`
	Target string `json:"target"`
}

type Context struct {
	General          []ContextGeneralItem     `json:"general,omitempty"`
	Text             string                   `json:"text,omitempty"`
	Terms            []string                 `json:"terms,omitempty"`
	TranslationTerms []ContextTranslationTerm `json:"translation_terms,omitempty"`
}

type STTConfig struct {
	APIKey                       string
	URL                          string
	Model                        string
	SampleRate                   int
	AudioFormat                  string
	NumChannels                  int
	LanguageHints                []string
	LanguageHintsStrict          *bool
	Context                      any
	EnableSpeakerDiarization     bool
	EnableLanguageIdentification bool
	MaxEndpointDelayMs           *int
	EndpointSensitivity          *float64
	EndpointLatencyLevel         *int
	ClientReferenceID            string
	DisableVADForceTurnEndpoint  bool
	DisableInterruptions         bool
}

type STTService struct {
	*stt.Base
	apiKey          string
	url             string
	audioFormat     string
	channels        int
	vadForceEnd     bool
	shouldInterrupt bool

	session atomic.Pointer[reconnect.Session]
	sink    stt.Sink

	mu       sync.Mutex
	final    []token
	turnOpen bool
}

func NewSTTService(config STTConfig) *STTService {
	model := config.Model
	if model == "" {
		model = DefaultModel
	}
	url := config.URL
	if url == "" {
		url = DefaultURL
	}
	format := config.AudioFormat
	if format == "" {
		format = "pcm_s16le"
	}
	channels := config.NumChannels
	if channels <= 0 {
		channels = 1
	}
	extra := map[string]any{
		ExtraSpeakerDiarization:     config.EnableSpeakerDiarization,
		ExtraLanguageIdentification: config.EnableLanguageIdentification,
	}
	if len(config.LanguageHints) > 0 {
		extra[ExtraLanguageHints] = config.LanguageHints
	}
	if config.LanguageHintsStrict != nil {
		extra[ExtraLanguageHintsStrict] = *config.LanguageHintsStrict
	}
	if config.Context != nil {
		extra[ExtraContext] = config.Context
	}
	if config.MaxEndpointDelayMs != nil {
		extra[ExtraMaxEndpointDelayMs] = *config.MaxEndpointDelayMs
	}
	if config.EndpointSensitivity != nil {
		extra[ExtraEndpointSensitivity] = *config.EndpointSensitivity
	}
	if config.EndpointLatencyLevel != nil {
		extra[ExtraEndpointLatencyLevel] = *config.EndpointLatencyLevel
	}
	if config.ClientReferenceID != "" {
		extra[ExtraClientReferenceID] = config.ClientReferenceID
	}
	s := &STTService{
		apiKey:          config.APIKey,
		url:             url,
		audioFormat:     format,
		channels:        channels,
		vadForceEnd:     !config.DisableVADForceTurnEndpoint,
		shouldInterrupt: !config.DisableInterruptions,
	}
	s.Base = stt.NewStreaming("SonioxSTT", s, stt.Options{
		Provider:   "soniox",
		Settings:   frames.STTSettings{Model: model, Extra: extra},
		SampleRate: config.SampleRate,
		TTFSP99:    350 * time.Millisecond,
	})
	return s
}

func (s *STTService) endpoint() (string, http.Header) {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.apiKey)
	return s.url, header
}

func (s *STTService) config() map[string]any {
	settings := s.Settings()
	hints := languageHints(settings)
	cfg := map[string]any{
		"model":                             settings.Model,
		"audio_format":                      s.audioFormat,
		"num_channels":                      s.channels,
		"enable_endpoint_detection":         true,
		"sample_rate":                       s.SampleRate(),
		"max_endpoint_delay_ms":             settings.Extra[ExtraMaxEndpointDelayMs],
		"endpoint_sensitivity":              settings.Extra[ExtraEndpointSensitivity],
		"endpoint_latency_adjustment_level": settings.Extra[ExtraEndpointLatencyLevel],
		"language_hints":                    hints,
		"language_hints_strict":             settings.Extra[ExtraLanguageHintsStrict],
		"context":                           settings.Extra[ExtraContext],
		"enable_speaker_diarization":        settings.Extra[ExtraSpeakerDiarization],
		"enable_language_identification":    settings.Extra[ExtraLanguageIdentification],
		"client_reference_id":               settings.Extra[ExtraClientReferenceID],
	}
	return cfg
}

func languageHints(settings frames.STTSettings) []string {
	var raw []string
	switch v := settings.Extra[ExtraLanguageHints].(type) {
	case []string:
		raw = v
	case []any:
		for _, x := range v {
			if str, ok := x.(string); ok {
				raw = append(raw, str)
			}
		}
	}
	if len(raw) == 0 && settings.Language != "" {
		raw = []string{settings.Language}
	}
	if len(raw) == 0 {
		return nil
	}
	seen := map[string]bool{}
	out := make([]string, 0, len(raw))
	for _, l := range raw {
		code := strings.ToLower(l)
		if i := strings.IndexAny(code, "-_"); i > 0 {
			code = code[:i]
		}
		if code != "" && !seen[code] {
			seen[code] = true
			out = append(out, code)
		}
	}
	return out
}

func (s *STTService) Connect(ctx context.Context, sink stt.Sink) error {
	s.sink = sink
	if !s.vadForceEnd {
		meta := frames.NewServiceMetadataFrame(s.Name())
		strategies := turns.UserTurnStrategies{
			StartStrategies: []user_start.UserTurnStartStrategy{proposedStart{interrupt: s.shouldInterrupt}},
			StopStrategies:  []user_stop.UserTurnStopStrategy{proposedStop{}},
		}
		meta.UserTurnStrategies = &strategies
		sink.Push(meta, frames.Downstream)
	}
	session := reconnect.New(reconnect.Config{
		Name:          "SonioxSTT",
		Endpoint:      s.endpoint,
		OnOpen:        s.onOpen,
		OnMessage:     func(_ int, data []byte) { s.onMessage(data) },
		OnDisconnect:  func(error) { s.finalizeTurn() },
		OnError:       sink.Error,
		CloseMessage:  &reconnect.Message{Type: websocket.TextMessage, Data: stopRecording},
		KeepaliveIdle: keepaliveIdle,
		Keepalive: func() (reconnect.Message, bool) {
			return reconnect.Message{Type: websocket.TextMessage, Data: keepaliveMessage}, true
		},
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *STTService) onOpen(w reconnect.Writer) error {
	s.mu.Lock()
	s.final = s.final[:0]
	s.mu.Unlock()
	data, err := json.Marshal(s.config())
	if err != nil {
		return err
	}
	return w.WriteMessage(websocket.TextMessage, data)
}

func (s *STTService) SendAudio(audio []byte) {
	if session := s.session.Load(); session != nil {
		session.SendDroppable(websocket.BinaryMessage, audio)
	}
}

func (s *STTService) Finalize() {
	if !s.vadForceEnd {
		return
	}
	if session := s.session.Load(); session != nil && session.Connected() {
		session.Send(websocket.TextMessage, finalizeMessage)
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

type token struct {
	Text     string `json:"text"`
	IsFinal  bool   `json:"is_final"`
	Language string `json:"language"`
	Speaker  string `json:"speaker"`
}

type response struct {
	Tokens       []token `json:"tokens"`
	ErrorCode    any     `json:"error_code"`
	ErrorMessage string  `json:"error_message"`
	Finished     bool    `json:"finished"`
}

func isEnd(t *token) bool {
	return t.Text == endToken || t.Text == finalizedToken
}

func (s *STTService) onMessage(data []byte) {
	var resp response
	if err := json.Unmarshal(data, &resp); err != nil {
		return
	}
	var interim strings.Builder
	hasNonFinal := false
	for i := range resp.Tokens {
		t := &resp.Tokens[i]
		if !isEnd(t) {
			s.turnStarted()
		}
		if t.IsFinal {
			if isEnd(t) {
				s.finalizeTurn()
			} else {
				s.mu.Lock()
				s.final = append(s.final, *t)
				s.mu.Unlock()
			}
		} else {
			hasNonFinal = true
			interim.WriteString(t.Text)
		}
	}
	s.mu.Lock()
	pending := len(s.final) > 0
	var text string
	if pending || hasNonFinal {
		var b strings.Builder
		for i := range s.final {
			b.WriteString(s.final[i].Text)
		}
		b.WriteString(interim.String())
		text = b.String()
	}
	lang := dominantLanguage(s.final)
	s.mu.Unlock()
	if text != "" {
		s.sink.Transcript(stt.Transcript{Text: text, Language: lang})
	}
	if resp.ErrorCode != nil || resp.ErrorMessage != "" {
		s.finalizeTurn()
		s.sink.Error(fmt.Errorf("soniox error %v: %s", resp.ErrorCode, resp.ErrorMessage), false)
	}
	if resp.Finished {
		s.finalizeTurn()
	}
}

func (s *STTService) turnStarted() {
	if s.vadForceEnd {
		return
	}
	s.mu.Lock()
	if s.turnOpen {
		s.mu.Unlock()
		return
	}
	s.turnOpen = true
	s.mu.Unlock()
	s.sink.Push(frames.NewProposedUserStartedSpeakingFrame(), frames.Downstream)
	s.sink.Push(frames.NewProposedUserStartedSpeakingFrame(), frames.Upstream)
}

func (s *STTService) finalizeTurn() {
	s.mu.Lock()
	var text, lang string
	if len(s.final) > 0 {
		var b strings.Builder
		for i := range s.final {
			b.WriteString(s.final[i].Text)
		}
		text = b.String()
		lang = dominantLanguage(s.final)
		s.final = s.final[:0]
	}
	closeTurn := !s.vadForceEnd && s.turnOpen
	if closeTurn {
		s.turnOpen = false
	}
	s.mu.Unlock()
	if text != "" {
		s.sink.Transcript(stt.Transcript{Text: text, Final: true, Finalized: true, Language: lang})
	}
	if closeTurn {
		s.sink.Push(frames.NewProposedUserStoppedSpeakingFrame(), frames.Downstream)
		s.sink.Push(frames.NewProposedUserStoppedSpeakingFrame(), frames.Upstream)
	}
}

func dominantLanguage(tokens []token) string {
	best, bestN := "", 0
	for i := range tokens {
		l := tokens[i].Language
		if l == "" || l == best {
			continue
		}
		n := 0
		for j := range tokens {
			if tokens[j].Language == l {
				n++
			}
		}
		if n > bestN {
			best, bestN = l, n
		}
	}
	return best
}

type proposedStart struct{ interrupt bool }

func (proposedStart) ShouldStart(frame any) bool {
	_, ok := frame.(*frames.ProposedUserStartedSpeakingFrame)
	return ok
}

func (p proposedStart) EnableInterruptions() bool { return p.interrupt }

func (proposedStart) Reset() {}

type proposedStop struct{}

func (proposedStop) ShouldStop(frame any) bool {
	_, ok := frame.(*frames.ProposedUserStoppedSpeakingFrame)
	return ok
}

func (proposedStop) Reset() {}
