package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
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
	DeepgramFluxURL          = "wss://api.deepgram.com/v2/listen"
	DefaultFluxModel         = "flux-general-en"
	FluxMultilingualModel    = "flux-general-multi"
	defaultEagerEOTThreshold = 0.5
	fluxConnectTimeout       = 10 * time.Second
	fluxConfigureAckTimeout  = 5 * time.Second
	fluxSilence              = 500 * time.Millisecond
)

const (
	FluxEagerEOTThreshold = "eager_eot_threshold"
	FluxEOTThreshold      = "eot_threshold"
	FluxEOTTimeoutMs      = "eot_timeout_ms"
	FluxKeyterm           = "keyterm"
	FluxMinConfidence     = "min_confidence"
	FluxNumerals          = "numerals"
	FluxProfanityFilter   = "profanity_filter"
	FluxRedact            = "redact"
	FluxRedactUsage       = "redact_usage"
	FluxLanguageHints     = "language_hints"
)

var (
	fluxConfigureFields = []string{FluxKeyterm, FluxEOTThreshold, FluxEagerEOTThreshold, FluxEOTTimeoutMs, FluxLanguageHints, FluxNumerals, FluxProfanityFilter, FluxRedactUsage}
	fluxURLFields       = []string{FluxRedact}
)

type FluxConfig struct {
	APIKey               string
	Model                string
	SampleRate           int
	BaseURL              string
	EagerEOTThreshold    *float64
	EOTThreshold         *float64
	EOTTimeoutMs         *int
	Keyterms             []string
	MinConfidence        *float64
	Numerals             *bool
	ProfanityFilter      *bool
	Redact               string
	RedactUsage          *bool
	LanguageHints        []string
	MIPOptOut            *bool
	Tags                 []string
	DisableInterruptions bool
	EnableEagerEndOfTurn bool
	EagerMatchPolicy     user_stop.EagerMatchPolicy
	WatchdogMinTimeout   time.Duration
	OnStartOfTurn        func(transcript string)
	OnTurnResumed        func()
	OnEndOfTurn          func(transcript string)
	OnEagerEndOfTurn     func(transcript string)
	OnUpdate             func(transcript string)
}

type FluxSTTService struct {
	*stt.Base
	apiKey          string
	baseURL         string
	mipOptOut       *bool
	tags            []string
	shouldInterrupt bool
	eager           bool
	watchdogMin     time.Duration
	cfg             FluxConfig

	session atomic.Pointer[reconnect.Session]
	sink    stt.Sink

	lastAudio     atomic.Int64
	chunkDuration atomic.Int64

	mu               sync.Mutex
	applied          frames.STTSettings
	userSpeaking     bool
	turnTranscript   string
	turnLanguage     string
	eagerPending     bool
	pendingReconnect bool
	confirmed        bool
	confirmTimer     *time.Timer
	watchdog         *time.Timer
	configInFlight   bool
	configSentAt     time.Time
	configPending    map[string]bool
	closed           bool
}

func NewFluxSTTService(config FluxConfig) *FluxSTTService {
	model := config.Model
	if model == "" {
		model = DefaultFluxModel
	}
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DeepgramFluxURL
	}
	watchdog := config.WatchdogMinTimeout
	if watchdog <= 0 {
		watchdog = 500 * time.Millisecond
	}
	extra := map[string]any{}
	eagerThreshold := config.EagerEOTThreshold
	if eagerThreshold == nil && config.EnableEagerEndOfTurn {
		v := defaultEagerEOTThreshold
		eagerThreshold = &v
	}
	if eagerThreshold != nil {
		extra[FluxEagerEOTThreshold] = *eagerThreshold
	}
	if config.EOTThreshold != nil {
		extra[FluxEOTThreshold] = *config.EOTThreshold
	}
	if config.EOTTimeoutMs != nil {
		extra[FluxEOTTimeoutMs] = *config.EOTTimeoutMs
	}
	if len(config.Keyterms) > 0 {
		extra[FluxKeyterm] = config.Keyterms
	}
	if config.MinConfidence != nil {
		extra[FluxMinConfidence] = *config.MinConfidence
	}
	if config.Numerals != nil {
		extra[FluxNumerals] = *config.Numerals
	}
	if config.ProfanityFilter != nil {
		extra[FluxProfanityFilter] = *config.ProfanityFilter
	}
	if config.Redact != "" {
		extra[FluxRedact] = config.Redact
	}
	if config.RedactUsage != nil {
		extra[FluxRedactUsage] = *config.RedactUsage
	}
	if config.LanguageHints != nil {
		extra[FluxLanguageHints] = config.LanguageHints
	}
	s := &FluxSTTService{
		apiKey:          config.APIKey,
		baseURL:         baseURL,
		mipOptOut:       config.MIPOptOut,
		tags:            config.Tags,
		shouldInterrupt: !config.DisableInterruptions,
		eager:           config.EnableEagerEndOfTurn,
		watchdogMin:     watchdog,
		cfg:             config,
	}
	settings := frames.STTSettings{Model: model, Extra: extra}
	s.applied = settings
	s.Base = stt.NewStreaming("DeepgramFluxSTT", s, stt.Options{
		Provider:   "deepgram-flux",
		Settings:   settings,
		SampleRate: config.SampleRate,
		TTFSP99:    300 * time.Millisecond,
	})
	return s
}

func FluxUserTurnStrategies(enableInterruptions bool) turns.UserTurnStrategies {
	return turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{&fluxTurnStart{interrupt: enableInterruptions}},
		StopStrategies:  []user_stop.UserTurnStopStrategy{fluxTurnStop{}},
	}
}

func FluxEagerUserTurnStrategies(enableInterruptions bool, policy user_stop.EagerMatchPolicy) turns.UserTurnStrategies {
	return turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{&fluxTurnStart{interrupt: enableInterruptions}},
		StopStrategies:  []user_stop.UserTurnStopStrategy{user_stop.NewEagerUserTurnStopStrategy(policy, 0)},
	}
}

type fluxTurnStart struct{ interrupt bool }

func (s *fluxTurnStart) ShouldStart(frame any) bool {
	_, ok := frame.(*frames.ProposedUserStartedSpeakingFrame)
	return ok
}

func (s *fluxTurnStart) EnableInterruptions() bool { return s.interrupt }

func (s *fluxTurnStart) Reset() {}

type fluxTurnStop struct{}

func (fluxTurnStop) ShouldStop(frame any) bool {
	_, ok := frame.(*frames.ProposedUserStoppedSpeakingFrame)
	return ok
}

func (fluxTurnStop) Reset() {}

func (s *FluxSTTService) endpoint() (string, http.Header) {
	settings := s.Settings()
	q := url.Values{}
	q.Set("model", settings.Model)
	q.Set("sample_rate", strconv.Itoa(s.SampleRate()))
	q.Set("encoding", "linear16")
	if v, ok := extraFloat(settings.Extra, FluxEagerEOTThreshold); ok {
		q.Set(FluxEagerEOTThreshold, strconv.FormatFloat(v, 'f', -1, 64))
	}
	if v, ok := extraFloat(settings.Extra, FluxEOTThreshold); ok {
		q.Set(FluxEOTThreshold, strconv.FormatFloat(v, 'f', -1, 64))
	}
	if v, ok := extraFloat(settings.Extra, FluxEOTTimeoutMs); ok {
		q.Set(FluxEOTTimeoutMs, strconv.Itoa(int(v)))
	}
	for _, key := range []string{FluxNumerals, FluxProfanityFilter, FluxRedactUsage} {
		if v, ok := extraBool(settings.Extra, key); ok {
			q.Set(key, strconv.FormatBool(v))
		}
	}
	if v, ok := settings.Extra[FluxRedact].(string); ok && v != "" {
		q.Set(FluxRedact, v)
	}
	if s.mipOptOut != nil {
		q.Set("mip_opt_out", strconv.FormatBool(*s.mipOptOut))
	}
	for _, k := range extraStrings(settings.Extra, FluxKeyterm) {
		q.Add("keyterm", k)
	}
	for _, t := range s.tags {
		q.Add("tag", t)
	}
	if settings.Model == FluxMultilingualModel {
		for _, h := range dedupe(extraStrings(settings.Extra, FluxLanguageHints)) {
			q.Add("language_hint", h)
		}
	}
	header := http.Header{}
	header.Set("Authorization", "Token "+s.apiKey)
	return s.baseURL + "?" + q.Encode(), header
}

func (s *FluxSTTService) Connect(ctx context.Context, sink stt.Sink) error {
	s.sink = sink
	meta := frames.NewServiceMetadataFrame(s.Name())
	strategies := FluxUserTurnStrategies(s.shouldInterrupt)
	if s.eager {
		strategies = FluxEagerUserTurnStrategies(s.shouldInterrupt, s.cfg.EagerMatchPolicy)
	}
	meta.UserTurnStrategies = &strategies
	sink.Push(meta, frames.Downstream)
	session := reconnect.New(reconnect.Config{
		Name:         "DeepgramFluxSTT",
		Endpoint:     s.endpoint,
		OnOpen:       s.onOpen,
		OnMessage:    func(_ int, data []byte) { s.onMessage(data) },
		OnDisconnect: func(error) { s.onDisconnect() },
		OnError:      sink.Error,
		CloseMessage: &reconnect.Message{Type: websocket.TextMessage, Data: closeStream},
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *FluxSTTService) onOpen(reconnect.Writer) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.confirmed = false
	s.resetConfigureLocked()
	s.applied = s.Settings()
	if s.confirmTimer == nil {
		s.confirmTimer = time.AfterFunc(fluxConnectTimeout, s.confirmTimeout)
	} else {
		s.confirmTimer.Reset(fluxConnectTimeout)
	}
	return nil
}

func (s *FluxSTTService) confirmTimeout() {
	s.mu.Lock()
	expired := !s.confirmed && !s.closed
	s.mu.Unlock()
	if !expired {
		return
	}
	s.sink.Error(fmt.Errorf("flux did not confirm the connection within %s", fluxConnectTimeout), false)
	s.endTurnCutShort()
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
}

func (s *FluxSTTService) SendAudio(audio []byte) {
	session := s.session.Load()
	if session == nil {
		return
	}
	s.lastAudio.Store(time.Now().UnixNano())
	if rate := s.SampleRate(); rate > 0 {
		s.chunkDuration.Store(int64(time.Duration(len(audio)) * time.Second / time.Duration(rate*2)))
	}
	session.SendDroppable(websocket.BinaryMessage, audio)
}

func (s *FluxSTTService) Disconnect(graceful bool) {
	s.mu.Lock()
	s.closed = true
	if s.watchdog != nil {
		s.watchdog.Stop()
	}
	if s.confirmTimer != nil {
		s.confirmTimer.Stop()
	}
	s.mu.Unlock()
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func (s *FluxSTTService) ApplySettings(current frames.STTSettings) error {
	s.mu.Lock()
	prev := s.applied
	s.applied = current
	changed := map[string]bool{}
	for _, k := range fluxConfigureFields {
		if !reflect.DeepEqual(prev.Extra[k], current.Extra[k]) {
			changed[k] = true
		}
	}
	reconnectNeeded := prev.Model != current.Model
	for _, k := range fluxURLFields {
		if !reflect.DeepEqual(prev.Extra[k], current.Extra[k]) {
			reconnectNeeded = true
		}
	}
	if changed[FluxRedactUsage] {
		if on, _ := extraBool(current.Extra, FluxRedactUsage); !on {
			delete(changed, FluxRedactUsage)
			reconnectNeeded = true
		}
	}
	var msg []byte
	if len(changed) > 0 && s.confirmed {
		msg = s.configureLocked(changed, current)
	}
	doReconnect := false
	if reconnectNeeded {
		if s.userSpeaking {
			s.pendingReconnect = true
		} else {
			doReconnect = true
		}
	}
	s.mu.Unlock()
	session := s.session.Load()
	if session == nil {
		return nil
	}
	if msg != nil {
		session.Send(websocket.TextMessage, msg)
	}
	if doReconnect {
		session.Reconnect()
	}
	return nil
}

func (s *FluxSTTService) configureLocked(fields map[string]bool, settings frames.STTSettings) []byte {
	if s.configInFlight && time.Since(s.configSentAt) < fluxConfigureAckTimeout {
		if s.configPending == nil {
			s.configPending = map[string]bool{}
		}
		for k := range fields {
			s.configPending[k] = true
		}
		return nil
	}
	msg := map[string]any{"type": "Configure"}
	if fields[FluxKeyterm] {
		msg["keyterms"] = nonNilStrings(extraStrings(settings.Extra, FluxKeyterm))
	}
	for _, key := range []string{FluxNumerals, FluxProfanityFilter, FluxRedactUsage} {
		if fields[key] {
			msg[key] = settings.Extra[key]
		}
	}
	thresholds := map[string]any{}
	for _, key := range []string{FluxEOTThreshold, FluxEagerEOTThreshold, FluxEOTTimeoutMs} {
		if fields[key] {
			thresholds[key] = settings.Extra[key]
		}
	}
	if len(thresholds) > 0 {
		msg["thresholds"] = thresholds
	}
	if fields[FluxLanguageHints] && settings.Model == FluxMultilingualModel {
		msg["language_hints"] = nonNilStrings(dedupe(extraStrings(settings.Extra, FluxLanguageHints)))
	}
	if len(msg) == 1 {
		return nil
	}
	data, err := json.Marshal(msg)
	if err != nil {
		return nil
	}
	s.configInFlight = true
	s.configSentAt = time.Now()
	return data
}

func (s *FluxSTTService) resetConfigureLocked() {
	s.configInFlight = false
	s.configSentAt = time.Time{}
	s.configPending = nil
}

func (s *FluxSTTService) onConfigureAck() {
	s.mu.Lock()
	s.configInFlight = false
	var msg []byte
	if pending := s.configPending; pending != nil {
		s.configPending = nil
		msg = s.configureLocked(pending, s.Settings())
	}
	s.mu.Unlock()
	if msg != nil {
		if session := s.session.Load(); session != nil && session.Connected() {
			session.Send(websocket.TextMessage, msg)
		}
	}
}

type fluxMessage struct {
	Type        string   `json:"type"`
	Event       string   `json:"event"`
	Transcript  string   `json:"transcript"`
	Languages   []string `json:"languages"`
	Code        string   `json:"code"`
	ErrorCode   string   `json:"error_code"`
	Description string   `json:"description"`
	Words       []struct {
		Confidence *float64 `json:"confidence"`
	} `json:"words"`
}

type FluxError struct {
	Code        string
	Description string
}

func (e *FluxError) Error() string {
	return "deepgram flux: [" + e.Code + "] " + e.Description
}

func (s *FluxSTTService) onMessage(data []byte) {
	var msg fluxMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return
	}
	switch msg.Type {
	case "Connected":
		s.mu.Lock()
		s.confirmed = true
		if s.confirmTimer != nil {
			s.confirmTimer.Stop()
		}
		s.mu.Unlock()
	case "Error":
		s.sink.Error(&FluxError{Code: orDefault(msg.Code, "unknown"), Description: orDefault(msg.Description, "no description")}, false)
		s.endTurnCutShort()
		if session := s.session.Load(); session != nil {
			session.Reconnect()
		}
	case "ConfigureSuccess":
		s.onConfigureAck()
	case "ConfigureFailure":
		s.onConfigureAck()
		s.sink.Error(errors.New("deepgram flux: configure rejected: ["+orDefault(msg.ErrorCode, "unknown")+"] "+orDefault(msg.Description, "no description")), false)
	case "TurnInfo":
		s.onTurnInfo(&msg)
	}
}

func (s *FluxSTTService) language(msg *fluxMessage) string {
	if len(msg.Languages) > 0 {
		return msg.Languages[0]
	}
	if s.Settings().Model == DefaultFluxModel {
		return "en"
	}
	return ""
}

func (s *FluxSTTService) onTurnInfo(msg *fluxMessage) {
	switch msg.Event {
	case "StartOfTurn":
		s.mu.Lock()
		s.userSpeaking = true
		s.turnTranscript = msg.Transcript
		s.turnLanguage = ""
		s.armWatchdogLocked()
		s.mu.Unlock()
		s.broadcast(func() frames.Frame { return frames.NewProposedUserStartedSpeakingFrame() })
		if s.cfg.OnStartOfTurn != nil {
			s.cfg.OnStartOfTurn(msg.Transcript)
		}
	case "TurnResumed":
		s.mu.Lock()
		pending := s.eagerPending
		s.eagerPending = false
		s.mu.Unlock()
		if pending {
			s.sink.Push(frames.NewUserTurnResponseCancelFrame(), frames.Downstream)
		}
		if s.cfg.OnTurnResumed != nil {
			s.cfg.OnTurnResumed()
		}
	case "EndOfTurn":
		s.mu.Lock()
		s.userSpeaking = false
		s.turnTranscript = ""
		s.eagerPending = false
		reconnectNow := s.pendingReconnect
		s.pendingReconnect = false
		if s.watchdog != nil {
			s.watchdog.Stop()
		}
		s.mu.Unlock()
		if s.confident(msg) {
			s.sink.Transcript(stt.Transcript{Text: msg.Transcript, Final: true, Finalized: true, Language: s.language(msg)})
		}
		s.broadcast(func() frames.Frame { return frames.NewProposedUserStoppedSpeakingFrame() })
		if s.cfg.OnEndOfTurn != nil {
			s.cfg.OnEndOfTurn(msg.Transcript)
		}
		if reconnectNow {
			if session := s.session.Load(); session != nil {
				session.Reconnect()
			}
		}
	case "EagerEndOfTurn":
		if s.eager {
			s.mu.Lock()
			s.eagerPending = true
			s.mu.Unlock()
			f := frames.NewEagerTranscriptionFrame(msg.Transcript, "")
			f.Language = s.language(msg)
			s.sink.Push(f, frames.Downstream)
		}
		if s.cfg.OnEagerEndOfTurn != nil {
			s.cfg.OnEagerEndOfTurn(msg.Transcript)
		}
	case "Update":
		if msg.Transcript != "" {
			lang := s.language(msg)
			s.mu.Lock()
			s.turnTranscript = msg.Transcript
			s.turnLanguage = lang
			s.mu.Unlock()
			s.sink.Transcript(stt.Transcript{Text: msg.Transcript, Language: lang})
		}
		if s.cfg.OnUpdate != nil {
			s.cfg.OnUpdate(msg.Transcript)
		}
	}
}

func (s *FluxSTTService) confident(msg *fluxMessage) bool {
	minConf, ok := extraFloat(s.Settings().Extra, FluxMinConfidence)
	if !ok || minConf == 0 {
		return true
	}
	var sum float64
	n := 0
	for _, w := range msg.Words {
		if w.Confidence != nil {
			sum += *w.Confidence
			n++
		}
	}
	return n > 0 && sum/float64(n) > minConf
}

func (s *FluxSTTService) broadcast(build func() frames.Frame) {
	s.sink.Push(build(), frames.Downstream)
	s.sink.Push(build(), frames.Upstream)
}

func (s *FluxSTTService) onDisconnect() {
	s.mu.Lock()
	s.confirmed = false
	s.resetConfigureLocked()
	s.mu.Unlock()
	s.endTurnCutShort()
}

func (s *FluxSTTService) endTurnCutShort() {
	s.mu.Lock()
	if !s.userSpeaking {
		s.mu.Unlock()
		return
	}
	s.userSpeaking = false
	s.eagerPending = false
	text, lang := s.turnTranscript, s.turnLanguage
	s.turnTranscript = ""
	if s.watchdog != nil {
		s.watchdog.Stop()
	}
	s.mu.Unlock()
	if text != "" {
		s.sink.Transcript(stt.Transcript{Text: text, Final: true, Language: lang})
	}
	s.broadcast(func() frames.Frame { return frames.NewProposedUserStoppedSpeakingFrame() })
}

func (s *FluxSTTService) watchdogThreshold() time.Duration {
	t := 2 * time.Duration(s.chunkDuration.Load())
	if t < s.watchdogMin {
		t = s.watchdogMin
	}
	return t
}

func (s *FluxSTTService) armWatchdogLocked() {
	if s.closed {
		return
	}
	d := s.watchdogThreshold()
	if s.watchdog == nil {
		s.watchdog = time.AfterFunc(d, s.checkWatchdog)
	} else {
		s.watchdog.Reset(d)
	}
}

func (s *FluxSTTService) checkWatchdog() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.userSpeaking || s.closed {
		return
	}
	threshold := s.watchdogThreshold()
	idle := time.Since(time.Unix(0, s.lastAudio.Load()))
	if idle < threshold {
		s.watchdog.Reset(threshold - idle)
		return
	}
	if session := s.session.Load(); session != nil && session.Connected() {
		silence := make([]byte, int(fluxSilence.Seconds()*float64(s.SampleRate()))*2)
		session.Send(websocket.BinaryMessage, silence)
	}
	s.lastAudio.Store(time.Now().UnixNano())
	s.watchdog.Reset(threshold)
}

func extraFloat(m map[string]any, key string) (float64, bool) {
	switch v := m[key].(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case *float64:
		if v != nil {
			return *v, true
		}
	case *int:
		if v != nil {
			return float64(*v), true
		}
	}
	return 0, false
}

func extraBool(m map[string]any, key string) (bool, bool) {
	switch v := m[key].(type) {
	case bool:
		return v, true
	case *bool:
		if v != nil {
			return *v, true
		}
	}
	return false, false
}

func extraStrings(m map[string]any, key string) []string {
	switch v := m[key].(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, x := range v {
			if str, ok := x.(string); ok {
				out = append(out, str)
			}
		}
		return out
	case string:
		if v != "" {
			return []string{v}
		}
	}
	return nil
}

func dedupe(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := in[:0:0]
	for _, v := range in {
		if _, ok := seen[v]; ok || v == "" {
			continue
		}
		seen[v] = struct{}{}
		out = append(out, v)
	}
	return out
}

func nonNilStrings(in []string) []string {
	if in == nil {
		return []string{}
	}
	return in
}

func orDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}
