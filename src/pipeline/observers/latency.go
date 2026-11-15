package observers

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

const (
	DefaultMinContribution = 5 * time.Millisecond
	printsAsZero           = 500 * time.Microsecond
	maxMoments             = 64
)

type TTFBBreakdown struct {
	Processor string
	Model     string
	StartTime time.Time
	Duration  time.Duration
}

type TextAggregationBreakdown struct {
	Processor string
	StartTime time.Time
	Duration  time.Duration
}

type FunctionCallLatency struct {
	FunctionName string
	StartTime    time.Time
	Duration     time.Duration
}

type MeasuredFrom string

const (
	MeasuredFromUserSilence     MeasuredFrom = "user_silence"
	MeasuredFromClientConnected MeasuredFrom = "client_connected"
)

type LatencyOwnerKind string

const (
	OwnerService  LatencyOwnerKind = "service"
	OwnerSetting  LatencyOwnerKind = "setting"
	OwnerBot      LatencyOwnerKind = "bot"
	OwnerPipeline LatencyOwnerKind = "pipeline"
)

type LatencyContribution struct {
	Key       string
	Label     string
	Owner     string
	OwnerKind LatencyOwnerKind
	StartTime time.Time
	Duration  time.Duration
}

type LatencyBreakdown struct {
	TTFB            []TTFBBreakdown
	Contributions   []LatencyContribution
	MeasuredFrom    MeasuredFrom
	Total           time.Duration
	TextAggregation *TextAggregationBreakdown
	UserTurnStart   time.Time
	UserTurn        time.Duration
	FunctionCalls   []FunctionCallLatency
}

func (b LatencyBreakdown) Contribution(key string) time.Duration {
	var d time.Duration
	for _, c := range b.Contributions {
		if c.Key == key {
			d += c.Duration
		}
	}
	return d
}

func (b LatencyBreakdown) Lines(byCost bool) []string {
	if len(b.Contributions) == 0 {
		return nil
	}
	ordered := b.Contributions
	if byCost {
		ordered = append([]LatencyContribution(nil), ordered...)
		sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Duration > ordered[j].Duration })
	}
	lines := make([]string, 0, len(ordered)+1)
	var total time.Duration
	for _, c := range ordered {
		total += c.Duration
		lines = append(lines, fmt.Sprintf("%6.3fs  %-20s [%s]", c.Duration.Seconds(), c.Label, c.Owner))
	}
	anchor := ""
	if b.MeasuredFrom == MeasuredFromClientConnected {
		anchor = " (from client connected)"
	}
	lines = append(lines, fmt.Sprintf("%6.3fs  TOTAL%s", total.Seconds(), anchor))
	return lines
}

type momentKind uint8

const (
	momentSilence momentKind = iota
	momentReady
	momentVADStop
	momentTranscript
	momentLLMRequest
	momentLLMChunk
	momentMarkerComplete
	momentMarkerIncomplete
	momentHandlersStart
	momentHandlersEnd
	momentFirstText
	momentSentence
	momentFirstAudio
	momentBotSpeaking
)

type moment struct {
	at      time.Time
	kind    momentKind
	source  string
	derived bool
}

type ownerRule uint8

const (
	ownerLiteral ownerRule = iota
	ownerOpener
	ownerCloser
)

type spanDef struct {
	key   string
	label string
	rule  ownerRule
	owner string
	core  bool
}

const turnCompletionOwner = "config: filter_incomplete_user_turns"

var (
	spanEndpointing     = spanDef{"endpointing_wait", "endpointing wait", ownerLiteral, "config: VAD stop_secs", true}
	spanFirstRequest    = spanDef{"first_request", "first request", ownerLiteral, "bot", true}
	spanTranscription   = spanDef{"transcription", "transcription", ownerCloser, "", true}
	spanTurnDetection   = spanDef{"turn_detection", "turn detection", ownerLiteral, "config: user turn strategies", true}
	spanLLMInference    = spanDef{"llm_inference", "LLM inference", ownerOpener, "", true}
	spanTurnCompletion  = spanDef{"turn_completion", "turn completion", ownerLiteral, turnCompletionOwner, false}
	spanWaitingForUser  = spanDef{"waiting_for_user", "waiting for user", ownerLiteral, turnCompletionOwner, false}
	spanLLMToolCall     = spanDef{"llm_tool_call", "LLM tool call", ownerOpener, "", false}
	spanFunctionHandler = spanDef{"function_handler", "function handler", ownerCloser, "", false}
	spanSentenceAgg     = spanDef{"sentence_aggregation", "sentence aggregation", ownerLiteral, "config: text_aggregation_mode", false}
	spanSpeechSynthesis = spanDef{"speech_synthesis", "speech synthesis", ownerCloser, "", true}
	spanOutputTransport = spanDef{"output_transport", "output transport", ownerCloser, "", false}
	spanAwaitingText    = spanDef{"awaiting_speakable_text", "awaiting speakable text", ownerCloser, "", false}
)

func spanFor(a, b momentKind, markersSeen bool) (spanDef, bool) {
	switch {
	case a == momentSilence && b == momentVADStop:
		return spanEndpointing, true
	case a == momentReady && b == momentLLMRequest:
		return spanFirstRequest, true
	case a == momentVADStop && b == momentTranscript:
		return spanTranscription, true
	case a == momentTranscript && b == momentLLMRequest:
		return spanTurnDetection, true
	case a == momentLLMRequest && (b == momentLLMChunk || b == momentFirstText):
		return spanLLMInference, true
	case a == momentLLMChunk && (b == momentMarkerComplete || b == momentMarkerIncomplete):
		return spanTurnCompletion, true
	case a == momentMarkerComplete && b == momentFirstText:
		return spanTurnCompletion, true
	case a == momentMarkerIncomplete && b == momentLLMRequest:
		return spanWaitingForUser, true
	case a == momentLLMChunk && b == momentHandlersStart:
		return spanLLMToolCall, true
	case a == momentHandlersStart && b == momentHandlersEnd:
		return spanFunctionHandler, true
	case a == momentFirstText && b == momentSentence:
		return spanSentenceAgg, true
	case (a == momentSentence || a == momentFirstText) && b == momentFirstAudio:
		return spanSpeechSynthesis, true
	case a == momentFirstAudio && b == momentBotSpeaking:
		return spanOutputTransport, true
	case a == momentLLMChunk && b == momentFirstText:
		if markersSeen {
			return spanAwaitingText, true
		}
		return spanLLMInference, true
	}
	return spanDef{}, false
}

type fnStart struct {
	name string
	at   time.Time
}

type LatencyOption func(*UserBotLatencyObserver)

func WithMinContribution(d time.Duration) LatencyOption {
	return func(o *UserBotLatencyObserver) { o.minContribution = d }
}

type UserBotLatencyObserver struct {
	minContribution time.Duration
	audioSeen       atomic.Int32

	mu   sync.Mutex
	seen RecentFrames

	userStoppedAt     time.Time
	userTurnStart     time.Time
	userTurn          time.Duration
	clientConnectedAt time.Time
	pipelineStartedAt time.Time
	firstSpeechDone   bool

	moments     [maxMoments]moment
	nMoments    int
	llmRequest  int
	markersSeen bool
	ttfb        []TTFBBreakdown
	textAgg     *TextAggregationBreakdown
	fnStarts    map[string]fnStart
	fnCalls     []FunctionCallLatency

	measuredHandlers    []func(time.Duration)
	breakdownHandlers   []func(LatencyBreakdown)
	firstSpeechHandlers []func(time.Duration)
}

func NewUserBotLatencyObserver(opts ...LatencyOption) *UserBotLatencyObserver {
	o := &UserBotLatencyObserver{
		minContribution: DefaultMinContribution,
		llmRequest:      -1,
		fnStarts:        make(map[string]fnStart),
	}
	for _, opt := range opts {
		opt(o)
	}
	return o
}

func (o *UserBotLatencyObserver) OnLatencyMeasured(fn func(time.Duration)) {
	if fn == nil {
		return
	}
	o.mu.Lock()
	o.measuredHandlers = append(o.measuredHandlers[:len(o.measuredHandlers):len(o.measuredHandlers)], fn)
	o.mu.Unlock()
}

func (o *UserBotLatencyObserver) OnLatencyBreakdown(fn func(LatencyBreakdown)) {
	if fn == nil {
		return
	}
	o.mu.Lock()
	o.breakdownHandlers = append(o.breakdownHandlers[:len(o.breakdownHandlers):len(o.breakdownHandlers)], fn)
	o.mu.Unlock()
}

func (o *UserBotLatencyObserver) OnFirstBotSpeechLatency(fn func(time.Duration)) {
	if fn == nil {
		return
	}
	o.mu.Lock()
	o.firstSpeechHandlers = append(o.firstSpeechHandlers[:len(o.firstSpeechHandlers):len(o.firstSpeechHandlers)], fn)
	o.mu.Unlock()
}

func (o *UserBotLatencyObserver) OnProcessFrame(pipeline.ProcessFrameEvent) {}

const firstAudioCandidates = 4

func (o *UserBotLatencyObserver) Interested(frame frames.Frame, direction frames.FrameDirection, kind pipeline.EventKind) bool {
	if kind != pipeline.PushEvent || direction != frames.Downstream {
		return false
	}
	switch frame.(type) {
	case *frames.TTSAudioFrame:
		if o.audioSeen.Load() >= firstAudioCandidates {
			return false
		}
		o.audioSeen.Add(1)
		return true
	case *frames.ClientConnectedFrame, *frames.VADUserStartedSpeakingFrame, *frames.UserStartedSpeakingFrame,
		*frames.VADUserStoppedSpeakingFrame, *frames.UserStoppedSpeakingFrame, *frames.InterruptionFrame,
		*frames.BotStartedSpeakingFrame:
		o.audioSeen.Store(0)
		return true
	case *frames.LLMTextFrame, *frames.TranscriptionFrame, *frames.LLMFullResponseStartFrame, *frames.LLMMarkerFrame,
		*frames.FunctionCallInProgressFrame, *frames.FunctionCallResultFrame, *frames.MetricsFrame:
		return true
	}
	return false
}

func (o *UserBotLatencyObserver) OnPipelineStarted() {
	o.mu.Lock()
	if o.pipelineStartedAt.IsZero() {
		o.pipelineStartedAt = time.Now()
	}
	o.mu.Unlock()
}

func (o *UserBotLatencyObserver) OnPipelineStopped() {}

func (o *UserBotLatencyObserver) OnPushFrame(event pipeline.PushFrameEvent) {
	if event.Direction != frames.Downstream {
		return
	}
	switch event.Frame.(type) {
	case *frames.LLMTextFrame:
		o.mu.Lock()
		o.mark(momentFirstText, event.Timestamp, event.ProcessorName, true, false)
		o.mu.Unlock()
		return
	case *frames.TTSAudioFrame:
		o.mu.Lock()
		o.mark(momentFirstAudio, event.Timestamp, event.ProcessorName, true, false)
		o.mu.Unlock()
		return
	case *frames.BotStartedSpeakingFrame:
		o.botStartedSpeaking(event)
		return
	case *frames.ClientConnectedFrame, *frames.VADUserStartedSpeakingFrame, *frames.UserStartedSpeakingFrame,
		*frames.VADUserStoppedSpeakingFrame, *frames.UserStoppedSpeakingFrame, *frames.TranscriptionFrame,
		*frames.LLMFullResponseStartFrame, *frames.LLMMarkerFrame, *frames.InterruptionFrame,
		*frames.FunctionCallInProgressFrame, *frames.FunctionCallResultFrame, *frames.MetricsFrame:
	default:
		return
	}

	now := event.Timestamp
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.seen.FirstSeen(event.Frame.ID()) {
		return
	}

	switch f := event.Frame.(type) {
	case *frames.ClientConnectedFrame:
		if o.clientConnectedAt.IsZero() {
			o.clientConnectedAt = now
		}
	case *frames.VADUserStartedSpeakingFrame, *frames.UserStartedSpeakingFrame:
		o.userStoppedAt = time.Time{}
		o.userTurn = 0
		o.resetAccumulators()
		o.firstSpeechDone = true
	case *frames.VADUserStoppedSpeakingFrame:
		at := f.Timestamp
		if at.IsZero() {
			at = now
		}
		o.userStoppedAt = at.Add(-time.Duration(f.StopSecs * float64(time.Second)))
		o.userTurnStart = o.userStoppedAt
		o.mark(momentVADStop, at, "", false, false)
	case *frames.UserStoppedSpeakingFrame:
		if o.userStoppedAt.IsZero() {
			o.userStoppedAt = now
			o.userTurnStart = now
			o.mark(momentVADStop, now, "", false, false)
		} else {
			o.userTurn = now.Sub(o.userStoppedAt)
		}
	case *frames.TranscriptionFrame:
		if !o.hasMoment(momentLLMRequest) {
			o.dropMoments(momentTranscript)
			o.mark(momentTranscript, now, event.ProcessorName, false, false)
		}
	case *frames.LLMFullResponseStartFrame:
		if o.mark(momentLLMRequest, now, event.ProcessorName, false, false) {
			o.llmRequest = o.nMoments - 1
		}
	case *frames.LLMMarkerFrame:
		o.markersSeen = true
		if f.AppendToContextImmediately {
			o.mark(momentMarkerIncomplete, now, "", false, false)
		} else {
			o.mark(momentMarkerComplete, now, "", true, false)
		}
	case *frames.InterruptionFrame:
		o.resetAccumulators()
	case *frames.FunctionCallInProgressFrame:
		o.fnStarts[f.ToolCallID] = fnStart{name: f.FunctionName, at: now}
	case *frames.FunctionCallResultFrame:
		if start, ok := o.fnStarts[f.ToolCallID]; ok {
			delete(o.fnStarts, f.ToolCallID)
			o.fnCalls = append(o.fnCalls, FunctionCallLatency{FunctionName: start.name, StartTime: start.at, Duration: now.Sub(start.at)})
		}
	case *frames.MetricsFrame:
		o.handleMetrics(f, now)
	}
}

func (o *UserBotLatencyObserver) measuring() bool {
	if !o.userStoppedAt.IsZero() {
		return true
	}
	return !o.clientConnectedAt.IsZero() && !o.firstSpeechDone
}

func (o *UserBotLatencyObserver) mark(kind momentKind, at time.Time, source string, once, derived bool) bool {
	if !o.measuring() || o.nMoments == maxMoments {
		return false
	}
	if once && o.hasMoment(kind) {
		return false
	}
	o.moments[o.nMoments] = moment{at: at, kind: kind, source: source, derived: derived}
	o.nMoments++
	return true
}

func (o *UserBotLatencyObserver) hasMoment(kind momentKind) bool {
	for i := 0; i < o.nMoments; i++ {
		if o.moments[i].kind == kind {
			return true
		}
	}
	return false
}

func (o *UserBotLatencyObserver) dropMoments(kind momentKind) {
	kept := 0
	for i := 0; i < o.nMoments; i++ {
		if o.moments[i].kind == kind {
			continue
		}
		if i == o.llmRequest {
			o.llmRequest = kept
		}
		o.moments[kept] = o.moments[i]
		kept++
	}
	for i := kept; i < o.nMoments; i++ {
		o.moments[i] = moment{}
	}
	o.nMoments = kept
}

func (o *UserBotLatencyObserver) handleMetrics(f *frames.MetricsFrame, now time.Time) {
	if !o.measuring() {
		return
	}
	for _, d := range f.Data {
		switch m := d.(type) {
		case frames.TTFBMetricsData:
			if m.Value <= 0 {
				continue
			}
			o.ttfb = append(o.ttfb, TTFBBreakdown{Processor: m.Processor, Model: m.Model, StartTime: now.Add(-m.Value), Duration: m.Value})
			if o.llmRequest >= 0 && m.Processor == o.moments[o.llmRequest].source {
				at := o.moments[o.llmRequest].at.Add(m.Value)
				if at.After(now) {
					at = now
				}
				o.mark(momentLLMChunk, at, m.Processor, false, true)
			}
		case frames.TextAggregationMetricsData:
			if o.textAgg == nil {
				o.textAgg = &TextAggregationBreakdown{Processor: m.Processor, StartTime: now.Add(-m.Value), Duration: m.Value}
			}
		}
	}
}

func (o *UserBotLatencyObserver) botStartedSpeaking(event pipeline.PushFrameEvent) {
	now := event.Timestamp
	o.mu.Lock()
	o.mark(momentBotSpeaking, now, event.ProcessorName, true, false)

	var firstSpeech, measured time.Duration
	hasFirst, hasMeasured := false, false
	if !o.clientConnectedAt.IsZero() && !o.firstSpeechDone {
		o.firstSpeechDone = true
		firstSpeech = now.Sub(o.clientConnectedAt)
		hasFirst = true
	}
	if !o.userStoppedAt.IsZero() {
		measured = now.Sub(o.userStoppedAt)
		o.userStoppedAt = time.Time{}
		hasMeasured = true
	}
	var breakdown LatencyBreakdown
	if hasFirst || hasMeasured {
		breakdown = o.buildBreakdown()
		o.resetAccumulators()
	}
	firstHandlers := o.firstSpeechHandlers
	measuredHandlers := o.measuredHandlers
	breakdownHandlers := o.breakdownHandlers
	o.mu.Unlock()

	if hasFirst {
		for _, h := range firstHandlers {
			h(firstSpeech)
		}
	}
	if hasMeasured {
		for _, h := range measuredHandlers {
			h(measured)
		}
	}
	if hasFirst || hasMeasured {
		for _, h := range breakdownHandlers {
			h(breakdown)
		}
	}
}

func (o *UserBotLatencyObserver) buildBreakdown() LatencyBreakdown {
	contributions := o.buildContributions()
	b := LatencyBreakdown{
		TTFB:          append([]TTFBBreakdown(nil), o.ttfb...),
		Contributions: contributions,
		UserTurnStart: o.userTurnStart,
		UserTurn:      o.userTurn,
		FunctionCalls: append([]FunctionCallLatency(nil), o.fnCalls...),
	}
	if o.textAgg != nil {
		ta := *o.textAgg
		b.TextAggregation = &ta
	}
	if len(contributions) > 0 {
		b.MeasuredFrom = MeasuredFromClientConnected
		if !o.userTurnStart.IsZero() {
			b.MeasuredFrom = MeasuredFromUserSilence
		}
	}
	for _, c := range contributions {
		b.Total += c.Duration
	}
	return b
}

func (o *UserBotLatencyObserver) derivedMoments(dst []moment) []moment {
	for i := 0; i < o.nMoments; i++ {
		if o.moments[i].kind == momentFirstText {
			if o.textAgg != nil && o.textAgg.Duration > 0 {
				dst = append(dst, moment{at: o.moments[i].at.Add(o.textAgg.Duration), kind: momentSentence, derived: true})
			}
			break
		}
	}
	if len(o.fnCalls) == 0 {
		return dst
	}
	calls := append([]FunctionCallLatency(nil), o.fnCalls...)
	sort.Slice(calls, func(i, j int) bool { return calls[i].StartTime.Before(calls[j].StartTime) })
	groupStart := 0
	groupEnd := calls[0].StartTime.Add(calls[0].Duration)
	flush := func(end int) {
		group := calls[groupStart:end]
		slowest := group[0]
		finish := group[0].StartTime.Add(group[0].Duration)
		for _, c := range group[1:] {
			if c.Duration > slowest.Duration {
				slowest = c
			}
			if e := c.StartTime.Add(c.Duration); e.After(finish) {
				finish = e
			}
		}
		owner := slowest.FunctionName
		if len(group) > 1 {
			owner += fmt.Sprintf(" +%d", len(group)-1)
		}
		dst = append(dst,
			moment{at: group[0].StartTime, kind: momentHandlersStart, derived: true},
			moment{at: finish, kind: momentHandlersEnd, source: owner, derived: true},
		)
	}
	for i := 1; i < len(calls); i++ {
		if !calls[i].StartTime.After(groupEnd) {
			if e := calls[i].StartTime.Add(calls[i].Duration); e.After(groupEnd) {
				groupEnd = e
			}
			continue
		}
		flush(i)
		groupStart = i
		groupEnd = calls[i].StartTime.Add(calls[i].Duration)
	}
	flush(len(calls))
	return dst
}

func (o *UserBotLatencyObserver) buildContributions() []LatencyContribution {
	start := o.userTurnStart
	startKind := momentSilence
	if start.IsZero() {
		startKind = momentReady
		if o.clientConnectedAt.After(start) {
			start = o.clientConnectedAt
		}
		if o.pipelineStartedAt.After(start) {
			start = o.pipelineStartedAt
		}
	}
	if start.IsZero() || !o.hasMoment(momentBotSpeaking) {
		return nil
	}

	moments := make([]moment, 0, o.nMoments+8)
	moments = append(moments, moment{at: start, kind: startKind})
	for i := 0; i < o.nMoments; i++ {
		if !o.moments[i].at.Before(start) {
			moments = append(moments, o.moments[i])
		}
	}
	moments = o.derivedMoments(moments)
	sort.SliceStable(moments, func(i, j int) bool {
		if !moments[i].at.Equal(moments[j].at) {
			return moments[i].at.Before(moments[j].at)
		}
		return moments[i].derived && !moments[j].derived
	})

	var spans []LatencyContribution
	core := map[string]bool{}
	for i := 0; i+1 < len(moments); i++ {
		opener, closer := moments[i], moments[i+1]
		if !closer.at.After(opener.at) {
			continue
		}
		def, ok := spanFor(opener.kind, closer.kind, o.markersSeen)
		if !ok {
			continue
		}
		owner, kind := def.owner, OwnerService
		switch def.rule {
		case ownerOpener:
			owner = opener.source
		case ownerCloser:
			owner = closer.source
		default:
			kind = OwnerBot
			if strings.HasPrefix(owner, "config: ") {
				kind = OwnerSetting
			}
		}
		if def.core {
			core[def.label] = true
		}
		spans = append(spans, LatencyContribution{
			Key: def.key, Label: def.label, Owner: owner, OwnerKind: kind,
			StartTime: opener.at, Duration: closer.at.Sub(opener.at),
		})
	}

	merged := spans[:0]
	for _, s := range spans {
		if n := len(merged); n > 0 {
			prev := &merged[n-1]
			gap := s.StartTime.Sub(prev.StartTime.Add(prev.Duration))
			if gap < printsAsZero && prev.Label == s.Label && prev.Owner == s.Owner {
				prev.Duration = s.StartTime.Add(s.Duration).Sub(prev.StartTime)
				continue
			}
		}
		merged = append(merged, s)
	}

	named := make([]LatencyContribution, 0, len(merged)+1)
	var sum time.Duration
	for _, c := range merged {
		if c.Duration >= o.minContribution || (core[c.Label] && c.Duration >= printsAsZero) {
			named = append(named, c)
			sum += c.Duration
		}
	}
	if rest := moments[len(moments)-1].at.Sub(start) - sum; rest >= printsAsZero {
		named = append(named, LatencyContribution{
			Key: "pipeline", Label: "pipeline", Owner: "strawgo", OwnerKind: OwnerPipeline,
			StartTime: start, Duration: rest,
		})
	}
	return named
}

func (o *UserBotLatencyObserver) resetAccumulators() {
	for i := 0; i < o.nMoments; i++ {
		o.moments[i] = moment{}
	}
	o.nMoments = 0
	o.llmRequest = -1
	o.markersSeen = false
	o.ttfb = o.ttfb[:0]
	o.textAgg = nil
	o.userTurnStart = time.Time{}
	o.userTurn = 0
	clear(o.fnStarts)
	o.fnCalls = o.fnCalls[:0]
}
