package tracing

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/pipeline/observers"
)

const InstrumentationName = "github.com/square-key-labs/strawgo-ai"

type Config struct {
	Tracer           trace.Tracer
	ConversationID   string
	ConversationType string
	Attributes       []attribute.KeyValue
	RecordText       bool
	TurnTracker      *observers.TurnTrackingObserver
	LatencyTracker   *observers.UserBotLatencyObserver
	TurnEndTimeout   time.Duration
	ProviderName     func(processorName string) string
}

type serviceSpan struct {
	kind      string
	processor string
	span      trace.Span
	closing   bool
	endAt     time.Time
	text      strings.Builder
}

type Observer struct {
	cfg         Config
	tracer      trace.Tracer
	provider    func(string) string
	turn        *observers.TurnTrackingObserver
	latency     *observers.UserBotLatencyObserver
	ownsTurn    bool
	ownsLatency bool
	inert       atomic.Bool

	mu           sync.Mutex
	seen         observers.RecentFrames
	convSpan     trace.Span
	convCtx      context.Context
	convID       string
	turnSpan     trace.Span
	turnCtx      context.Context
	turnNumber   int
	spans        []*serviceSpan
	models       map[string]string
	userSpeechAt time.Time
	lastContext  *llm.Context
}

func NewObserver(cfg Config) *Observer {
	o := &Observer{
		cfg:      cfg,
		tracer:   cfg.Tracer,
		provider: cfg.ProviderName,
		turn:     cfg.TurnTracker,
		latency:  cfg.LatencyTracker,
		models:   make(map[string]string),
	}
	if o.tracer == nil {
		o.tracer = otel.Tracer(InstrumentationName)
	}
	if o.provider == nil {
		o.provider = ProviderName
	}
	if o.turn == nil {
		var opts []observers.TurnTrackingOption
		if cfg.TurnEndTimeout > 0 {
			opts = append(opts, observers.WithTurnEndTimeout(cfg.TurnEndTimeout))
		}
		o.turn = observers.NewTurnTrackingObserver(opts...)
		o.ownsTurn = true
	}
	if o.latency == nil {
		o.latency = observers.NewUserBotLatencyObserver()
		o.ownsLatency = true
	}
	o.turn.OnTurnStarted(o.onTurnStarted)
	o.turn.OnTurnEnded(o.onTurnEnded)
	o.latency.OnLatencyMeasured(o.onLatencyMeasured)
	return o
}

func (o *Observer) TurnTracker() *observers.TurnTrackingObserver { return o.turn }

func (o *Observer) LatencyTracker() *observers.UserBotLatencyObserver { return o.latency }

func (o *Observer) Enabled() bool { return !o.inert.Load() }

func (o *Observer) ConversationID() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.convID
}

func (o *Observer) ConversationContext() context.Context {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.convCtx == nil {
		return context.Background()
	}
	return o.convCtx
}

func (o *Observer) TurnContext() context.Context {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.parentLocked()
}

func (o *Observer) TurnNumber() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.turnNumber
}

func (o *Observer) OnProcessFrame(pipeline.ProcessFrameEvent) {}

func (o *Observer) Interested(frame frames.Frame, direction frames.FrameDirection, kind pipeline.EventKind) bool {
	if kind != pipeline.PushEvent {
		return false
	}
	if o.inert.Load() {
		_, ok := frame.(*frames.StartFrame)
		return ok
	}
	interested := o.ownsLatency && o.latency.Interested(frame, direction, kind)
	if !interested && o.ownsTurn {
		interested = o.turn.Interested(frame, direction, kind)
	}
	if interested {
		return true
	}
	switch frame.(type) {
	case *frames.LLMTextFrame, *frames.TTSTextFrame:
		return o.cfg.RecordText
	case *frames.LLMContextFrame, *frames.StartFrame, *frames.EndFrame, *frames.CancelFrame,
		*frames.UserStartedSpeakingFrame, *frames.InterimTranscriptionFrame, *frames.TranscriptionFrame,
		*frames.LLMFullResponseStartFrame, *frames.LLMFullResponseEndFrame, *frames.FunctionCallsStartedFrame,
		*frames.TTSStartedFrame, *frames.TTSStoppedFrame, *frames.InterruptionFrame, *frames.MetricsFrame:
		return true
	}
	return false
}

func (o *Observer) OnPipelineStarted() {
	o.inert.Store(false)
	if o.ownsTurn {
		o.turn.OnPipelineStarted()
	}
	if o.ownsLatency {
		o.latency.OnPipelineStarted()
	}
}

func (o *Observer) OnPipelineStopped() {
	if o.ownsTurn {
		o.turn.OnPipelineStopped()
	}
	if o.ownsLatency {
		o.latency.OnPipelineStopped()
	}
	o.mu.Lock()
	o.endConversationLocked()
	o.mu.Unlock()
}

func (o *Observer) StartConversation(id string) {
	o.mu.Lock()
	o.startConversationLocked(id)
	o.mu.Unlock()
}

func (o *Observer) EndConversation() {
	o.mu.Lock()
	o.endConversationLocked()
	o.mu.Unlock()
}

func (o *Observer) OnPushFrame(event pipeline.PushFrameEvent) {
	if o.inert.Load() {
		if _, ok := event.Frame.(*frames.StartFrame); !ok {
			return
		}
	}
	if o.ownsTurn {
		o.turn.OnPushFrame(event)
	}
	if o.ownsLatency {
		o.latency.OnPushFrame(event)
	}

	switch event.Frame.(type) {
	case *frames.LLMTextFrame:
		if o.cfg.RecordText {
			o.appendText(KindLLM, event)
		}
		return
	case *frames.TTSTextFrame:
		if o.cfg.RecordText {
			o.appendText(KindTTS, event)
		}
		return
	case *frames.LLMContextFrame:
		o.mu.Lock()
		if f := event.Frame.(*frames.LLMContextFrame); f.Context != nil {
			o.lastContext = f.Context
		}
		o.mu.Unlock()
		return
	case *frames.StartFrame, *frames.EndFrame, *frames.CancelFrame, *frames.UserStartedSpeakingFrame,
		*frames.InterimTranscriptionFrame, *frames.TranscriptionFrame,
		*frames.LLMFullResponseStartFrame, *frames.LLMFullResponseEndFrame, *frames.FunctionCallsStartedFrame,
		*frames.TTSStartedFrame, *frames.TTSStoppedFrame, *frames.InterruptionFrame, *frames.MetricsFrame:
	default:
		return
	}

	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.seen.FirstSeen(event.Frame.ID()) {
		return
	}
	ts := event.Timestamp
	if ts.IsZero() {
		ts = time.Now()
	}
	src := event.ProcessorName

	switch f := event.Frame.(type) {
	case *frames.StartFrame:
		if o.convSpan == nil {
			o.startConversationLocked(o.cfg.ConversationID)
		}
	case *frames.EndFrame, *frames.CancelFrame:
		o.endConversationLocked()
	case *frames.UserStartedSpeakingFrame:
		if o.userSpeechAt.IsZero() {
			o.userSpeechAt = ts
		}
		o.finalizeClosingLocked(KindSTT)
	case *frames.InterimTranscriptionFrame:
		o.sttSpanLocked(src, ts)
	case *frames.TranscriptionFrame:
		s := o.sttSpanLocked(src, ts)
		if s == nil || !f.IsFinal {
			return
		}
		if o.cfg.RecordText && f.Text != "" {
			if s.text.Len() > 0 {
				s.text.WriteByte(' ')
			}
			s.text.WriteString(strings.TrimSpace(f.Text))
		}
		s.span.SetAttributes(attribute.Bool("is_final", true))
		if f.Language != "" {
			s.span.SetAttributes(attribute.String("language", f.Language))
		}
		s.closing, s.endAt = true, ts
		o.userSpeechAt = time.Time{}
	case *frames.LLMFullResponseStartFrame:
		o.finalizeClosingLocked(KindSTT)
		o.finalizeLocked(KindLLM, src)
		s := o.openLocked(KindLLM, src, ts)
		if s != nil {
			o.describeContextLocked(s.span)
		}
	case *frames.LLMFullResponseEndFrame:
		if s := o.findLocked(KindLLM, src); s != nil && !s.closing {
			s.closing, s.endAt = true, ts
		}
	case *frames.FunctionCallsStartedFrame:
		if event.Direction != frames.Downstream {
			return
		}
		if s := o.findLocked(KindLLM, src); s != nil && len(f.FunctionCalls) > 0 {
			names := make([]string, len(f.FunctionCalls))
			for i, c := range f.FunctionCalls {
				names[i] = c.FunctionName
			}
			s.span.SetAttributes(
				attribute.Int("function_calls.count", len(f.FunctionCalls)),
				attribute.String("function_calls.first_name", names[0]),
				attribute.StringSlice("function_calls.names", names),
			)
		}
	case *frames.TTSStartedFrame:
		o.finalizeLocked(KindTTS, src)
		o.openLocked(KindTTS, src, ts)
	case *frames.TTSStoppedFrame:
		if s := o.findLocked(KindTTS, src); s != nil && !s.closing {
			s.closing, s.endAt = true, ts
		}
	case *frames.InterruptionFrame:
		for _, s := range o.spans {
			if s.kind == KindTTS && !s.closing {
				s.span.SetAttributes(attribute.Bool("tts.interrupted", true))
				s.closing, s.endAt = true, ts
			}
		}
	case *frames.MetricsFrame:
		o.applyMetricsLocked(f)
	}
}

func (o *Observer) appendText(kind string, event pipeline.PushFrameEvent) {
	var text string
	switch f := event.Frame.(type) {
	case *frames.LLMTextFrame:
		text = f.Text
	case *frames.TTSTextFrame:
		text = f.Text
	}
	if text == "" {
		return
	}
	o.mu.Lock()
	if s := o.findLocked(kind, event.ProcessorName); s != nil && !s.closing {
		if kind == KindTTS && s.text.Len() > 0 && !strings.HasPrefix(text, " ") {
			s.text.WriteByte(' ')
		}
		s.text.WriteString(text)
	}
	o.mu.Unlock()
}

func (o *Observer) startConversationLocked(id string) {
	if o.convSpan != nil {
		return
	}
	if id == "" {
		id = uuid.NewString()
	}
	convType := o.cfg.ConversationType
	if convType == "" {
		convType = "voice"
	}
	attrs := make([]attribute.KeyValue, 0, 2+len(o.cfg.Attributes))
	attrs = append(attrs, attribute.String("conversation.id", id), attribute.String("conversation.type", convType))
	attrs = append(attrs, o.cfg.Attributes...)
	ctx, span := o.tracer.Start(context.Background(), "conversation", trace.WithAttributes(attrs...), trace.WithNewRoot())
	if !span.IsRecording() {
		span.End()
		o.inert.Store(true)
		return
	}
	o.convSpan, o.convCtx, o.convID = span, ctx, id
}

func (o *Observer) endConversationLocked() {
	o.finalizeAllLocked()
	if o.turnSpan != nil {
		o.turnSpan.SetAttributes(
			attribute.Bool("turn.was_interrupted", true),
			attribute.Bool("turn.ended_by_conversation_end", true),
		)
		o.turnSpan.End()
		o.turnSpan, o.turnCtx = nil, nil
	}
	if o.convSpan != nil {
		o.convSpan.End()
		o.convSpan, o.convCtx = nil, nil
	}
	o.lastContext = nil
	o.userSpeechAt = time.Time{}
}

func (o *Observer) onTurnStarted(turn int) {
	if o.inert.Load() {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.convSpan == nil {
		o.startConversationLocked(o.cfg.ConversationID)
		if o.convSpan == nil {
			return
		}
	}
	if o.turnSpan != nil {
		o.turnSpan.End()
	}
	ctx, span := o.tracer.Start(o.convCtx, "turn", trace.WithAttributes(
		attribute.Int("turn.number", turn),
		attribute.String("turn.type", "conversation"),
		attribute.String("conversation.id", o.convID),
	))
	o.turnSpan, o.turnCtx, o.turnNumber = span, ctx, turn
}

func (o *Observer) onTurnEnded(turn int, duration time.Duration, interrupted bool) {
	if o.inert.Load() {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.turnSpan == nil || turn != o.turnNumber {
		return
	}
	o.finalizeAllLocked()
	o.turnSpan.SetAttributes(
		attribute.Float64("turn.duration_seconds", duration.Seconds()),
		attribute.Bool("turn.was_interrupted", interrupted),
	)
	o.turnSpan.End()
	o.turnSpan, o.turnCtx = nil, nil
}

func (o *Observer) onLatencyMeasured(latency time.Duration) {
	if o.inert.Load() {
		return
	}
	o.mu.Lock()
	if o.turnSpan != nil {
		o.turnSpan.SetAttributes(attribute.Float64("turn.user_bot_latency_seconds", latency.Seconds()))
	}
	o.mu.Unlock()
}

func (o *Observer) parentLocked() context.Context {
	if o.turnCtx != nil {
		return o.turnCtx
	}
	if o.convCtx != nil {
		return o.convCtx
	}
	return context.Background()
}

func (o *Observer) openLocked(kind, processor string, start time.Time) *serviceSpan {
	if o.convSpan == nil {
		return nil
	}
	_, span := o.tracer.Start(o.parentLocked(), kind,
		trace.WithTimestamp(start),
		trace.WithAttributes(serviceStartAttributes(kind, o.provider(processor), o.models[processor])...),
	)
	s := &serviceSpan{kind: kind, processor: processor, span: span}
	o.spans = append(o.spans, s)
	return s
}

func (o *Observer) sttSpanLocked(processor string, ts time.Time) *serviceSpan {
	if s := o.findLocked(KindSTT, processor); s != nil {
		if !s.closing {
			return s
		}
		if o.userSpeechAt.IsZero() || !o.userSpeechAt.After(s.endAt) {
			s.closing = false
			return s
		}
		o.finalizeLocked(KindSTT, processor)
	}
	start := ts
	if !o.userSpeechAt.IsZero() && o.userSpeechAt.Before(ts) {
		start = o.userSpeechAt
	}
	return o.openLocked(KindSTT, processor, start)
}

func (o *Observer) findLocked(kind, processor string) *serviceSpan {
	for i := len(o.spans) - 1; i >= 0; i-- {
		if s := o.spans[i]; s.kind == kind && s.processor == processor {
			return s
		}
	}
	return nil
}

func (o *Observer) findAnyLocked(processor string) *serviceSpan {
	for i := len(o.spans) - 1; i >= 0; i-- {
		if o.spans[i].processor == processor {
			return o.spans[i]
		}
	}
	return nil
}

func (o *Observer) finalizeLocked(kind, processor string) {
	kept := o.spans[:0]
	for _, s := range o.spans {
		if s.kind == kind && s.processor == processor {
			o.endSpan(s)
			continue
		}
		kept = append(kept, s)
	}
	o.trimSpans(kept)
}

func (o *Observer) finalizeClosingLocked(kind string) {
	kept := o.spans[:0]
	for _, s := range o.spans {
		if s.kind == kind && s.closing {
			o.endSpan(s)
			continue
		}
		kept = append(kept, s)
	}
	o.trimSpans(kept)
}

func (o *Observer) finalizeAllLocked() {
	for _, s := range o.spans {
		o.endSpan(s)
	}
	o.trimSpans(o.spans[:0])
}

func (o *Observer) trimSpans(kept []*serviceSpan) {
	for i := len(kept); i < len(o.spans); i++ {
		o.spans[i] = nil
	}
	o.spans = kept
}

func (o *Observer) endSpan(s *serviceSpan) {
	if o.cfg.RecordText && s.text.Len() > 0 {
		key := "text"
		switch s.kind {
		case KindSTT:
			key = "transcript"
		case KindLLM:
			key = "output"
		}
		s.span.SetAttributes(attribute.String(key, s.text.String()))
		if s.kind == KindTTS {
			s.span.SetAttributes(attribute.Int("metrics.character_count", s.text.Len()))
		}
	}
	if s.closing && !s.endAt.IsZero() {
		s.span.End(trace.WithTimestamp(s.endAt))
		return
	}
	s.span.End()
}

func (o *Observer) describeContextLocked(span trace.Span) {
	c := o.lastContext
	if c == nil {
		return
	}
	snap := c.Snapshot()
	if names := toolNames(snap.Tools); len(names) > 0 {
		span.SetAttributes(attribute.Int("tool_count", len(names)), attribute.StringSlice("tools", names))
	}
	if !o.cfg.RecordText {
		return
	}
	var system []string
	for _, m := range snap.Messages {
		if m.Role == llm.RoleSystem || m.Role == llm.RoleDeveloper {
			system = append(system, m.Text())
		}
	}
	if len(system) > 0 {
		span.SetAttributes(attribute.String("gen_ai.system_instructions", strings.Join(system, "\n")))
	}
	if b, err := json.Marshal(messagesForTrace(snap.Messages)); err == nil {
		span.SetAttributes(attribute.String("input", string(b)))
	}
}

type traceMessage struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
}

func messagesForTrace(msgs []llm.Message) []traceMessage {
	out := make([]traceMessage, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, traceMessage{Role: string(m.Role), Content: m.Text()})
	}
	return out
}

func toolNames(t llm.Tools) []string {
	names := make([]string, 0, len(t.Functions))
	for _, f := range t.Functions {
		names = append(names, f.Name)
	}
	return names
}

func (o *Observer) applyMetricsLocked(f *frames.MetricsFrame) {
	for _, d := range f.Data {
		if d == nil {
			continue
		}
		base := d.Base()
		if base.Model != "" {
			o.models[base.Processor] = base.Model
		}
		s := o.findAnyLocked(base.Processor)
		if s == nil {
			continue
		}
		if base.Model != "" {
			s.span.SetAttributes(attribute.String("gen_ai.request.model", base.Model))
		}
		switch m := d.(type) {
		case frames.TTFBMetricsData:
			setTTFB(s.span, m.Value)
		case frames.LLMUsageMetricsData:
			s.span.SetAttributes(UsageAttributes(m.Value)...)
		case frames.STTUsageMetricsData:
			s.span.SetAttributes(attribute.Float64("metrics.audio_seconds", m.Value.AudioSeconds))
		case frames.TTSUsageMetricsData:
			s.span.SetAttributes(attribute.Int("metrics.character_count", m.Value))
		case frames.ProcessingMetricsData:
			s.span.SetAttributes(attribute.Float64("metrics.processing_time", m.Value.Seconds()))
		}
	}
}
