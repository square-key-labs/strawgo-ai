package observers

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

type logClock struct {
	mu      sync.Mutex
	started time.Time
	seen    RecentFrames
}

func (c *logClock) start() {
	c.mu.Lock()
	c.started = time.Now()
	c.seen.Reset()
	c.mu.Unlock()
}

func (c *logClock) first(f frames.Frame, ts time.Time) (float64, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.seen.FirstSeen(f.ID()) {
		return 0, false
	}
	if c.started.IsZero() {
		c.started = ts
	}
	return ts.Sub(c.started).Seconds(), true
}

func arrow(d frames.FrameDirection) string {
	if d == frames.Upstream {
		return "←"
	}
	return "→"
}

type MetricsLogObserver struct {
	clock   logClock
	include []reflect.Type
}

func NewMetricsLogObserver(include ...frames.MetricsData) *MetricsLogObserver {
	o := &MetricsLogObserver{}
	for _, m := range include {
		if m != nil {
			o.include = append(o.include, reflect.TypeOf(m))
		}
	}
	return o
}

func (o *MetricsLogObserver) OnProcessFrame(pipeline.ProcessFrameEvent) {}

func (o *MetricsLogObserver) Interested(frame frames.Frame, _ frames.FrameDirection, kind pipeline.EventKind) bool {
	if kind != pipeline.PushEvent {
		return false
	}
	_, ok := frame.(*frames.MetricsFrame)
	return ok && logger.IsDebugEnabled()
}

func (o *MetricsLogObserver) OnPipelineStarted() { o.clock.start() }
func (o *MetricsLogObserver) OnPipelineStopped() {}

func (o *MetricsLogObserver) OnPushFrame(event pipeline.PushFrameEvent) {
	f, ok := event.Frame.(*frames.MetricsFrame)
	if !ok || !logger.IsDebugEnabled() {
		return
	}
	at, first := o.clock.first(f, event.Timestamp)
	if !first {
		return
	}
	for _, d := range f.Data {
		if d != nil && o.wants(d) {
			logger.Debug("%s at %.3fs", FormatMetrics(d), at)
		}
	}
}

func (o *MetricsLogObserver) wants(d frames.MetricsData) bool {
	if len(o.include) == 0 {
		return true
	}
	t := reflect.TypeOf(d)
	for _, want := range o.include {
		if t == want {
			return true
		}
	}
	return false
}

func FormatMetrics(d frames.MetricsData) string {
	base := d.Base()
	model := ""
	if base.Model != "" {
		model = " (" + base.Model + ")"
	}
	p := "[" + base.Processor + "]"
	switch m := d.(type) {
	case frames.TTFBMetricsData:
		return fmt.Sprintf("📊 %s TTFB%s: %.3fs", p, model, m.Value.Seconds())
	case frames.TTFAMetricsData:
		return fmt.Sprintf("📊 %s TTFA%s: %.3fs (%.3fs leading silence)", p, model, m.TTFA.Seconds(), m.LeadingSilence.Seconds())
	case frames.TTFATMetricsData:
		return fmt.Sprintf("📊 %s TTFAT%s: %.3fs (%.3fs thinking)", p, model, m.TTFAT.Seconds(), m.ThinkingTime.Seconds())
	case frames.ProcessingMetricsData:
		return fmt.Sprintf("📊 %s PROCESSING TIME%s: %.3fs", p, model, m.Value.Seconds())
	case frames.LLMUsageMetricsData:
		u := m.Value
		details := []string{
			fmt.Sprintf("prompt: %d", u.PromptTokens),
			fmt.Sprintf("completion: %d", u.CompletionTokens),
			fmt.Sprintf("total: %d", u.TotalTokens),
		}
		if u.CacheReadTokens > 0 {
			details = append(details, fmt.Sprintf("cache_read: %d", u.CacheReadTokens))
		}
		if u.CacheCreationTokens > 0 {
			details = append(details, fmt.Sprintf("cache_creation: %d", u.CacheCreationTokens))
		}
		if u.ReasoningTokens > 0 {
			details = append(details, fmt.Sprintf("reasoning: %d", u.ReasoningTokens))
		}
		return fmt.Sprintf("📊 %s LLM TOKEN USAGE%s: %s", p, model, strings.Join(details, ", "))
	case frames.STTUsageMetricsData:
		return fmt.Sprintf("📊 %s STT USAGE%s: %.3fs audio", p, model, m.Value.AudioSeconds)
	case frames.TTSUsageMetricsData:
		return fmt.Sprintf("📊 %s TTS USAGE%s: %d characters", p, model, m.Value)
	case frames.TextAggregationMetricsData:
		return fmt.Sprintf("📊 %s TEXT AGGREGATION%s: %.3fs", p, model, m.Value.Seconds())
	case frames.TurnMetricsData:
		status := "INCOMPLETE"
		if m.IsComplete {
			status = "COMPLETE"
		}
		return fmt.Sprintf("📊 %s TURN%s: %s (probability: %.2f%%, e2e: %.1fms)", p, model, status, m.Probability*100, float64(m.E2EProcessingTime)/float64(time.Millisecond))
	}
	return fmt.Sprintf("📊 %s METRICS%s: %+v", p, model, d)
}

type LLMLogObserver struct {
	clock logClock
	mu    sync.Mutex
	llms  map[string]struct{}
}

func NewLLMLogObserver() *LLMLogObserver {
	return &LLMLogObserver{llms: make(map[string]struct{})}
}

func (o *LLMLogObserver) OnProcessFrame(pipeline.ProcessFrameEvent) {}

func (o *LLMLogObserver) Interested(frame frames.Frame, _ frames.FrameDirection, kind pipeline.EventKind) bool {
	if kind != pipeline.PushEvent {
		return false
	}
	switch frame.(type) {
	case *frames.LLMFullResponseStartFrame, *frames.LLMFullResponseEndFrame, *frames.LLMTextFrame,
		*frames.FunctionCallInProgressFrame, *frames.FunctionCallResultFrame, *frames.LLMContextFrame:
		return logger.IsDebugEnabled()
	}
	return false
}

func (o *LLMLogObserver) OnPipelineStarted() { o.clock.start() }
func (o *LLMLogObserver) OnPipelineStopped() {}

func (o *LLMLogObserver) isLLM(name string) bool {
	o.mu.Lock()
	_, ok := o.llms[name]
	o.mu.Unlock()
	return ok
}

func (o *LLMLogObserver) OnPushFrame(event pipeline.PushFrameEvent) {
	switch event.Frame.(type) {
	case *frames.LLMFullResponseStartFrame, *frames.LLMFullResponseEndFrame, *frames.LLMTextFrame,
		*frames.FunctionCallInProgressFrame, *frames.FunctionCallResultFrame, *frames.LLMContextFrame:
	default:
		return
	}
	if !logger.IsDebugEnabled() {
		return
	}
	src := event.ProcessorName
	if _, ok := event.Frame.(*frames.LLMTextFrame); ok && !o.isLLM(src) {
		return
	}
	at, first := o.clock.first(event.Frame, event.Timestamp)
	if !first {
		return
	}
	a := arrow(event.Direction)
	switch f := event.Frame.(type) {
	case *frames.LLMFullResponseStartFrame:
		o.mu.Lock()
		o.llms[src] = struct{}{}
		o.mu.Unlock()
		logger.Debug("🧠 %s %s LLM START RESPONSE at %.2fs", src, a, at)
	case *frames.LLMFullResponseEndFrame:
		if o.isLLM(src) {
			logger.Debug("🧠 %s %s LLM END RESPONSE at %.2fs", src, a, at)
		}
	case *frames.LLMTextFrame:
		logger.Debug("🧠 %s %s LLM GENERATING: %q at %.2fs", src, a, f.Text, at)
	case *frames.FunctionCallInProgressFrame:
		if event.Direction == frames.Upstream {
			logger.Debug("🧠 %s %s LLM FUNCTION CALL (%s): %q(%v) at %.2fs", src, a, f.ToolCallID, f.FunctionName, f.Arguments, at)
		}
	case *frames.FunctionCallResultFrame:
		if event.Direction == frames.Upstream {
			logger.Debug("🧠 %s %s LLM FUNCTION CALL RESULT (%s): %v at %.2fs", src, a, f.ToolCallID, f.Result, at)
		}
	case *frames.LLMContextFrame:
		if f.Context != nil {
			logger.Debug("🧠 %s %s LLM CONTEXT FRAME: %d messages %v at %.2fs", src, a, f.Context.Len(), f.Context.Messages(), at)
		}
	}
}

type TranscriptionLogObserver struct {
	clock logClock
}

func NewTranscriptionLogObserver() *TranscriptionLogObserver {
	return &TranscriptionLogObserver{}
}

func (o *TranscriptionLogObserver) OnProcessFrame(pipeline.ProcessFrameEvent) {}

func (o *TranscriptionLogObserver) Interested(frame frames.Frame, _ frames.FrameDirection, kind pipeline.EventKind) bool {
	if kind != pipeline.PushEvent {
		return false
	}
	switch frame.(type) {
	case *frames.TranscriptionFrame, *frames.InterimTranscriptionFrame:
		return logger.IsDebugEnabled()
	}
	return false
}

func (o *TranscriptionLogObserver) OnPipelineStarted() { o.clock.start() }
func (o *TranscriptionLogObserver) OnPipelineStopped() {}

func (o *TranscriptionLogObserver) OnPushFrame(event pipeline.PushFrameEvent) {
	switch event.Frame.(type) {
	case *frames.TranscriptionFrame, *frames.InterimTranscriptionFrame:
	default:
		return
	}
	if !logger.IsDebugEnabled() {
		return
	}
	at, first := o.clock.first(event.Frame, event.Timestamp)
	if !first {
		return
	}
	switch f := event.Frame.(type) {
	case *frames.TranscriptionFrame:
		logger.Debug("💬 %s → TRANSCRIPTION: %q (final=%t) at %.2fs", event.ProcessorName, f.Text, f.IsFinal, at)
	case *frames.InterimTranscriptionFrame:
		logger.Debug("💬 %s → INTERIM TRANSCRIPTION: %q from %q at %.2fs", event.ProcessorName, f.Text, f.UserID, at)
	}
}

type DebugLogFilter struct {
	Frame     frames.Frame
	Processor string
}

type DebugLogObserver struct {
	clock         logClock
	filters       []debugFilter
	excludeFields map[string]bool
}

type debugFilter struct {
	t         reflect.Type
	processor string
}

func NewDebugLogObserver(filters []DebugLogFilter, excludeFields ...string) *DebugLogObserver {
	o := &DebugLogObserver{excludeFields: map[string]bool{"Data": true, "Audio": true, "Image": true, "Images": true}}
	if len(excludeFields) > 0 {
		o.excludeFields = make(map[string]bool, len(excludeFields))
		for _, f := range excludeFields {
			o.excludeFields[f] = true
		}
	}
	for _, f := range filters {
		if f.Frame != nil {
			o.filters = append(o.filters, debugFilter{t: reflect.TypeOf(f.Frame), processor: f.Processor})
		}
	}
	return o
}

func NewDebugLogObserverFor(examples ...frames.Frame) *DebugLogObserver {
	filters := make([]DebugLogFilter, 0, len(examples))
	for _, ex := range examples {
		filters = append(filters, DebugLogFilter{Frame: ex})
	}
	return NewDebugLogObserver(filters)
}

func (o *DebugLogObserver) OnProcessFrame(pipeline.ProcessFrameEvent) {}

func (o *DebugLogObserver) Interested(frame frames.Frame, _ frames.FrameDirection, kind pipeline.EventKind) bool {
	if kind != pipeline.PushEvent || !logger.IsDebugEnabled() {
		return false
	}
	if len(o.filters) == 0 {
		return true
	}
	t := reflect.TypeOf(frame)
	for _, flt := range o.filters {
		if flt.t == t {
			return true
		}
	}
	return false
}

func (o *DebugLogObserver) OnPipelineStarted() { o.clock.start() }
func (o *DebugLogObserver) OnPipelineStopped() {}

func (o *DebugLogObserver) wants(f frames.Frame, processor string) bool {
	if len(o.filters) == 0 {
		return true
	}
	t := reflect.TypeOf(f)
	for _, flt := range o.filters {
		if flt.t == t && (flt.processor == "" || flt.processor == processor) {
			return true
		}
	}
	return false
}

func (o *DebugLogObserver) OnPushFrame(event pipeline.PushFrameEvent) {
	if !logger.IsDebugEnabled() || !o.wants(event.Frame, event.ProcessorName) {
		return
	}
	at := event.Timestamp.Sub(o.startedAt(event.Timestamp)).Seconds()
	details := o.describe(event.Frame)
	if details != "" {
		logger.Debug("%s %s %s %s at %.2fs", event.ProcessorName, arrow(event.Direction), event.Frame.Name(), details, at)
		return
	}
	logger.Debug("%s %s %s at %.2fs", event.ProcessorName, arrow(event.Direction), event.Frame.Name(), at)
}

func (o *DebugLogObserver) startedAt(ts time.Time) time.Time {
	o.clock.mu.Lock()
	defer o.clock.mu.Unlock()
	if o.clock.started.IsZero() {
		o.clock.started = ts
	}
	return o.clock.started
}

func (o *DebugLogObserver) describe(f frames.Frame) string {
	v := reflect.ValueOf(f)
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return ""
		}
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return ""
	}
	t := v.Type()
	var parts []string
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if field.Anonymous || !field.IsExported() || o.excludeFields[field.Name] {
			continue
		}
		fv := v.Field(i)
		if fv.IsZero() {
			continue
		}
		parts = append(parts, field.Name+": "+formatValue(fv))
	}
	return strings.Join(parts, ", ")
}

func formatValue(v reflect.Value) string {
	switch v.Kind() {
	case reflect.String:
		return fmt.Sprintf("%q", v.String())
	case reflect.Slice:
		if v.Type().Elem().Kind() == reflect.Uint8 {
			return fmt.Sprintf("%d bytes", v.Len())
		}
		if v.Len() > 3 {
			return fmt.Sprintf("%d items", v.Len())
		}
	case reflect.Interface, reflect.Pointer:
		if s, ok := v.Interface().(fmt.Stringer); ok {
			return s.String()
		}
	}
	return fmt.Sprintf("%v", v.Interface())
}
