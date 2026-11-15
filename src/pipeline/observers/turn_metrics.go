package observers

import (
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

type TurnMetricsData struct {
	ProcessorName string
	StartTime     time.Time
	EndTime       time.Time
	Duration      time.Duration
}

func NewTurnMetricsData(processorName string) *TurnMetricsData {
	return &TurnMetricsData{ProcessorName: processorName}
}

func (m *TurnMetricsData) Start() {
	m.StartTime = time.Now()
	m.EndTime = time.Time{}
	m.Duration = 0
}

func (m *TurnMetricsData) Stop() {
	m.EndTime = time.Now()
	if m.StartTime.IsZero() {
		m.Duration = 0
		return
	}
	m.Duration = m.EndTime.Sub(m.StartTime)
}

func (m *TurnMetricsData) ToFrame() *frames.TurnMetricsFrame {
	return frames.NewTurnMetricsFrame(m.ProcessorName, m.StartTime, m.EndTime, m.Duration)
}

type TurnMetricsObserver struct {
	mu       sync.Mutex
	seen     RecentFrames
	metrics  []TurnMetricsData
	handlers []func(TurnMetricsData)
}

func NewTurnMetricsObserver() *TurnMetricsObserver {
	return &TurnMetricsObserver{}
}

func (o *TurnMetricsObserver) OnTurnMetrics(fn func(TurnMetricsData)) {
	if fn == nil {
		return
	}
	o.mu.Lock()
	o.handlers = append(o.handlers[:len(o.handlers):len(o.handlers)], fn)
	o.mu.Unlock()
}

func (o *TurnMetricsObserver) OnProcessFrame(pipeline.ProcessFrameEvent) {}

func (o *TurnMetricsObserver) Interested(frame frames.Frame, _ frames.FrameDirection, kind pipeline.EventKind) bool {
	if kind != pipeline.PushEvent {
		return false
	}
	_, ok := frame.(*frames.TurnMetricsFrame)
	return ok
}

func (o *TurnMetricsObserver) OnPushFrame(event pipeline.PushFrameEvent) {
	f, ok := event.Frame.(*frames.TurnMetricsFrame)
	if !ok {
		return
	}
	m := TurnMetricsData{
		ProcessorName: f.ProcessorName,
		StartTime:     f.StartTime,
		EndTime:       f.EndTime,
		Duration:      f.Duration,
	}

	o.mu.Lock()
	if !o.seen.FirstSeen(f.ID()) {
		o.mu.Unlock()
		return
	}
	o.metrics = append(o.metrics, m)
	handlers := o.handlers
	o.mu.Unlock()

	for _, h := range handlers {
		h(m)
	}
}

func (o *TurnMetricsObserver) OnPipelineStarted() {}

func (o *TurnMetricsObserver) OnPipelineStopped() {}

func (o *TurnMetricsObserver) Metrics() []TurnMetricsData {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]TurnMetricsData(nil), o.metrics...)
}
