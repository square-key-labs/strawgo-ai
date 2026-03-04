package observers

import (
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

type TurnMetricsData struct {
	mu            sync.Mutex
	ProcessorName string
	StartTime     time.Time
	EndTime       time.Time
	Duration      time.Duration
}

func NewTurnMetricsData(processorName string) *TurnMetricsData {
	return &TurnMetricsData{ProcessorName: processorName}
}

func (m *TurnMetricsData) Start() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.StartTime = time.Now()
	m.EndTime = time.Time{}
	m.Duration = 0
}

func (m *TurnMetricsData) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.EndTime = time.Now()
	if !m.StartTime.IsZero() {
		m.Duration = m.EndTime.Sub(m.StartTime)
		return
	}
	m.Duration = 0
}

func (m *TurnMetricsData) ToFrame() *frames.TurnMetricsFrame {
	m.mu.Lock()
	defer m.mu.Unlock()

	return frames.NewTurnMetricsFrame(m.ProcessorName, m.StartTime, m.EndTime, m.Duration)
}

type TurnMetricsObserver struct {
	mu sync.Mutex

	OnTurnMetrics func(metrics TurnMetricsData)

	metrics []TurnMetricsData
}

func NewTurnMetricsObserver() *TurnMetricsObserver {
	return &TurnMetricsObserver{metrics: make([]TurnMetricsData, 0)}
}

func (o *TurnMetricsObserver) OnProcessFrame(event pipeline.ProcessFrameEvent) {
	o.handleFrame(event.Frame)
}

func (o *TurnMetricsObserver) OnPushFrame(event pipeline.PushFrameEvent) {
	o.handleFrame(event.Frame)
}

func (o *TurnMetricsObserver) OnPipelineStarted() {}

func (o *TurnMetricsObserver) OnPipelineStopped() {}

func (o *TurnMetricsObserver) Metrics() []TurnMetricsData {
	o.mu.Lock()
	defer o.mu.Unlock()

	copyMetrics := make([]TurnMetricsData, len(o.metrics))
	copy(copyMetrics, o.metrics)
	return copyMetrics
}

func (o *TurnMetricsObserver) handleFrame(frame frames.Frame) {
	metricsFrame, ok := frame.(*frames.TurnMetricsFrame)
	if !ok {
		return
	}

	metrics := TurnMetricsData{
		ProcessorName: metricsFrame.ProcessorName,
		StartTime:     metricsFrame.StartTime,
		EndTime:       metricsFrame.EndTime,
		Duration:      metricsFrame.Duration,
	}

	o.mu.Lock()
	o.metrics = append(o.metrics, metrics)
	cb := o.OnTurnMetrics
	o.mu.Unlock()

	if cb != nil {
		go cb(metrics)
	}
}
