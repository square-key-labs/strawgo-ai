package pipeline

import (
	"sync"
	"time"
)

type FrameProcessorMetrics struct {
	mu sync.Mutex

	ttfbStartTime       time.Time
	ttfbDuration        time.Duration
	ttfbMetricsRunning  bool
	processingStartTime time.Time
	processingDuration  time.Duration
	processingRunning   bool
}

func (m *FrameProcessorMetrics) StartTTFBMetrics() time.Duration {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.ttfbStartTime = time.Now()
	m.ttfbMetricsRunning = true

	return 0
}

func (m *FrameProcessorMetrics) StopTTFBMetrics() time.Duration {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.ttfbMetricsRunning {
		return m.ttfbDuration
	}

	m.ttfbDuration = time.Since(m.ttfbStartTime)
	m.ttfbMetricsRunning = false

	return m.ttfbDuration
}

func (m *FrameProcessorMetrics) StartProcessingMetrics() time.Duration {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.processingStartTime = time.Now()
	m.processingRunning = true

	return 0
}

func (m *FrameProcessorMetrics) StopProcessingMetrics() time.Duration {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.processingRunning {
		return m.processingDuration
	}

	m.processingDuration = time.Since(m.processingStartTime)
	m.processingRunning = false

	return m.processingDuration
}
