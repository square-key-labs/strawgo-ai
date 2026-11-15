package observers

import (
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

type TransportTimingReport struct {
	StartupDuration time.Duration
	ProcessorName   string
}

type StartupTimingObserver struct {
	mu          sync.Mutex
	startAt     time.Time
	hasReported bool
	handlers    []func(TransportTimingReport)
}

func NewStartupTimingObserver() *StartupTimingObserver {
	return &StartupTimingObserver{}
}

func (o *StartupTimingObserver) OnTransportTimingReport(fn func(TransportTimingReport)) {
	if fn == nil {
		return
	}
	o.mu.Lock()
	o.handlers = append(o.handlers[:len(o.handlers):len(o.handlers)], fn)
	o.mu.Unlock()
}

func (o *StartupTimingObserver) OnProcessFrame(pipeline.ProcessFrameEvent) {}

func (o *StartupTimingObserver) Interested(frame frames.Frame, _ frames.FrameDirection, kind pipeline.EventKind) bool {
	if kind != pipeline.PushEvent {
		return false
	}
	switch frame.(type) {
	case *frames.StartFrame, *frames.ClientConnectedFrame:
		return true
	}
	return false
}

func (o *StartupTimingObserver) OnPushFrame(event pipeline.PushFrameEvent) {
	switch event.Frame.(type) {
	case *frames.StartFrame:
		o.mu.Lock()
		if o.startAt.IsZero() {
			o.startAt = event.Timestamp
			o.hasReported = false
		}
		o.mu.Unlock()
	case *frames.ClientConnectedFrame:
		o.mu.Lock()
		if o.startAt.IsZero() || o.hasReported {
			o.mu.Unlock()
			return
		}
		o.hasReported = true
		report := TransportTimingReport{
			StartupDuration: event.Timestamp.Sub(o.startAt),
			ProcessorName:   event.ProcessorName,
		}
		handlers := o.handlers
		o.mu.Unlock()
		for _, h := range handlers {
			h(report)
		}
	}
}

func (o *StartupTimingObserver) OnPipelineStarted() {
	o.reset()
}

func (o *StartupTimingObserver) OnPipelineStopped() {
	o.reset()
}

func (o *StartupTimingObserver) reset() {
	o.mu.Lock()
	o.startAt = time.Time{}
	o.hasReported = false
	o.mu.Unlock()
}
