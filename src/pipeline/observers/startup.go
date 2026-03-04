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
	mu sync.Mutex

	OnTransportTimingReport func(report TransportTimingReport)

	startAt     time.Time
	hasStartAt  bool
	hasReported bool
}

func NewStartupTimingObserver() *StartupTimingObserver {
	return &StartupTimingObserver{}
}

func (o *StartupTimingObserver) OnProcessFrame(event pipeline.ProcessFrameEvent) {
	o.handleFrame(event.Frame, event.ProcessorName, event.Timestamp)
}

func (o *StartupTimingObserver) OnPushFrame(event pipeline.PushFrameEvent) {
	o.handleFrame(event.Frame, event.ProcessorName, event.Timestamp)
}

func (o *StartupTimingObserver) OnPipelineStarted() {
	o.reset()
}

func (o *StartupTimingObserver) OnPipelineStopped() {
	o.reset()
}

func (o *StartupTimingObserver) handleFrame(frame frames.Frame, processorName string, now time.Time) {
	o.mu.Lock()

	switch frame.(type) {
	case *frames.StartFrame:
		o.startAt = now
		o.hasStartAt = true
		o.hasReported = false
		o.mu.Unlock()
		return
	case *frames.ClientConnectedFrame:
		if o.hasStartAt && !o.hasReported {
			report := TransportTimingReport{
				StartupDuration: now.Sub(o.startAt),
				ProcessorName:   processorName,
			}
			cb := o.OnTransportTimingReport
			o.hasReported = true
			o.mu.Unlock()

			if cb != nil {
				go cb(report)
			}
			return
		}
	}

	o.mu.Unlock()
}

func (o *StartupTimingObserver) reset() {
	o.mu.Lock()
	defer o.mu.Unlock()

	o.startAt = time.Time{}
	o.hasStartAt = false
	o.hasReported = false
}
