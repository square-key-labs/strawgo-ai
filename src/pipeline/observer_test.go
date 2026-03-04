package pipeline

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type queueOnlyProcessor struct{}

func (p *queueOnlyProcessor) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}

func (p *queueOnlyProcessor) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}

func (p *queueOnlyProcessor) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}

func (p *queueOnlyProcessor) Link(next processors.FrameProcessor) {}

func (p *queueOnlyProcessor) SetPrev(prev processors.FrameProcessor) {}

func (p *queueOnlyProcessor) Start(ctx context.Context) error { return nil }

func (p *queueOnlyProcessor) Stop() error { return nil }

func (p *queueOnlyProcessor) Name() string { return "queue-only" }

type slowObserver struct{}

func (o *slowObserver) OnProcessFrame(event ProcessFrameEvent) { time.Sleep(100 * time.Millisecond) }

func (o *slowObserver) OnPushFrame(event PushFrameEvent) { time.Sleep(100 * time.Millisecond) }

func (o *slowObserver) OnPipelineStarted() {}

func (o *slowObserver) OnPipelineStopped() {}

type panicObserver struct{}

func (o *panicObserver) OnProcessFrame(event ProcessFrameEvent) { panic("process panic") }

func (o *panicObserver) OnPushFrame(event PushFrameEvent) { panic("push panic") }

func (o *panicObserver) OnPipelineStarted() { panic("started panic") }

func (o *panicObserver) OnPipelineStopped() { panic("stopped panic") }

type countingObserver struct {
	processCount atomic.Int64
	pushCount    atomic.Int64
}

func (o *countingObserver) OnProcessFrame(event ProcessFrameEvent) { o.processCount.Add(1) }

func (o *countingObserver) OnPushFrame(event PushFrameEvent) { o.pushCount.Add(1) }

func (o *countingObserver) OnPipelineStarted() {}

func (o *countingObserver) OnPipelineStopped() {}

func TestObserverNonBlocking(t *testing.T) {
	taskObserver := NewTaskObserver()
	taskObserver.AddObserver(&slowObserver{})

	base := processors.NewBaseProcessor("observer-non-blocking", nil)
	base.SetObserver(taskObserver)
	base.Link(&queueOnlyProcessor{})

	start := time.Now()
	for i := 0; i < 10; i++ {
		if err := base.ProcessFrame(context.Background(), frames.NewTextFrame("payload"), frames.Downstream); err != nil {
			t.Fatalf("process frame %d: %v", i, err)
		}
	}

	elapsed := time.Since(start)
	if elapsed >= 50*time.Millisecond {
		t.Fatalf("expected non-blocking observer processing <50ms, got %v", elapsed)
	}
}

func TestObserverPanicRecovery(t *testing.T) {
	taskObserver := NewTaskObserver()
	taskObserver.AddObserver(&panicObserver{})

	base := processors.NewBaseProcessor("observer-panic-recovery", nil)
	base.SetObserver(taskObserver)
	base.Link(&queueOnlyProcessor{})

	for i := 0; i < 10; i++ {
		if err := base.ProcessFrame(context.Background(), frames.NewTextFrame("payload"), frames.Downstream); err != nil {
			t.Fatalf("process frame %d: %v", i, err)
		}
	}
}

func TestObserverReceivesEvents(t *testing.T) {
	obs := &countingObserver{}
	taskObserver := NewTaskObserver()
	taskObserver.AddObserver(obs)

	base := processors.NewBaseProcessor("observer-receives-events", nil)
	base.SetObserver(taskObserver)
	base.Link(&queueOnlyProcessor{})

	if err := base.ProcessFrame(context.Background(), frames.NewTextFrame("payload"), frames.Downstream); err != nil {
		t.Fatalf("process frame: %v", err)
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if obs.processCount.Load() > 0 && obs.pushCount.Load() > 0 {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}

	t.Fatalf("expected process and push events, got process=%d push=%d", obs.processCount.Load(), obs.pushCount.Load())
}
