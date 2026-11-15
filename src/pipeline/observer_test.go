package pipeline

import (
	"context"
	"fmt"
	"sync"
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

type benchSink struct {
	queueOnlyProcessor
	count  atomic.Int64
	target atomic.Int64
	done   chan struct{}
}

func (s *benchSink) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	if s.count.Add(1) == s.target.Load() {
		s.done <- struct{}{}
	}
	return nil
}

type metricsOnlyObserver struct {
	seen atomic.Int64
}

func (o *metricsOnlyObserver) OnProcessFrame(event ProcessFrameEvent) {}

func (o *metricsOnlyObserver) OnPushFrame(event PushFrameEvent) {
	if _, ok := event.Frame.(*frames.MetricsFrame); ok {
		o.seen.Add(1)
	}
}

func (o *metricsOnlyObserver) Interested(frame frames.Frame, _ frames.FrameDirection, kind EventKind) bool {
	if kind != PushEvent {
		return false
	}
	_, ok := frame.(*frames.MetricsFrame)
	return ok
}

func (o *metricsOnlyObserver) OnPipelineStarted() {}

func (o *metricsOnlyObserver) OnPipelineStopped() {}

func BenchmarkAudioThroughPipelineWithObservers(b *testing.B) {
	for _, n := range []int{0, 1, 5} {
		b.Run(fmt.Sprintf("observers=%d", n), func(b *testing.B) {
			procs := make([]processors.FrameProcessor, 8)
			for i := range procs {
				procs[i] = processors.NewBaseProcessor(fmt.Sprintf("bench-%d", i), nil)
			}
			pl := NewPipeline(procs)
			sink := &benchSink{done: make(chan struct{}, 1)}
			pl.Link(sink)
			taskObserver := NewTaskObserver()
			for i := 0; i < n; i++ {
				taskObserver.AddObserver(&metricsOnlyObserver{})
			}
			if n > 0 {
				pl.SetObserver(taskObserver)
			}
			ctx, cancel := context.WithCancel(context.Background())
			if err := pl.Start(ctx); err != nil {
				b.Fatal(err)
			}
			frame := frames.NewAudioFrame(make([]byte, 320), 16000, 1)
			sink.target.Store(int64(b.N))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := pl.QueueFrame(frame, frames.Downstream); err != nil {
					b.Fatal(err)
				}
			}
			<-sink.done
			b.StopTimer()
			cancel()
			_ = pl.Stop()
			taskObserver.Close()
		})
	}
}

type gatedObserver struct {
	gate    chan struct{}
	mu      sync.Mutex
	names   []string
	started atomic.Int64
}

func (o *gatedObserver) OnProcessFrame(event ProcessFrameEvent) {}

func (o *gatedObserver) OnPushFrame(event PushFrameEvent) {
	o.started.Add(1)
	<-o.gate
	o.mu.Lock()
	o.names = append(o.names, event.Frame.Name())
	o.mu.Unlock()
}

func (o *gatedObserver) OnPipelineStarted() {}

func (o *gatedObserver) OnPipelineStopped() {}

func (o *gatedObserver) snapshot() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]string(nil), o.names...)
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestObserverFilterSkipsUninterestingEvents(t *testing.T) {
	selective := &metricsOnlyObserver{}
	all := &countingObserver{}
	taskObserver := NewTaskObserver()
	defer taskObserver.Close()
	taskObserver.AddObserver(selective)
	taskObserver.AddObserver(all)

	audio := frames.NewAudioFrame(make([]byte, 320), 16000, 1)
	for i := 0; i < 100; i++ {
		taskObserver.OnProcessFrame("p", audio, frames.Downstream)
		taskObserver.OnPushFrame("p", audio, frames.Downstream)
	}
	metrics := frames.NewMetricsFrame()
	taskObserver.OnProcessFrame("p", metrics, frames.Downstream)
	taskObserver.OnPushFrame("p", metrics, frames.Downstream)

	waitFor(t, "unfiltered observer to receive everything", func() bool {
		return all.processCount.Load() == 101 && all.pushCount.Load() == 101
	})
	waitFor(t, "selective observer to receive the metrics push", func() bool { return selective.seen.Load() == 1 })

	var queued int
	for _, w := range *taskObserver.snapshot.Load() {
		if w.observer == Observer(selective) {
			w.mu.Lock()
			queued = int(w.seq)
			w.mu.Unlock()
		}
	}
	if queued != 1 {
		t.Fatalf("selective observer had %d events enqueued, want 1", queued)
	}
	if n := taskObserver.Dropped(); n != 0 {
		t.Fatalf("dropped %d events, want 0", n)
	}
}

func TestObserverFilteredDispatchDoesNotAllocate(t *testing.T) {
	taskObserver := NewTaskObserver()
	defer taskObserver.Close()
	for i := 0; i < 5; i++ {
		taskObserver.AddObserver(&metricsOnlyObserver{})
	}
	var audio frames.Frame = frames.NewAudioFrame(make([]byte, 320), 16000, 1)
	allocs := testing.AllocsPerRun(1000, func() {
		taskObserver.OnProcessFrame("p", audio, frames.Downstream)
		taskObserver.OnPushFrame("p", audio, frames.Downstream)
	})
	if allocs != 0 {
		t.Fatalf("filtered dispatch allocated %.1f times per run, want 0", allocs)
	}
}

func TestObserverOverflowDropsMediaButKeepsControlInOrder(t *testing.T) {
	obs := &gatedObserver{gate: make(chan struct{})}
	taskObserver := NewTaskObserver()
	defer taskObserver.Close()
	taskObserver.AddObserver(obs)

	audio := frames.NewAudioFrame(make([]byte, 320), 16000, 1)
	taskObserver.OnPushFrame("p", audio, frames.Downstream)
	waitFor(t, "observer to block", func() bool { return obs.started.Load() == 1 })

	const controlEvents = 3000
	const dataEvents = DataObserverQueueSize + 500
	for i := 0; i < controlEvents; i++ {
		for j := 0; j < 10; j++ {
			taskObserver.OnPushFrame("p", audio, frames.Downstream)
		}
		taskObserver.OnPushFrame("p", frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	}
	for i := 0; i < dataEvents; i++ {
		taskObserver.OnPushFrame("p", frames.NewTextFrame("x"), frames.Downstream)
	}
	taskObserver.OnPushFrame("p", frames.NewUserStoppedSpeakingFrame(), frames.Downstream)

	stats := taskObserver.DropStats()
	if stats.Control != 0 {
		t.Fatalf("dropped %d control events, want 0", stats.Control)
	}
	if want := uint64(controlEvents*10 - MediaObserverQueueSize); stats.Media != want {
		t.Fatalf("dropped %d media events, want %d", stats.Media, want)
	}
	if stats.Data != 500 {
		t.Fatalf("dropped %d data events, want 500", stats.Data)
	}
	if taskObserver.Dropped() != stats.Total() {
		t.Fatalf("Dropped() = %d, want %d", taskObserver.Dropped(), stats.Total())
	}

	close(obs.gate)
	total := 1 + MediaObserverQueueSize + controlEvents + DataObserverQueueSize + 1
	waitFor(t, "queued events to drain", func() bool { return len(obs.snapshot()) == total })

	names := obs.snapshot()
	started, firstText := 0, -1
	for i, name := range names {
		switch name {
		case "UserStartedSpeakingFrame":
			started++
			if firstText >= 0 {
				t.Fatalf("control event %d delivered after later data event %d", i, firstText)
			}
		case "TextFrame":
			if firstText < 0 {
				firstText = i
			}
		}
	}
	if started != controlEvents {
		t.Fatalf("delivered %d control events, want %d", started, controlEvents)
	}
	if last := names[len(names)-1]; last != "UserStoppedSpeakingFrame" {
		t.Fatalf("last delivered event is %s, want UserStoppedSpeakingFrame", last)
	}
}

func TestObserverRemoveStopsDeliveryAndCloseIsBounded(t *testing.T) {
	removed := &countingObserver{}
	blocked := &gatedObserver{gate: make(chan struct{})}
	defer close(blocked.gate)
	taskObserver := NewTaskObserver()
	taskObserver.AddObserver(removed)
	taskObserver.AddObserver(blocked)

	text := frames.NewTextFrame("x")
	taskObserver.OnPushFrame("p", text, frames.Downstream)
	waitFor(t, "first delivery", func() bool { return removed.pushCount.Load() == 1 && blocked.started.Load() == 1 })

	taskObserver.RemoveObserver(removed)
	taskObserver.OnPushFrame("p", text, frames.Downstream)
	time.Sleep(20 * time.Millisecond)
	if n := removed.pushCount.Load(); n != 1 {
		t.Fatalf("removed observer received %d push events, want 1", n)
	}

	start := time.Now()
	taskObserver.Close()
	if elapsed := time.Since(start); elapsed > observerCloseTimeout+500*time.Millisecond {
		t.Fatalf("Close took %v with a blocked observer", elapsed)
	}
	taskObserver.OnPushFrame("p", text, frames.Downstream)
	taskObserver.AddObserver(&countingObserver{})
	if taskObserver.snapshot.Load() != nil {
		t.Fatal("observer registered after Close")
	}
}
