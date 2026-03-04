package processors

import (
	"context"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type countingProcessor struct {
	mu    sync.Mutex
	count int
}

func (p *countingProcessor) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}

func (p *countingProcessor) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	p.mu.Lock()
	p.count++
	p.mu.Unlock()
	return nil
}

func (p *countingProcessor) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}

func (p *countingProcessor) Link(next FrameProcessor) {}

func (p *countingProcessor) SetPrev(prev FrameProcessor) {}

func (p *countingProcessor) Start(ctx context.Context) error { return nil }

func (p *countingProcessor) Stop() error { return nil }

func (p *countingProcessor) Name() string { return "counting" }

func TestBaseProcessorConcurrentProcessFrame(t *testing.T) {
	p := NewBaseProcessor("concurrent-process", nil)

	var wg sync.WaitGroup
	workers := 5
	framesPerWorker := 100

	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < framesPerWorker; i++ {
				frame := frames.NewTextFrame("process")
				if err := p.ProcessFrame(context.Background(), frame, frames.Downstream); err != nil {
					t.Errorf("worker %d process frame error: %v", id, err)
					return
				}
			}
		}(worker)
	}

	wg.Wait()
}

func TestBaseProcessorConcurrentQueueFrame(t *testing.T) {
	p := NewBaseProcessor("concurrent-queue", nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := p.Start(ctx); err != nil {
		t.Fatalf("start processor: %v", err)
	}
	defer func() {
		if err := p.Stop(); err != nil {
			t.Fatalf("stop processor: %v", err)
		}
	}()

	var wg sync.WaitGroup
	workers := 5
	framesPerWorker := 100

	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < framesPerWorker; i++ {
				var frame frames.Frame
				if i%2 == 0 {
					frame = frames.NewStartFrame()
				} else {
					frame = frames.NewTextFrame("queue")
				}

				if err := p.QueueFrame(frame, frames.Downstream); err != nil {
					t.Errorf("worker %d queue frame error: %v", id, err)
					return
				}
			}
		}(worker)
	}

	wg.Wait()

	time.Sleep(20 * time.Millisecond)
}

func TestBaseProcessorConcurrentPushFrame(t *testing.T) {
	p := NewBaseProcessor("concurrent-push", nil)
	target := &countingProcessor{}
	p.Link(target)

	var wg sync.WaitGroup
	workers := 5
	framesPerWorker := 100

	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < framesPerWorker; i++ {
				if err := p.PushFrame(frames.NewTextFrame("push"), frames.Downstream); err != nil {
					t.Errorf("worker %d push frame error: %v", id, err)
					return
				}
			}
		}(worker)
	}

	wg.Wait()
}

type interruptionCaptureProcessor struct {
	*BaseProcessor
	mu     sync.Mutex
	frames []frames.Frame
}

func newInterruptionCaptureProcessor(name string) *interruptionCaptureProcessor {
	p := &interruptionCaptureProcessor{}
	p.BaseProcessor = NewBaseProcessor(name, p)
	return p
}

func (p *interruptionCaptureProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	p.mu.Lock()
	p.frames = append(p.frames, frame)
	p.mu.Unlock()
	return nil
}

func (p *interruptionCaptureProcessor) latestFrame() frames.Frame {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.frames) == 0 {
		return nil
	}
	return p.frames[len(p.frames)-1]
}

func waitForCapturedFrame(t *testing.T, p *interruptionCaptureProcessor) frames.Frame {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		frame := p.latestFrame()
		if frame != nil {
			return frame
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("timed out waiting for captured interruption frame")
	return nil
}

func TestBroadcastInterruption(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	source := newInterruptionCaptureProcessor("source-capture")
	middle := NewBaseProcessor("middle-broadcast", nil)
	sink := newInterruptionCaptureProcessor("sink-capture")

	source.Link(middle)
	middle.Link(sink)

	if err := source.Start(ctx); err != nil {
		t.Fatalf("start source: %v", err)
	}
	defer func() {
		if err := source.Stop(); err != nil {
			t.Fatalf("stop source: %v", err)
		}
	}()

	if err := middle.Start(ctx); err != nil {
		t.Fatalf("start middle: %v", err)
	}
	defer func() {
		if err := middle.Stop(); err != nil {
			t.Fatalf("stop middle: %v", err)
		}
	}()

	if err := sink.Start(ctx); err != nil {
		t.Fatalf("start sink: %v", err)
	}
	defer func() {
		if err := sink.Stop(); err != nil {
			t.Fatalf("stop sink: %v", err)
		}
	}()

	if err := middle.BroadcastInterruption(ctx); err != nil {
		t.Fatalf("broadcast interruption: %v", err)
	}

	upstreamFrame := waitForCapturedFrame(t, source)
	downstreamFrame := waitForCapturedFrame(t, sink)

	if _, ok := upstreamFrame.(*frames.InterruptionFrame); !ok {
		t.Fatalf("expected upstream frame to be InterruptionFrame, got %T", upstreamFrame)
	}
	if _, ok := downstreamFrame.(*frames.InterruptionFrame); !ok {
		t.Fatalf("expected downstream frame to be InterruptionFrame, got %T", downstreamFrame)
	}

	if upstreamFrame.GetBroadcastSiblingID() != strconv.FormatUint(downstreamFrame.ID(), 10) {
		t.Fatalf("upstream sibling mismatch: got %s expected %d", upstreamFrame.GetBroadcastSiblingID(), downstreamFrame.ID())
	}
	if downstreamFrame.GetBroadcastSiblingID() != strconv.FormatUint(upstreamFrame.ID(), 10) {
		t.Fatalf("downstream sibling mismatch: got %s expected %d", downstreamFrame.GetBroadcastSiblingID(), upstreamFrame.ID())
	}
}
