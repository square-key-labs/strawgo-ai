package pipeline

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

func newConcurrentTestTask() *PipelineTask {
	pipe := NewPipeline([]processors.FrameProcessor{
		processors.NewPassthroughProcessor("test-pass", false),
	})
	return NewPipelineTask(pipe)
}

func queueWhenReady(task *PipelineTask, frame frames.Frame, direction ...frames.FrameDirection) error {
	deadline := time.Now().Add(2 * time.Second)
	for {
		var err error
		if len(direction) > 0 {
			err = task.QueueFrame(frame, direction[0])
		} else {
			err = task.QueueFrame(frame)
		}
		if err == nil {
			return nil
		}
		if !strings.Contains(err.Error(), "not started") {
			return err
		}
		if time.Now().After(deadline) {
			return err
		}
		time.Sleep(2 * time.Millisecond)
	}
}

func waitRunResult(t *testing.T, done <-chan error) error {
	t.Helper()
	select {
	case err := <-done:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("pipeline task run timed out")
		return nil
	}
}

func TestPipelineTaskConcurrentQueueFrame(t *testing.T) {
	task := newConcurrentTestTask()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	runDone := make(chan error, 1)
	go func() {
		runDone <- task.Run(ctx)
	}()

	var wg sync.WaitGroup
	workers := 5
	framesPerWorker := 100

	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < framesPerWorker; i++ {
				frame := frames.NewTextFrame("worker-frame")
				if err := queueWhenReady(task, frame); err != nil {
					t.Errorf("worker %d queue frame error: %v", id, err)
					return
				}
			}
		}(worker)
	}

	wg.Wait()

	if err := queueWhenReady(task, frames.NewEndFrame()); err != nil {
		t.Fatalf("queue end frame: %v", err)
	}

	if err := waitRunResult(t, runDone); err != nil {
		t.Fatalf("run returned error: %v", err)
	}
}

func TestPipelineTaskConcurrentCancel(t *testing.T) {
	task := newConcurrentTestTask()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	runDone := make(chan error, 1)
	go func() {
		runDone <- task.Run(ctx)
	}()

	if err := queueWhenReady(task, frames.NewTextFrame("warmup")); err != nil {
		t.Fatalf("queue warmup frame: %v", err)
	}

	var wg sync.WaitGroup
	workers := 10
	cancelsPerWorker := 100

	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < cancelsPerWorker; i++ {
				task.Cancel()
			}
		}()
	}

	wg.Wait()

	if err := waitRunResult(t, runDone); err != nil {
		t.Fatalf("run returned error: %v", err)
	}
}

func TestQueueFrame_Direction(t *testing.T) {
	// Create a test processor that tracks frames and their directions
	tracker := newDirectionTrackingProcessor("tracker")

	pipe := NewPipeline([]processors.FrameProcessor{tracker})
	task := NewPipelineTask(pipe)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	runDone := make(chan error, 1)
	go func() {
		runDone <- task.Run(ctx)
	}()

	// Queue a downstream frame (default)
	downstreamFrame := frames.NewTextFrame("downstream-test")
	if err := queueWhenReady(task, downstreamFrame); err != nil {
		t.Fatalf("queue downstream frame: %v", err)
	}

	// Queue an upstream frame
	upstreamFrame := frames.NewTextFrame("upstream-test")
	if err := queueWhenReady(task, upstreamFrame, frames.Upstream); err != nil {
		t.Fatalf("queue upstream frame: %v", err)
	}

	// Give frames time to process
	time.Sleep(100 * time.Millisecond)

	// Queue end frame
	if err := queueWhenReady(task, frames.NewEndFrame()); err != nil {
		t.Fatalf("queue end frame: %v", err)
	}

	if err := waitRunResult(t, runDone); err != nil {
		t.Fatalf("run returned error: %v", err)
	}

	// Verify frames were tracked with correct directions
	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	if len(tracker.frames) < 3 {
		t.Fatalf("expected at least 3 tracked frames, got %d", len(tracker.frames))
	}

	// Check that we have frames with both directions
	hasDownstream := false
	hasUpstream := false
	for _, tf := range tracker.frames {
		if tf.direction == frames.Downstream {
			hasDownstream = true
		}
		if tf.direction == frames.Upstream {
			hasUpstream = true
		}
	}

	if !hasDownstream {
		t.Error("no downstream frames tracked")
	}
	if !hasUpstream {
		t.Error("no upstream frames tracked")
	}
}

func TestLegacyInterruptionTaskFrame(t *testing.T) {
	tracker := newDirectionTrackingProcessor("legacy-interruption-tracker")
	pipe := NewPipeline([]processors.FrameProcessor{tracker})
	task := NewPipelineTask(pipe)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	runDone := make(chan error, 1)
	go func() {
		runDone <- task.Run(ctx)
	}()

	if err := queueWhenReady(task, frames.NewInterruptionTaskFrame(), frames.Upstream); err != nil {
		t.Fatalf("queue interruption task frame upstream: %v", err)
	}

	deadline := time.Now().Add(2 * time.Second)
	found := false
	for time.Now().Before(deadline) {
		tracker.mu.Lock()
		for _, tf := range tracker.frames {
			if _, ok := tf.frame.(*frames.InterruptionFrame); ok && tf.direction == frames.Downstream {
				found = true
				break
			}
		}
		tracker.mu.Unlock()
		if found {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	if !found {
		t.Fatal("expected deprecated InterruptionTaskFrame path to emit downstream InterruptionFrame")
	}

	if err := queueWhenReady(task, frames.NewEndFrame()); err != nil {
		t.Fatalf("queue end frame: %v", err)
	}

	if err := waitRunResult(t, runDone); err != nil {
		t.Fatalf("run returned error: %v", err)
	}
}

// directionTrackingProcessor tracks frames and their directions
type directionTrackingProcessor struct {
	*processors.BaseProcessor
	name   string
	frames []trackedFrame
	mu     sync.Mutex
}

type trackedFrame struct {
	frame     frames.Frame
	direction frames.FrameDirection
}

func newDirectionTrackingProcessor(name string) *directionTrackingProcessor {
	p := &directionTrackingProcessor{
		name:   name,
		frames: make([]trackedFrame, 0),
	}
	p.BaseProcessor = processors.NewBaseProcessor(name, p)
	return p
}

func (p *directionTrackingProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	p.mu.Lock()
	p.frames = append(p.frames, trackedFrame{frame: frame, direction: direction})
	p.mu.Unlock()
	return p.PushFrame(frame, direction)
}
