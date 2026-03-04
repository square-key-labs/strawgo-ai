package processors

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// frameCaptureProcessor captures all frames pushed to it for test assertions.
type frameCaptureProcessor struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (p *frameCaptureProcessor) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}

func (p *frameCaptureProcessor) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	p.mu.Lock()
	p.frames = append(p.frames, frame)
	p.mu.Unlock()
	return nil
}

func (p *frameCaptureProcessor) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}

func (p *frameCaptureProcessor) Link(next FrameProcessor)        {}
func (p *frameCaptureProcessor) SetPrev(prev FrameProcessor)     {}
func (p *frameCaptureProcessor) Start(ctx context.Context) error { return nil }
func (p *frameCaptureProcessor) Stop() error                     { return nil }
func (p *frameCaptureProcessor) Name() string                    { return "frame-capture" }

func (p *frameCaptureProcessor) capturedFrames() []frames.Frame {
	p.mu.Lock()
	defer p.mu.Unlock()
	result := make([]frames.Frame, len(p.frames))
	copy(result, p.frames)
	return result
}

func (p *frameCaptureProcessor) hasFrameOfType(name string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, f := range p.frames {
		if f.Name() == name {
			return true
		}
	}
	return false
}

// waitForFrame polls until a frame with the given name appears or timeout.
func (p *frameCaptureProcessor) waitForFrame(t *testing.T, name string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if p.hasFrameOfType(name) {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s frame", name)
}

// startIdleController is a helper that creates, links, starts, and returns
// the controller + capture processor for testing.
func startIdleController(t *testing.T, timeout time.Duration) (*UserIdleController, *frameCaptureProcessor, context.CancelFunc) {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())

	controller := NewUserIdleController(timeout)
	capture := &frameCaptureProcessor{}
	controller.Link(capture)

	if err := controller.Start(ctx); err != nil {
		cancel()
		t.Fatalf("start controller: %v", err)
	}

	t.Cleanup(func() {
		cancel()
		controller.Stop()
	})

	return controller, capture, cancel
}

func TestUserIdleController_TimerFires(t *testing.T) {
	idleTimeout := 30 * time.Millisecond
	controller, capture, _ := startIdleController(t, idleTimeout)

	// BotStoppedSpeaking should start the idle timer
	if err := controller.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Downstream); err != nil {
		t.Fatalf("queue BotStoppedSpeakingFrame: %v", err)
	}

	// Wait for the idle timeout frame to appear (with margin)
	capture.waitForFrame(t, "UserIdleTimeoutFrame", 2*time.Second)

	// Verify it was pushed
	found := false
	for _, f := range capture.capturedFrames() {
		if _, ok := f.(*frames.UserIdleTimeoutFrame); ok {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected UserIdleTimeoutFrame to be pushed downstream")
	}
}

func TestUserIdleController_TimerSuppressed(t *testing.T) {
	idleTimeout := 40 * time.Millisecond
	controller, capture, _ := startIdleController(t, idleTimeout)

	// BotStoppedSpeaking starts the idle timer
	if err := controller.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Downstream); err != nil {
		t.Fatalf("queue BotStoppedSpeakingFrame: %v", err)
	}

	// Quickly send UserStartedSpeaking to cancel the timer
	time.Sleep(10 * time.Millisecond)
	if err := controller.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream); err != nil {
		t.Fatalf("queue UserStartedSpeakingFrame: %v", err)
	}

	// Wait well past the original timeout
	time.Sleep(idleTimeout * 2)

	// Verify NO UserIdleTimeoutFrame was pushed
	if capture.hasFrameOfType("UserIdleTimeoutFrame") {
		t.Fatal("UserIdleTimeoutFrame should NOT have been pushed when user started speaking")
	}
}

func TestUserIdleController_Disabled(t *testing.T) {
	// timeout=0 means disabled
	controller, capture, _ := startIdleController(t, 0)

	// BotStoppedSpeaking with disabled timeout should NOT trigger anything
	if err := controller.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Downstream); err != nil {
		t.Fatalf("queue BotStoppedSpeakingFrame: %v", err)
	}

	// Wait a generous amount of time
	time.Sleep(100 * time.Millisecond)

	// Verify NO UserIdleTimeoutFrame was pushed
	if capture.hasFrameOfType("UserIdleTimeoutFrame") {
		t.Fatal("UserIdleTimeoutFrame should NOT have been pushed when idle detection is disabled")
	}
}

func TestUserIdleController_RuntimeUpdate(t *testing.T) {
	// Start disabled
	controller, capture, _ := startIdleController(t, 0)

	// Enable at runtime with a short timeout
	newTimeout := 30 * time.Millisecond
	updateFrame := frames.NewUserIdleTimeoutUpdateFrame(newTimeout)
	if err := controller.QueueFrame(updateFrame, frames.Downstream); err != nil {
		t.Fatalf("queue UserIdleTimeoutUpdateFrame: %v", err)
	}

	// Allow the update frame to be processed
	time.Sleep(20 * time.Millisecond)

	// Now BotStoppedSpeaking should trigger idle detection
	if err := controller.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Downstream); err != nil {
		t.Fatalf("queue BotStoppedSpeakingFrame: %v", err)
	}

	// Wait for the timeout frame
	capture.waitForFrame(t, "UserIdleTimeoutFrame", 2*time.Second)

	found := false
	for _, f := range capture.capturedFrames() {
		if _, ok := f.(*frames.UserIdleTimeoutFrame); ok {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected UserIdleTimeoutFrame after runtime update enabled idle detection")
	}
}

func TestUserIdleController_BotStartedSpeakingCancels(t *testing.T) {
	idleTimeout := 40 * time.Millisecond
	controller, capture, _ := startIdleController(t, idleTimeout)

	// BotStoppedSpeaking starts the idle timer
	if err := controller.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Downstream); err != nil {
		t.Fatalf("queue BotStoppedSpeakingFrame: %v", err)
	}

	// Quickly send BotStartedSpeaking (bot started speaking again) to cancel
	time.Sleep(10 * time.Millisecond)
	if err := controller.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Downstream); err != nil {
		t.Fatalf("queue BotStartedSpeakingFrame: %v", err)
	}

	// Wait well past the original timeout
	time.Sleep(idleTimeout * 2)

	if capture.hasFrameOfType("UserIdleTimeoutFrame") {
		t.Fatal("UserIdleTimeoutFrame should NOT have been pushed when bot started speaking again")
	}
}

func TestUserIdleController_FunctionCallCancels(t *testing.T) {
	idleTimeout := 40 * time.Millisecond
	controller, capture, _ := startIdleController(t, idleTimeout)

	// BotStoppedSpeaking starts the idle timer
	if err := controller.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Downstream); err != nil {
		t.Fatalf("queue BotStoppedSpeakingFrame: %v", err)
	}

	// Quickly send FunctionCallInProgress to cancel
	time.Sleep(10 * time.Millisecond)
	fcFrame := frames.NewFunctionCallInProgressFrame("call-1", "lookup", nil, false)
	if err := controller.QueueFrame(fcFrame, frames.Downstream); err != nil {
		t.Fatalf("queue FunctionCallInProgressFrame: %v", err)
	}

	// Wait well past the original timeout
	time.Sleep(idleTimeout * 2)

	if capture.hasFrameOfType("UserIdleTimeoutFrame") {
		t.Fatal("UserIdleTimeoutFrame should NOT have been pushed when function call is in progress")
	}
}
