package processors

import (
	"context"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// UserIdleController detects when the user goes silent for too long after the
// bot stops speaking. When the configured idle timeout elapses without the user
// starting to speak, a UserIdleTimeoutFrame is pushed downstream.
//
// The idle timer:
//   - Starts after receiving BotStoppedSpeakingFrame
//   - Resets/cancels on UserStartedSpeakingFrame (user started speaking)
//   - Resets/cancels on BotStartedSpeakingFrame (bot started speaking again)
//   - Resets/cancels on FunctionCallInProgressFrame (function call in progress)
//   - Can be reconfigured at runtime via UserIdleTimeoutUpdateFrame
//
// A timeout of 0 (default) disables idle detection entirely.
type UserIdleController struct {
	*BaseProcessor

	mu      sync.Mutex
	timeout time.Duration
	timer   *time.Timer
}

// NewUserIdleController creates a new UserIdleController with the given idle timeout.
// A timeout of 0 disables idle detection.
func NewUserIdleController(timeout time.Duration) *UserIdleController {
	c := &UserIdleController{
		timeout: timeout,
	}
	c.BaseProcessor = NewBaseProcessor("UserIdleController", c)
	return c
}

// HandleFrame processes frames to manage the idle timer state.
func (c *UserIdleController) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch frame.(type) {
	case *frames.BotStoppedSpeakingFrame:
		c.startIdleTimer()

	case *frames.UserStartedSpeakingFrame:
		c.cancelIdleTimer()

	case *frames.BotStartedSpeakingFrame:
		c.cancelIdleTimer()

	case *frames.FunctionCallInProgressFrame:
		c.cancelIdleTimer()

	case *frames.UserIdleTimeoutUpdateFrame:
		updateFrame := frame.(*frames.UserIdleTimeoutUpdateFrame)
		c.mu.Lock()
		c.timeout = updateFrame.Timeout
		c.mu.Unlock()
		logger.Debug("[%s] Idle timeout updated to %v", c.Name(), updateFrame.Timeout)
	}

	// All frames pass through unchanged
	return c.PushFrame(frame, direction)
}

// startIdleTimer starts (or restarts) the idle detection timer.
// If timeout is 0, this is a no-op.
func (c *UserIdleController) startIdleTimer() {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Cancel any existing timer first
	if c.timer != nil {
		c.timer.Stop()
		c.timer = nil
	}

	if c.timeout <= 0 {
		return
	}

	timeout := c.timeout
	c.timer = time.AfterFunc(timeout, func() {
		c.mu.Lock()
		c.timer = nil
		c.mu.Unlock()

		logger.Info("[%s] User idle timeout fired after %v", c.Name(), timeout)
		if err := c.PushFrame(frames.NewUserIdleTimeoutFrame(), frames.Downstream); err != nil {
			logger.Error("[%s] Failed to push UserIdleTimeoutFrame: %v", c.Name(), err)
		}
	})

	logger.Debug("[%s] Idle timer started (%v)", c.Name(), timeout)
}

// cancelIdleTimer stops the idle timer if it's running.
func (c *UserIdleController) cancelIdleTimer() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.timer != nil {
		c.timer.Stop()
		c.timer = nil
		logger.Debug("[%s] Idle timer cancelled", c.Name())
	}
}
