package aggregators

import (
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type UserIdleController struct {
	onIdle func()

	mu           sync.Mutex
	timeout      time.Duration
	waiting      bool
	botSpeaking  bool
	userTurn     bool
	pendingCalls int
	timer        *time.Timer
	gen          uint64
	closed       bool
}

func NewUserIdleController(timeout time.Duration, onIdle func()) *UserIdleController {
	return &UserIdleController{timeout: timeout, onIdle: onIdle}
}

func (c *UserIdleController) WaitingForUser() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.waiting
}

func (c *UserIdleController) Timeout() time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.timeout
}

func (c *UserIdleController) ProcessFrame(frame frames.Frame) {
	c.mu.Lock()
	defer c.mu.Unlock()

	switch f := frame.(type) {
	case *frames.UserIdleTimeoutUpdateFrame:
		c.timeout = f.Timeout
		if c.timeout <= 0 {
			c.cancelLocked()
		} else if c.waiting {
			c.startLocked()
		}
	case *frames.BotStoppedSpeakingFrame:
		c.botSpeaking = false
		if !c.userTurn && c.pendingCalls == 0 {
			c.waiting = true
			c.startLocked()
		}
	case *frames.BotStartedSpeakingFrame:
		c.botSpeaking = true
		c.waiting = false
		c.cancelLocked()
	case *frames.UserStartedSpeakingFrame:
		c.waiting = false
		c.userTurn = true
		c.cancelLocked()
	case *frames.UserStoppedSpeakingFrame:
		c.userTurn = false
	case *frames.FunctionCallsStartedFrame:
		c.waiting = false
		c.pendingCalls += len(f.FunctionCalls)
		c.cancelLocked()
	case *frames.FunctionCallResultFrame:
		if f.Properties == nil || f.Properties.IsFinal {
			c.callFinishedLocked()
		}
	case *frames.FunctionCallCancelFrame:
		c.callFinishedLocked()
	}
}

func (c *UserIdleController) UserTurnStarted() {
	c.ProcessFrame(frames.NewUserStartedSpeakingFrame())
}

func (c *UserIdleController) UserTurnStopped() {
	c.ProcessFrame(frames.NewUserStoppedSpeakingFrame())
}

func (c *UserIdleController) WaitForUser() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.botSpeaking || c.userTurn || c.pendingCalls > 0 {
		return
	}
	c.waiting = true
	c.startLocked()
}

func (c *UserIdleController) Stop() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	c.cancelLocked()
}

func (c *UserIdleController) callFinishedLocked() {
	if c.pendingCalls > 0 {
		c.pendingCalls--
	}
}

func (c *UserIdleController) startLocked() {
	c.cancelLocked()
	if c.timeout <= 0 || c.closed || c.onIdle == nil {
		return
	}
	gen := c.gen
	c.timer = time.AfterFunc(c.timeout, func() {
		c.mu.Lock()
		if gen != c.gen || c.closed {
			c.mu.Unlock()
			return
		}
		c.timer = nil
		c.mu.Unlock()
		c.onIdle()
	})
}

func (c *UserIdleController) cancelLocked() {
	c.gen++
	if c.timer != nil {
		c.timer.Stop()
		c.timer = nil
	}
}
