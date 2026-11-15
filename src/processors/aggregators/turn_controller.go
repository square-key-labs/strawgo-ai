package aggregators

import (
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

const DefaultUserTurnStopTimeout = 5 * time.Second

type userTurnHooks struct {
	started          func(strategy user_start.UserTurnStartStrategy, enableInterruptions bool)
	stopped          func(timedOut bool)
	resetAggregation func()
	wake             func()
	turnMetrics      func(turn.TurnMetrics)
}

type userTurnController struct {
	strategies  turns.UserTurnStrategies
	stopTimeout time.Duration
	hooks       userTurnHooks
	now         func() time.Time

	userSpeaking        bool
	turnActive          bool
	stopPending         bool
	enableInterruptions bool
	lastActivity        time.Time

	timerMu sync.Mutex
	timer   *time.Timer
	armedAt time.Time
	closed  bool
}

func newUserTurnController(strategies turns.UserTurnStrategies, stopTimeout time.Duration, hooks userTurnHooks) *userTurnController {
	return &userTurnController{
		strategies:  strategies,
		stopTimeout: stopTimeout,
		hooks:       hooks,
		now:         time.Now,
	}
}

func (c *userTurnController) setStrategies(strategies turns.UserTurnStrategies) {
	c.strategies = strategies
}

func (c *userTurnController) active() bool {
	return c.turnActive
}

func (c *userTurnController) process(frame frames.Frame) {
	now := c.now()
	stopAfterSpeech := false

	switch frame.(type) {
	case *frames.UserStartedSpeakingFrame, *frames.VADUserStartedSpeakingFrame:
		c.userSpeaking = true
		c.stopPending = false
		c.lastActivity = now
	case *frames.UserStoppedSpeakingFrame, *frames.VADUserStoppedSpeakingFrame:
		c.userSpeaking = false
		c.lastActivity = now
		stopAfterSpeech = c.stopPending
	case *frames.TranscriptionFrame, *frames.InterimTranscriptionFrame:
		c.lastActivity = now
	}

	var starter user_start.UserTurnStartStrategy
	for _, s := range c.strategies.StartStrategies {
		if s.ShouldStart(frame) && starter == nil && !c.turnActive {
			starter = s
		}
		if r, ok := s.(user_start.AggregationResetter); ok && r.ConsumeAggregationReset() && !c.turnActive && starter == nil {
			c.hooks.resetAggregation()
		}
	}
	if starter != nil {
		c.startTurn(starter, now)
	}

	decided := false
	for _, s := range c.strategies.StopStrategies {
		if s.ShouldStop(frame) {
			decided = true
		}
		if r, ok := s.(user_stop.TurnMetricsReporter); ok && c.hooks.turnMetrics != nil {
			if m, ok := r.TakeTurnMetrics(); ok {
				c.hooks.turnMetrics(m)
			}
		}
	}

	switch {
	case decided && c.turnActive:
		if c.userSpeaking {
			c.stopPending = true
		} else {
			c.endTurn(false)
		}
	case stopAfterSpeech:
		c.endTurn(false)
	}

	if c.turnActive && !c.userSpeaking && c.stopTimeout > 0 && !c.strategyScheduled() && !now.Before(c.lastActivity.Add(c.stopTimeout)) {
		c.endTurn(true)
	}

	c.reschedule(now)
}

func (c *userTurnController) trackSpeaking(frame frames.Frame) {
	now := c.now()
	switch frame.(type) {
	case *frames.UserStartedSpeakingFrame, *frames.VADUserStartedSpeakingFrame:
		c.userSpeaking = true
	case *frames.UserStoppedSpeakingFrame, *frames.VADUserStoppedSpeakingFrame:
		c.userSpeaking = false
		if c.stopPending {
			c.endTurn(false)
		}
	default:
		return
	}
	c.lastActivity = now
	c.reschedule(now)
}

func (c *userTurnController) startTurn(s user_start.UserTurnStartStrategy, now time.Time) {
	c.turnActive = true
	c.stopPending = false
	c.enableInterruptions = s.EnableInterruptions()
	c.lastActivity = now
	for _, st := range c.strategies.StartStrategies {
		st.Reset()
	}
	for _, st := range c.strategies.StopStrategies {
		if lc, ok := st.(user_stop.TurnLifecycle); ok {
			lc.HandleUserTurnStarted()
		}
	}
	c.hooks.started(s, c.enableInterruptions)
}

func (c *userTurnController) endTurn(timedOut bool) {
	if !c.turnActive {
		return
	}
	c.turnActive = false
	c.stopPending = false
	for _, st := range c.strategies.StopStrategies {
		if lc, ok := st.(user_stop.TurnLifecycle); ok {
			lc.HandleUserTurnStopped()
		} else {
			st.Reset()
		}
	}
	c.hooks.stopped(timedOut)
}

func (c *userTurnController) strategyScheduled() bool {
	for _, s := range c.strategies.StopStrategies {
		if sch, ok := s.(user_stop.Scheduler); ok {
			if _, pending := sch.NextDeadline(); pending {
				return true
			}
		}
	}
	return false
}

func (c *userTurnController) nextWake() time.Time {
	var next time.Time
	for _, s := range c.strategies.StopStrategies {
		if sch, ok := s.(user_stop.Scheduler); ok {
			if t, pending := sch.NextDeadline(); pending && (next.IsZero() || t.Before(next)) {
				next = t
			}
		}
	}
	if c.turnActive && !c.userSpeaking && c.stopTimeout > 0 {
		if t := c.lastActivity.Add(c.stopTimeout); next.IsZero() || t.Before(next) {
			next = t
		}
	}
	return next
}

func (c *userTurnController) reschedule(now time.Time) {
	next := c.nextWake()

	c.timerMu.Lock()
	defer c.timerMu.Unlock()
	if c.closed {
		return
	}
	if !c.armedAt.IsZero() && !now.Before(c.armedAt) {
		c.armedAt = time.Time{}
	}
	if next.Equal(c.armedAt) {
		return
	}
	if next.IsZero() {
		if c.timer != nil {
			c.timer.Stop()
		}
		c.armedAt = time.Time{}
		return
	}
	d := next.Sub(now)
	if d < 0 {
		d = 0
	}
	if c.timer == nil {
		c.timer = time.AfterFunc(d, c.hooks.wake)
	} else {
		c.timer.Reset(d)
	}
	c.armedAt = next
}

func (c *userTurnController) close() {
	c.timerMu.Lock()
	defer c.timerMu.Unlock()
	c.closed = true
	if c.timer != nil {
		c.timer.Stop()
	}
}
