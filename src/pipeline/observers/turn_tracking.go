package observers

import (
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

const DefaultTurnEndTimeout = 2500 * time.Millisecond

type TurnTrackingOption func(*TurnTrackingObserver)

func WithTurnEndTimeout(d time.Duration) TurnTrackingOption {
	return func(o *TurnTrackingObserver) {
		if d > 0 {
			o.timeout = d
		}
	}
}

type turnEvent struct {
	started     bool
	turn        int
	duration    time.Duration
	interrupted bool
}

type TurnTrackingObserver struct {
	timeout time.Duration

	emitMu sync.Mutex
	mu     sync.Mutex

	turnCount     int
	active        bool
	botSpeaking   bool
	botSpoken     bool
	turnStartedAt time.Time
	timer         *time.Timer
	timerGen      uint64

	startedHandlers []func(turn int)
	endedHandlers   []func(turn int, duration time.Duration, interrupted bool)
}

func NewTurnTrackingObserver(opts ...TurnTrackingOption) *TurnTrackingObserver {
	o := &TurnTrackingObserver{timeout: DefaultTurnEndTimeout}
	for _, opt := range opts {
		opt(o)
	}
	return o
}

func (o *TurnTrackingObserver) OnTurnStarted(fn func(turn int)) {
	if fn == nil {
		return
	}
	o.mu.Lock()
	o.startedHandlers = append(o.startedHandlers[:len(o.startedHandlers):len(o.startedHandlers)], fn)
	o.mu.Unlock()
}

func (o *TurnTrackingObserver) OnTurnEnded(fn func(turn int, duration time.Duration, interrupted bool)) {
	if fn == nil {
		return
	}
	o.mu.Lock()
	o.endedHandlers = append(o.endedHandlers[:len(o.endedHandlers):len(o.endedHandlers)], fn)
	o.mu.Unlock()
}

func (o *TurnTrackingObserver) TurnCount() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.turnCount
}

func (o *TurnTrackingObserver) TurnActive() bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.active
}

func (o *TurnTrackingObserver) OnProcessFrame(pipeline.ProcessFrameEvent) {}

func (o *TurnTrackingObserver) Interested(frame frames.Frame, _ frames.FrameDirection, kind pipeline.EventKind) bool {
	if kind != pipeline.PushEvent {
		return false
	}
	switch frame.(type) {
	case *frames.StartFrame, *frames.UserStartedSpeakingFrame, *frames.BotStartedSpeakingFrame,
		*frames.BotStoppedSpeakingFrame, *frames.EndFrame, *frames.CancelFrame:
		return true
	}
	return false
}

func (o *TurnTrackingObserver) OnPushFrame(event pipeline.PushFrameEvent) {
	switch event.Frame.(type) {
	case *frames.StartFrame, *frames.UserStartedSpeakingFrame, *frames.BotStartedSpeakingFrame,
		*frames.BotStoppedSpeakingFrame, *frames.EndFrame, *frames.CancelFrame:
	default:
		return
	}

	o.emitMu.Lock()
	defer o.emitMu.Unlock()

	var events [2]turnEvent
	n := 0
	ts := event.Timestamp

	o.mu.Lock()
	switch event.Frame.(type) {
	case *frames.StartFrame:
		if o.turnCount == 0 {
			events[n] = o.startTurn(ts)
			n++
		}
	case *frames.UserStartedSpeakingFrame:
		switch {
		case o.botSpeaking:
			o.cancelTimer()
			if ev, ok := o.endTurn(ts, true); ok {
				events[n] = ev
				n++
			}
			o.botSpeaking = false
			events[n] = o.startTurn(ts)
			n++
		case o.active && o.botSpoken:
			o.cancelTimer()
			if ev, ok := o.endTurn(ts, false); ok {
				events[n] = ev
				n++
			}
			events[n] = o.startTurn(ts)
			n++
		case !o.active:
			events[n] = o.startTurn(ts)
			n++
		}
	case *frames.BotStartedSpeakingFrame:
		o.botSpeaking = true
		o.botSpoken = true
		o.cancelTimer()
	case *frames.BotStoppedSpeakingFrame:
		if o.botSpeaking {
			o.botSpeaking = false
			o.scheduleTurnEnd(ts)
		}
	case *frames.EndFrame, *frames.CancelFrame:
		o.cancelTimer()
		if ev, ok := o.endTurn(ts, true); ok {
			events[n] = ev
			n++
		}
	}
	o.mu.Unlock()

	o.emit(events[:n])
}

func (o *TurnTrackingObserver) OnPipelineStarted() {
	o.emitMu.Lock()
	o.mu.Lock()
	o.cancelTimer()
	o.turnCount = 0
	o.active = false
	o.botSpeaking = false
	o.botSpoken = false
	o.turnStartedAt = time.Time{}
	o.mu.Unlock()
	o.emitMu.Unlock()
}

func (o *TurnTrackingObserver) OnPipelineStopped() {
	o.mu.Lock()
	o.cancelTimer()
	o.mu.Unlock()
}

func (o *TurnTrackingObserver) startTurn(ts time.Time) turnEvent {
	o.active = true
	o.botSpoken = false
	o.turnCount++
	o.turnStartedAt = ts
	return turnEvent{started: true, turn: o.turnCount}
}

func (o *TurnTrackingObserver) endTurn(ts time.Time, interrupted bool) (turnEvent, bool) {
	if !o.active {
		return turnEvent{}, false
	}
	o.active = false
	return turnEvent{turn: o.turnCount, duration: ts.Sub(o.turnStartedAt), interrupted: interrupted}, true
}

func (o *TurnTrackingObserver) scheduleTurnEnd(stoppedAt time.Time) {
	o.cancelTimer()
	gen := o.timerGen
	o.timer = time.AfterFunc(o.timeout, func() { o.endAfterTimeout(gen, stoppedAt) })
}

func (o *TurnTrackingObserver) cancelTimer() {
	o.timerGen++
	if o.timer != nil {
		o.timer.Stop()
		o.timer = nil
	}
}

func (o *TurnTrackingObserver) endAfterTimeout(gen uint64, stoppedAt time.Time) {
	o.emitMu.Lock()
	defer o.emitMu.Unlock()

	o.mu.Lock()
	if gen != o.timerGen || !o.active || o.botSpeaking {
		o.mu.Unlock()
		return
	}
	o.timer = nil
	ev, ok := o.endTurn(stoppedAt, false)
	o.mu.Unlock()
	if ok {
		o.emit([]turnEvent{ev})
	}
}

func (o *TurnTrackingObserver) emit(events []turnEvent) {
	if len(events) == 0 {
		return
	}
	o.mu.Lock()
	started := o.startedHandlers
	ended := o.endedHandlers
	o.mu.Unlock()
	for _, ev := range events {
		if ev.started {
			for _, h := range started {
				h(ev.turn)
			}
			continue
		}
		for _, h := range ended {
			h(ev.turn, ev.duration, ev.interrupted)
		}
	}
}
