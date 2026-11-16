package aggregators

import (
	"context"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	DefaultDTMFTimeout = 2 * time.Second
	DefaultDTMFPrefix  = "DTMF: "
)

type DTMFOption func(*DTMFAggregator)

func WithDTMFTimeout(d time.Duration) DTMFOption {
	return func(a *DTMFAggregator) { a.timeout = d }
}

func WithDTMFTerminationDigit(k frames.KeypadEntry) DTMFOption {
	return func(a *DTMFAggregator) { a.terminator = k }
}

func WithDTMFPrefix(prefix string) DTMFOption {
	return func(a *DTMFAggregator) { a.prefix = prefix }
}

type DTMFAggregator struct {
	*processors.BaseProcessor

	timeout    time.Duration
	terminator frames.KeypadEntry
	prefix     string

	digits []byte
	gen    uint64

	timerMu sync.Mutex
	timer   *time.Timer
}

func NewDTMFAggregator(opts ...DTMFOption) *DTMFAggregator {
	a := &DTMFAggregator{
		timeout:    DefaultDTMFTimeout,
		terminator: frames.KeypadPound,
		prefix:     DefaultDTMFPrefix,
	}
	for _, opt := range opts {
		opt(a)
	}
	a.BaseProcessor = processors.NewBaseProcessor("DTMFAggregator", a)
	return a
}

func (a *DTMFAggregator) Stop() error {
	a.stopTimer()
	return a.BaseProcessor.Stop()
}

func (a *DTMFAggregator) HandleFrame(_ context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *selfFrame:
		if f.kind == selfDTMFTimeout && f.gen == a.gen {
			return a.flush()
		}
		return nil
	case *frames.InputDTMFFrame:
		if err := a.PushFrame(frame, direction); err != nil {
			return err
		}
		return a.handleDigit(f.Button)
	case *frames.EndFrame, *frames.CancelFrame:
		a.stopTimer()
		if err := a.flush(); err != nil {
			logger.Error("[%s] flush failed: %v", a.Name(), err)
		}
		return a.PushFrame(frame, direction)
	}
	return a.PushFrame(frame, direction)
}

func (a *DTMFAggregator) handleDigit(button frames.KeypadEntry) error {
	if !button.Valid() {
		return nil
	}
	if len(a.digits) == 0 && a.InterruptionsAllowed() {
		if err := a.BroadcastInterruption(nil); err != nil {
			logger.Error("[%s] interruption broadcast failed: %v", a.Name(), err)
		}
	}
	a.digits = append(a.digits, button...)
	if button == a.terminator {
		a.stopTimer()
		return a.flush()
	}
	a.armTimer()
	return nil
}

func (a *DTMFAggregator) flush() error {
	a.gen++
	if len(a.digits) == 0 {
		return nil
	}
	text := a.prefix + string(a.digits)
	a.digits = a.digits[:0]
	return a.PushFrame(frames.NewTranscriptionFrame(text, true), frames.Downstream)
}

func (a *DTMFAggregator) armTimer() {
	a.gen++
	gen := a.gen
	a.timerMu.Lock()
	defer a.timerMu.Unlock()
	if a.timer != nil {
		a.timer.Stop()
	}
	if a.timeout <= 0 {
		return
	}
	a.timer = time.AfterFunc(a.timeout, func() {
		_ = a.QueueFrame(newSelfFrame(selfDTMFTimeout, gen, nil), frames.Downstream)
	})
}

func (a *DTMFAggregator) stopTimer() {
	a.timerMu.Lock()
	defer a.timerMu.Unlock()
	if a.timer != nil {
		a.timer.Stop()
		a.timer = nil
	}
}
