package processors

import (
	"context"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type GatedAggregator struct {
	*BaseProcessor
	openFn      func(frames.Frame) bool
	closeFn     func(frames.Frame) bool
	open        bool
	direction   frames.FrameDirection
	accumulated []frames.Frame
}

type GatedAggregatorOption func(*GatedAggregator)

func WithGateDirection(direction frames.FrameDirection) GatedAggregatorOption {
	return func(g *GatedAggregator) { g.direction = direction }
}

func NewGatedAggregator(openFn, closeFn func(frames.Frame) bool, startOpen bool, opts ...GatedAggregatorOption) *GatedAggregator {
	g := &GatedAggregator{
		openFn:    openFn,
		closeFn:   closeFn,
		open:      startOpen,
		direction: frames.Downstream,
	}
	for _, opt := range opts {
		opt(g)
	}
	g.BaseProcessor = NewBaseProcessor("GatedAggregator", g)
	return g
}

func (g *GatedAggregator) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if frames.IsSystem(frame) || direction != g.direction {
		return g.PushFrame(frame, direction)
	}
	if isTerminal(frame) {
		g.flush()
		return g.PushFrame(frame, direction)
	}

	wasOpen := g.open
	if g.open {
		g.open = g.closeFn == nil || !g.closeFn(frame)
	} else {
		g.open = g.openFn != nil && g.openFn(frame)
	}
	if wasOpen != g.open {
		state := "closed"
		if g.open {
			state = "open"
		}
		logger.Debug("[%s] Gate is now %s because of %s", g.Name(), state, frame.Name())
	}

	if !g.open {
		g.accumulated = append(g.accumulated, frame)
		return nil
	}
	err := g.PushFrame(frame, direction)
	g.flush()
	return err
}

func (g *GatedAggregator) flush() {
	for _, f := range g.accumulated {
		_ = g.PushFrame(f, g.direction)
	}
	clear(g.accumulated)
	g.accumulated = g.accumulated[:0]
}
