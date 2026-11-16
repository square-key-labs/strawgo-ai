package llmservice

import (
	"context"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type speculationGate struct {
	mu        sync.Mutex
	active    *llm.Context
	confirmed *llm.Context
	holding   bool
	dropping  bool
	held      []frames.Frame
	cancel    context.CancelFunc
}

func (s *Service) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	if direction != frames.Downstream {
		return s.BaseProcessor.PushFrame(frame, direction)
	}
	g := &s.gate
	g.mu.Lock()
	defer g.mu.Unlock()
	return s.pushAll(g.process(frame))
}

func (s *Service) pushAll(out []frames.Frame) error {
	var first error
	for _, f := range out {
		if err := s.BaseProcessor.PushFrame(f, frames.Downstream); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func (s *Service) beginGeneration(ctx context.Context, f *frames.LLMContextFrame) (context.Context, func()) {
	g := &s.gate
	g.mu.Lock()
	defer g.mu.Unlock()
	kept := g.dropHeld()
	g.dropping = false
	g.active = nil
	speculative := f.Speculation && f.Context != g.confirmed
	if f.Context == g.confirmed {
		g.confirmed = nil
	}
	var cancel context.CancelFunc
	if speculative {
		ctx, cancel = context.WithCancel(ctx)
		g.active, g.holding, g.cancel = f.Context, true, cancel
	}
	_ = s.pushAll(kept)
	return ctx, func() {
		g.mu.Lock()
		g.dropping = false
		g.cancel = nil
		g.mu.Unlock()
		if cancel != nil {
			cancel()
		}
	}
}

func (s *Service) withdrawSpeculation(reason string) bool {
	g := &s.gate
	g.mu.Lock()
	if g.active == nil {
		g.mu.Unlock()
		return false
	}
	_ = s.pushAll(g.discard())
	g.mu.Unlock()
	logger.Debug("[%s] withdrawing the speculative response: %s", s.Name(), reason)
	s.broadcast(func() frames.Frame { return frames.NewUserTurnResponseCancelFrame() })
	return true
}

func (g *speculationGate) process(frame frames.Frame) []frames.Frame {
	if frames.IsSystem(frame) {
		out := []frames.Frame{frame}
		switch f := frame.(type) {
		case *frames.UserTurnResponseCancelFrame, *frames.InterruptionFrame:
			out = append(out, g.discard()...)
		case *frames.UserTurnSpeculationConfirmedFrame:
			out = append(out, g.release(f.Context)...)
		}
		return out
	}
	if _, ok := frame.(*frames.EndFrame); ok {
		return append(g.discard(), frame)
	}
	switch {
	case g.holding:
		g.held = append(g.held, frame)
		return nil
	case g.dropping && !frame.SurvivesInterruption():
		return nil
	}
	return []frames.Frame{frame}
}

func (g *speculationGate) release(c *llm.Context) []frames.Frame {
	if g.active == nil || g.active != c {
		g.confirmed = c
		return nil
	}
	g.active = nil
	if !g.holding {
		return nil
	}
	logger.Debug("speculative response confirmed, releasing %d frames", len(g.held))
	out := g.held
	g.held, g.holding = nil, false
	return out
}

func (g *speculationGate) discard() []frames.Frame {
	g.confirmed = nil
	if g.active == nil {
		return nil
	}
	g.active = nil
	if g.cancel != nil {
		g.cancel()
		g.dropping = true
	}
	if g.holding {
		logger.Debug("speculative response withdrawn, discarding %d frames", len(g.held))
	}
	return g.dropHeld()
}

func (g *speculationGate) dropHeld() []frames.Frame {
	var kept []frames.Frame
	for _, f := range g.held {
		if f.SurvivesInterruption() {
			kept = append(kept, f)
		}
	}
	g.held, g.holding = nil, false
	return kept
}
