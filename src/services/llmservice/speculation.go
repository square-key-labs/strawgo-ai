package llmservice

import (
	"context"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type speculationGate struct {
	mu           sync.Mutex
	running      *generation
	spec         *generation
	held         []frames.Frame
	confirmed    *llm.Context
	withdrawn    *llm.Context
	withdrawNext bool
}

type generation struct {
	context   *llm.Context
	discarded bool
	cancel    context.CancelFunc
}

func (s *Service) pushAll(out []frames.Frame) {
	for _, f := range out {
		_ = s.BaseProcessor.PushFrame(f, frames.Downstream)
	}
}

func (s *Service) beginGeneration(ctx context.Context, f *frames.LLMContextFrame) (context.Context, func(), bool) {
	g := &s.gate
	g.mu.Lock()
	var superseded *llm.Context
	if g.spec != nil {
		if !f.Speculation {
			superseded = g.spec.context
		}
		s.pushAll(g.withdraw(g.spec.context))
	}
	gen := &generation{}
	if f.Speculation && (f.Context == g.withdrawn || g.withdrawNext) {
		g.withdrawn, g.withdrawNext = nil, false
		g.mu.Unlock()
		logger.Debug("[%s] speculative request withdrawn before it ran, skipping it", s.Name())
		s.broadcastCancel(f.Context)
		return ctx, func() {}, false
	}
	if f.Speculation {
		switch f.Context {
		case g.confirmed:
			g.confirmed = nil
		default:
			gen.context = f.Context
			ctx, gen.cancel = context.WithCancel(ctx)
			g.spec = gen
		}
	}
	g.running = gen
	g.mu.Unlock()
	if superseded != nil {
		logger.Debug("[%s] a new inference superseded the held speculative response", s.Name())
		s.broadcastCancel(superseded)
	}
	return ctx, func() {
		g.mu.Lock()
		if g.running == gen {
			g.running = nil
		}
		g.mu.Unlock()
		if gen.cancel != nil {
			gen.cancel()
		}
	}, true
}

func (s *Service) broadcastCancel(c *llm.Context) {
	s.broadcast(func() frames.Frame {
		f := frames.NewUserTurnResponseCancelFrame()
		f.Context = c
		return f
	})
}

func (s *Service) emitGenerated(frame frames.Frame) {
	g := &s.gate
	g.mu.Lock()
	defer g.mu.Unlock()
	gen := g.running
	switch {
	case gen != nil && gen.discarded:
		s.pushAll(usageOnly([]frames.Frame{frame}))
	case gen != nil && gen == g.spec:
		g.held = append(g.held, frame)
	default:
		_ = s.BaseProcessor.PushFrame(frame, frames.Downstream)
	}
}

func (s *Service) pushVerdict(frame frames.Frame, direction frames.FrameDirection) error {
	if direction != frames.Downstream {
		return s.PushFrame(frame, direction)
	}
	g := &s.gate
	g.mu.Lock()
	defer g.mu.Unlock()
	var out []frames.Frame
	switch f := frame.(type) {
	case *frames.UserTurnResponseCancelFrame:
		g.withdrawNext = f.Context == nil && g.spec == nil
		out = g.withdraw(f.Context)
	case *frames.UserTurnSpeculationConfirmedFrame:
		out = g.release(f.Context)
	case *frames.EndFrame, *frames.StopFrame:
		s.pushAll(g.withdraw(nil))
		return s.PushFrame(frame, direction)
	default:
		out = g.withdraw(nil)
	}
	err := s.PushFrame(frame, direction)
	s.pushAll(out)
	return err
}

func (s *Service) withdrawSpeculation(reason string) bool {
	g := &s.gate
	g.mu.Lock()
	gen := g.running
	if gen == nil || gen.context == nil {
		g.mu.Unlock()
		return false
	}
	if gen.discarded {
		g.mu.Unlock()
		return true
	}
	if g.spec != gen {
		g.mu.Unlock()
		return false
	}
	s.pushAll(g.withdraw(gen.context))
	g.mu.Unlock()
	logger.Debug("[%s] withdrawing the speculative response: %s", s.Name(), reason)
	s.broadcastCancel(gen.context)
	return true
}

func (g *speculationGate) release(c *llm.Context) []frames.Frame {
	if g.spec == nil || g.spec.context != c {
		g.confirmed = c
		return nil
	}
	logger.Debug("speculative response confirmed, releasing %d frames", len(g.held))
	out := g.held
	g.spec, g.held = nil, nil
	return out
}

func (g *speculationGate) withdraw(c *llm.Context) []frames.Frame {
	spec := g.spec
	if spec == nil || (c != nil && spec.context != c) {
		if c != nil {
			g.withdrawn = c
		}
		return nil
	}
	logger.Debug("speculative response withdrawn, discarding %d frames", len(g.held))
	spec.discarded = true
	if spec.cancel != nil {
		spec.cancel()
	}
	out := usageOnly(g.held)
	g.spec, g.held = nil, nil
	return out
}

func usageOnly(held []frames.Frame) []frames.Frame {
	var usage []frames.MetricsData
	for _, f := range held {
		m, ok := f.(*frames.MetricsFrame)
		if !ok {
			continue
		}
		for _, d := range m.Data {
			if _, ok := d.(frames.LLMUsageMetricsData); ok {
				usage = append(usage, d)
			}
		}
	}
	if len(usage) == 0 {
		return nil
	}
	return []frames.Frame{frames.NewMetricsFrame(usage...)}
}
