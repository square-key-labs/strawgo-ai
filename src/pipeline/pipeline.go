package pipeline

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type edgeHandler func(frame frames.Frame, direction frames.FrameDirection) error

type edge struct {
	*processors.BaseProcessor
	handle edgeHandler
}

func newEdge(name string, handle edgeHandler) *edge {
	e := &edge{handle: handle}
	e.BaseProcessor = processors.NewBaseProcessor(name, e, processors.WithDirectMode())
	return e
}

func (e *edge) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return e.handle(frame, direction)
}

type procRef struct {
	p processors.FrameProcessor
}

type Pipeline struct {
	name       string
	processors []processors.FrameProcessor
	source     *edge
	sink       *edge
	next       atomic.Pointer[procRef]
	prev       atomic.Pointer[procRef]
	started    []processors.FrameProcessor
}

var pipelineCounter atomic.Uint64

func NewPipeline(procs []processors.FrameProcessor) *Pipeline {
	p := &Pipeline{
		name:       fmt.Sprintf("Pipeline#%d", pipelineCounter.Add(1)),
		processors: procs,
	}

	p.source = newEdge(p.name+"::Source", func(frame frames.Frame, direction frames.FrameDirection) error {
		if direction == frames.Upstream {
			return p.PushFrame(frame, frames.Upstream)
		}
		return p.source.PushFrame(frame, frames.Downstream)
	})
	p.sink = newEdge(p.name+"::Sink", func(frame frames.Frame, direction frames.FrameDirection) error {
		if direction == frames.Downstream {
			return p.PushFrame(frame, frames.Downstream)
		}
		return p.sink.PushFrame(frame, frames.Upstream)
	})

	chain := make([]processors.FrameProcessor, 0, len(procs)+2)
	chain = append(chain, p.source)
	chain = append(chain, procs...)
	chain = append(chain, p.sink)
	for i := 0; i < len(chain)-1; i++ {
		chain[i].Link(chain[i+1])
	}

	return p
}

func (p *Pipeline) Name() string {
	return p.name
}

func (p *Pipeline) Processors() []processors.FrameProcessor {
	return p.processors
}

func (p *Pipeline) Link(next processors.FrameProcessor) {
	if next == nil {
		p.next.Store(nil)
		return
	}
	p.next.Store(&procRef{p: next})
	next.SetPrev(p)
}

func (p *Pipeline) SetPrev(prev processors.FrameProcessor) {
	if prev == nil {
		p.prev.Store(nil)
		return
	}
	p.prev.Store(&procRef{p: prev})
}

func (p *Pipeline) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	if direction == frames.Upstream {
		return p.sink.QueueFrame(frame, frames.Upstream)
	}
	return p.source.QueueFrame(frame, frames.Downstream)
}

func (p *Pipeline) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return p.QueueFrame(frame, direction)
}

func (p *Pipeline) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	var ref *procRef
	if direction == frames.Downstream {
		ref = p.next.Load()
	} else {
		ref = p.prev.Load()
	}
	if ref == nil {
		return nil
	}
	return ref.p.QueueFrame(frame, direction)
}

func (p *Pipeline) SetObserver(observer processors.FrameObserver) {
	for _, proc := range p.processors {
		if observerAware, ok := proc.(processors.ObserverAwareProcessor); ok {
			observerAware.SetObserver(observer)
		}
	}
}

func (p *Pipeline) Start(ctx context.Context) error {
	chain := make([]processors.FrameProcessor, 0, len(p.processors)+2)
	chain = append(chain, p.source)
	chain = append(chain, p.processors...)
	chain = append(chain, p.sink)

	for _, proc := range chain {
		if err := proc.Start(ctx); err != nil {
			for i := len(p.started) - 1; i >= 0; i-- {
				_ = p.started[i].Stop()
			}
			p.started = nil
			return fmt.Errorf("failed to start processor %s: %w", proc.Name(), err)
		}
		p.started = append(p.started, proc)
	}

	logger.Debug("[%s] Started %d processors", p.name, len(p.processors))
	return nil
}

func (p *Pipeline) Stop() error {
	for i := len(p.started) - 1; i >= 0; i-- {
		if err := p.started[i].Stop(); err != nil {
			logger.Error("[%s] Error stopping processor %s: %v", p.name, p.started[i].Name(), err)
		}
	}
	p.started = nil
	return nil
}
