package processors

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type handoffQueue struct {
	mu   sync.Mutex
	q    frameQueue
	wake chan struct{}
}

func newHandoffQueue() *handoffQueue {
	return &handoffQueue{wake: make(chan struct{}, 1)}
}

func (h *handoffQueue) put(frame frames.Frame) {
	h.mu.Lock()
	h.q.push(queuedFrame{frame: frame})
	h.mu.Unlock()
	wake(h.wake)
}

func (h *handoffQueue) take(ctx context.Context) (frames.Frame, bool) {
	for {
		h.mu.Lock()
		qf, ok := h.q.pop()
		h.mu.Unlock()
		if ok {
			return qf.frame, true
		}
		select {
		case <-h.wake:
		case <-ctx.Done():
			return nil, false
		}
	}
}

func (h *handoffQueue) reset() {
	h.mu.Lock()
	h.q.retain(func(queuedFrame) bool { return false })
	h.mu.Unlock()
}

type ProducerProcessor struct {
	*BaseProcessor
	filter      func(frames.Frame) bool
	transformer func(frames.Frame) frames.Frame
	passthrough bool

	consumersMu sync.Mutex
	consumers   atomic.Pointer[[]*handoffQueue]
}

type ProducerOption func(*ProducerProcessor)

func WithProducerTransformer(transformer func(frames.Frame) frames.Frame) ProducerOption {
	return func(p *ProducerProcessor) { p.transformer = transformer }
}

func WithProducerPassthrough(passthrough bool) ProducerOption {
	return func(p *ProducerProcessor) { p.passthrough = passthrough }
}

func NewProducerProcessor(filter func(frames.Frame) bool, opts ...ProducerOption) *ProducerProcessor {
	p := &ProducerProcessor{filter: filter, passthrough: true}
	for _, opt := range opts {
		opt(p)
	}
	p.BaseProcessor = NewBaseProcessor("ProducerProcessor", p)
	return p
}

func (p *ProducerProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if p.filter == nil || !p.filter(frame) {
		return p.PushFrame(frame, direction)
	}
	p.produce(frame)
	if p.passthrough {
		return p.PushFrame(frame, direction)
	}
	return nil
}

func (p *ProducerProcessor) produce(frame frames.Frame) {
	list := p.consumers.Load()
	if list == nil {
		return
	}
	for _, q := range *list {
		out := frame
		if p.transformer != nil {
			out = p.transformer(frame)
		}
		if out != nil {
			q.put(out)
		}
	}
}

func (p *ProducerProcessor) addConsumer(q *handoffQueue) {
	p.consumersMu.Lock()
	defer p.consumersMu.Unlock()
	var cur []*handoffQueue
	if list := p.consumers.Load(); list != nil {
		cur = *list
	}
	for _, existing := range cur {
		if existing == q {
			return
		}
	}
	next := make([]*handoffQueue, len(cur), len(cur)+1)
	copy(next, cur)
	next = append(next, q)
	p.consumers.Store(&next)
}

func (p *ProducerProcessor) removeConsumer(q *handoffQueue) {
	p.consumersMu.Lock()
	defer p.consumersMu.Unlock()
	list := p.consumers.Load()
	if list == nil {
		return
	}
	next := make([]*handoffQueue, 0, len(*list))
	for _, existing := range *list {
		if existing != q {
			next = append(next, existing)
		}
	}
	p.consumers.Store(&next)
}

type ConsumerProcessor struct {
	*BaseProcessor
	producer    *ProducerProcessor
	transformer func(frames.Frame) frames.Frame
	direction   frames.FrameDirection
	queue       *handoffQueue

	lifeMu sync.Mutex
	wg     sync.WaitGroup
}

type ConsumerOption func(*ConsumerProcessor)

func WithConsumerTransformer(transformer func(frames.Frame) frames.Frame) ConsumerOption {
	return func(c *ConsumerProcessor) { c.transformer = transformer }
}

func WithConsumerDirection(direction frames.FrameDirection) ConsumerOption {
	return func(c *ConsumerProcessor) { c.direction = direction }
}

func NewConsumerProcessor(producer *ProducerProcessor, opts ...ConsumerOption) *ConsumerProcessor {
	c := &ConsumerProcessor{
		producer:  producer,
		direction: frames.Downstream,
		queue:     newHandoffQueue(),
	}
	for _, opt := range opts {
		opt(c)
	}
	c.BaseProcessor = NewBaseProcessor("ConsumerProcessor", c)
	return c
}

func (c *ConsumerProcessor) Start(ctx context.Context) error {
	c.lifeMu.Lock()
	defer c.lifeMu.Unlock()
	if err := c.BaseProcessor.Start(ctx); err != nil {
		return err
	}
	runCtx := c.BaseProcessor.Context()
	c.wg.Add(1)
	go c.consume(runCtx)
	return nil
}

func (c *ConsumerProcessor) Stop() error {
	c.lifeMu.Lock()
	defer c.lifeMu.Unlock()
	err := c.BaseProcessor.Stop()
	c.wg.Wait()
	c.detach()
	return err
}

func (c *ConsumerProcessor) consume(ctx context.Context) {
	defer c.wg.Done()
	for {
		frame, ok := c.queue.take(ctx)
		if !ok {
			return
		}
		if c.transformer != nil {
			frame = c.transformer(frame)
		}
		if frame == nil {
			continue
		}
		if err := c.QueueFrame(frame, c.direction); err != nil {
			return
		}
	}
}

func (c *ConsumerProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch frame.(type) {
	case *frames.StartFrame:
		if c.producer != nil {
			c.producer.addConsumer(c.queue)
		}
	case *frames.EndFrame, *frames.StopFrame, *frames.CancelFrame:
		c.detach()
	}
	return c.PushFrame(frame, direction)
}

func (c *ConsumerProcessor) detach() {
	if c.producer != nil {
		c.producer.removeConsumer(c.queue)
	}
	c.queue.reset()
}
