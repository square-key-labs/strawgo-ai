package processors

import (
	"context"
	"fmt"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/interruptions"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// FrameProcessor is the interface that all processors must implement
type FrameProcessor interface {
	// ProcessFrame processes a single frame
	ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error

	// QueueFrame adds a frame to this processor's queue
	QueueFrame(frame frames.Frame, direction frames.FrameDirection) error

	// PushFrame sends a frame to the next/previous processor
	PushFrame(frame frames.Frame, direction frames.FrameDirection) error

	// Link connects this processor to the next one in the chain
	Link(next FrameProcessor)

	// SetPrev sets the previous processor in the chain
	SetPrev(prev FrameProcessor)

	// Start begins processing frames
	Start(ctx context.Context) error

	// Stop gracefully stops the processor
	Stop() error

	// Name returns the processor name
	Name() string
}

// BaseProcessor provides the common functionality for all processors
type BaseProcessor struct {
	name string
	next FrameProcessor
	prev FrameProcessor

	// Separate channels for system (high priority) and other frames
	systemChan chan frameWithDirection
	dataChan   chan frameWithDirection

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	mu     sync.RWMutex

	// Interruption support
	allowInterruptions     bool
	interruptionStrategies []interruptions.InterruptionStrategy

	// Handler for subclasses
	handler ProcessHandler
}

type frameWithDirection struct {
	frame     frames.Frame
	direction frames.FrameDirection
}

// ProcessHandler is the interface that subclasses implement for custom processing
type ProcessHandler interface {
	HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error
}

// NewBaseProcessor creates a new BaseProcessor
func NewBaseProcessor(name string, handler ProcessHandler) *BaseProcessor {
	return &BaseProcessor{
		name:       name,
		systemChan: make(chan frameWithDirection, 100),
		dataChan:   make(chan frameWithDirection, 1000),
		handler:    handler,
	}
}

func (p *BaseProcessor) Name() string {
	return p.name
}

func (p *BaseProcessor) Link(next FrameProcessor) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.next = next
	if next != nil {
		next.SetPrev(p)
	}
}

func (p *BaseProcessor) SetPrev(prev FrameProcessor) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.prev = prev
}

func (p *BaseProcessor) Start(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.ctx != nil {
		return fmt.Errorf("processor %s already started", p.name)
	}

	p.ctx, p.cancel = context.WithCancel(ctx)

	// Start system frame handler (high priority)
	p.wg.Add(1)
	go p.systemFrameHandler()

	// Start data frame handler (normal priority)
	p.wg.Add(1)
	go p.dataFrameHandler()

	logger.Info("[%s] Started", p.name)
	logger.Debug("[%s] Processor initialized with system and data channels", p.name)
	return nil
}

func (p *BaseProcessor) Stop() error {
	p.mu.Lock()
	if p.cancel != nil {
		p.cancel()
	}
	p.mu.Unlock()

	p.wg.Wait()

	logger.Info("[%s] Stopped", p.name)
	logger.Debug("[%s] All goroutines terminated", p.name)
	return nil
}

func (p *BaseProcessor) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	fwd := frameWithDirection{frame: frame, direction: direction}

	// Check if frame is categorizable
	if categorizable, ok := frame.(frames.Categorizable); ok {
		if categorizable.Category() == frames.SystemCategory {
			select {
			case p.systemChan <- fwd:
				return nil
			case <-p.ctx.Done():
				return p.ctx.Err()
			}
		}
	}

	// All other frames go to data channel
	select {
	case p.dataChan <- fwd:
		return nil
	case <-p.ctx.Done():
		return p.ctx.Err()
	}
}

func (p *BaseProcessor) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	p.mu.RLock()
	var target FrameProcessor
	if direction == frames.Downstream {
		target = p.next
	} else {
		target = p.prev
	}
	p.mu.RUnlock()

	if target == nil {
		// End of chain
		return nil
	}

	return target.QueueFrame(frame, direction)
}

func (p *BaseProcessor) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if p.handler != nil {
		return p.handler.HandleFrame(ctx, frame, direction)
	}
	// Default: pass through
	return p.PushFrame(frame, direction)
}

// systemFrameHandler processes high-priority system frames immediately
func (p *BaseProcessor) systemFrameHandler() {
	defer p.wg.Done()

	for {
		select {
		case <-p.ctx.Done():
			logger.Debug("[%s] System frame handler shutting down", p.name)
			return
		case fwd := <-p.systemChan:
			logger.Debug("[%s] Processing system frame: %s", p.name, fwd.frame.Name())
			if err := p.ProcessFrame(p.ctx, fwd.frame, fwd.direction); err != nil {
				logger.Error("[%s] Error processing system frame %s: %v", p.name, fwd.frame.Name(), err)
			}
		}
	}
}

// dataFrameHandler processes normal priority data/control frames
func (p *BaseProcessor) dataFrameHandler() {
	defer p.wg.Done()

	for {
		select {
		case <-p.ctx.Done():
			logger.Debug("[%s] Data frame handler shutting down", p.name)
			return
		case fwd := <-p.dataChan:
			logger.Debug("[%s] Processing data frame: %s", p.name, fwd.frame.Name())
			if err := p.ProcessFrame(p.ctx, fwd.frame, fwd.direction); err != nil {
				logger.Error("[%s] Error processing data frame %s: %v", p.name, fwd.frame.Name(), err)
			}
		}
	}
}

// HandleStartFrame processes StartFrame and configures interruption settings
// This should be called by processors when they receive a StartFrame
func (p *BaseProcessor) HandleStartFrame(frame *frames.StartFrame) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.allowInterruptions = frame.AllowInterruptions
	p.interruptionStrategies = frame.InterruptionStrategies

	logger.Debug("[%s] Interruptions configured: allowed=%v, strategies=%d",
		p.name, p.allowInterruptions, len(p.interruptionStrategies))
}

// InterruptionsAllowed returns whether interruptions are enabled
func (p *BaseProcessor) InterruptionsAllowed() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.allowInterruptions
}

// InterruptionStrategies returns the configured interruption strategies
func (p *BaseProcessor) InterruptionStrategies() []interruptions.InterruptionStrategy {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.interruptionStrategies
}

// PushInterruptionTaskFrame pushes an InterruptionTaskFrame upstream
// This is a helper method for processors that need to trigger an interruption
func (p *BaseProcessor) PushInterruptionTaskFrame() error {
	logger.Debug("[%s] Pushing InterruptionTaskFrame upstream", p.name)
	return p.PushFrame(frames.NewInterruptionTaskFrame(), frames.Upstream)
}

// HandleInterruptionFrame processes an InterruptionFrame
// This should be called by processors when they receive an InterruptionFrame
func (p *BaseProcessor) HandleInterruptionFrame() {
	logger.Debug("[%s] Handling interruption - clearing queues", p.name)

	// Drain the data channel to clear any pending frames
	p.mu.Lock()
	defer p.mu.Unlock()

	// Drain data channel
	for {
		select {
		case <-p.dataChan:
			// Discard frame
		default:
			return
		}
	}
}
