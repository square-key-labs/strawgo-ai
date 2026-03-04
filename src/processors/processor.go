package processors

import (
	"context"
	"fmt"
	"reflect"
	"runtime"
	"strconv"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/turns"
)

// ErrorHandler is a callback function type for handling errors in processors
// file and line provide the source location where the error occurred
type ErrorHandler func(processor FrameProcessor, err error, file string, line int)

type FrameObserver interface {
	OnProcessFrame(processorName string, frame frames.Frame, direction frames.FrameDirection)
	OnPushFrame(processorName string, frame frames.Frame, direction frames.FrameDirection)
}

type ObserverAwareProcessor interface {
	SetObserver(observer FrameObserver)
}

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

	observer FrameObserver

	// Separate channels for system (high priority) and other frames
	systemChan chan frameWithDirection
	dataChan   chan frameWithDirection

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	mu     sync.RWMutex

	// Interruption support
	allowInterruptions bool
	turnStrategies     turns.UserTurnStrategies

	// Handler for subclasses
	handler ProcessHandler

	// Error handling callback
	// Called when push_error is invoked or an unexpected exception occurs
	onError ErrorHandler
}

type frameWithDirection struct {
	frame     frames.Frame
	direction frames.FrameDirection
}

type InterruptionStrategy interface {
	AppendAudio(audio []byte, sampleRate int) error
	AppendText(text string) error
	ShouldInterrupt() (bool, error)
	Reset() error
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

func (p *BaseProcessor) SetObserver(observer FrameObserver) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.observer = observer
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

	if err := target.QueueFrame(frame, direction); err != nil {
		return err
	}

	p.notifyPushFrame(frame, direction)

	return nil
}

func (p *BaseProcessor) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	p.notifyProcessFrame(frame, direction)

	if p.handler != nil {
		return p.handler.HandleFrame(ctx, frame, direction)
	}
	// Default: pass through
	return p.PushFrame(frame, direction)
}

func (p *BaseProcessor) notifyProcessFrame(frame frames.Frame, direction frames.FrameDirection) {
	defer func() {
		if r := recover(); r != nil {
			logger.Error("[%s] Recovered from observer panic in process notification: %v", p.name, r)
		}
	}()

	p.mu.RLock()
	observer := p.observer
	name := p.name
	p.mu.RUnlock()

	if observer != nil {
		observer.OnProcessFrame(name, frame, direction)
	}
}

func (p *BaseProcessor) notifyPushFrame(frame frames.Frame, direction frames.FrameDirection) {
	defer func() {
		if r := recover(); r != nil {
			logger.Error("[%s] Recovered from observer panic in push notification: %v", p.name, r)
		}
	}()

	p.mu.RLock()
	observer := p.observer
	name := p.name
	p.mu.RUnlock()

	if observer != nil {
		observer.OnPushFrame(name, frame, direction)
	}
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
			// Only log non-AudioFrame processing to reduce noise
			if fwd.frame.Name() != "AudioFrame" && fwd.frame.Name() != "TTSAudioFrame" {
				logger.Debug("[%s] Processing data frame: %s", p.name, fwd.frame.Name())
			}
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
	p.turnStrategies = frame.TurnStrategies

	totalStrategies := len(p.turnStrategies.StartStrategies) + len(p.turnStrategies.StopStrategies) + len(p.turnStrategies.MuteStrategies)
	logger.Debug("[%s] Interruptions configured: allowed=%v, turn_strategies=%d", p.name, p.allowInterruptions, totalStrategies)
}

// InterruptionsAllowed returns whether interruptions are enabled
func (p *BaseProcessor) InterruptionsAllowed() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.allowInterruptions
}

func (p *BaseProcessor) TurnStrategies() turns.UserTurnStrategies {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.turnStrategies
}

func (p *BaseProcessor) InterruptionStrategies() []InterruptionStrategy {
	return nil
}

// PushInterruptionTaskFrame pushes an InterruptionTaskFrame upstream
// This is a helper method for processors that need to trigger an interruption
func (p *BaseProcessor) PushInterruptionTaskFrame() error {
	logger.Debug("[%s] Pushing InterruptionTaskFrame upstream", p.name)
	return p.PushFrame(frames.NewInterruptionTaskFrame(), frames.Upstream)
}

func (p *BaseProcessor) BroadcastFrame(ctx context.Context, frameConstructor func() frames.Frame) error {
	if frameConstructor == nil {
		return fmt.Errorf("frame constructor cannot be nil")
	}

	frameDownstream := frameConstructor()
	frameUpstream := frameConstructor()
	if frameDownstream == nil || frameUpstream == nil {
		return fmt.Errorf("frame constructor returned nil frame")
	}

	downstreamSiblingID := strconv.FormatUint(frameUpstream.ID(), 10)
	upstreamSiblingID := strconv.FormatUint(frameDownstream.ID(), 10)

	if err := setBroadcastSiblingID(frameDownstream, downstreamSiblingID); err != nil {
		return err
	}
	if err := setBroadcastSiblingID(frameUpstream, upstreamSiblingID); err != nil {
		return err
	}

	if ctx != nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}

	if err := p.PushFrame(frameDownstream, frames.Downstream); err != nil {
		return err
	}

	if ctx != nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}

	return p.PushFrame(frameUpstream, frames.Upstream)
}

func (p *BaseProcessor) BroadcastInterruption(ctx context.Context) error {
	logger.Debug("[%s] Broadcasting paired InterruptionFrame in both directions", p.name)
	return p.BroadcastFrame(ctx, func() frames.Frame {
		return frames.NewInterruptionFrame()
	})
}

func setBroadcastSiblingID(frame frames.Frame, siblingID string) error {
	value := reflect.ValueOf(frame)
	if !value.IsValid() || value.Kind() != reflect.Ptr || value.IsNil() {
		return fmt.Errorf("cannot set BroadcastSiblingID on invalid frame")
	}

	if setBroadcastSiblingIDOnValue(value.Elem(), siblingID) {
		return nil
	}

	return fmt.Errorf("frame %s does not expose BroadcastSiblingID", frame.Name())
}

func setBroadcastSiblingIDOnValue(value reflect.Value, siblingID string) bool {
	if !value.IsValid() {
		return false
	}

	if value.Kind() == reflect.Ptr {
		if value.IsNil() {
			return false
		}
		return setBroadcastSiblingIDOnValue(value.Elem(), siblingID)
	}

	if value.Kind() != reflect.Struct {
		return false
	}

	broadcastSiblingIDField := value.FieldByName("BroadcastSiblingID")
	if broadcastSiblingIDField.IsValid() && broadcastSiblingIDField.CanSet() && broadcastSiblingIDField.Kind() == reflect.String {
		broadcastSiblingIDField.SetString(siblingID)
		return true
	}

	for i := 0; i < value.NumField(); i++ {
		if !value.Type().Field(i).Anonymous {
			continue
		}
		if setBroadcastSiblingIDOnValue(value.Field(i), siblingID) {
			return true
		}
	}

	return false
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

// SetOnError sets the error handler callback for this processor
// The callback is invoked when PushError is called or an unexpected error occurs
func (p *BaseProcessor) SetOnError(handler ErrorHandler) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.onError = handler
}

// PushError creates and pushes an ErrorFrame upstream with simplified error reporting
// It also calls the on_error callback if set, and logs the error with file/line info
// Parameters:
//   - errorMsg: Human-readable error message
//   - err: Optional underlying error (can be nil)
//   - fatal: If true, indicates a fatal error that should stop the pipeline
func (p *BaseProcessor) PushError(errorMsg string, err error, fatal bool) error {
	// Get caller information for debugging (skip 1 frame to get the actual caller)
	_, file, line, ok := runtime.Caller(1)
	if !ok {
		file = "unknown"
		line = 0
	}

	// Create the error
	var fullErr error
	if err != nil {
		fullErr = fmt.Errorf("%s: %w", errorMsg, err)
	} else {
		fullErr = fmt.Errorf("%s", errorMsg)
	}

	// Log with file/line information
	if fatal {
		logger.Error("[%s] FATAL ERROR at %s:%d - %v", p.name, file, line, fullErr)
	} else {
		logger.Error("[%s] Error at %s:%d - %v", p.name, file, line, fullErr)
	}

	// Call the on_error callback if set
	p.mu.RLock()
	handler := p.onError
	p.mu.RUnlock()

	if handler != nil {
		handler(p, fullErr, file, line)
	}

	// Create and push ErrorFrame upstream
	errorFrame := frames.NewErrorFrame(fullErr)
	if fatal {
		errorFrame.SetMetadata("fatal", true)
	}
	errorFrame.SetMetadata("file", file)
	errorFrame.SetMetadata("line", line)
	errorFrame.SetMetadata("processor", p.name) // Include processor reference

	return p.PushFrame(errorFrame, frames.Upstream)
}

// PushErrorFrame pushes an existing ErrorFrame upstream (for backwards compatibility)
// Prefer using PushError for new code as it provides better error context
func (p *BaseProcessor) PushErrorFrame(errorFrame *frames.ErrorFrame) error {
	// Get caller information
	_, file, line, ok := runtime.Caller(1)
	if !ok {
		file = "unknown"
		line = 0
	}

	// Log with file/line information
	logger.Error("[%s] Error at %s:%d - %v", p.name, file, line, errorFrame.Error)

	// Call the on_error callback if set
	p.mu.RLock()
	handler := p.onError
	p.mu.RUnlock()

	if handler != nil {
		handler(p, errorFrame.Error, file, line)
	}

	return p.PushFrame(errorFrame, frames.Upstream)
}
