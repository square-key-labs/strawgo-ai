package processors

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"runtime/debug"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/turns"
)

type ErrorHandler func(processor FrameProcessor, err error, file string, line int)

type FrameObserver interface {
	OnProcessFrame(processorName string, frame frames.Frame, direction frames.FrameDirection)
	OnPushFrame(processorName string, frame frames.Frame, direction frames.FrameDirection)
}

type ObserverAwareProcessor interface {
	SetObserver(observer FrameObserver)
}

type FrameProcessor interface {
	ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error
	QueueFrame(frame frames.Frame, direction frames.FrameDirection) error
	PushFrame(frame frames.Frame, direction frames.FrameDirection) error
	Link(next FrameProcessor)
	SetPrev(prev FrameProcessor)
	Start(ctx context.Context) error
	Stop() error
	Name() string
}

type ProcessHandler interface {
	HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error
}

type InterruptionStrategy interface {
	AppendAudio(audio []byte, sampleRate int) error
	AppendText(text string) error
	ShouldInterrupt() (bool, error)
	Reset() error
}

type StartConfig struct {
	AllowInterruptions    bool
	TurnStrategies        turns.UserTurnStrategies
	AudioInSampleRate     int
	AudioOutSampleRate    int
	EnableMetrics         bool
	EnableUsageMetrics    bool
	ReportOnlyInitialTTFB bool
}

type Option func(*BaseProcessor)

func WithDirectMode() Option {
	return func(p *BaseProcessor) { p.direct = true }
}

func WithConcurrentSystemFrames() Option {
	return func(p *BaseProcessor) { p.concurrentSystem = true }
}

func WithLegacyConcurrency() Option {
	return func(p *BaseProcessor) { p.legacy = true }
}

type queuedFrame struct {
	frame     frames.Frame
	direction frames.FrameDirection
	epoch     uint64
}

type frameQueue struct {
	buf  []queuedFrame
	head int
	n    int
}

func (q *frameQueue) push(f queuedFrame) {
	if q.n == len(q.buf) {
		size := len(q.buf) * 2
		if size == 0 {
			size = 64
		}
		grown := make([]queuedFrame, size)
		for i := 0; i < q.n; i++ {
			grown[i] = q.buf[(q.head+i)%len(q.buf)]
		}
		q.buf = grown
		q.head = 0
	}
	q.buf[(q.head+q.n)%len(q.buf)] = f
	q.n++
}

func (q *frameQueue) pop() (queuedFrame, bool) {
	if q.n == 0 {
		return queuedFrame{}, false
	}
	f := q.buf[q.head]
	q.buf[q.head] = queuedFrame{}
	q.head = (q.head + 1) % len(q.buf)
	q.n--
	return f, true
}

func (q *frameQueue) retain(keep func(queuedFrame) bool) int {
	dropped := 0
	kept := 0
	for i := 0; i < q.n; i++ {
		idx := (q.head + i) % len(q.buf)
		f := q.buf[idx]
		q.buf[idx] = queuedFrame{}
		if keep(f) {
			q.buf[(q.head+kept)%len(q.buf)] = f
			kept++
		} else {
			dropped++
		}
	}
	q.n = kept
	return dropped
}

type linkRef struct {
	p FrameProcessor
}

type observerRef struct {
	o FrameObserver
}

type BaseProcessor struct {
	name    string
	handler ProcessHandler
	direct  bool
	legacy  bool

	concurrentSystem bool

	next     atomic.Pointer[linkRef]
	prev     atomic.Pointer[linkRef]
	observer atomic.Pointer[observerRef]

	qmu          sync.Mutex
	sysQ         frameQueue
	dataQ        frameQueue
	pendingStart int
	epoch        uint64
	epochCtx     context.Context
	epochCancel  context.CancelFunc
	sysWake      chan struct{}
	dataWake     chan struct{}

	handlerMu     sync.Mutex
	systemWaiting atomic.Int32

	lifeMu  sync.Mutex
	ctx     context.Context
	cancel  context.CancelFunc
	wg      sync.WaitGroup
	stopped atomic.Bool

	cfgMu        sync.RWMutex
	startCfg     StartConfig
	startApplied bool

	onError atomic.Pointer[ErrorHandler]
}

func NewBaseProcessor(name string, handler ProcessHandler, opts ...Option) *BaseProcessor {
	p := &BaseProcessor{
		name:     name,
		handler:  handler,
		sysWake:  make(chan struct{}, 1),
		dataWake: make(chan struct{}, 1),
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

func NewLegacyProcessor(name string, handler ProcessHandler) *BaseProcessor {
	return NewBaseProcessor(name, handler, WithLegacyConcurrency())
}

func (p *BaseProcessor) Name() string {
	return p.name
}

func (p *BaseProcessor) Link(next FrameProcessor) {
	if next == nil {
		p.next.Store(nil)
		return
	}
	p.next.Store(&linkRef{p: next})
	next.SetPrev(p.self())
}

func (p *BaseProcessor) self() FrameProcessor {
	if fp, ok := p.handler.(FrameProcessor); ok {
		return fp
	}
	return p
}

func (p *BaseProcessor) SetPrev(prev FrameProcessor) {
	if prev == nil {
		p.prev.Store(nil)
		return
	}
	p.prev.Store(&linkRef{p: prev})
}

func (p *BaseProcessor) Next() FrameProcessor {
	if ref := p.next.Load(); ref != nil {
		return ref.p
	}
	return nil
}

func (p *BaseProcessor) Prev() FrameProcessor {
	if ref := p.prev.Load(); ref != nil {
		return ref.p
	}
	return nil
}

func (p *BaseProcessor) SetObserver(observer FrameObserver) {
	if observer == nil {
		p.observer.Store(nil)
		return
	}
	p.observer.Store(&observerRef{o: observer})
}

func (p *BaseProcessor) Start(ctx context.Context) error {
	p.lifeMu.Lock()
	defer p.lifeMu.Unlock()

	if p.ctx != nil {
		return fmt.Errorf("processor %s already started", p.name)
	}

	p.ctx, p.cancel = context.WithCancel(ctx)

	p.qmu.Lock()
	p.epochCtx, p.epochCancel = context.WithCancel(p.ctx)
	p.qmu.Unlock()

	if !p.direct {
		p.wg.Add(2)
		go p.systemLoop(p.ctx)
		go p.dataLoop(p.ctx)
	}

	logger.Debug("[%s] Started", p.name)
	return nil
}

func (p *BaseProcessor) Stop() error {
	p.stopped.Store(true)

	p.lifeMu.Lock()
	cancel := p.cancel
	p.lifeMu.Unlock()

	if cancel != nil {
		cancel()
	}
	p.wg.Wait()

	logger.Debug("[%s] Stopped", p.name)
	return nil
}

func (p *BaseProcessor) Context() context.Context {
	p.lifeMu.Lock()
	defer p.lifeMu.Unlock()
	if p.ctx == nil {
		return context.Background()
	}
	return p.ctx
}

func (p *BaseProcessor) QueueLen() (system, data int) {
	p.qmu.Lock()
	defer p.qmu.Unlock()
	return p.sysQ.n, p.dataQ.n
}

func (p *BaseProcessor) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	if frame == nil {
		return errors.New("nil frame")
	}
	if p.stopped.Load() {
		return context.Canceled
	}

	if p.direct {
		return p.invoke(p.Context(), queuedFrame{frame: frame, direction: direction}, false)
	}

	if frames.IsSystem(frame) {
		if _, ok := frame.(*frames.InterruptionFrame); ok {
			p.resetForInterruption()
		}
		p.qmu.Lock()
		if _, ok := frame.(*frames.StartFrame); ok {
			p.pendingStart++
		}
		p.sysQ.push(queuedFrame{frame: frame, direction: direction})
		p.qmu.Unlock()
		wake(p.sysWake)
		return nil
	}

	p.qmu.Lock()
	p.dataQ.push(queuedFrame{frame: frame, direction: direction, epoch: p.epoch})
	p.qmu.Unlock()
	wake(p.dataWake)
	return nil
}

func wake(ch chan struct{}) {
	select {
	case ch <- struct{}{}:
	default:
	}
}

func (p *BaseProcessor) systemLoop(ctx context.Context) {
	defer p.wg.Done()

	for {
		if ctx.Err() != nil {
			return
		}

		p.qmu.Lock()
		qf, ok := p.sysQ.pop()
		p.qmu.Unlock()

		if !ok {
			select {
			case <-p.sysWake:
				continue
			case <-ctx.Done():
				return
			}
		}

		_ = p.invoke(ctx, qf, false)

		if _, isStart := qf.frame.(*frames.StartFrame); isStart {
			p.qmu.Lock()
			p.pendingStart--
			p.qmu.Unlock()
			wake(p.dataWake)
		}
	}
}

func (p *BaseProcessor) dataLoop(ctx context.Context) {
	defer p.wg.Done()

	for {
		if ctx.Err() != nil {
			return
		}

		p.qmu.Lock()
		var qf queuedFrame
		ok := false
		if p.pendingStart == 0 {
			qf, ok = p.dataQ.pop()
		}
		frameCtx := p.epochCtx
		p.qmu.Unlock()

		if !ok {
			select {
			case <-p.dataWake:
				continue
			case <-ctx.Done():
				return
			}
		}

		if p.legacy {
			frameCtx = ctx
		}
		_ = p.invoke(frameCtx, qf, true)
	}
}

func (p *BaseProcessor) resetForInterruption() {
	p.lifeMu.Lock()
	life := p.ctx
	p.lifeMu.Unlock()

	p.qmu.Lock()
	p.epoch++
	oldCancel := p.epochCancel
	if life != nil {
		p.epochCtx, p.epochCancel = context.WithCancel(life)
	}
	dropped := p.dataQ.retain(func(qf queuedFrame) bool {
		return qf.frame.SurvivesInterruption()
	})
	p.qmu.Unlock()

	if oldCancel != nil {
		oldCancel()
	}
	if dropped > 0 {
		logger.Debug("[%s] Interruption dropped %d queued frames", p.name, dropped)
	}
}

func (p *BaseProcessor) lockHandlerForData() {
	for {
		p.handlerMu.Lock()
		if p.systemWaiting.Load() == 0 {
			return
		}
		p.handlerMu.Unlock()
		runtime.Gosched()
	}
}

func (p *BaseProcessor) invoke(ctx context.Context, qf queuedFrame, checkEpoch bool) (err error) {
	if !p.legacy && !p.direct && !p.concurrentSystem {
		if checkEpoch {
			p.lockHandlerForData()
		} else {
			p.systemWaiting.Add(1)
			p.handlerMu.Lock()
			p.systemWaiting.Add(-1)
		}
		defer p.handlerMu.Unlock()
	}

	if checkEpoch && !qf.frame.SurvivesInterruption() {
		p.qmu.Lock()
		stale := qf.epoch != p.epoch
		p.qmu.Unlock()
		if stale {
			return nil
		}
	}

	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic processing %s: %v", qf.frame.Name(), r)
			logger.Error("[%s] %v\n%s", p.name, err, debug.Stack())
			_ = p.PushError("processor panic", err, false)
		}
	}()

	if start, ok := qf.frame.(*frames.StartFrame); ok {
		p.HandleStartFrame(start)
	}

	err = p.ProcessFrame(ctx, qf.frame, qf.direction)
	if err != nil && !errors.Is(err, context.Canceled) {
		logger.Error("[%s] Error processing %s: %v", p.name, qf.frame.Name(), err)
		if h := p.onError.Load(); h != nil && *h != nil {
			(*h)(p.self(), err, "", 0)
		}
	}
	return err
}

func (p *BaseProcessor) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if ref := p.observer.Load(); ref != nil {
		p.safeObserve(func() { ref.o.OnProcessFrame(p.name, frame, direction) })
	}

	if p.handler != nil {
		return p.handler.HandleFrame(ctx, frame, direction)
	}
	return p.PushFrame(frame, direction)
}

func (p *BaseProcessor) safeObserve(fn func()) {
	defer func() {
		if r := recover(); r != nil {
			logger.Error("[%s] Recovered from observer panic: %v", p.name, r)
		}
	}()
	fn()
}

func (p *BaseProcessor) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	var ref *linkRef
	if direction == frames.Downstream {
		ref = p.next.Load()
	} else {
		ref = p.prev.Load()
	}
	if ref == nil {
		return nil
	}

	if obs := p.observer.Load(); obs != nil {
		p.safeObserve(func() { obs.o.OnPushFrame(p.name, frame, direction) })
	}

	return ref.p.QueueFrame(frame, direction)
}

func (p *BaseProcessor) HandleStartFrame(frame *frames.StartFrame) {
	p.cfgMu.Lock()
	defer p.cfgMu.Unlock()

	if p.startApplied {
		return
	}
	p.startApplied = true
	p.startCfg = StartConfig{
		AllowInterruptions:    frame.AllowInterruptions,
		TurnStrategies:        frame.TurnStrategies,
		AudioInSampleRate:     frame.AudioInSampleRate,
		AudioOutSampleRate:    frame.AudioOutSampleRate,
		EnableMetrics:         frame.EnableMetrics,
		EnableUsageMetrics:    frame.EnableUsageMetrics,
		ReportOnlyInitialTTFB: frame.ReportOnlyInitialTTFB,
	}
}

func (p *BaseProcessor) StartConfig() StartConfig {
	p.cfgMu.RLock()
	defer p.cfgMu.RUnlock()
	return p.startCfg
}

func (p *BaseProcessor) InterruptionsAllowed() bool {
	p.cfgMu.RLock()
	defer p.cfgMu.RUnlock()
	return p.startCfg.AllowInterruptions
}

func (p *BaseProcessor) TurnStrategies() turns.UserTurnStrategies {
	p.cfgMu.RLock()
	defer p.cfgMu.RUnlock()
	return p.startCfg.TurnStrategies
}

func (p *BaseProcessor) InterruptionStrategies() []InterruptionStrategy {
	return nil
}

func (p *BaseProcessor) PushInterruptionTaskFrame() error {
	return p.PushFrame(frames.NewInterruptionTaskFrame(), frames.Upstream)
}

type siblingSetter interface {
	SetBroadcastSiblingID(id string)
}

func (p *BaseProcessor) BroadcastFrame(ctx context.Context, frameConstructor func() frames.Frame) error {
	if frameConstructor == nil {
		return fmt.Errorf("frame constructor cannot be nil")
	}

	down := frameConstructor()
	up := frameConstructor()
	if down == nil || up == nil {
		return fmt.Errorf("frame constructor returned nil frame")
	}

	downSetter, okDown := down.(siblingSetter)
	upSetter, okUp := up.(siblingSetter)
	if !okDown || !okUp {
		return fmt.Errorf("frame %s does not expose BroadcastSiblingID", down.Name())
	}
	downSetter.SetBroadcastSiblingID(strconv.FormatUint(up.ID(), 10))
	upSetter.SetBroadcastSiblingID(strconv.FormatUint(down.ID(), 10))

	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	if err := p.PushFrame(down, frames.Downstream); err != nil {
		return err
	}
	return p.PushFrame(up, frames.Upstream)
}

func (p *BaseProcessor) BroadcastInterruption(ctx context.Context) error {
	return p.BroadcastFrame(ctx, func() frames.Frame {
		return frames.NewInterruptionFrame()
	})
}

func (p *BaseProcessor) HandleInterruptionFrame() {
	p.qmu.Lock()
	p.dataQ.retain(func(qf queuedFrame) bool {
		return qf.frame.SurvivesInterruption()
	})
	p.qmu.Unlock()
}

func (p *BaseProcessor) SetOnError(handler ErrorHandler) {
	p.onError.Store(&handler)
}

func (p *BaseProcessor) PushError(errorMsg string, err error, fatal bool) error {
	_, file, line, ok := runtime.Caller(1)
	if !ok {
		file = "unknown"
	}

	var fullErr error
	if err != nil {
		fullErr = fmt.Errorf("%s: %w", errorMsg, err)
	} else {
		fullErr = errors.New(errorMsg)
	}

	if fatal {
		logger.Error("[%s] FATAL ERROR at %s:%d - %v", p.name, file, line, fullErr)
	} else {
		logger.Error("[%s] Error at %s:%d - %v", p.name, file, line, fullErr)
	}

	if h := p.onError.Load(); h != nil && *h != nil {
		(*h)(p.self(), fullErr, file, line)
	}

	errorFrame := frames.NewErrorFrame(fullErr)
	errorFrame.Fatal = fatal
	errorFrame.Processor = p.name
	if fatal {
		errorFrame.SetMetadata("fatal", true)
	}
	errorFrame.SetMetadata("file", file)
	errorFrame.SetMetadata("line", line)
	errorFrame.SetMetadata("processor", p.name)

	return p.PushFrame(errorFrame, frames.Upstream)
}

func (p *BaseProcessor) PushErrorFrame(errorFrame *frames.ErrorFrame) error {
	_, file, line, ok := runtime.Caller(1)
	if !ok {
		file = "unknown"
	}

	logger.Error("[%s] Error at %s:%d - %v", p.name, file, line, errorFrame.Error)

	if h := p.onError.Load(); h != nil && *h != nil {
		(*h)(p.self(), errorFrame.Error, file, line)
	}
	if errorFrame.Processor == "" {
		errorFrame.Processor = p.name
	}

	return p.PushFrame(errorFrame, frames.Upstream)
}
