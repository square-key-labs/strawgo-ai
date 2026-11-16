package pipeline

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/turns"
)

type PipelineTaskConfig struct {
	AllowInterruptions      bool
	TurnStrategies          turns.UserTurnStrategies
	AudioInSampleRate       int
	AudioOutSampleRate      int
	EnableMetrics           bool
	EnableUsageMetrics      bool
	ReportOnlyInitialTTFB   bool
	IdleTimeout             time.Duration
	CancelOnIdleTimeout     bool
	CancelTimeout           time.Duration
	HeartbeatInterval       time.Duration
	KeepRunningOnFatal      bool
	ProcessorUnusablePolicy ProcessorUnusablePolicy
}

type ProcessorUnusablePolicy int

const (
	ProcessorUnusableContinue ProcessorUnusablePolicy = iota
	ProcessorUnusableEnd
	ProcessorUnusableCancel
)

func DefaultPipelineTaskConfig() *PipelineTaskConfig {
	return &PipelineTaskConfig{
		AllowInterruptions:  true,
		TurnStrategies:      turns.UserTurnStrategies{},
		AudioInSampleRate:   16000,
		AudioOutSampleRate:  24000,
		CancelOnIdleTimeout: true,
		CancelTimeout:       5 * time.Second,
	}
}

type FrameHandler func(frame frames.Frame)

type frameSubscription struct {
	types []reflect.Type
	fn    FrameHandler
}

func (s frameSubscription) matches(frame frames.Frame) bool {
	if len(s.types) == 0 {
		return true
	}
	t := reflect.TypeOf(frame)
	for _, want := range s.types {
		if t == want {
			return true
		}
	}
	return false
}

type queuedTaskFrame struct {
	frame     frames.Frame
	direction frames.FrameDirection
}

type PipelineTask struct {
	pipeline *Pipeline
	source   *edge
	sink     *edge
	config   *PipelineTaskConfig
	log      *logger.Logger

	mu          sync.Mutex
	ctx         context.Context
	cancel      context.CancelFunc
	started     bool
	finished    bool
	cancelling  bool
	pending     []queuedTaskFrame
	observer    *TaskObserver
	onStarted   func()
	onFinished  func()
	onError     func(error)
	onIdle      func()
	downstream  []frameSubscription
	upstream    []frameSubscription
	done        chan struct{}
	finishOnce  sync.Once
	lastActive  atomic.Int64
	lastBeat    atomic.Int64
	hasSubs     atomic.Bool
	finishCause atomic.Value
	fatalErr    error
	unusable    map[string]struct{}
}

func NewPipelineTask(pipeline *Pipeline) *PipelineTask {
	return NewPipelineTaskWithConfig(pipeline, DefaultPipelineTaskConfig())
}

func NewPipelineTaskWithConfig(pipeline *Pipeline, config *PipelineTaskConfig) *PipelineTask {
	if config == nil {
		config = DefaultPipelineTaskConfig()
	}
	if config.CancelTimeout <= 0 {
		config.CancelTimeout = 5 * time.Second
	}

	t := &PipelineTask{
		pipeline: pipeline,
		config:   config,
		log:      logger.WithPrefix("PipelineTask"),
		done:     make(chan struct{}),
	}

	t.source = newEdge("PipelineTask::Source", func(frame frames.Frame, direction frames.FrameDirection) error {
		if direction == frames.Upstream {
			return t.handleUpstreamFrame(frame)
		}
		return t.source.PushFrame(frame, frames.Downstream)
	})
	t.sink = newEdge("PipelineTask::Sink", func(frame frames.Frame, direction frames.FrameDirection) error {
		if direction == frames.Downstream {
			return t.handleDownstreamFrame(frame)
		}
		return t.sink.PushFrame(frame, frames.Upstream)
	})

	t.source.Link(pipeline)
	pipeline.Link(t.sink)

	return t
}

func (t *PipelineTask) OnStarted(callback func()) {
	t.mu.Lock()
	t.onStarted = callback
	t.mu.Unlock()
}

func (t *PipelineTask) OnFinished(callback func()) {
	t.mu.Lock()
	t.onFinished = callback
	t.mu.Unlock()
}

func (t *PipelineTask) OnError(callback func(error)) {
	t.mu.Lock()
	t.onError = callback
	t.mu.Unlock()
}

func (t *PipelineTask) OnIdleTimeout(callback func()) {
	t.mu.Lock()
	t.onIdle = callback
	t.mu.Unlock()
}

func (t *PipelineTask) OnFrameReachedDownstream(fn FrameHandler, examples ...frames.Frame) {
	t.subscribe(&t.downstream, fn, examples)
}

func (t *PipelineTask) OnFrameReachedUpstream(fn FrameHandler, examples ...frames.Frame) {
	t.subscribe(&t.upstream, fn, examples)
}

func (t *PipelineTask) subscribe(list *[]frameSubscription, fn FrameHandler, examples []frames.Frame) {
	sub := frameSubscription{fn: fn}
	for _, ex := range examples {
		sub.types = append(sub.types, reflect.TypeOf(ex))
	}
	t.mu.Lock()
	*list = append(append([]frameSubscription(nil), *list...), sub)
	t.mu.Unlock()
	t.hasSubs.Store(true)
}

func (t *PipelineTask) SetObserver(observer *TaskObserver) {
	t.mu.Lock()
	t.observer = observer
	t.mu.Unlock()

	if observer == nil {
		t.pipeline.SetObserver(nil)
		return
	}
	t.pipeline.SetObserver(observer)
}

func (t *PipelineTask) Pipeline() *Pipeline {
	return t.pipeline
}

func (t *PipelineTask) HasFinished() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.finished
}

func (t *PipelineTask) Done() <-chan struct{} {
	return t.done
}

func (t *PipelineTask) QueueFrame(frame frames.Frame, direction ...frames.FrameDirection) error {
	dir := frames.Downstream
	if len(direction) > 0 {
		dir = direction[0]
	}

	t.mu.Lock()
	if t.finished {
		t.mu.Unlock()
		return fmt.Errorf("pipeline already finished")
	}
	if !t.started {
		t.pending = append(t.pending, queuedTaskFrame{frame: frame, direction: dir})
		t.mu.Unlock()
		return nil
	}
	t.mu.Unlock()

	return t.inject(frame, dir)
}

func (t *PipelineTask) QueueFrames(list []frames.Frame) error {
	for _, f := range list {
		if err := t.QueueFrame(f); err != nil {
			return err
		}
	}
	return nil
}

func (t *PipelineTask) inject(frame frames.Frame, dir frames.FrameDirection) error {
	if dir == frames.Upstream {
		return t.sink.QueueFrame(frame, frames.Upstream)
	}
	return t.source.QueueFrame(frame, frames.Downstream)
}

func (t *PipelineTask) StopWhenDone() error {
	return t.QueueFrame(frames.NewEndFrame())
}

func (t *PipelineTask) Run(ctx context.Context) error {
	t.mu.Lock()
	if t.started {
		t.mu.Unlock()
		return fmt.Errorf("pipeline already started")
	}
	runCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	t.ctx, t.cancel = runCtx, cancel
	t.mu.Unlock()

	defer cancel()

	if err := t.source.Start(runCtx); err != nil {
		return fmt.Errorf("failed to start pipeline: %w", err)
	}
	if err := t.pipeline.Start(runCtx); err != nil {
		_ = t.source.Stop()
		return fmt.Errorf("failed to start pipeline: %w", err)
	}
	if err := t.sink.Start(runCtx); err != nil {
		_ = t.pipeline.Stop()
		_ = t.source.Stop()
		return fmt.Errorf("failed to start pipeline: %w", err)
	}

	observer := t.getObserver()
	if observer != nil {
		observer.OnPipelineStarted()
		defer func() {
			observer.OnPipelineStopped()
			observer.Close()
		}()
	}

	now := time.Now().UnixNano()
	t.lastActive.Store(now)
	t.lastBeat.Store(now)

	start := frames.NewStartFrameWithConfig(t.config.AllowInterruptions, t.config.TurnStrategies)
	start.AudioInSampleRate = t.config.AudioInSampleRate
	start.AudioOutSampleRate = t.config.AudioOutSampleRate
	start.EnableMetrics = t.config.EnableMetrics
	start.EnableUsageMetrics = t.config.EnableUsageMetrics
	start.ReportOnlyInitialTTFB = t.config.ReportOnlyInitialTTFB
	if err := t.source.QueueFrame(start, frames.Downstream); err != nil {
		_ = t.pipeline.Stop()
		return fmt.Errorf("failed to queue start frame: %w", err)
	}

	t.mu.Lock()
	t.started = true
	pending := t.pending
	t.pending = nil
	wasCancelling := t.cancelling
	t.mu.Unlock()

	for _, item := range pending {
		if err := t.inject(item.frame, item.direction); err != nil {
			t.reportError(err)
		}
	}
	if wasCancelling {
		t.pushCancel("cancelled before start")
	}

	var monitors sync.WaitGroup
	if t.config.IdleTimeout > 0 || t.config.HeartbeatInterval > 0 {
		monitors.Add(1)
		go func() {
			defer monitors.Done()
			t.monitor(runCtx)
		}()
	}

	select {
	case <-t.done:
	case <-ctx.Done():
		t.pushCancel("context cancelled")
		select {
		case <-t.done:
		case <-time.After(t.config.CancelTimeout):
			t.log.Warn("Cancel frame did not reach the end of the pipeline within %v", t.config.CancelTimeout)
			t.finish("cancel timeout")
		}
	}

	cancel()
	monitors.Wait()

	if err := t.pipeline.Stop(); err != nil {
		t.log.Warn("Error stopping pipeline: %v", err)
	}
	_ = t.sink.Stop()
	_ = t.source.Stop()

	t.log.Info("Pipeline finished")
	t.mu.Lock()
	err := t.fatalErr
	t.mu.Unlock()
	return err
}

func (t *PipelineTask) monitor(ctx context.Context) {
	interval := time.Second
	if t.config.HeartbeatInterval > 0 && t.config.HeartbeatInterval < interval {
		interval = t.config.HeartbeatInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	lastSent := time.Now()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.done:
			return
		case now := <-ticker.C:
			if hb := t.config.HeartbeatInterval; hb > 0 {
				if now.Sub(lastSent) >= hb {
					lastSent = now
					_ = t.source.QueueFrame(frames.NewHeartbeatFrame(), frames.Downstream)
				}
				if since := now.Sub(time.Unix(0, t.lastBeat.Load())); since > 10*hb {
					t.log.Warn("No heartbeat reached the end of the pipeline for %v", since.Round(time.Millisecond))
					t.lastBeat.Store(now.UnixNano())
				}
			}
			if idle := t.config.IdleTimeout; idle > 0 {
				if now.Sub(time.Unix(0, t.lastActive.Load())) >= idle {
					t.lastActive.Store(now.UnixNano())
					t.mu.Lock()
					cb := t.onIdle
					t.mu.Unlock()
					if cb != nil {
						cb()
					}
					if t.config.CancelOnIdleTimeout {
						t.log.Warn("Idle timeout after %v, cancelling pipeline", idle)
						t.Cancel()
					}
				}
			}
		}
	}
}

func (t *PipelineTask) getObserver() *TaskObserver {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.observer
}

func (t *PipelineTask) Cancel() {
	t.mu.Lock()
	if t.finished || t.cancelling {
		t.mu.Unlock()
		return
	}
	t.cancelling = true
	started := t.started
	t.mu.Unlock()

	if !started {
		return
	}
	t.pushCancel("cancel requested")
}

func (t *PipelineTask) pushCancel(reason string) {
	t.mu.Lock()
	t.cancelling = true
	t.mu.Unlock()

	t.log.Info("Cancelling pipeline: %s", reason)
	frame := frames.NewCancelFrame()
	frame.Reason = reason
	_ = t.source.QueueFrame(frame, frames.Downstream)

	timeout := t.config.CancelTimeout
	go func() {
		select {
		case <-t.done:
		case <-time.After(timeout):
			t.finish("cancel timeout")
		}
	}()
}

func isActivity(frame frames.Frame) bool {
	switch frame.(type) {
	case *frames.BotStartedSpeakingFrame, *frames.BotStoppedSpeakingFrame,
		*frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame,
		*frames.TTSAudioFrame, *frames.TranscriptionFrame, *frames.LLMTextFrame:
		return true
	}
	return false
}

func (t *PipelineTask) dispatch(list *[]frameSubscription, frame frames.Frame) {
	if !t.hasSubs.Load() {
		return
	}
	t.mu.Lock()
	subs := *list
	t.mu.Unlock()
	for _, sub := range subs {
		if sub.matches(frame) {
			sub.fn(frame)
		}
	}
}

func (t *PipelineTask) handleDownstreamFrame(frame frames.Frame) error {
	if isActivity(frame) {
		t.lastActive.Store(time.Now().UnixNano())
	}
	t.dispatch(&t.downstream, frame)

	switch f := frame.(type) {
	case *frames.StartFrame:
		t.mu.Lock()
		cb := t.onStarted
		t.mu.Unlock()
		if cb != nil {
			cb()
		}
	case *frames.HeartbeatFrame:
		t.lastBeat.Store(time.Now().UnixNano())
	case *frames.EndFrame:
		t.finish("end frame")
	case *frames.StopFrame:
		t.finish("stop frame")
	case *frames.CancelFrame:
		t.finish("cancel frame")
	case *frames.ErrorFrame:
		t.handleError(f)
	}
	return nil
}

func (t *PipelineTask) handleUpstreamFrame(frame frames.Frame) error {
	if isActivity(frame) {
		t.lastActive.Store(time.Now().UnixNano())
	}
	t.dispatch(&t.upstream, frame)

	switch f := frame.(type) {
	case *frames.InterruptionTaskFrame:
		return t.source.QueueFrame(frames.NewInterruptionFrame(), frames.Downstream)
	case *frames.EndTaskFrame:
		end := frames.NewEndFrame()
		end.Reason = f.Reason
		return t.source.QueueFrame(end, frames.Downstream)
	case *frames.CancelTaskFrame:
		t.Cancel()
	case *frames.ErrorFrame:
		t.handleError(f)
	}
	return nil
}

func (t *PipelineTask) handleError(f *frames.ErrorFrame) {
	t.reportError(f.Error)
	if f.Fatal && !t.config.KeepRunningOnFatal {
		t.log.Error("Fatal error from %s, cancelling pipeline: %v", f.Processor, f.Error)
		t.recordFatal(f)
		t.Cancel()
		return
	}
	if f.Fatal || !f.Category.IsPermanent() || !t.firstUnusable(f.Processor) {
		return
	}
	switch t.config.ProcessorUnusablePolicy {
	case ProcessorUnusableEnd:
		t.log.Error("%s can no longer do its job (%s), ending pipeline: %v", f.Processor, f.Category, f.Error)
		_ = t.StopWhenDone()
	case ProcessorUnusableCancel:
		t.log.Error("%s can no longer do its job (%s), cancelling pipeline: %v", f.Processor, f.Category, f.Error)
		t.recordFatal(f)
		t.Cancel()
	default:
		t.log.Error("%s can no longer do its job (%s): %v", f.Processor, f.Category, f.Error)
	}
}

func (t *PipelineTask) firstUnusable(processor string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, seen := t.unusable[processor]; seen {
		return false
	}
	if t.unusable == nil {
		t.unusable = make(map[string]struct{})
	}
	t.unusable[processor] = struct{}{}
	return true
}

type FatalError struct {
	Processor string
	Err       error
}

func (e *FatalError) Error() string {
	return fmt.Sprintf("pipeline task ended by fatal error from %s: %v", e.Processor, e.Err)
}

func (e *FatalError) Unwrap() error {
	return e.Err
}

func (t *PipelineTask) recordFatal(f *frames.ErrorFrame) {
	cause := f.Error
	if cause == nil {
		cause = errors.New("fatal error")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.finished || t.cancelling || t.fatalErr != nil {
		return
	}
	t.fatalErr = &FatalError{Processor: f.Processor, Err: cause}
}

func (t *PipelineTask) reportError(err error) {
	t.mu.Lock()
	cb := t.onError
	t.mu.Unlock()
	if cb != nil {
		cb(err)
	}
}

func (t *PipelineTask) finish(cause string) {
	t.finishOnce.Do(func() {
		t.finishCause.Store(cause)
		t.mu.Lock()
		t.finished = true
		cb := t.onFinished
		t.mu.Unlock()

		t.log.Info("Pipeline finishing: %s", cause)
		if cb != nil {
			cb()
		}
		close(t.done)
	})
}
