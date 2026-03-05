package pipeline

import (
	"context"
	"fmt"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/turns"
)

// PipelineTaskConfig holds configuration for pipeline task
type PipelineTaskConfig struct {
	AllowInterruptions bool
	TurnStrategies     turns.UserTurnStrategies
}

// DefaultPipelineTaskConfig returns default configuration
func DefaultPipelineTaskConfig() *PipelineTaskConfig {
	return &PipelineTaskConfig{
		AllowInterruptions: true,
		TurnStrategies:     turns.UserTurnStrategies{},
	}
}

// PipelineTask orchestrates the execution of a pipeline
type PipelineTask struct {
	pipeline *Pipeline
	ctx      context.Context
	cancel   context.CancelFunc
	wg       sync.WaitGroup
	observer *TaskObserver
	log      *logger.Logger

	// Configuration
	config *PipelineTaskConfig

	// Frame queuing
	userFrameQueue chan userFrameQueueItem

	// Lifecycle tracking
	started  bool
	finished bool
	mu       sync.RWMutex

	// Event handlers
	onStarted  func()
	onFinished func()
	onError    func(error)
}

// userFrameQueueItem wraps a frame with its direction
type userFrameQueueItem struct {
	frame     frames.Frame
	direction frames.FrameDirection
}

// NewPipelineTask creates a new pipeline task with default configuration
func NewPipelineTask(pipeline *Pipeline) *PipelineTask {
	return NewPipelineTaskWithConfig(pipeline, DefaultPipelineTaskConfig())
}

// NewPipelineTaskWithConfig creates a new pipeline task with custom configuration
func NewPipelineTaskWithConfig(pipeline *Pipeline, config *PipelineTaskConfig) *PipelineTask {
	task := &PipelineTask{
		pipeline:       pipeline,
		config:         config,
		userFrameQueue: make(chan userFrameQueueItem, 100),
		log:            logger.WithPrefix("PipelineTask"),
	}

	// Initialize the pipeline with this task
	pipeline.Initialize(task)

	return task
}

// OnStarted sets a callback for when the pipeline starts
func (t *PipelineTask) OnStarted(callback func()) {
	t.onStarted = callback
}

// OnFinished sets a callback for when the pipeline finishes
func (t *PipelineTask) OnFinished(callback func()) {
	t.onFinished = callback
}

// OnError sets a callback for errors
func (t *PipelineTask) OnError(callback func(error)) {
	t.onError = callback
}

func (t *PipelineTask) SetObserver(observer *TaskObserver) {
	t.mu.Lock()
	t.observer = observer
	t.mu.Unlock()

	t.pipeline.SetObserver(observer)
}

// QueueFrame adds a frame to be processed by the pipeline
// direction is optional; defaults to Downstream if not specified
func (t *PipelineTask) QueueFrame(frame frames.Frame, direction ...frames.FrameDirection) error {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if !t.started {
		return fmt.Errorf("pipeline not started")
	}

	if t.finished {
		return fmt.Errorf("pipeline already finished")
	}

	// Default to Downstream if not specified
	dir := frames.Downstream
	if len(direction) > 0 {
		dir = direction[0]
	}

	select {
	case t.userFrameQueue <- userFrameQueueItem{frame: frame, direction: dir}:
		return nil
	case <-t.ctx.Done():
		return t.ctx.Err()
	}
}

// Run starts the pipeline and runs until completion
func (t *PipelineTask) Run(ctx context.Context) error {
	t.mu.Lock()
	if t.started {
		t.mu.Unlock()
		return fmt.Errorf("pipeline already started")
	}
	t.started = true
	t.ctx, t.cancel = context.WithCancel(ctx)
	t.mu.Unlock()

	t.log.Info("Starting pipeline")

	// Start the pipeline
	if err := t.pipeline.Start(t.ctx); err != nil {
		return fmt.Errorf("failed to start pipeline: %w", err)
	}
	if observer := t.getObserver(); observer != nil {
		observer.OnPipelineStarted()
	}
	defer func() {
		if observer := t.getObserver(); observer != nil {
			observer.OnPipelineStopped()
		}
	}()

	// Start frame processor
	t.wg.Add(1)
	go t.processUserFrames()

	// Send StartFrame to initialize the pipeline with interruption configuration
	startFrame := frames.NewStartFrameWithConfig(
		t.config.AllowInterruptions,
		t.config.TurnStrategies,
	)
	if err := t.pipeline.QueueFrame(startFrame); err != nil {
		return fmt.Errorf("failed to queue start frame: %w", err)
	}

	// Wait for completion
	t.wg.Wait()

	// Stop the pipeline
	if err := t.pipeline.Stop(); err != nil {
		t.log.Warn("Error stopping pipeline: %v", err)
	}

	t.log.Info("Pipeline finished")
	return nil
}

func (t *PipelineTask) getObserver() *TaskObserver {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.observer
}

// Cancel stops the pipeline immediately
func (t *PipelineTask) Cancel() {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.cancel != nil {
		t.log.Info("Cancelling pipeline")
		t.cancel()
	}
}

// processUserFrames processes frames queued by the user
func (t *PipelineTask) processUserFrames() {
	defer t.wg.Done()

	for {
		select {
		case <-t.ctx.Done():
			return
		case item := <-t.userFrameQueue:
			var err error
			if item.direction == frames.Upstream {
				// For upstream frames, push from the end of the pipeline going upstream
				err = t.pipeline.QueueFrameUpstream(item.frame)
			} else {
				// For downstream frames (default), use normal downstream routing
				err = t.pipeline.QueueFrame(item.frame)
			}
			if err != nil {
				t.log.Warn("Error queuing user frame: %v", err)
				if t.onError != nil {
					t.onError(err)
				}
			}
		}
	}
}

// handleDownstreamFrame handles frames that reach the sink
func (t *PipelineTask) handleDownstreamFrame(frame frames.Frame) error {
	t.log.Debug("Frame reached sink: %s", frame.Name())

	// Handle lifecycle frames
	switch frame.(type) {
	case *frames.StartFrame:
		t.log.Info("Pipeline started")
		if t.onStarted != nil {
			t.onStarted()
		}

	case *frames.EndFrame:
		t.log.Info("End frame reached, finishing pipeline")
		t.markFinished()
		t.Cancel()

	case *frames.CancelFrame:
		t.log.Info("Cancel frame reached, stopping immediately")
		t.markFinished()
		t.Cancel()

	case *frames.ErrorFrame:
		errorFrame := frame.(*frames.ErrorFrame)
		t.log.Error("Error frame received: %v", errorFrame.Error)
		if t.onError != nil {
			t.onError(errorFrame.Error)
		}
	}

	return nil
}

// handleUpstreamFrame handles frames going back up the pipeline
func (t *PipelineTask) handleUpstreamFrame(frame frames.Frame) error {
	t.log.Debug("Upstream frame from pipeline: %s", frame.Name())

	// Handle InterruptionTaskFrame - convert to InterruptionFrame and send downstream
	if _, ok := frame.(*frames.InterruptionTaskFrame); ok {
		t.log.Warn("InterruptionTaskFrame is deprecated; use BaseProcessor.BroadcastInterruption() instead")
		t.log.Warn("Received InterruptionTaskFrame, sending InterruptionFrame downstream")
		// Send interruption frame downstream to all processors
		if err := t.pipeline.QueueFrame(frames.NewInterruptionFrame()); err != nil {
			t.log.Error("Error queuing interruption frame: %v", err)
			return err
		}
		return nil
	}

	// Handle error frames
	if errorFrame, ok := frame.(*frames.ErrorFrame); ok {
		if t.onError != nil {
			t.onError(errorFrame.Error)
		}
	}

	return nil
}

func (t *PipelineTask) markFinished() {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.finished {
		t.finished = true
		if t.onFinished != nil {
			t.onFinished()
		}
	}
}
