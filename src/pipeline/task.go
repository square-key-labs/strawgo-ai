package pipeline

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/interruptions"
)

// PipelineTaskConfig holds configuration for pipeline task
type PipelineTaskConfig struct {
	AllowInterruptions     bool
	InterruptionStrategies []interruptions.InterruptionStrategy
}

// DefaultPipelineTaskConfig returns default configuration
func DefaultPipelineTaskConfig() *PipelineTaskConfig {
	return &PipelineTaskConfig{
		AllowInterruptions:     true,
		InterruptionStrategies: []interruptions.InterruptionStrategy{},
	}
}

// PipelineTask orchestrates the execution of a pipeline
type PipelineTask struct {
	pipeline *Pipeline
	ctx      context.Context
	cancel   context.CancelFunc
	wg       sync.WaitGroup

	// Configuration
	config *PipelineTaskConfig

	// Frame queuing
	userFrameQueue chan frames.Frame

	// Lifecycle tracking
	started  bool
	finished bool
	mu       sync.RWMutex

	// Event handlers
	onStarted  func()
	onFinished func()
	onError    func(error)
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
		userFrameQueue: make(chan frames.Frame, 100),
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

// QueueFrame adds a frame to be processed by the pipeline
func (t *PipelineTask) QueueFrame(frame frames.Frame) error {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if !t.started {
		return fmt.Errorf("pipeline not started")
	}

	if t.finished {
		return fmt.Errorf("pipeline already finished")
	}

	select {
	case t.userFrameQueue <- frame:
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

	log.Printf("[PipelineTask] Starting pipeline")

	// Start the pipeline
	if err := t.pipeline.Start(t.ctx); err != nil {
		return fmt.Errorf("failed to start pipeline: %w", err)
	}

	// Start frame processor
	t.wg.Add(1)
	go t.processUserFrames()

	// Send StartFrame to initialize the pipeline with interruption configuration
	startFrame := frames.NewStartFrameWithConfig(
		t.config.AllowInterruptions,
		t.config.InterruptionStrategies,
	)
	if err := t.pipeline.QueueFrame(startFrame); err != nil {
		return fmt.Errorf("failed to queue start frame: %w", err)
	}

	// Wait for completion
	t.wg.Wait()

	// Stop the pipeline
	if err := t.pipeline.Stop(); err != nil {
		log.Printf("[PipelineTask] Error stopping pipeline: %v", err)
	}

	log.Printf("[PipelineTask] Pipeline finished")
	return nil
}

// Cancel stops the pipeline immediately
func (t *PipelineTask) Cancel() {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.cancel != nil {
		log.Printf("[PipelineTask] Cancelling pipeline")
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
		case frame := <-t.userFrameQueue:
			if err := t.pipeline.QueueFrame(frame); err != nil {
				log.Printf("[PipelineTask] Error queuing user frame: %v", err)
				if t.onError != nil {
					t.onError(err)
				}
			}
		}
	}
}

// handleDownstreamFrame handles frames that reach the sink
func (t *PipelineTask) handleDownstreamFrame(frame frames.Frame) error {
	log.Printf("[PipelineTask] Frame reached sink: %s", frame.Name())

	// Handle lifecycle frames
	switch frame.(type) {
	case *frames.StartFrame:
		log.Printf("[PipelineTask] Pipeline started")
		if t.onStarted != nil {
			t.onStarted()
		}

	case *frames.EndFrame:
		log.Printf("[PipelineTask] End frame reached, finishing pipeline")
		t.markFinished()
		t.Cancel()

	case *frames.CancelFrame:
		log.Printf("[PipelineTask] Cancel frame reached, stopping immediately")
		t.markFinished()
		t.Cancel()

	case *frames.ErrorFrame:
		errorFrame := frame.(*frames.ErrorFrame)
		log.Printf("[PipelineTask] Error frame received: %v", errorFrame.Error)
		if t.onError != nil {
			t.onError(errorFrame.Error)
		}
	}

	return nil
}

// handleUpstreamFrame handles frames going back up the pipeline
func (t *PipelineTask) handleUpstreamFrame(frame frames.Frame) error {
	log.Printf("[PipelineTask] Upstream frame from pipeline: %s", frame.Name())

	// Handle InterruptionTaskFrame - convert to InterruptionFrame and send downstream
	if _, ok := frame.(*frames.InterruptionTaskFrame); ok {
		log.Printf("[PipelineTask] Received InterruptionTaskFrame, sending InterruptionFrame downstream")
		// Send interruption frame downstream to all processors
		if err := t.pipeline.QueueFrame(frames.NewInterruptionFrame()); err != nil {
			log.Printf("[PipelineTask] Error queuing interruption frame: %v", err)
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
