package pipeline

import (
	"context"
	"fmt"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

// PipelineSource is the entry point for frames into the pipeline
type PipelineSource struct {
	*processors.BaseProcessor
	task *PipelineTask
}

func newPipelineSource(task *PipelineTask) *PipelineSource {
	ps := &PipelineSource{
		task: task,
	}
	ps.BaseProcessor = processors.NewBaseProcessor("PipelineSource", ps)
	return ps
}

func (p *PipelineSource) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if direction == frames.Upstream {
		// Frames going upstream from the pipeline go to the task
		if p.task != nil {
			return p.task.handleUpstreamFrame(frame)
		}
		return nil
	}

	// Downstream frames just pass through
	return p.PushFrame(frame, direction)
}

// PipelineSink is the exit point for frames from the pipeline
type PipelineSink struct {
	*processors.BaseProcessor
	task *PipelineTask
}

func newPipelineSink(task *PipelineTask) *PipelineSink {
	ps := &PipelineSink{
		task: task,
	}
	ps.BaseProcessor = processors.NewBaseProcessor("PipelineSink", ps)
	return ps
}

func (p *PipelineSink) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if direction == frames.Downstream {
		// Frames reaching the end of the pipeline are handled by the task
		if p.task != nil {
			return p.task.handleDownstreamFrame(frame)
		}
		return nil
	}

	// Upstream frames pass back through
	return p.PushFrame(frame, direction)
}

// Pipeline connects multiple processors in a linear chain
type Pipeline struct {
	processors []processors.FrameProcessor
	source     *PipelineSource
	sink       *PipelineSink
}

// NewPipeline creates a new pipeline with the given processors
func NewPipeline(procs []processors.FrameProcessor) *Pipeline {
	p := &Pipeline{
		processors: procs,
	}
	return p
}

// Initialize sets up the pipeline with source and sink
func (p *Pipeline) Initialize(task *PipelineTask) error {
	p.source = newPipelineSource(task)
	p.sink = newPipelineSink(task)

	// Build the chain: source -> processors -> sink
	chain := []processors.FrameProcessor{p.source}
	chain = append(chain, p.processors...)
	chain = append(chain, p.sink)

	// Link all processors
	for i := 0; i < len(chain)-1; i++ {
		chain[i].Link(chain[i+1])
	}

	logger.Info("[Pipeline] Initialized with %d processors", len(p.processors))
	logger.Debug("[Pipeline] Processor chain: source -> %d processors -> sink", len(p.processors))
	return nil
}

// Start begins processing in all processors
func (p *Pipeline) Start(ctx context.Context) error {
	// Start source
	if err := p.source.Start(ctx); err != nil {
		return fmt.Errorf("failed to start source: %w", err)
	}

	// Start all user processors
	for _, proc := range p.processors {
		if err := proc.Start(ctx); err != nil {
			return fmt.Errorf("failed to start processor %s: %w", proc.Name(), err)
		}
	}

	// Start sink
	if err := p.sink.Start(ctx); err != nil {
		return fmt.Errorf("failed to start sink: %w", err)
	}

	logger.Info("[Pipeline] Started all processors")
	logger.Debug("[Pipeline] Pipeline is running and ready to process frames")
	return nil
}

// Stop gracefully stops all processors
func (p *Pipeline) Stop() error {
	logger.Debug("[Pipeline] Beginning graceful shutdown")

	// Stop in reverse order
	if err := p.sink.Stop(); err != nil {
		logger.Error("[Pipeline] Error stopping sink: %v", err)
	}

	for i := len(p.processors) - 1; i >= 0; i-- {
		if err := p.processors[i].Stop(); err != nil {
			logger.Error("[Pipeline] Error stopping processor %s: %v", p.processors[i].Name(), err)
		}
	}

	if err := p.source.Stop(); err != nil {
		logger.Error("[Pipeline] Error stopping source: %v", err)
	}

	logger.Info("[Pipeline] Stopped all processors")
	return nil
}

// QueueFrame queues a frame at the source of the pipeline
func (p *Pipeline) QueueFrame(frame frames.Frame) error {
	return p.source.QueueFrame(frame, frames.Downstream)
}
