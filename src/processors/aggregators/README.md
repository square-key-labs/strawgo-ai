# LLM Context Aggregators

Aggregators for managing conversation context and enabling intelligent interruptions in StrawGo voice pipelines.

## Overview

This package provides two core aggregators:

- **LLMUserAggregator**: Accumulates user transcriptions and decides when to interrupt
- **LLMAssistantAggregator**: Accumulates bot responses and updates conversation context

## Quick Start

```go
import (
    "github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
    "github.com/square-key-labs/strawgo-ai/src/services"
    "github.com/square-key-labs/strawgo-ai/src/interruptions"
    "github.com/square-key-labs/strawgo-ai/src/pipeline"
)

// 1. Create shared context
llmContext := services.NewLLMContext("You are a helpful assistant.")

// 2. Create aggregators
userAgg := aggregators.NewLLMUserAggregator(llmContext, nil)
assistantAgg := aggregators.NewLLMAssistantAggregator(llmContext, nil)

// 3. Add to pipeline
pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    stt,
    userAgg,        // User input → LLM
    llm,
    tts,
    assistantAgg,   // LLM output → Context
})

// 4. Enable interruptions
config := &pipeline.PipelineTaskConfig{
    AllowInterruptions: true,
    InterruptionStrategies: []interruptions.InterruptionStrategy{
        interruptions.NewMinWordsInterruptionStrategy(3),
    },
}

task := pipeline.NewPipelineTaskWithConfig(pipe, config)
```

## Components

### LLMUserAggregator

Manages user input and interruption decisions.

**Features:**
- Accumulates transcriptions from STT
- Tracks bot speaking state
- Evaluates interruption strategies
- Decides whether to interrupt or discard input
- Outputs LLMContextFrame to trigger LLM

**Configuration:**
```go
params := &aggregators.UserAggregatorParams{
    AggregationTimeout: 500 * time.Millisecond,
}
userAgg := aggregators.NewLLMUserAggregator(context, params)
```

### LLMAssistantAggregator

Manages bot responses and context updates.

**Features:**
- Accumulates LLM response tokens
- Updates conversation context
- Handles interruptions
- Tracks function calls

**Configuration:**
```go
assistantAgg := aggregators.NewLLMAssistantAggregator(context, nil)
```

## Interruption Flow

```
User speaks → STT → TranscriptionFrame
    ↓
UserAggregator accumulates text
    ↓
Bot speaking? YES → Check strategies
    ↓
3+ words? YES → INTERRUPT!
    ↓
Push InterruptionTaskFrame upstream
    ↓
PipelineTask → InterruptionFrame downstream
    ↓
TTS stops + AssistantAgg clears queue
    ↓
Process user input → LLM
```

## Examples

### Basic Usage

See `examples/voice_call_complete.go` for a complete working example.

### Standalone Demo

See `examples/aggregators_with_interruptions.go` for an isolated demonstration.

## Documentation

- **[Complete Guide](../../../docs/AGGREGATORS.md)** - Comprehensive documentation
- **[Implementation Plan](../../../AGGREGATOR_IMPLEMENTATION_PLAN.md)** - Architecture details

## API Reference

### LLMUserAggregator

```go
func NewLLMUserAggregator(
    context *services.LLMContext,
    params *UserAggregatorParams,
) *LLMUserAggregator

func (u *LLMUserAggregator) HandleFrame(
    ctx context.Context,
    frame frames.Frame,
    direction frames.FrameDirection,
) error
```

### LLMAssistantAggregator

```go
func NewLLMAssistantAggregator(
    context *services.LLMContext,
    params *AssistantAggregatorParams,
) *LLMAssistantAggregator

func (a *LLMAssistantAggregator) HandleFrame(
    ctx context.Context,
    frame frames.Frame,
    direction frames.FrameDirection,
) error
```

## Testing

```bash
# Run standalone demo
go run examples/aggregators_with_interruptions.go

# Run voice pipeline
go run examples/voice_call_complete.go
```

## License

Same as StrawGo parent project.

## Version

1.0.0 - Production Ready
