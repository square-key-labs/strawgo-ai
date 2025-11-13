# Interruption Support in StrawGo

This document describes how to use `allow_interruptions` and `interruption_strategy` features in StrawGo, similar to Pipecat.

## Overview

Interruptions allow users to interrupt the bot while it's speaking. This is essential for creating natural, conversational AI agents that can be stopped mid-sentence when the user wants to interject.

## Key Concepts

### 1. **Allow Interruptions**
A boolean flag that enables or disables interruption handling in the pipeline.

- `true`: Users can interrupt the bot
- `false`: Interruptions are disabled (default)

### 2. **Interruption Strategies**
Strategies determine *when* a user should be allowed to interrupt the bot. They analyze user input (audio, text, or both) to decide if the interruption should be triggered.

### 3. **Interruption Flow**

```
User speaks -> Strategy checks conditions -> InterruptionTaskFrame (upstream)
                                                      ↓
Pipeline Task receives InterruptionTaskFrame -> Converts to InterruptionFrame
                                                      ↓
                                           InterruptionFrame (downstream)
                                                      ↓
                              All processors clear queues and stop current tasks
```

## Usage

### Basic Configuration

```go
import (
    "github.com/square-key-labs/strawgo-ai/src/interruptions"
    "github.com/square-key-labs/strawgo-ai/src/pipeline"
)

// Configure interruption strategies
config := &pipeline.PipelineTaskConfig{
    AllowInterruptions: true,
    InterruptionStrategies: []interruptions.InterruptionStrategy{
        interruptions.NewMinWordsInterruptionStrategy(3), // Interrupt after 3 words
    },
}

// Create pipeline task with config
task := pipeline.NewPipelineTaskWithConfig(pipe, config)
```

### Default Configuration

```go
// Uses default config: allow_interruptions=true, no strategies
task := pipeline.NewPipelineTask(pipe)
```

## Built-in Interruption Strategies

### MinWordsInterruptionStrategy

Interrupts when the user has spoken at least a minimum number of words.

```go
strategy := interruptions.NewMinWordsInterruptionStrategy(3)
```

**Parameters:**
- `minWords`: Minimum number of words required to trigger an interruption

**Use Case:** Prevent accidental interruptions from brief utterances like "um" or "uh"

## Implementing Custom Strategies

### Strategy Interface

```go
type InterruptionStrategy interface {
    // AppendAudio adds audio data for analysis
    AppendAudio(audio []byte, sampleRate int) error

    // AppendText adds text data for analysis
    AppendText(text string) error

    // ShouldInterrupt determines if interruption should occur
    ShouldInterrupt() (bool, error)

    // Reset clears accumulated data
    Reset() error
}
```

### Example: Custom Time-Based Strategy

```go
type TimeBasedStrategy struct {
    interruptions.BaseInterruptionStrategy
    minDuration time.Duration
    startTime   time.Time
}

func NewTimeBasedStrategy(minDuration time.Duration) *TimeBasedStrategy {
    return &TimeBasedStrategy{
        minDuration: minDuration,
    }
}

func (s *TimeBasedStrategy) AppendText(text string) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    if s.startTime.IsZero() {
        s.startTime = time.Now()
    }
    return nil
}

func (s *TimeBasedStrategy) ShouldInterrupt() (bool, error) {
    s.mu.Lock()
    defer s.mu.Unlock()

    if s.startTime.IsZero() {
        return false, nil
    }

    elapsed := time.Since(s.startTime)
    return elapsed >= s.minDuration, nil
}

func (s *TimeBasedStrategy) Reset() error {
    s.mu.Lock()
    defer s.mu.Unlock()

    s.startTime = time.Time{}
    return nil
}
```

## Using Interruptions in Custom Processors

### Handling StartFrame

Processors must handle `StartFrame` to receive interruption configuration:

```go
func (p *MyProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
    if startFrame, ok := frame.(*frames.StartFrame); ok {
        p.HandleStartFrame(startFrame) // Configures interruption settings
        return p.PushFrame(frame, direction)
    }
    // ... handle other frames
}
```

### Checking Interruption Strategies

```go
// When user stops speaking, check if we should interrupt
if _, ok := frame.(*frames.UserStoppedSpeakingFrame); ok {
    if p.botSpeaking && p.InterruptionsAllowed() {
        shouldInterrupt := false

        // Check all strategies
        for _, strategy := range p.InterruptionStrategies() {
            interrupt, err := strategy.ShouldInterrupt()
            if err != nil {
                log.Printf("Error checking strategy: %v", err)
                continue
            }
            if interrupt {
                shouldInterrupt = true
                break
            }
        }

        if shouldInterrupt {
            // Trigger interruption
            p.PushInterruptionTaskFrame()

            // Reset all strategies
            for _, strategy := range p.InterruptionStrategies() {
                strategy.Reset()
            }
        }
    }
}
```

### Appending Data to Strategies

```go
// Append text from transcription
if textFrame, ok := frame.(*frames.TextFrame); ok {
    for _, strategy := range p.InterruptionStrategies() {
        strategy.AppendText(textFrame.Text)
    }
}

// Append audio data
if audioFrame, ok := frame.(*frames.InputAudioRawFrame); ok {
    for _, strategy := range p.InterruptionStrategies() {
        strategy.AppendAudio(audioFrame.Audio, audioFrame.SampleRate)
    }
}
```

### Handling InterruptionFrame

When an interruption occurs, processors receive an `InterruptionFrame`:

```go
if _, ok := frame.(*frames.InterruptionFrame); ok {
    p.HandleInterruptionFrame() // Clears queues
    // Stop current TTS/processing
    // Reset state
    return p.PushFrame(frame, direction)
}
```

## Frame Types

### InterruptionTaskFrame (Control Frame)
- Pushed **upstream** by processors
- Signals to PipelineTask that an interruption should occur
- Converted to `InterruptionFrame` by PipelineTask

### InterruptionFrame (System Frame)
- Pushed **downstream** by PipelineTask
- Received by all processors
- Triggers queue clearing and state reset

## Helper Methods

### BaseProcessor Methods

```go
// Check if interruptions are enabled
allowed := processor.InterruptionsAllowed()

// Get configured strategies
strategies := processor.InterruptionStrategies()

// Trigger interruption
processor.PushInterruptionTaskFrame()

// Handle interruption (clear queues)
processor.HandleInterruptionFrame()
```

## Best Practices

1. **Always Reset Strategies**: After an interruption is triggered, reset all strategies to clear accumulated data

2. **Track Bot State**: Keep track of whether the bot is speaking to avoid unnecessary interruption checks

3. **Multiple Strategies**: You can use multiple strategies together. If ANY strategy returns `true`, the interruption is triggered:
   ```go
   InterruptionStrategies: []interruptions.InterruptionStrategy{
       interruptions.NewMinWordsInterruptionStrategy(2),
       NewTimeBasedStrategy(500 * time.Millisecond),
   }
   ```

4. **Error Handling**: Always handle errors from strategy methods gracefully

5. **Thread Safety**: Use the `BaseInterruptionStrategy` which provides a mutex for thread-safe operations

## Example

See `examples/interruption_strategies.go` for a complete working example demonstrating:
- Configuring interruption strategies
- Checking interruption conditions
- Triggering interruptions
- Handling InterruptionFrame

## Comparison with Pipecat

StrawGo's interruption system is designed to be compatible with Pipecat's approach:

| Feature | Pipecat | StrawGo |
|---------|---------|---------|
| Configuration | `allow_interruptions` in `PipelineTaskParams` | `AllowInterruptions` in `PipelineTaskConfig` |
| Strategies | `interruption_strategies` list | `InterruptionStrategies` slice |
| Strategy Interface | `BaseInterruptionStrategy` | `InterruptionStrategy` interface |
| Min Words Strategy | `MinWordsInterruptionStrategy` | `MinWordsInterruptionStrategy` |
| Trigger Frame | `InterruptionTaskFrame` | `InterruptionTaskFrame` |
| System Frame | `InterruptionFrame` | `InterruptionFrame` |

## Future Enhancements

Potential additional strategies to implement:

1. **Volume-Based Strategy**: Interrupt based on audio volume/energy
2. **Keyword Strategy**: Interrupt when specific keywords are detected
3. **Sentiment Strategy**: Interrupt based on detected urgency/sentiment
4. **Hybrid Strategies**: Combine multiple conditions (e.g., min words AND min duration)
