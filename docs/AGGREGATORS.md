# LLM Context Aggregators

## Overview

Aggregators are specialized processors that manage conversation context and enable intelligent interruptions in StrawGo voice pipelines. They accumulate text from various sources (STT, LLM), maintain conversation history, and decide when to interrupt the bot based on configurable strategies.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    LLMContext                                │
│  - Shared conversation state                                 │
│  - System prompt, messages, tools                            │
│  - Updated by both aggregators                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────┴──────────────────┐
        ↓                                      ↓
┌──────────────────┐                  ┌──────────────────┐
│  UserAggregator  │                  │ AssistantAggr.   │
│                  │                  │                  │
│ - Accumulates    │                  │ - Accumulates    │
│   transcriptions │                  │   LLM responses  │
│ - Tracks bot     │                  │ - Updates        │
│   speaking       │                  │   context        │
│ - Checks         │                  │ - Handles        │
│   interruptions  │                  │   interruptions  │
│ - Triggers       │                  │ - Tracks         │
│   interruptions  │                  │   function calls │
└──────────────────┘                  └──────────────────┘
```

### Data Flow

```
1. Audio Input → STT → TranscriptionFrame
2. TranscriptionFrame → UserAggregator → accumulate text
3. UserAggregator checks: bot speaking? strategies satisfied?
4. If interrupt: InterruptionTaskFrame → PipelineTask → InterruptionFrame
5. InterruptionFrame → TTS (stop) + AssistantAggregator (clear)
6. UserAggregator → LLMContextFrame → LLM
7. LLM → TextFrame → AssistantAggregator → accumulate
8. AssistantAggregator → update context on completion
```

---

## User Aggregator

### Purpose

The **LLMUserAggregator** is responsible for:
1. Accumulating user speech transcriptions
2. Tracking whether the bot is currently speaking
3. Evaluating interruption strategies
4. Deciding whether to interrupt or discard user input
5. Converting accumulated text into LLM context

### Key Features

- **Intelligent Interruptions**: Only interrupts when strategies are satisfied
- **Input Discarding**: Discards user input if interruption conditions aren't met
- **Background Task**: Handles timeouts for late transcriptions
- **State Tracking**: Monitors bot speaking state via TTS frames

### Configuration

```go
type UserAggregatorParams struct {
    AggregationTimeout             time.Duration // Default: 500ms
    TurnEmulatedVADTimeout         time.Duration // Default: 800ms
    EnableEmulatedVADInterruptions bool          // Default: false
}

// Create with defaults
userAgg := aggregators.NewLLMUserAggregator(context, nil)

// Or customize
params := &aggregators.UserAggregatorParams{
    AggregationTimeout: 1 * time.Second,
}
userAgg := aggregators.NewLLMUserAggregator(context, params)
```

### Frame Processing

| Input Frame | Action |
|-------------|--------|
| `StartFrame` | Configure interruptions, start background task |
| `EndFrame` | Stop background task, cleanup |
| `TranscriptionFrame` | Accumulate text, feed to strategies |
| `TTSStartedFrame` | Set `botSpeaking = true` |
| `TTSStoppedFrame` | Set `botSpeaking = false` |

| Output Frame | When |
|--------------|------|
| `LLMContextFrame` | When aggregation is pushed (user input processed) |
| `InterruptionTaskFrame` | When interruption conditions met |

### Interruption Decision Logic

```go
func (u *LLMUserAggregator) pushAggregation() error {
    if len(u.aggregation) == 0 {
        return nil
    }

    // If bot is speaking AND we have strategies
    if len(u.InterruptionStrategies()) > 0 && u.botSpeaking {
        shouldInterrupt := u.shouldInterruptBasedOnStrategies()

        if shouldInterrupt {
            // INTERRUPT: Push interruption frame and process input
            u.PushInterruptionTaskFrameAndWait()
            return u.processAggregation()
        } else {
            // DON'T INTERRUPT: Discard user input
            return u.Reset()
        }
    }

    // Bot not speaking or no strategies - always process
    return u.processAggregation()
}
```

### Example Usage

```go
// Create shared context
llmContext := services.NewLLMContext("You are a helpful assistant.")

// Create user aggregator
userAgg := aggregators.NewLLMUserAggregator(llmContext, nil)

// Add to pipeline
pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    transport.Input(),
    deepgramSTT,
    userAgg,        // ← UserAggregator here
    openaiLLM,
    elevenLabsTTS,
    transport.Output(),
})
```

---

## Assistant Aggregator

### Purpose

The **LLMAssistantAggregator** is responsible for:
1. Accumulating LLM response tokens
2. Updating conversation context when responses complete
3. Handling interruptions by clearing queued responses
4. Tracking function call execution

### Key Features

- **Response Accumulation**: Collects streaming LLM tokens
- **Context Updates**: Adds complete responses to conversation history
- **Interruption Handling**: Clears queue when interrupted
- **Function Call Support**: Tracks parallel function executions

### Configuration

```go
type AssistantAggregatorParams struct {
    // Currently no specific params needed
}

// Create with defaults
assistantAgg := aggregators.NewLLMAssistantAggregator(context, nil)
```

### Frame Processing

| Input Frame | Action |
|-------------|--------|
| `InterruptionFrame` | Clear aggregation, reset state |
| `LLMFullResponseStartFrame` | Increment nesting counter |
| `LLMFullResponseEndFrame` | Decrement counter, push aggregation |
| `TextFrame` | Accumulate if response active |
| `LLMTextFrame` | Accumulate if response active (legacy) |
| `FunctionCallInProgressFrame` | Track function call, update context |
| `FunctionCallResultFrame` | Update result, potentially trigger LLM |

| Output Frame | When |
|--------------|------|
| `LLMContextFrame` | After aggregation pushed (response complete) |

### Example Usage

```go
// Create shared context (same one as user aggregator)
llmContext := services.NewLLMContext("You are a helpful assistant.")

// Create assistant aggregator
assistantAgg := aggregators.NewLLMAssistantAggregator(llmContext, nil)

// Add to pipeline (at the END, after output transport)
pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    transport.Input(),
    deepgramSTT,
    userAgg,
    openaiLLM,
    elevenLabsTTS,
    transport.Output(),
    assistantAgg,    // ← AssistantAggregator here (LAST)
})
```

---

## LLM Context

### Structure

```go
type LLMContext struct {
    Messages     []LLMMessage
    SystemPrompt string
    Model        string
    Temperature  float64
    Tools        []Tool      // Available functions
    ToolChoice   interface{} // "auto", "none", "required", or specific function
}

type LLMMessage struct {
    Role       string     // "user", "assistant", "system", "tool"
    Content    string
    ToolCalls  []ToolCall // For assistant messages with function calls
    ToolCallID string     // For tool response messages
}
```

### Methods

```go
// Basic message management
context.AddUserMessage("Hello")
context.AddAssistantMessage("Hi there!")
context.AddSystemMessage("Be concise")
context.Clear()

// Tool/function support
context.AddMessageWithToolCalls(toolCalls)
context.AddToolMessage(toolCallID, result)
context.SetTools(tools)
context.SetToolChoice("auto")

// Cloning
clonedContext := context.Clone()
```

### Example: Complete Setup

```go
// Create context with system prompt
llmContext := services.NewLLMContext(`You are a helpful voice assistant.
Keep responses brief and conversational.`)

// Optionally set temperature/model
llmContext.Temperature = 0.7
llmContext.Model = "gpt-4-turbo-preview"

// Add tools (optional)
llmContext.SetTools([]services.Tool{
    {
        Type: "function",
        Function: services.ToolFunction{
            Name:        "get_weather",
            Description: "Get the current weather in a location",
            Parameters: map[string]interface{}{
                "type": "object",
                "properties": map[string]interface{}{
                    "location": map[string]interface{}{
                        "type":        "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": []string{"location"},
            },
        },
    },
})
```

---

## Complete Pipeline Integration

### Minimal Setup (Interruptions Enabled)

```go
package main

import (
    "context"
    "github.com/square-key-labs/strawgo-ai/src/interruptions"
    "github.com/square-key-labs/strawgo-ai/src/pipeline"
    "github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
    "github.com/square-key-labs/strawgo-ai/src/services"
    // ... other imports
)

func main() {
    // 1. Create shared LLM context
    llmContext := services.NewLLMContext("You are a helpful assistant.")

    // 2. Create aggregators
    userAgg := aggregators.NewLLMUserAggregator(llmContext, nil)
    assistantAgg := aggregators.NewLLMAssistantAggregator(llmContext, nil)

    // 3. Create services
    stt := deepgram.NewSTTService(/* config */)
    llm := openai.NewLLMService(/* config */)
    tts := elevenlabs.NewTTSService(/* config */)

    // 4. Build pipeline with aggregators
    pipe := pipeline.NewPipeline([]processors.FrameProcessor{
        transport.Input(),
        stt,
        userAgg,        // ← User aggregator (enables interruptions)
        llm,
        tts,
        transport.Output(),
        assistantAgg,   // ← Assistant aggregator (updates context)
    })

    // 5. Configure interruptions
    config := &pipeline.PipelineTaskConfig{
        AllowInterruptions: true,
        InterruptionStrategies: []interruptions.InterruptionStrategy{
            interruptions.NewMinWordsInterruptionStrategy(3),
        },
    }

    // 6. Create and run task
    task := pipeline.NewPipelineTaskWithConfig(pipe, config)
    task.Run(context.Background())
}
```

---

## Interruption Strategies

### MinWordsInterruptionStrategy

Interrupts when the user speaks a minimum number of words while the bot is speaking.

```go
// Interrupt after 3 words
strategy := interruptions.NewMinWordsInterruptionStrategy(3)

// Interrupt after 5 words
strategy := interruptions.NewMinWordsInterruptionStrategy(5)
```

**Example:**
- Bot is speaking
- User says: "Hey" (1 word) → No interruption
- User says: "Hey wait" (2 words) → No interruption
- User says: "Hey wait stop" (3 words) → **INTERRUPTION!**

### Custom Strategies

You can create custom interruption strategies by implementing the `InterruptionStrategy` interface:

```go
type InterruptionStrategy interface {
    AppendText(text string) error
    AppendAudio(audio []byte, sampleRate int) error
    ShouldInterrupt() (bool, error)
    Reset() error
}
```

**Example: Sentiment-Based Strategy**

```go
type SentimentInterruptionStrategy struct {
    accumulatedText string
    threshold       float64
}

func (s *SentimentInterruptionStrategy) AppendText(text string) error {
    s.accumulatedText += " " + text
    return nil
}

func (s *SentimentInterruptionStrategy) ShouldInterrupt() (bool, error) {
    // Analyze sentiment of accumulated text
    sentiment := analyzeSentiment(s.accumulatedText)

    // Interrupt if negative sentiment above threshold
    return sentiment < s.threshold, nil
}

func (s *SentimentInterruptionStrategy) Reset() error {
    s.accumulatedText = ""
    return nil
}
```

---

## Debugging and Logging

### Key Log Messages

When debugging interruptions, look for these log messages:

**UserAggregator:**
```
[LLMUserAggregator] Interruptions: allowed=true, strategies=1
[LLMUserAggregator] Bot started speaking
[LLMUserAggregator] Transcription (final=true): 'Hey wait stop'
[LLMUserAggregator] pushAggregation called: bot_speaking=true, has_strategies=true
[LLMUserAggregator] 🔴 Interruption conditions MET - triggering interruption
[LLMUserAggregator] ⚪ Interruption conditions NOT met - discarding input
```

**MinWordsStrategy:**
```
[MinWordsStrategy] should_interrupt=true num_spoken_words=3 min_words=3
```

**PipelineTask:**
```
[PipelineTask] Received InterruptionTaskFrame, sending InterruptionFrame downstream
```

**AssistantAggregator:**
```
[LLMAssistantAggregator] ⚠️  Interruption received - clearing aggregation
```

### Troubleshooting

**Problem: Interruptions not triggering**

Check:
1. ✅ `AllowInterruptions: true` in config?
2. ✅ Strategies configured in config?
3. ✅ UserAggregator in pipeline?
4. ✅ TTS emitting TTSStarted/StoppedFrame?
5. ✅ Bot actually speaking when user interrupts?

**Problem: All input discarded**

Check:
1. ✅ Is bot always marked as speaking?
2. ✅ Are strategies too strict (e.g., requiring too many words)?
3. ✅ Check logs for "Interruption conditions NOT met"

**Problem: Context not persisting**

Check:
1. ✅ Same LLMContext shared by both aggregators?
2. ✅ AssistantAggregator in pipeline?
3. ✅ AssistantAggregator at END of pipeline?

---

## Best Practices

### 1. Always Use Both Aggregators

```go
// ✅ CORRECT
userAgg := aggregators.NewLLMUserAggregator(context, nil)
assistantAgg := aggregators.NewLLMAssistantAggregator(context, nil)

pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    stt,
    userAgg,       // User input handling
    llm,
    tts,
    output,
    assistantAgg,  // Response handling
})
```

```go
// ❌ INCORRECT (missing assistant aggregator)
pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    stt,
    userAgg,
    llm,
    tts,
    output,
    // Missing assistantAgg - context won't update!
})
```

### 2. Share LLMContext

```go
// ✅ CORRECT - same context
llmContext := services.NewLLMContext("...")
userAgg := aggregators.NewLLMUserAggregator(llmContext, nil)
assistantAgg := aggregators.NewLLMAssistantAggregator(llmContext, nil)
```

```go
// ❌ INCORRECT - different contexts
userContext := services.NewLLMContext("...")
assistantContext := services.NewLLMContext("...")
userAgg := aggregators.NewLLMUserAggregator(userContext, nil)
assistantAgg := aggregators.NewLLMAssistantAggregator(assistantContext, nil)
```

### 3. Configure Reasonable Interruption Thresholds

```go
// ✅ GOOD - 3 words is reasonable
interruptions.NewMinWordsInterruptionStrategy(3)

// ⚠️  TOO STRICT - users will get frustrated
interruptions.NewMinWordsInterruptionStrategy(10)

// ⚠️  TOO LOOSE - every utterance interrupts
interruptions.NewMinWordsInterruptionStrategy(1)
```

### 4. Place AssistantAggregator Last

```go
// ✅ CORRECT - assistant aggregator at end
pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    input,
    stt,
    userAgg,
    llm,
    tts,
    output,
    assistantAgg,  // LAST - sees all frames
})
```

### 5. Monitor Logs During Development

Enable verbose logging to understand interruption decisions:

```go
log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)
```

---

## Advanced Topics

### Multi-Strategy Interruptions

Combine multiple strategies for intelligent interruptions:

```go
config := &pipeline.PipelineTaskConfig{
    AllowInterruptions: true,
    InterruptionStrategies: []interruptions.InterruptionStrategy{
        interruptions.NewMinWordsInterruptionStrategy(3),
        // Custom strategies:
        NewSentimentInterruptionStrategy(-0.5),
        NewUrgencyInterruptionStrategy(0.8),
    },
}
```

**Behavior**: Interrupts if **ANY** strategy returns true.

### Function Calling with Aggregators

```go
// Define tools
llmContext.SetTools([]services.Tool{
    {
        Type: "function",
        Function: services.ToolFunction{
            Name:        "get_current_time",
            Description: "Get the current time",
            Parameters:  map[string]interface{}{"type": "object"},
        },
    },
})

// OpenAI will emit FunctionCallInProgressFrame
// AssistantAggregator will track it
// Your code should emit FunctionCallResultFrame with the result
```

### Customizing Aggregation Timeouts

```go
params := &aggregators.UserAggregatorParams{
    // Wait 1 second for late transcriptions
    AggregationTimeout: 1 * time.Second,
}

userAgg := aggregators.NewLLMUserAggregator(context, params)
```

---

## Performance Considerations

### Memory Usage

- **Context Size**: LLMContext grows with conversation length
- **Recommendation**: Clear old messages periodically for long conversations

```go
// Keep only last 20 messages
if len(llmContext.Messages) > 20 {
    llmContext.Messages = llmContext.Messages[len(llmContext.Messages)-20:]
}
```

### Latency

- **Interruption Detection**: < 10ms overhead
- **Aggregation**: Minimal overhead (string concatenation)
- **Background Task**: 500ms polling interval (configurable)

### Concurrency

- **Thread-Safe**: Context is shared but accessed sequentially
- **Background Task**: Runs in separate goroutine
- **No Locks**: Frame-based synchronization (no mutexes needed)

---

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

func (u *LLMUserAggregator) Reset() error
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

func (a *LLMAssistantAggregator) Reset() error
```

### LLMContext

```go
func NewLLMContext(systemPrompt string) *LLMContext

func (c *LLMContext) AddUserMessage(content string)
func (c *LLMContext) AddAssistantMessage(content string)
func (c *LLMContext) AddSystemMessage(content string)
func (c *LLMContext) AddMessageWithToolCalls(toolCalls []ToolCall)
func (c *LLMContext) AddToolMessage(toolCallID, content string)
func (c *LLMContext) SetTools(tools []Tool)
func (c *LLMContext) SetToolChoice(choice interface{})
func (c *LLMContext) Clear()
func (c *LLMContext) Clone() *LLMContext
```

---

## Migration Guide

### From Old Pipeline (No Aggregators)

**Old Code:**
```go
openaiLLM := openai.NewLLMService(openai.LLMConfig{
    SystemPrompt: "You are helpful",
})

pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    stt,
    openaiLLM,
    tts,
})

task := pipeline.NewPipelineTask(pipe)
```

**New Code (With Aggregators):**
```go
// Create shared context
llmContext := services.NewLLMContext("You are helpful")

// Create aggregators
userAgg := aggregators.NewLLMUserAggregator(llmContext, nil)
assistantAgg := aggregators.NewLLMAssistantAggregator(llmContext, nil)

openaiLLM := openai.NewLLMService(openai.LLMConfig{})

pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    stt,
    userAgg,        // NEW
    openaiLLM,
    tts,
    assistantAgg,   // NEW
})

// Enable interruptions
config := &pipeline.PipelineTaskConfig{
    AllowInterruptions: true,
    InterruptionStrategies: []interruptions.InterruptionStrategy{
        interruptions.NewMinWordsInterruptionStrategy(3),
    },
}

task := pipeline.NewPipelineTaskWithConfig(pipe, config)
```

---

## Examples

See:
- `examples/voice_call_complete.go` - Complete voice pipeline with interruptions
- `examples/aggregators_with_interruptions.go` - Standalone aggregator example

---

## References

- [Interruption Strategies](./INTERRUPTIONS.md)
- [Pipeline Architecture](./PIPELINE.md)
- [Frame Types](./FRAMES.md)
- [Pipecat Reference](../.local_context/pipecat/processors/aggregators/)

---

**Last Updated**: 2025-11-14
**Version**: 1.0.0
**Status**: Production Ready
