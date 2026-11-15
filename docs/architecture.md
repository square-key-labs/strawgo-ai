# Architecture

This page describes the engine as implemented in `src/frames`, `src/processors`, `src/pipeline`, `src/llm` and `src/services/llmservice`. File references are to the current tree.

## Pieces

- A **frame** (`frames.Frame`) is a unit of data or signalling: audio, text, a transcription, a context update, a lifecycle event.
- A **processor** (`processors.FrameProcessor`) receives frames, acts on them and pushes frames downstream (towards the output transport) or upstream (towards the input).
- A **pipeline** (`pipeline.NewPipeline`) links processors in order. A pipeline is itself a processor, so pipelines nest. `ParallelPipeline`, `SyncParallelPipeline`, `ServiceSwitcher` and `LLMSwitcher` (all in `src/pipeline`) compose branches.
- A **task** (`pipeline.NewPipelineTask` / `NewPipelineTaskWithConfig`) owns one pipeline: it adds a source and a sink edge, sends the `StartFrame`, injects frames from application code, and decides when the run is over.

A typical voice pipeline:

```
session.Input -> [VAD] -> STT -> user aggregator -> LLM -> TTS -> session.Output -> assistant aggregator
```

The assistant aggregator sits after the output transport so that it sees what was actually spoken (`TTSTextFrame`s and the bot speaking frames).

## Frames and categories

Every frame has one of three categories (`frames.FrameCategory`):

| Category | Queue | Examples |
| --- | --- | --- |
| System | system queue, handled ahead of queued data | `StartFrame`, `CancelFrame`, `InterruptionFrame`, `ErrorFrame`, `MetricsFrame`, `UserStartedSpeakingFrame`, `BotStartedSpeakingFrame`, `FunctionCallsStartedFrame`, `FunctionCallCancelFrame`, `CancelTaskFrame` |
| Control | data queue, in order | `EndFrame`, `StopFrame`, `LLMContextFrame`, `LLMFullResponseStartFrame`/`EndFrame`, `TTSStartedFrame`/`TTSStoppedFrame`, `FunctionCallInProgressFrame`, `FunctionCallResultFrame`, `EndTaskFrame`, `LLMMessagesAppendFrame` |
| Data | data queue, in order | `AudioFrame`, `TTSAudioFrame`, `TextFrame`, `LLMTextFrame`, `TranscriptionFrame`, `TTSSpeakFrame`, `LLMRunFrame` |

`frames.IsSystem(f)` tests the category.

Independently of category, a frame either survives an interruption or not (`Frame.SurvivesInterruption()`). Frames built with `NewUninterruptibleBaseFrame` survive: `EndFrame`, `StopFrame`, `EndTaskFrame`, `FunctionCallInProgressFrame`, `FunctionCallResultFrame`, `LLMUpdateSettingsFrame`, `TTSUpdateSettingsFrame`, `STTUpdateSettingsFrame`, `PipelineFlushFrame`, `LLMContextSummaryResultFrame`, the audio-buffer recording start/stop frames, `VolumeFrame`, `VolumeGainFrame`, `BotBackchannelFrame`, and input `AudioFrame`. Any frame can be marked with `SetSurvivesInterruption(true)` before it is queued (the flows manager does this for its context updates).

## Processors and queues

`processors.NewBaseProcessor(name, handler, opts...)` gives a processor two queues and two goroutines (`src/processors/processor.go`):

- the **system queue**, drained by one goroutine, receives system frames;
- the **data queue**, drained by another goroutine, receives control and data frames in order.

Both queues are growable ring buffers: `QueueFrame` never blocks and there is no fixed capacity. A slow processor therefore builds a backlog rather than applying back-pressure to the processor before it.

Data frames are held back while a `StartFrame` is still being handled, so a processor never sees data before its start configuration.

By default `HandleFrame` calls are serialized by a per-processor mutex: a system frame and a data frame are never handled at the same time, so handler state needs no extra locking. Options change this:

| Option | Effect |
| --- | --- |
| `WithConcurrentSystemFrames()` | System and data handlers may run at the same time. The processor guards its own state. Used by `llmservice.Service` and the transport output so that an interruption can be acted on while a data handler is blocked. |
| `WithDirectMode()` | No queues or goroutines; `QueueFrame` calls the handler on the caller's goroutine. Used for the pipeline and task edges. |
| `NewLegacyProcessor` / `WithLegacyConcurrency()` | Pre-rebuild behaviour: no handler mutex, and data handlers get the processor-lifetime context instead of the epoch context. |

A handler must forward every frame it does not consume, including `EndFrame`, `CancelFrame` and frames it does not recognise. `PushFrame(frame, direction)` sends to the next or previous processor. `BroadcastFrame(ctx, ctor)` builds two instances with `ctor` and pushes one each way, linking them through `BroadcastSiblingID`. `PushError(msg, err, fatal)` sends an `ErrorFrame` upstream. A panic in a handler is recovered and turned into a non-fatal `ErrorFrame`.

`p.StartConfig()` returns the configuration carried by the first `StartFrame` (`AllowInterruptions`, `TurnStrategies`, `AudioInSampleRate`, `AudioOutSampleRate`, `EnableMetrics`, `EnableUsageMetrics`, `ReportOnlyInitialTTFB`); later `StartFrame`s are forwarded but not applied. `p.Context()` is the processor-lifetime context, for connections and goroutines that must outlive one frame.

A processor that ends the call when the caller says goodbye:

```go
package main

import (
	"context"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type goodbye struct {
	*processors.BaseProcessor
}

func newGoodbye() *goodbye {
	g := &goodbye{}
	g.BaseProcessor = processors.NewBaseProcessor("Goodbye", g)
	return g
}

func (g *goodbye) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if err := g.PushFrame(frame, direction); err != nil {
		return err
	}
	f, ok := frame.(*frames.TranscriptionFrame)
	if ok && f.IsFinal && strings.Contains(strings.ToLower(f.Text), "goodbye") {
		return g.PushFrame(frames.NewEndTaskFrame("caller said goodbye"), frames.Upstream)
	}
	return nil
}

func main() {
	_ = newGoodbye()
}
```

## Interruptions and the epoch context

An interruption starts when the user aggregator decides a user turn has started while the bot is speaking or a response is pending, interruptions are allowed (`PipelineTaskConfig.AllowInterruptions` and the start strategy's `EnableInterruptions()`), and no mute strategy is active. It then calls `BroadcastInterruption`, which sends an `InterruptionFrame` both ways.

Each processor keeps an **epoch**. When an `InterruptionFrame` is queued on a processor (`QueueFrame`, before it is handled):

1. the epoch is incremented;
2. the context passed to the data handler for the previous epoch is cancelled;
3. queued control and data frames that do not survive interruption are dropped;
4. frames already dequeued but stamped with an older epoch are skipped instead of handled.

So a data handler blocked on network I/O (an LLM stream, an HTTP TTS request, a function-call wait) is released simply by honouring the `ctx` it was given. It does not need to watch for `InterruptionFrame` itself. Long-lived work that must survive interruptions uses `p.Context()` instead.

System frames are handled with the lifetime context. The `InterruptionFrame` is then forwarded like any other frame and reaches the next processor, which repeats the process.

Effects downstream of the user aggregator:

- `llmservice.Service` stops the current generation (its ctx is cancelled) and cancels only the running function calls registered with `CancelOnInterruption: true`. By default a running call keeps going.
- The TTS base (`src/services/tts`) drops queued sentences and buffered audio, asks streaming providers to `Cancel` the open contexts, and emits `TTSStoppedFrame` upstream if it was speaking.
- The output transport discards queued audio of the old epoch, writes the serializer's interruption message (for example a Twilio `clear`), and emits `BotStoppedSpeakingFrame`.
- The assistant aggregator commits whatever text had been spoken so far, then resets.

TTS services still stamp a context id on `TTSStartedFrame` and `TTSAudioFrame`. `processors.NewAudioGuard` still exists, but no example or transport uses it; stale audio is removed by the epoch mechanism above.

## Task lifecycle and shutdown

`PipelineTaskConfig` (`DefaultPipelineTaskConfig()` sets `AllowInterruptions: true`, 16 kHz in, 24 kHz out, `CancelOnIdleTimeout: true`, `CancelTimeout: 5s`):

| Field | Meaning |
| --- | --- |
| `AllowInterruptions`, `TurnStrategies`, `AudioInSampleRate`, `AudioOutSampleRate`, `EnableMetrics`, `EnableUsageMetrics`, `ReportOnlyInitialTTFB` | copied into the `StartFrame` |
| `IdleTimeout`, `CancelOnIdleTimeout` | no activity frame (speech, transcription, LLM text, TTS audio) for `IdleTimeout` calls `OnIdleTimeout` and, if enabled, cancels the task |
| `HeartbeatInterval` | sends `HeartbeatFrame`s and warns when they stop reaching the sink |
| `CancelTimeout` | how long a `CancelFrame` may take to reach the sink before the task finishes anyway |
| `KeepRunningOnFatal` | by default a fatal `ErrorFrame` cancels the task |

`task.Run(ctx)`:

1. starts the source edge, every processor, and the sink edge, under a context derived from `context.WithoutCancel(ctx)`;
2. queues the `StartFrame`, then any frames queued with `task.QueueFrame` before `Run`;
3. waits until the task finishes;
4. stops processors in reverse order and returns.

The task finishes when an `EndFrame`, `StopFrame` or `CancelFrame` reaches the sink.

- **Graceful end**: `task.StopWhenDone()` or `task.QueueFrame(frames.NewEndFrame())` from outside; `frames.NewEndTaskFrame(reason)` pushed upstream from inside the pipeline. The `EndFrame` is a control frame that survives interruption, so it is handled after everything queued before it. The TTS base waits for its audio to drain; the output transport flushes audio, waits for a playback acknowledgement when the serializer supports one, sends the serializer's end message, hangs up through `serializers.Hangupper` if implemented, and closes the session.
- **Cancel**: cancelling the `ctx` passed to `Run`, `task.Cancel()`, a `CancelTaskFrame` pushed upstream, a fatal `ErrorFrame`, or the idle timeout. A `CancelFrame` (system) is sent; if it has not reached the sink within `CancelTimeout`, the task finishes anyway.
- **Client disconnect**: the transport session pushes `ClientDisconnectedFrame` downstream and a `CancelTaskFrame` upstream, unless `TransportParams.OnClientDisconnected` is set, in which case the callback decides.

Task callbacks: `OnStarted`, `OnFinished`, `OnError`, `OnIdleTimeout`, `OnFrameReachedDownstream(fn, examples...)`, `OnFrameReachedUpstream(fn, examples...)`. `task.Done()` is closed when the task finishes. `pipeline.NewRunner(...).Run(ctx, tasks...)` runs several tasks and, with `WithSignalHandling(true)`, cancels them on SIGINT/SIGTERM.

## Conversation context

The conversation is an `*llm.Context` (`src/llm/context.go`): messages (`llm.Message` with typed parts: text, image, audio, file, tool call, tool result, thought), tools and tool choice. It is guarded by a lock and copy-on-write: `Snapshot()` returns an immutable view that a provider can read while the context keeps changing.

**Single-writer rule**: only the context aggregators (`aggregators.NewContextAggregatorPair`) change the context while a pipeline runs.

- The user aggregator appends user turns and applies `LLMMessagesAppendFrame`, `LLMMessagesUpdateFrame`, `LLMMessagesTransformFrame`, `LLMSetToolsFrame` and `LLMSetToolChoiceFrame`.
- The assistant aggregator appends assistant text (what was spoken, or the LLM text when no TTS text exists), thoughts, tool calls and tool results.
- LLM services read a snapshot and never write to the context. Application code changes the context by queueing the frames above instead of calling `Add`/`Set` while the pipeline runs.

The user aggregator pushes an `LLMContextFrame` downstream when a user turn ends. The assistant aggregator pushes one upstream (the user aggregator forwards it downstream) after a function result that should run the LLM, or on `LLMRunFrame`.

## Function calling

```
LLM provider emits llm.ToolCall events
  -> llmservice.Service, after the stream ends:
       broadcast FunctionCallsStartedFrame (system)
       broadcast FunctionCallInProgressFrame per call (uninterruptible control)
       run each registered handler in its own goroutine
       broadcast FunctionCallResultFrame (uninterruptible) or FunctionCallCancelFrame (system)
  -> assistant aggregator:
       on InProgress: add the tool call and a placeholder result to the context
       on Result: replace the placeholder; if the result asks for it, push LLMContextFrame upstream
       on Cancel: replace the placeholder with llm.ToolResultInterrupted
```

- Register handlers with `svc.RegisterFunction(name, handler, llm.FunctionOptions{...})`; `RegisterFunction("", h)` registers a catch-all. A call to an unregistered function produces a result with an error.
- The handler's ctx derives from the service lifetime, not the epoch. It is cancelled on `CancelFrame`, after `FunctionOptions.Timeout` / `llmservice.WithFunctionCallTimeout`, and on interruption only when the function was registered with `llm.FunctionOptions{CancelOnInterruption: true}`.
- Interruptions do not cancel function calls by default (`llm.DefaultFunctionOptions()` has `CancelOnInterruption: false`; Pipecat's default is the opposite). User speech during a running call, including a backchannel, leaves the call running: its result is recorded when it arrives and the LLM reruns once the user has stopped speaking. The realtime function runner (`src/services/realtime`) and `flows.Function` follow the same default.
- When an opted-in call is cancelled, the context records `llm.ToolResultInterrupted`, which tells the model the outcome is unknown, because the side effect may already have happened. The call settles exactly once: a handler that returns after the cancellation has its result dropped by the service, and the assistant aggregator ignores any result for a call that is already cancelled or completed.
- Calls from one response run in parallel unless the service was built with `llmservice.WithSequentialFunctionCalls()`.
- `llm.FunctionResult{Value, RunLLM}`: a nil `RunLLM` means "run the LLM again". Calls from one response are grouped; the LLM reruns once, after the last call in the group completes. The rerun waits while the user is speaking, and until `BotStoppedSpeakingFrame` while the bot is speaking.
- The upstream copies of these frames reach the user aggregator, where `FunctionCallUserMuteStrategy` can mute the user until every call in progress has finished.

See [examples/function_calling](../examples/function_calling/main.go).

## Metrics, observers and tracing

With `EnableMetrics`, services push `frames.MetricsFrame` (system) carrying typed values from `src/frames/metrics.go`: `TTFBMetricsData` and `ProcessingMetricsData` (`TTFAMetricsData`, `TTFATMetricsData` and `TextAggregationMetricsData` are defined, but no service in the tree emits them yet). With `EnableUsageMetrics`: `LLMUsageMetricsData`, `TTSUsageMetricsData` (characters), `STTUsageMetricsData` (audio seconds). Durations are `time.Duration`. `ReportOnlyInitialTTFB` reports TTFB once per service.

Observers implement `pipeline.Observer` (`OnProcessFrame`, `OnPushFrame`, `OnPipelineStarted`, `OnPipelineStopped`). Add them to a `pipeline.TaskObserver` and attach it with `task.SetObserver`. Each observer gets its own goroutine and a 1024-event buffer; when an observer falls behind, events for it are dropped and counted (`TaskObserver.Dropped()`), so a slow observer never blocks the pipeline. Ready-made observers are in `src/pipeline/observers`: `UserBotLatencyObserver`, `TurnTrackingObserver`, `TurnMetricsObserver`, `StartupTimingObserver`, `MetricsLogObserver`, `TranscriptionLogObserver`, `LLMLogObserver`, `DebugLogObserver`.

`src/tracing` turns the same events into OpenTelemetry spans (conversation, turn, and STT/LLM/TTS service spans) through a tracer you supply:

```go
package main

import (
	"log"
	"time"

	"go.opentelemetry.io/otel"

	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/pipeline/observers"
	"github.com/square-key-labs/strawgo-ai/src/tracing"
)

func attachObservers(task *pipeline.PipelineTask, callID string) {
	turnTracker := observers.NewTurnTrackingObserver()
	latency := observers.NewUserBotLatencyObserver()
	latency.OnLatencyMeasured(func(d time.Duration) {
		log.Printf("call %s: user stopped speaking -> bot audio in %v", callID, d)
	})

	obs := pipeline.NewTaskObserver()
	obs.AddObserver(turnTracker)
	obs.AddObserver(latency)
	obs.AddObserver(observers.NewMetricsLogObserver())
	obs.AddObserver(tracing.NewObserver(tracing.Config{
		Tracer:         otel.Tracer(tracing.InstrumentationName),
		ConversationID: callID,
		TurnTracker:    turnTracker,
		LatencyTracker: latency,
	}))
	task.SetObserver(obs)
}

func main() {}
```

Exporter setup (OTLP or otherwise) is up to the application; `src/tracing` only needs a `trace.Tracer`.
