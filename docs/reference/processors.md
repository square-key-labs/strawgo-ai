# Processors reference

This page documents the `processors` package: the `FrameProcessor` and `ProcessHandler` interfaces, `BaseProcessor` and its concurrency modes, the rules for writing a correct processor, and the built-in processors.

A **frame** is one typed message in the pipeline (audio, text, a control signal). A **processor** is one stage that receives frames, acts on them and pushes them on. A **pipeline** is the ordered chain of processors. **Downstream** means toward the output; **upstream** means back toward the input. An **interruption** happens when the user talks over the bot and pending bot output is thrown away.

Import path: `github.com/square-key-labs/strawgo-ai/src/processors`.

## Interfaces

```text
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
```

`FrameProcessor` is what a pipeline links together. You almost never implement it from scratch. Instead you write a struct that embeds `*processors.BaseProcessor` (which implements `FrameProcessor`) and implements `ProcessHandler`. `BaseProcessor` owns the queues and goroutines and calls your `HandleFrame` for each frame.

| Method | What it does |
| --- | --- |
| `QueueFrame` | Called by the previous (or next) processor to hand you a frame. Returns at once in queued modes. |
| `ProcessFrame` | Runs one frame: notifies the observer, then calls `HandleFrame` (or forwards if there is no handler). |
| `PushFrame` | Sends a frame to the next processor (downstream) or the previous one (upstream). |
| `Link`, `SetPrev` | Wire neighbours. `pipeline.NewPipeline` calls these for you. |
| `Start`, `Stop` | Start and stop the processor's goroutines. The task calls these. |

Other interfaces in the package:

| Interface | Methods | Use |
| --- | --- | --- |
| `FrameObserver` | `OnProcessFrame(processorName string, frame frames.Frame, direction frames.FrameDirection)`, `OnPushFrame(...)` | Low-level hook that `pipeline.TaskObserver` implements. Use [`pipeline.Observer`](observers.md) instead. |
| `ObserverAwareProcessor` | `SetObserver(observer FrameObserver)` | Implemented by `BaseProcessor`; the pipeline uses it to attach observers. |
| `InterruptionStrategy` | `AppendAudio`, `AppendText`, `ShouldInterrupt`, `Reset` | Declared for parity. `BaseProcessor.InterruptionStrategies()` returns nil and nothing calls it. |
| `TextCarrier` | `GetText() string`, `SetText(text string)` | Lets your own frame types work with `TextTransformProcessor`. |
| `ErrorHandler` | `func(processor FrameProcessor, err error, file string, line int)` | Callback type for `SetOnError`. |

This frame type implements `TextCarrier`, so `TextTransformProcessor` rewrites it like the built-in text frames:

```go
type NoteFrame struct {
	*frames.DataFrame
	Text string
}

func (f *NoteFrame) GetText() string     { return f.Text }
func (f *NoteFrame) SetText(text string) { f.Text = text }

var _ processors.TextCarrier = (*NoteFrame)(nil)
```

## Constructors and options

```text
func NewBaseProcessor(name string, handler ProcessHandler, opts ...Option) *BaseProcessor
func NewLegacyProcessor(name string, handler ProcessHandler) *BaseProcessor
func WithDirectMode() Option
func WithConcurrentSystemFrames() Option
func WithLegacyConcurrency() Option
```

`name` is what logs, observers, metrics and `ProcessorRef` matching use. `handler` is usually the struct that embeds the `BaseProcessor`. A nil handler forwards every frame unchanged. `NewLegacyProcessor(name, h)` is `NewBaseProcessor(name, h, WithLegacyConcurrency())`.

| Option | Default | Effect |
| --- | --- | --- |
| none | yes | Queued mode, described below. |
| `WithDirectMode()` | off | No queues and no goroutines. `QueueFrame` runs `HandleFrame` on the caller's goroutine. |
| `WithConcurrentSystemFrames()` | off | Queued mode, but a system frame may run while a data frame's handler is still running. |
| `WithLegacyConcurrency()` | off | Queued mode with no handler lock, and data handlers get a context that interruptions do not cancel. |

Pass options after the handler. This processor lets system frames, such as an interruption, run while a slow data frame is still being handled:

```go
type meter struct {
	*processors.BaseProcessor
}

func newMeter() *meter {
	m := &meter{}
	m.BaseProcessor = processors.NewBaseProcessor("Meter", m, processors.WithConcurrentSystemFrames())
	return m
}

func (m *meter) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return m.PushFrame(frame, direction)
}
```

### Guarantees of each mode

| Guarantee | Default | Concurrent system frames | Direct | Legacy |
| --- | --- | --- | --- | --- |
| Goroutines per processor | 2 (system loop, data loop) | 2 | 0 | 2 |
| `HandleFrame` calls never overlap | yes | no: a system frame can run during a data frame | no lock: depends on callers | no |
| System frames overtake queued data and control frames | yes | yes | no queue | yes |
| Data frames wait until a pending `StartFrame` is handled | yes | yes | no | yes |
| Interruption drops queued frames that do not survive it | yes | yes | no | yes |
| Interruption cancels the `ctx` of the running data handler | yes | yes | no | no |
| Panic in `HandleFrame` is recovered and reported as a non-fatal `ErrorFrame` | yes | yes | yes | yes |
| `QueueFrame` after `Stop` returns `context.Canceled` | yes | yes | yes | yes |

The default mode is the right choice for almost every processor: your handler behaves as if it were single-threaded. Built-in users of the other modes: filters and pipeline edges use direct mode; `llmservice.Service` and the WebSocket output transport use concurrent system frames; `SyncParallelPipeline` uses legacy.

> [!NOTE]
> In the default mode, the `ctx` passed to `HandleFrame` for a data or control frame is cancelled when an interruption arrives. Pass it to network calls so they stop when the user barges in. For system frames, `ctx` is the processor's lifetime context.

## `BaseProcessor` methods

| Method | Signature | Description |
| --- | --- | --- |
| `Name` | `func (p *BaseProcessor) Name() string` | The name given to the constructor. |
| `PushFrame` | `func (p *BaseProcessor) PushFrame(frame frames.Frame, direction frames.FrameDirection) error` | Queue a frame on the neighbour in `direction`. Returns nil and drops the frame if there is no neighbour. |
| `QueueFrame` | `func (p *BaseProcessor) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error` | Accept a frame. Errors on a nil frame or after `Stop`. |
| `ProcessFrame` | `func (p *BaseProcessor) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error` | Notify the observer, then run the handler. |
| `BroadcastFrame` | `func (p *BaseProcessor) BroadcastFrame(ctx context.Context, frameConstructor func() frames.Frame) error` | Build two frames, link them as siblings, push one downstream and one upstream. `ctx` may be nil. |
| `BroadcastInterruption` | `func (p *BaseProcessor) BroadcastInterruption(ctx context.Context) error` | `BroadcastFrame` with `frames.NewInterruptionFrame`. |
| `PushInterruptionTaskFrame` | `func (p *BaseProcessor) PushInterruptionTaskFrame() error` | Push an `InterruptionTaskFrame` upstream; the task then sends an `InterruptionFrame` through the whole pipeline. |
| `PushError` | `func (p *BaseProcessor) PushError(errorMsg string, err error, fatal bool) error` | Log, call the `SetOnError` handler, and push an `ErrorFrame` upstream. See [errors](errors.md). |
| `PushErrorFrame` | `func (p *BaseProcessor) PushErrorFrame(errorFrame *frames.ErrorFrame) error` | Push a prepared `ErrorFrame` upstream; fills `Processor` if empty. |
| `SetOnError` | `func (p *BaseProcessor) SetOnError(handler ErrorHandler)` | Called for every `PushError`, `PushErrorFrame` and every non-nil error returned by `HandleFrame` (except `context.Canceled`). |
| `Context` | `func (p *BaseProcessor) Context() context.Context` | The processor's lifetime context; `context.Background()` before `Start`. Use it for goroutines you start. |
| `StartConfig` | `func (p *BaseProcessor) StartConfig() StartConfig` | The values from the first `StartFrame`. |
| `InterruptionsAllowed` | `func (p *BaseProcessor) InterruptionsAllowed() bool` | `StartConfig().AllowInterruptions`. |
| `TurnStrategies` | `func (p *BaseProcessor) TurnStrategies() turns.UserTurnStrategies` | `StartConfig().TurnStrategies`. |
| `HandleStartFrame` | `func (p *BaseProcessor) HandleStartFrame(frame *frames.StartFrame)` | Records the start config. Called automatically before your handler sees a `StartFrame`. |
| `HandleInterruptionFrame` | `func (p *BaseProcessor) HandleInterruptionFrame()` | Drops queued frames that do not survive interruption. Queued modes already do this when the `InterruptionFrame` is queued. |
| `QueueLen` | `func (p *BaseProcessor) QueueLen() (system, data int)` | Current queue depths. |
| `Next`, `Prev` | `func (p *BaseProcessor) Next() FrameProcessor` | The linked neighbours, or nil. |
| `Link`, `SetPrev`, `SetObserver`, `Start`, `Stop` | | Lifecycle and wiring, called by the pipeline and task. `Start` errors if called twice. `Stop` cancels the context and waits for both loops to exit. |

`StartConfig` mirrors the `StartFrame` fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `AllowInterruptions` | `bool` | Whether user speech may interrupt the bot. |
| `TurnStrategies` | `turns.UserTurnStrategies` | Turn strategies set on the task. |
| `AudioInSampleRate` | `int` | Pipeline input sample rate in Hz. |
| `AudioOutSampleRate` | `int` | Pipeline output sample rate in Hz. |
| `EnableMetrics` | `bool` | Emit TTFB and processing metrics. |
| `EnableUsageMetrics` | `bool` | Emit token, character and audio-second usage metrics. |
| `ReportOnlyInitialTTFB` | `bool` | Emit only the first TTFB metric per service. |

This check reads the start config and reports a non-fatal error upstream when the pipeline runs at an unexpected rate:

```go
func checkInputRate(p *processors.BaseProcessor, want int) error {
	if got := p.StartConfig().AudioInSampleRate; got != want {
		return p.PushError("unexpected input rate", fmt.Errorf("got %d Hz, want %d Hz", got, want), false)
	}
	return nil
}
```

## Writing a processor

This complete program defines a processor that upper-cases text, counts what it changed, and forwards everything. Run it with `go run .`; it prints `HELLO`, `HOW ARE YOU?` and the count, then exits.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type Shouter struct {
	*processors.BaseProcessor
	count int
}

func NewShouter() *Shouter {
	s := &Shouter{}
	s.BaseProcessor = processors.NewBaseProcessor("Shouter", s)
	return s
}

func (s *Shouter) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.TextFrame:
		if direction == frames.Downstream {
			s.count++
			f.Text = strings.ToUpper(f.Text)
		}
	case *frames.EndFrame:
		log.Printf("shouted %d messages", s.count)
	}
	return s.PushFrame(frame, direction)
}

func main() {
	task := pipeline.NewPipelineTask(pipeline.NewPipeline([]processors.FrameProcessor{NewShouter()}))
	task.OnFrameReachedDownstream(func(f frames.Frame) {
		fmt.Println(f.(*frames.TextFrame).Text)
	}, (*frames.TextFrame)(nil))

	for _, text := range []string{"hello", "how are you?"} {
		if err := task.QueueFrame(frames.NewTextFrame(text)); err != nil {
			log.Fatal(err)
		}
	}
	if err := task.StopWhenDone(); err != nil {
		log.Fatal(err)
	}
	if err := task.Run(context.Background()); err != nil {
		log.Fatal(err)
	}
}
```

How it works:

1. `NewShouter` builds the struct first, then passes it to `NewBaseProcessor` as the handler. `BaseProcessor` then calls `Shouter.HandleFrame` for every frame. Because `Shouter` embeds `*BaseProcessor`, it is a `FrameProcessor` and neighbours link to the outer struct.
2. `HandleFrame` changes only what it cares about and ends with `PushFrame(frame, direction)` for every frame, including the ones it did not touch.
3. `count` needs no lock: in the default mode, `HandleFrame` never runs twice at the same time.

### Rules for a correct processor

1. **Forward what you do not consume.** Every frame you do not deliberately swallow must reach `PushFrame` in the same direction. Forgetting a `StartFrame`, `EndFrame` or `CancelFrame` hangs or breaks the pipeline: processors downstream never start, and the task only finishes when `EndFrame` or `CancelFrame` reaches its sink.
2. **Forward `StartFrame` before you start work that pushes frames.** Built-in services push the `StartFrame` first, then connect.
3. **Honour `ctx`.** Pass it to blocking calls. Stop work and return when it is done. Do not treat `context.Canceled` as an error to report; the base already ignores it.
4. **Do not block on system frames.** System frames share the handler lock with data frames in the default mode, so a slow system-frame branch delays everything. Hand long work to a goroutine that uses `p.Context()`.
5. **Do not keep `HandleFrame` busy with long work** such as a network stream. Either keep it short or accept that an interruption cancels `ctx`, as `llmservice.Service` does.
6. **Push from goroutines with `PushFrame`.** It is safe to call from any goroutine. To feed a frame back into your own handler from a goroutine (for example a timer), call your own `QueueFrame`.
7. **Report errors upstream with `PushError`.** Mark an error fatal only when the call cannot continue: a fatal error cancels the task by default.
8. **If you override `Start` or `Stop`, call the embedded `BaseProcessor.Start` or `Stop`.**

## Built-in processors

The built-in processor you are most likely to add is `TextTransformProcessor`. Placed between the LLM and the TTS, this one changes how a word is spoken. The context records the changed text too, because the assistant aggregator records what the TTS spoke:

```go
func pronunciationFix() *processors.TextTransformProcessor {
	return processors.NewTextTransformProcessor("Pronunciation", func(s string) string {
		return strings.ReplaceAll(s, "SQL", "sequel")
	})
}
```

### In package `processors`

| Type | Constructor | Purpose |
| --- | --- | --- |
| `FrameFilter` | `NewFrameFilter(examples ...frames.Frame)` | Passes only frames whose type matches one of `examples`, plus system frames, `EndFrame` and `StopFrame`. Direct mode. |
| `FunctionFilter` | `NewFunctionFilter(predicate func(frames.Frame) bool, opts ...FunctionFilterOption)` | Passes frames for which `predicate` returns true. Lifecycle frames always pass. Direct mode. |
| `IdentityFilter` | `NewIdentityFilter()` | Passes everything. Direct mode. |
| `NullFilter` | `NewNullFilter()` | Drops everything except system frames, `EndFrame` and `StopFrame`. Direct mode. |
| `GatedAggregator` | `NewGatedAggregator(openFn, closeFn func(frames.Frame) bool, startOpen bool, opts ...GatedAggregatorOption)` | Holds frames while the gate is closed and releases them, in order, when `openFn` opens it. System frames and the other direction pass. |
| `ProducerProcessor` | `NewProducerProcessor(filter func(frames.Frame) bool, opts ...ProducerOption)` | Copies matching frames to every attached `ConsumerProcessor`. Passes them through too unless `WithProducerPassthrough(false)`. |
| `ConsumerProcessor` | `NewConsumerProcessor(producer *ProducerProcessor, opts ...ConsumerOption)` | Injects frames copied by a producer, typically in another branch of a parallel pipeline. Attaches on `StartFrame`, detaches on `EndFrame`, `StopFrame`, `CancelFrame`. |
| `TextTransformProcessor` | `NewTextTransformProcessor(name string, transform func(string) string)` | Rewrites the text of every text-carrying frame (see [frames](frames.md#text)) and of frames implementing `TextCarrier`. |
| `FrameLogger` | `NewFrameLogger(config FrameLoggerConfig)` | Logs each frame at debug level and forwards it. |
| `PassthroughProcessor` | `NewPassthroughProcessor(name string, logFrames bool)` | Forwards everything; optionally logs frame names at debug level. |
| `TextGeneratorProcessor` | `NewTextGeneratorProcessor(messages []string)` | Demo source: after `StartFrame`, pushes each message as a `TextFrame` 200 ms apart. |
| `TextPrinterProcessor` | `NewTextPrinterProcessor()` | Demo sink: prints each `TextFrame` to stdout and forwards it. |
| `UserIdleController` | `NewUserIdleController(timeout time.Duration)` | Pushes `UserIdleTimeoutFrame` downstream when the user stays silent for `timeout` after the bot stops speaking. 0 disables. See [aggregators](aggregators.md#idle-detection). |
| `AudioGuard` | `NewAudioGuard()`, `NewAudioGuardShadow()` | Drops `TTSAudioFrame`s that do not belong to the current TTS context, so stale audio never plays after an interruption. The shadow variant only logs. Place it between TTS and the output transport. |

Option tables:

| Option | Default | Effect |
| --- | --- | --- |
| `WithFunctionFilterDirection(direction)` | `Downstream` | Filter only this direction; the other passes. |
| `WithFunctionFilterBothDirections()` | off | Filter both directions. |
| `WithFunctionFilterSystemFrames(enabled bool)` | false | Also run the predicate on system frames (lifecycle frames still pass). |
| `WithGateDirection(direction)` | `Downstream` | The direction the gate applies to. |
| `WithProducerTransformer(fn func(frames.Frame) frames.Frame)` | none | Transform each copy before a consumer gets it; returning nil skips that consumer. |
| `WithProducerPassthrough(bool)` | true | Whether matching frames also continue downstream. |
| `WithConsumerTransformer(fn func(frames.Frame) frames.Frame)` | none | Transform frames before injecting; nil skips. |
| `WithConsumerDirection(direction)` | `Downstream` | Direction the consumer injects in. |

`FrameLoggerConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Prefix` | `string` | `"Frame"` | Log prefix; the processor is named `FrameLogger:<Prefix>`. |
| `IgnoredFrameTypes` | `[]frames.Frame` | none | Example values of types to forward without logging. |
| `LogDirection` | `bool` | false | Prefix each line with an arrow for the direction. |
| `LogFrameDetails` | `bool` | false | Append exported field values (except `Data`, `Audio`, `Timestamp`). |
| `Logger` | `*logger.Logger` | the default logger | Logger to write to. |

### In other packages

| Package | Type | Purpose | Reference |
| --- | --- | --- | --- |
| `processors/aggregators` | `LLMUserAggregator`, `LLMAssistantAggregator`, `ContextAggregatorPair` | Build the conversation context from user and bot turns. | [aggregators](aggregators.md) |
| `processors/aggregators` | `SentenceAggregator`, `DTMFAggregator`, `UserTurnCompletionProcessor` | Sentence buffering, keypad input, LLM-judged turn completion. | [aggregators](aggregators.md) |
| `processors/filters` | `WakeCheckFilter` (`NewWakeCheckFilter(phrases []string, opts ...WakeCheckOption)`) | Drops `TranscriptionFrame`s until one contains a wake phrase, then passes transcriptions while the user keeps talking within the keepalive (`DefaultWakeKeepalive`, 3 s; change with `WithWakeKeepalive`). | this page |
| `processors/filters` | `WakeNotifierFilter` (`NewWakeNotifierFilter(notifier *Notifier, filter func(frames.Frame) bool, types ...frames.Frame)`) | Calls `notifier.Notify()` when a frame of one of `types` passes `filter`. Forwards everything. `Notifier` offers `Notify`, `Wait(ctx)` and `C()`. | this page |
| `processors/audiobuffer` | `Processor` (`audiobuffer.New(cfg Config, h Handlers)`) | Records user and bot audio, whole-call or per turn, with WAV helpers. | [recording guide](../guides/recording.md) |
| `processors/transcript` | `Processor` (`transcript.NewProcessor(...)`) with `User()` and `Assistant()` | Emits a `Message{Role, Content, Timestamp, UserID}` for each user transcription and each finished bot reply. | [recording guide](../guides/recording.md) |
| `processors/rtvi` | `Processor` (`rtvi.New(params Params)`), `Observer` | Speaks the RTVI client protocol to web and mobile front ends. | [RTVI guide](../guides/rtvi.md) |
| `audio/vad` | `VADInputProcessor` | Voice activity detection on caller audio. | [turns](turns.md#vad) |
| `audio` | `AudioConverterProcessor` | Codec and sample-rate conversion. | [audio](audio.md#converter) |
| `audio/classify` | `Processor` | Runs a classifier over windows of caller audio. | [audio](audio.md#classify) |
| `flows` | `Processor` | Coordinates node transitions with function calls. | [flows](flows.md) |
| `services/llmservice` | `Service` | Runs an LLM provider. | [LLM](llm.md) |
| `services/stt`, `services/tts` | `Base` | Base for every STT and TTS provider. | [STT and TTS bases](stt-tts-bases.md) |

## Next steps

- [Frames reference](frames.md): every frame your processor can receive.
- [Pipeline reference](pipeline.md): how processors are linked, started and stopped.
- [Observers reference](observers.md): watch frames without writing a processor.
- [Core concepts](../getting-started/core-concepts.md): the processing model with diagrams.
