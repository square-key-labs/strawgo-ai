# Observers reference

This page documents observers: the `pipeline.Observer` and `SelectiveObserver` interfaces, `TaskObserver`, every built-in observer in `pipeline/observers`, and OpenTelemetry tracing in `tracing`.

An **observer** watches frames moving through a pipeline without being part of it: it cannot change or block frames. A **frame** is one typed message in the pipeline. A **turn** is one exchange of the user or the bot talking. **Latency** here means the time from the user going quiet to the bot's voice starting.

Import paths: `github.com/square-key-labs/strawgo-ai/src/pipeline`, `.../src/pipeline/observers`, `.../src/tracing`.

## Quick example

This observer counts final transcriptions, and the function below attaches it with two built-in observers.

```go
type transcriptCounter struct {
	mu    sync.Mutex
	count int
}

func (c *transcriptCounter) Interested(f frames.Frame, _ frames.FrameDirection, kind pipeline.EventKind) bool {
	_, ok := f.(*frames.TranscriptionFrame)
	return ok && kind == pipeline.PushEvent
}

func (c *transcriptCounter) OnPushFrame(e pipeline.PushFrameEvent) {
	if t := e.Frame.(*frames.TranscriptionFrame); t.IsFinal {
		c.mu.Lock()
		c.count++
		c.mu.Unlock()
	}
}

func (c *transcriptCounter) OnProcessFrame(pipeline.ProcessFrameEvent) {}
func (c *transcriptCounter) OnPipelineStarted()                        {}
func (c *transcriptCounter) OnPipelineStopped()                        {}

func attachObservers(task *pipeline.PipelineTask, counter *transcriptCounter) {
	latency := observers.NewUserBotLatencyObserver()
	latency.OnLatencyMeasured(func(d time.Duration) { log.Printf("response latency %v", d) })

	obs := pipeline.NewTaskObserver()
	obs.AddObserver(counter)
	obs.AddObserver(latency)
	obs.AddObserver(observers.NewMetricsLogObserver())
	task.SetObserver(obs)
}
```

## Interfaces

```text
type Observer interface {
	OnProcessFrame(event ProcessFrameEvent)
	OnPushFrame(event PushFrameEvent)
	OnPipelineStarted()
	OnPipelineStopped()
}

type SelectiveObserver interface {
	Observer
	Interested(frame frames.Frame, direction frames.FrameDirection, kind EventKind) bool
}
```

| Item | Meaning |
| --- | --- |
| `OnProcessFrame` | A processor is about to handle a frame. |
| `OnPushFrame` | A processor pushed a frame to its neighbour. Most observers use this one. |
| `OnPipelineStarted`, `OnPipelineStopped` | The task started and stopped. |
| `Interested` | Optional filter, called synchronously on the pipeline goroutine for every frame. Return false to skip the event entirely. Keep it fast and allocation-free. |

`ProcessFrameEvent` and `PushFrameEvent` both have `ProcessorName string`, `Frame frames.Frame`, `Direction frames.FrameDirection`, `Timestamp time.Time`. `EventKind` is `ProcessEvent` or `PushEvent`. `ObserverInterested(observer, frame, direction, kind)` applies an observer's filter, or returns true if it has none.

A frame passes several processors, so an observer usually sees the same frame more than once. `observers.RecentFrames` (`FirstSeen(id uint64) bool`, `Reset()`) helps count each frame ID once.

Because an observer sees the same frame at every processor, use `RecentFrames` when you want to count each frame once:

```go
type interruptionCounter struct {
	seen  observers.RecentFrames
	count atomic.Int64
}

func (c *interruptionCounter) OnPushFrame(e pipeline.PushFrameEvent) {
	if _, ok := e.Frame.(*frames.InterruptionFrame); ok && c.seen.FirstSeen(e.Frame.ID()) {
		c.count.Add(1)
	}
}

func (c *interruptionCounter) OnProcessFrame(pipeline.ProcessFrameEvent) {}
func (c *interruptionCounter) OnPipelineStarted()                        {}
func (c *interruptionCounter) OnPipelineStopped()                        {}
```

## TaskObserver

```text
func NewTaskObserver() *TaskObserver
func (o *TaskObserver) AddObserver(observer Observer)
func (o *TaskObserver) RemoveObserver(observer Observer)
func (o *TaskObserver) DropStats() ObserverDropStats
func (o *TaskObserver) Dropped() uint64
func (o *TaskObserver) Close()
```

Attach it with `PipelineTask.SetObserver` before `Run`. The task calls `OnPipelineStarted`, `OnPipelineStopped` and `Close` for you.

| Behaviour | Detail |
| --- | --- |
| Delivery | Each observer has its own goroutine and receives events in order. A slow observer never blocks the pipeline or other observers. |
| Queues | Per observer, three bounded queues by event class: media (audio and image frames) 1024 (`MediaObserverQueueSize`), data 4096 (`DataObserverQueueSize`), control and system 65536 (`ControlObserverQueueSize`). |
| Overflow | When a queue is full the event is dropped and counted. `DropStats()` returns `ObserverDropStats{Media, Data, Control}`; `Total()` and `Dropped()` sum them. |
| Adding | `AddObserver` ignores nil, duplicates, observers added after `Close`, and observer values whose type is not comparable (use a pointer). |
| Close | Stops delivery and waits up to 1 s for observer goroutines to finish. |

`ClassifyFrame(frame) EventClass` and `IsMediaFrame(frame) bool` expose the classification (`MediaEvent`, `DataEvent`, `ControlEvent`).

This runs the task with observers attached and logs, after the call, whether any observer fell behind and lost events:

```go
func runObserved(ctx context.Context, task *pipeline.PipelineTask, list ...pipeline.Observer) error {
	obs := pipeline.NewTaskObserver()
	for _, o := range list {
		obs.AddObserver(o)
	}
	task.SetObserver(obs)
	err := task.Run(ctx)
	if d := obs.DropStats(); d.Total() > 0 {
		log.Printf("observer events dropped: media=%d data=%d control=%d", d.Media, d.Data, d.Control)
	}
	return err
}
```

## Built-in observers

All built-in observers are `SelectiveObserver`s that look only at push events.

| Observer | Constructor | Watches | Output |
| --- | --- | --- | --- |
| `UserBotLatencyObserver` | `NewUserBotLatencyObserver(opts ...LatencyOption)` | Downstream speaking, VAD, transcription, LLM, function-call, TTS and metrics frames | Callbacks with the latency of each response and a breakdown. |
| `TurnTrackingObserver` | `NewTurnTrackingObserver(opts ...TurnTrackingOption)` | `StartFrame`, `UserStartedSpeakingFrame`, `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`, `EndFrame`, `CancelFrame` | Turn start and end callbacks. |
| `TurnMetricsObserver` | `NewTurnMetricsObserver()` | `TurnMetricsFrame` | `OnTurnMetrics` callbacks and `Metrics()` history. |
| `StartupTimingObserver` | `NewStartupTimingObserver()` | `StartFrame`, `ClientConnectedFrame` | One `TransportTimingReport` per run. |
| `MetricsLogObserver` | `NewMetricsLogObserver(include ...frames.MetricsData)` | `MetricsFrame` | Debug log lines; `include` limits the metric types (pass zero values such as `frames.TTFBMetricsData{}`). |
| `LLMLogObserver` | `NewLLMLogObserver()` | LLM response, text, context and function-call frames | Debug log lines. |
| `TranscriptionLogObserver` | `NewTranscriptionLogObserver()` | `TranscriptionFrame`, `InterimTranscriptionFrame` | Debug log lines. |
| `DebugLogObserver` | `NewDebugLogObserver(filters []DebugLogFilter, excludeFields ...string)`, `NewDebugLogObserverFor(examples ...frames.Frame)` | All frames, or those matching the filters | A debug log line per push with the frame's fields. |
| `Multi` | `NewMulti(observers ...pipeline.Observer)` | Union of its children's interests | Forwards to each child; `Observers()` lists them. |

The log observers only do work when the logger is at debug level.

This logs the slowest stage of each response and the length of each turn:

```go
func timingObservers() []pipeline.Observer {
	latency := observers.NewUserBotLatencyObserver()
	latency.OnLatencyBreakdown(func(b observers.LatencyBreakdown) {
		var slowest observers.LatencyContribution
		for _, c := range b.Contributions {
			if c.Duration > slowest.Duration {
				slowest = c
			}
		}
		log.Printf("slowest stage: %s (%v)", slowest.Label, slowest.Duration)
	})
	tracker := observers.NewTurnTrackingObserver()
	tracker.OnTurnEnded(func(turn int, d time.Duration, interrupted bool) {
		log.Printf("turn %d took %v (interrupted=%v)", turn, d, interrupted)
	})
	return []pipeline.Observer{latency, tracker}
}
```

### UserBotLatencyObserver

| Callback | Signature | Fires |
| --- | --- | --- |
| `OnLatencyMeasured` | `func(time.Duration)` | For each bot response: time from the user going silent to the bot's first audio. |
| `OnLatencyBreakdown` | `func(LatencyBreakdown)` | With the same measurement split into contributions. |
| `OnFirstBotSpeechLatency` | `func(time.Duration)` | Once: time from client connected to the bot's first audio. |

| Option | Default | Effect |
| --- | --- | --- |
| `WithMinContribution(d)` | 5 ms (`DefaultMinContribution`) | Contributions shorter than this are left out of the breakdown. |

`LatencyBreakdown` fields: `TTFB []TTFBBreakdown` (per service: `Processor`, `Model`, `StartTime`, `Duration`), `Contributions []LatencyContribution` (`Key`, `Label`, `Owner`, `OwnerKind`, `StartTime`, `Duration`), `MeasuredFrom` (`MeasuredFromUserSilence` or `MeasuredFromClientConnected`), `Total`, `TextAggregation *TextAggregationBreakdown`, `UserTurnStart`, `UserTurn`, `FunctionCalls []FunctionCallLatency`. `Contribution(key)` sums one key; `Lines(byCost bool)` formats it for logs. `OwnerKind` is `OwnerService`, `OwnerSetting`, `OwnerBot` or `OwnerPipeline`.

### TurnTrackingObserver

| Method | Description |
| --- | --- |
| `OnTurnStarted(fn func(turn int))` | Called when a turn starts. The first turn starts with the pipeline. |
| `OnTurnEnded(fn func(turn int, duration time.Duration, interrupted bool))` | Called when a turn ends. |
| `TurnCount() int`, `TurnActive() bool` | Current state. |

| Option | Default | Effect |
| --- | --- | --- |
| `WithTurnEndTimeout(d)` | 2.5 s (`DefaultTurnEndTimeout`) | After the bot stops speaking, the turn ends if the user does not start within this time. |

A user starting to speak while the bot is speaking ends the turn as interrupted and starts a new one. `EndFrame` and `CancelFrame` end the current turn as interrupted.

### TurnMetricsObserver and StartupTimingObserver

`TurnMetricsObserver.OnTurnMetrics(fn func(TurnMetricsData))` receives each `TurnMetricsFrame` as `TurnMetricsData{ProcessorName, StartTime, EndTime, Duration}`; `Metrics()` returns all of them. `NewTurnMetricsData(name)` with `Start`, `Stop` and `ToFrame` builds such frames from your own processor.

`StartupTimingObserver.OnTransportTimingReport(fn func(TransportTimingReport))` receives `TransportTimingReport{StartupDuration, ProcessorName}`: the time from the `StartFrame` to the transport's `ClientConnectedFrame`.

`FormatMetrics(d frames.MetricsData) string` formats one metric the way `MetricsLogObserver` does.

`DebugLogFilter` has `Frame frames.Frame` (an example value of the type) and `Processor string` (only pushes by that processor; empty means any). `excludeFields` replaces the default exclusions (`Data`, `Audio`, `Image`, `Images`).

## Tracing

`tracing.Observer` turns a call into OpenTelemetry spans. It uses the global tracer provider unless you pass a tracer, so you configure exporters with the OpenTelemetry SDK as usual. With no SDK installed, spans are not recording and the observer goes inert.

```text
func NewObserver(cfg Config) *Observer
```

### tracing.Config

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Tracer` | `trace.Tracer` | `otel.Tracer(InstrumentationName)` | Tracer to create spans with. `InstrumentationName` is `github.com/square-key-labs/strawgo-ai`. |
| `ConversationID` | `string` | a random UUID | `conversation.id` attribute. |
| `ConversationType` | `string` | `"voice"` | `conversation.type` attribute. |
| `Attributes` | `[]attribute.KeyValue` | none | Extra attributes on the conversation span. |
| `RecordText` | `bool` | false | Record transcripts, LLM input and output, system instructions and TTS text as attributes. Off by default because it captures what callers say. |
| `TurnTracker` | `*observers.TurnTrackingObserver` | an internal one | Share your own turn tracker. If you pass one, add it to the `TaskObserver` yourself. |
| `LatencyTracker` | `*observers.UserBotLatencyObserver` | an internal one | Share your own latency observer; add it to the `TaskObserver` yourself. |
| `TurnEndTimeout` | `time.Duration` | 2.5 s | Turn-end timeout of the internal turn tracker. |
| `ProviderName` | `func(processorName string) string` | `tracing.ProviderName` | Maps a processor name to `gen_ai.provider.name`. The default strips suffixes such as `LLMService` and `TTS`, lower-cases, and maps a few names (`google` and `gemini` to `gcp.gemini`, `xai` to `x_ai`, and so on). |

Methods: `StartConversation(id string)` and `EndConversation()` (the conversation otherwise starts with the first turn and ends when the pipeline stops), `ConversationID()`, `ConversationContext()`, `TurnContext()`, `TurnNumber()`, `Enabled()`, `TurnTracker()`, `LatencyTracker()`.

```go
func attachTracing(obs *pipeline.TaskObserver, callID string) {
	obs.AddObserver(tracing.NewObserver(tracing.Config{
		ConversationID: callID,
		Attributes:     []attribute.KeyValue{attribute.String("app.tenant", "demo")},
	}))
}
```

### Spans and attributes

| Span | Parent | Attributes |
| --- | --- | --- |
| `conversation` | new root | `conversation.id`, `conversation.type`, plus `Config.Attributes` |
| `turn` | `conversation` | `turn.number`, `turn.type` (`"conversation"`), `conversation.id`, `turn.duration_seconds`, `turn.was_interrupted`, `turn.user_bot_latency_seconds`, `turn.ended_by_conversation_end` |
| `stt` | current turn | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.operation.name` (`"stt"`), `is_final`, `language`, `metrics.ttfb`, `metrics.audio_seconds`, `metrics.processing_time`; with `RecordText`: `transcript` |
| `llm` | current turn | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.operation.name` (`"chat"`), `gen_ai.output.type` (`"text"`), `stream`, `metrics.ttfb`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens`, `gen_ai.usage.reasoning.output_tokens`, `function_calls.count`, `function_calls.first_name`, `function_calls.names`, `tool_count`, `tools`, `metrics.processing_time`; with `RecordText`: `gen_ai.system_instructions`, `input` (JSON messages), `output` |
| `tts` | current turn | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.operation.name` (`"tts"`), `gen_ai.output.type` (`"speech"`), `metrics.ttfb`, `metrics.character_count`, `tts.interrupted`, `metrics.processing_time`; with `RecordText`: `text` |

Metric attributes appear only when the task has `EnableMetrics` (and `EnableUsageMetrics` for usage). `KindSTT`, `KindLLM` and `KindTTS` are the span-name constants. `UsageAttributes(u llm.Usage)` returns the usage attributes for your own spans.

## Next steps

- [Observability guide](../guides/observability.md): metrics, latency and tracing set up end to end.
- [Metrics dashboard recipe](../cookbook/metrics-dashboard.md): send observer data to your metrics system.
- [Pipeline reference](pipeline.md): `SetObserver` and the task lifecycle.
- [Frames reference](frames.md#metrics): `MetricsFrame` and its data types.
