# Pipeline reference

This page documents the `pipeline` package: `Pipeline`, the parallel variants, the service switchers, `PipelineTask` with every config field and event hook, and `Runner`.

A **pipeline** is an ordered chain of processors that frames flow through. A **frame** is one typed message (audio, text, a control signal). A **processor** is one stage in the chain. A **task** runs one pipeline for one conversation: it starts it, feeds it frames and decides when it is finished.

Import path: `github.com/square-key-labs/strawgo-ai/src/pipeline`.

## Quick example

This function runs any list of processors as one call with metrics on, a 5-minute idle limit, and error and finish logging. It is the shape most bots use inside a transport's session handler.

```go
func runCall(ctx context.Context, procs []processors.FrameProcessor) error {
	cfg := pipeline.DefaultPipelineTaskConfig()
	cfg.EnableMetrics = true
	cfg.IdleTimeout = 5 * time.Minute

	task := pipeline.NewPipelineTaskWithConfig(pipeline.NewPipeline(procs), cfg)
	task.OnError(func(err error) { log.Printf("pipeline error: %v", err) })
	task.OnIdleTimeout(func() { log.Print("caller went quiet, ending call") })
	task.OnFinished(func() { log.Print("call finished") })

	if err := task.QueueFrame(frames.NewTTSSpeakFrame("Hello! How can I help?")); err != nil {
		return err
	}
	return task.Run(ctx)
}
```

## Pipeline

```text
func NewPipeline(procs []processors.FrameProcessor) *Pipeline
func (p *Pipeline) Name() string
func (p *Pipeline) Processors() []processors.FrameProcessor
```

`NewPipeline` links the processors in order between an internal source and sink, and names the pipeline `Pipeline#N`. A `Pipeline` is itself a `FrameProcessor`, so you can nest one inside another or inside a parallel pipeline.

| Behaviour | Detail |
| --- | --- |
| Start | `Start(ctx)` starts every processor in order. If one fails, the ones already started are stopped in reverse order and the error names the processor. |
| Stop | `Stop()` stops processors in reverse order and logs (does not return) their errors. |
| Observers | `SetObserver` passes the observer to every direct child that implements `ObserverAwareProcessor`. |
| Frames in | `QueueFrame` sends downstream frames to the first processor and upstream frames to the last one. |
| Frames out | Frames that leave the last processor downstream (or the first one upstream) go to the pipeline's neighbour. |

## ParallelPipeline

```text
func NewParallelPipeline(branches ...[]processors.FrameProcessor) *ParallelPipeline
func (pp *ParallelPipeline) Name() string
func (b *ParallelPipeline) Branches() []*Pipeline
```

Every frame that enters is sent to every branch. Each branch runs independently, and frames leave as soon as a branch emits them.

| Rule | Detail |
| --- | --- |
| Branch count | 1 to 64. Any other count panics. |
| Lifecycle merge | A downstream `StartFrame`, `EndFrame`, `StopFrame`, `CancelFrame` or `InterruptionFrame` leaves once, after every branch has emitted it. |
| Start ordering | Downstream frames that branches emit before the merged `StartFrame` has left are held, then released after it. |
| Deduplication | A frame object that comes out of several branches (because each branch forwarded the same input) leaves only once. Frames a branch creates itself always leave. |
| Upstream | Upstream frames leave from the top of each branch, with the same deduplication. |

## SyncParallelPipeline

```text
func NewSyncParallelPipeline(branches ...[]processors.FrameProcessor) *SyncParallelPipeline
```

Like `ParallelPipeline`, but each non-system frame is processed as a round: it is sent to every branch, the pipeline waits until every branch has finished with it, and then releases everything the branches produced, branch 0's output first.

| Rule | Detail |
| --- | --- |
| Branch count | 1 to 64. Any other count panics. |
| System frames | Bypass rounds: forwarded at once and also given to every branch. |
| `EndFrame`, `StopFrame` | The round waits until every branch has emitted the frame, then forwards it once. |
| Interruptions | An `InterruptionFrame` during a round of an interruptible frame abandons the round; released outputs that do not survive interruption are discarded. |
| Concurrency | The input stage uses legacy concurrency, so a round's wait is not cancelled by an interruption context, only by the abandon rule above. |

## ServiceSwitcher and LLMSwitcher

A **service switcher** holds several interchangeable services (for example two TTS vendors) and sends frames to one active service at a time.

```text
func NewServiceSwitcher(services []processors.FrameProcessor, strategy ServiceSwitcherStrategy) *ServiceSwitcher
func (s *ServiceSwitcher) ActiveService() processors.FrameProcessor
func (s *ServiceSwitcher) Services() []processors.FrameProcessor
func (s *ServiceSwitcher) Strategy() ServiceSwitcherStrategy
func (s *ServiceSwitcher) SwitchTo(service frames.ProcessorRef) bool
func (s *ServiceSwitcher) OnServiceSwitched(fn func(service processors.FrameProcessor))
func (s *ServiceSwitcher) Name() string

func NewLLMSwitcher(llms []*llmservice.Service, strategy ServiceSwitcherStrategy) *LLMSwitcher
func (s *LLMSwitcher) ActiveLLM() *llmservice.Service
func (s *LLMSwitcher) LLMs() []*llmservice.Service
func (s *LLMSwitcher) RegisterFunction(name string, handler llm.FunctionHandler, opts ...llm.FunctionOptions)
func (s *LLMSwitcher) UnregisterFunction(name string)
func (s *LLMSwitcher) RunInference(ctx context.Context, snapshot llm.Snapshot, delta llm.Settings) (string, error)
```

| Rule | Detail |
| --- | --- |
| Service count | 1 to 64. Any other count panics. The first service starts active. |
| Strategy | A nil strategy means `ManualSwitchStrategy{}`. |
| Routing | Frames go only to the active service. Lifecycle frames (`StartFrame`, `EndFrame`, `StopFrame`, `CancelFrame`) go to every service and leave once. |
| Manual switch | A `ManuallySwitchServiceFrame` or `ServiceSwitcherFrame` is passed to `strategy.OnSwitchFrame`. If it returns a valid index, that service becomes active and the frame is consumed. `SwitchTo` does the same from Go. Services are matched by identity, then by `Name()`. |
| Settings | `LLMUpdateSettingsFrame`, `TTSUpdateSettingsFrame` and `STTUpdateSettingsFrame` go to the active service. A copy goes to the inactive service named in `Service`, or to every inactive service when `Service` is nil and `ReachInactiveServices` is true. |
| Errors | A non-fatal `ErrorFrame` from an inactive service is dropped. One from the active service is passed to `strategy.OnServiceError`; if that picks another service, the switcher activates it and drops the error. Fatal errors always pass upstream. |
| Metadata | `ServiceMetadataFrame`, `LLMServiceMetadataFrame` and `TTSMetadataFrame` from inactive services are dropped. |
| Callbacks | `OnServiceSwitched` handlers (appended, not replaced) run after each switch. |

`LLMSwitcher` is a `ServiceSwitcher` of `*llmservice.Service` values. `RegisterFunction` and `UnregisterFunction` apply to every LLM; `RunInference` uses the active one. It satisfies `flows.LLM`.

Strategies:

```text
type ServiceSwitcherStrategy interface {
	OnSwitchFrame(frame frames.Frame, services []processors.FrameProcessor, active int) int
	OnServiceError(err *frames.ErrorFrame, services []processors.FrameProcessor, active int) int
}
```

Return the index to activate, or -1 to keep the current service.

| Strategy | `OnSwitchFrame` | `OnServiceError` |
| --- | --- | --- |
| `ManualSwitchStrategy{}` | Activates the service named by a `ManuallySwitchServiceFrame`. | Never switches. |
| `FailoverSwitchStrategy{}` | Same as manual. | On a non-fatal error with at least two services, moves to the next service, wrapping around. |

See the [switching services guide](../guides/switching-services.md) for worked examples.

## PipelineTask

```text
func NewPipelineTask(pipeline *Pipeline) *PipelineTask
func NewPipelineTaskWithConfig(pipeline *Pipeline, config *PipelineTaskConfig) *PipelineTask
func DefaultPipelineTaskConfig() *PipelineTaskConfig
```

`NewPipelineTask(p)` is `NewPipelineTaskWithConfig(p, DefaultPipelineTaskConfig())`. A nil config also means the default. A task runs once.

### PipelineTaskConfig

The "Default" column is what `DefaultPipelineTaskConfig()` returns. If you build a `PipelineTaskConfig{}` literal yourself, every field starts at its zero value instead, except `CancelTimeout`, which the constructor raises to 5 s when it is zero or negative. Start from `DefaultPipelineTaskConfig()` and change fields.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `AllowInterruptions` | `bool` | `true` | Whether user speech may interrupt the bot. Copied into the `StartFrame`. |
| `TurnStrategies` | `turns.UserTurnStrategies` | empty | Turn strategies for the user aggregator. Empty means the aggregator's own or default strategies. See [turns](turns.md). |
| `AudioInSampleRate` | `int` | `16000` | Sample rate, in Hz, that input audio is resampled to. |
| `AudioOutSampleRate` | `int` | `24000` | Sample rate, in Hz, that TTS services produce unless configured otherwise. |
| `EnableMetrics` | `bool` | `false` | Emit TTFB and processing-time `MetricsFrame`s. |
| `EnableUsageMetrics` | `bool` | `false` | Emit token, character and audio-second usage metrics. |
| `ReportOnlyInitialTTFB` | `bool` | `false` | Emit only the first TTFB metric per service. |
| `IdleTimeout` | `time.Duration` | `0` (off) | End the task when no activity frame reaches either end of the pipeline for this long. |
| `CancelOnIdleTimeout` | `bool` | `true` | Whether an idle timeout cancels the task. When false, only the `OnIdleTimeout` callback runs. |
| `CancelTimeout` | `time.Duration` | `5s` | How long a cancel may take to reach the end of the pipeline before the task finishes anyway. |
| `HeartbeatInterval` | `time.Duration` | `0` (off) | Send a `HeartbeatFrame` through the pipeline this often. |
| `KeepRunningOnFatal` | `bool` | `false` | When true, a fatal `ErrorFrame` is reported but does not cancel the task. |

Activity frames, for the idle timeout, are `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`, `UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`, `TTSAudioFrame`, `TranscriptionFrame` and `LLMTextFrame`, seen when they leave the pipeline downstream or reach the task's source upstream. The idle and heartbeat checks run every second, or every `HeartbeatInterval` if that is shorter.

### Methods

| Method | Signature | Description |
| --- | --- | --- |
| `Run` | `func (t *PipelineTask) Run(ctx context.Context) error` | Start the pipeline, send `StartFrame`, and block until the task finishes. See [lifecycle](#lifecycle-and-shutdown). |
| `QueueFrame` | `func (t *PipelineTask) QueueFrame(frame frames.Frame, direction ...frames.FrameDirection) error` | Inject a frame. Downstream (the default) enters at the first processor; `frames.Upstream` enters at the last. Before `Run`, frames are buffered and sent right after the `StartFrame`. After the task finishes it returns an error. |
| `QueueFrames` | `func (t *PipelineTask) QueueFrames(list []frames.Frame) error` | `QueueFrame` for each frame, downstream. Stops at the first error. |
| `StopWhenDone` | `func (t *PipelineTask) StopWhenDone() error` | Queue an `EndFrame`: the pipeline finishes what is queued, then ends. |
| `Cancel` | `func (t *PipelineTask) Cancel()` | Send a `CancelFrame` now. Before `Run`, the task cancels right after starting. Safe to call more than once. |
| `Done` | `func (t *PipelineTask) Done() <-chan struct{}` | Closed when the task finishes. |
| `HasFinished` | `func (t *PipelineTask) HasFinished() bool` | Whether the task has finished. |
| `Pipeline` | `func (t *PipelineTask) Pipeline() *Pipeline` | The pipeline. |
| `SetObserver` | `func (t *PipelineTask) SetObserver(observer *TaskObserver)` | Attach observers. Call before `Run`. See [observers](observers.md). |

### Event hooks

| Hook | Signature | Fires when | Multiple calls |
| --- | --- | --- | --- |
| `OnStarted` | `func (t *PipelineTask) OnStarted(callback func())` | The `StartFrame` has passed through every processor. | Replaces |
| `OnFinished` | `func (t *PipelineTask) OnFinished(callback func())` | The task finishes, just before `Done()` closes. | Replaces |
| `OnError` | `func (t *PipelineTask) OnError(callback func(error))` | An `ErrorFrame` reaches either end of the pipeline, fatal or not, or a buffered frame fails to inject. | Replaces |
| `OnIdleTimeout` | `func (t *PipelineTask) OnIdleTimeout(callback func())` | `IdleTimeout` passes with no activity. Runs before the cancel. | Replaces |
| `OnFrameReachedDownstream` | `func (t *PipelineTask) OnFrameReachedDownstream(fn FrameHandler, examples ...frames.Frame)` | A frame leaves the last processor downstream. With `examples`, only frames of those exact types; with none, every frame. | Adds |
| `OnFrameReachedUpstream` | `func (t *PipelineTask) OnFrameReachedUpstream(fn FrameHandler, examples ...frames.Frame)` | A frame reaches the task's source moving upstream. | Adds |

`FrameHandler` is `func(frame frames.Frame)`. Pass typed nil pointers as examples, for example `(*frames.TranscriptionFrame)(nil)`. Hooks run on pipeline goroutines: keep them short and never block in them.

### Lifecycle and shutdown

1. `Run` starts the source, the pipeline and the sink. If any start fails, `Run` returns that error.
2. The observer's `OnPipelineStarted` runs, then the task queues the `StartFrame` and any frames you queued before `Run`.
3. While running, the task reacts to frames that reach its ends:

   | Frame | Where | Task action |
   | --- | --- | --- |
   | `EndFrame`, `StopFrame`, `CancelFrame` | leaves the pipeline downstream | Finish. |
   | `EndTaskFrame` | reaches the source upstream | Queue an `EndFrame` with the same `Reason`. |
   | `CancelTaskFrame` | reaches the source upstream | `Cancel()`. |
   | `InterruptionTaskFrame` | reaches the source upstream | Queue an `InterruptionFrame`. |
   | `ErrorFrame` | either end | Call `OnError`. If `Fatal` and not `KeepRunningOnFatal`, `Cancel()`. |
   | `HeartbeatFrame` | leaves downstream | Record that the pipeline is alive. |

4. The task finishes when a terminal frame leaves the pipeline, or when a cancel has not arrived within `CancelTimeout`. `OnFinished` runs and `Done()` closes.
5. If the `ctx` passed to `Run` is cancelled first, the task sends a `CancelFrame` and waits up to `CancelTimeout` for it to arrive.
6. `Run` then stops the pipeline (processors in reverse order), calls the observer's `OnPipelineStopped`, closes the `TaskObserver`, and returns nil.

Processors run on `context.WithoutCancel(ctx)`: cancelling your context does not tear processors down mid-frame; it starts an orderly cancel instead.

| You want | Do |
| --- | --- |
| Say goodbye, then hang up | `QueueFrame(frames.NewTTSSpeakFrame("Goodbye"))`, then `StopWhenDone()`. |
| End from inside a processor | Push `frames.NewEndTaskFrame(reason)` upstream. |
| Stop now, dropping queued output | `Cancel()`, or push `frames.NewCancelTaskFrame(reason)` upstream. |
| Stop when the process shuts down | Cancel the `ctx` passed to `Run`, or use `Runner`. |

`Run` returns nil after a fatal error too. Use `OnError` to learn why a task ended; see [errors](errors.md).

When `HeartbeatInterval` is set and no heartbeat reaches the end of the pipeline for 10 intervals, the task logs a warning. It does not cancel.

## Runner

```text
func NewRunner(opts ...RunnerOption) *Runner
func WithSignalHandling(enabled bool) RunnerOption
func (r *Runner) Run(ctx context.Context, tasks ...*PipelineTask) error
```

`Runner.Run` runs every task concurrently and returns when all have finished.

| Behaviour | Detail |
| --- | --- |
| Signals | With signal handling on (the default), SIGINT or SIGTERM cancels every task. `WithSignalHandling(false)` turns it off. |
| Context | When `ctx` is done, every task is cancelled and `Run` waits for them. |
| Errors | If any task's `Run` returns an error, the other tasks are cancelled and `Run` returns the first error. |
| Edge cases | No tasks returns nil at once. A nil task returns an error before anything starts. |

```go
func runAll(ctx context.Context, a, b *pipeline.PipelineTask) error {
	return pipeline.NewRunner(pipeline.WithSignalHandling(true)).Run(ctx, a, b)
}
```

## Other exported items

| Item | Purpose |
| --- | --- |
| `FrameProcessorMetrics` | A small timer with `StartTTFBMetrics`, `StopTTFBMetrics`, `StartProcessingMetrics` and `StopProcessingMetrics` (the stop methods return the elapsed `time.Duration`). |
| `Observer`, `SelectiveObserver`, `TaskObserver`, `EventKind`, `EventClass`, `ClassifyFrame`, `IsMediaFrame`, `ObserverInterested`, queue-size constants | Observer machinery. See [observers](observers.md). |

## Next steps

- [Processors reference](processors.md): what goes inside a pipeline.
- [Observers reference](observers.md): watch a running task.
- [Switching services guide](../guides/switching-services.md): `ServiceSwitcher` and failover in practice.
- [Deployment guide](../guides/deployment.md): many tasks per process and graceful shutdown.
