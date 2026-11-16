# Frames reference

This page lists every frame type in the `frames` package, grouped by topic. For each frame you get its category, whether it survives an interruption, its fields, and which parts of Strawgo emit and consume it.

A **frame** is one typed message that moves through a pipeline: a chunk of audio, a piece of text, a control signal. A **processor** is a stage that receives frames, acts on them and passes them on. An **interruption** (also called barge-in) happens when the user starts talking while the bot is speaking, and Strawgo throws away the bot's pending output.

Import path: `github.com/square-key-labs/strawgo-ai/src/frames`.

## How to read the tables

Every frame embeds one of three base types. The base type decides how a processor queues the frame.

| Category | Base type | Queue | Effect of an interruption |
| --- | --- | --- | --- |
| System | `*frames.SystemFrame` | Priority queue, handled before any queued data or control frame | Never dropped. The "Survives" column shows `n/a`. |
| Control | `*frames.ControlFrame` | Ordered queue, shared with data frames | Dropped unless it survives interruption. |
| Data | `*frames.DataFrame` | Ordered queue, shared with control frames | Dropped unless it survives interruption. |

"Survives" is the value of `SurvivesInterruption()` on a frame built by its constructor. When an `InterruptionFrame` reaches a processor, the processor drops every queued control or data frame that does not survive, and cancels the `ctx` passed to the handler that is running one.

"Emitted by" and "Consumed by" name the packages under `src/` that build or act on the frame, found by searching the source. "Your code" means no built-in processor emits it and you queue it yourself. "None built in" means no package in this repository acts on the frame; it passes through unchanged and you can catch it with your own processor or with `PipelineTask.OnFrameReachedDownstream`.

## The `Frame` interface and `BaseFrame`

Every frame implements `frames.Frame`.

```text
type Frame interface {
	ID() uint64
	Name() string
	PTS() time.Time
	Metadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	GetBroadcastSiblingID() string
	SurvivesInterruption() bool
	String() string
}
```

| Method | Returns |
| --- | --- |
| `ID()` | A process-wide unique number, assigned when the frame is built. |
| `Name()` | The type name, for example `"TTSAudioFrame"`. |
| `PTS()` | The creation time. `BaseFrame.SetPTS` changes it. |
| `Metadata()` | A lazily created map for extra values, for example `"codec"` on audio frames. |
| `GetBroadcastSiblingID()` | For a frame sent in both directions by `BroadcastFrame`, the ID of its twin. Empty otherwise. |
| `SurvivesInterruption()` | Whether an interruption leaves this frame in the queue. |

`BaseFrame` implements the interface. Its exported fields are `BroadcastSiblingID string`, `TransportSource string` and `TransportDestination string`. It also has `SetPTS`, `HasMetadata`, `SetSurvivesInterruption` and `SetBroadcastSiblingID`.

| Helper | Signature | Purpose |
| --- | --- | --- |
| `NewBaseFrame` | `func NewBaseFrame(name string) *BaseFrame` | A base that is dropped on interruption. |
| `NewUninterruptibleBaseFrame` | `func NewUninterruptibleBaseFrame(name string) *BaseFrame` | A base that survives interruption. |
| `IsSystem` | `func IsSystem(f Frame) bool` | True for frames whose `Category()` is `SystemCategory`. |
| `Downstream`, `Upstream` | `FrameDirection` constants | Downstream flows from input to output; upstream flows back toward the input. |

This helper prints any frame with the methods every frame has, for example in a debug processor:

```go
func describe(f frames.Frame) string {
	kind := "queued in order"
	if frames.IsSystem(f) {
		kind = "system"
	}
	return fmt.Sprintf("%s #%d (%s, survives interruption: %v)", f.Name(), f.ID(), kind, f.SurvivesInterruption())
}
```

## Lifecycle

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `StartFrame` | system | n/a | `AllowInterruptions bool`, `TurnStrategies turns.UserTurnStrategies`, `AudioInSampleRate int`, `AudioOutSampleRate int`, `EnableMetrics bool`, `EnableUsageMetrics bool`, `ReportOnlyInitialTTFB bool` | `pipeline` (`PipelineTask.Run`, from the task config) | Every `BaseProcessor` stores it as its `StartConfig`; transports, STT, TTS and aggregators start their work on it |
| `EndFrame` | control | yes | `Reason string` | `pipeline` (`StopWhenDone`, or from `EndTaskFrame`), `flows` (`EndConversation`), `serializers` (carrier "stop" or "hangup" events) | `pipeline` (task finishes when it reaches the end); TTS drains, transports flush and hang up |
| `StopFrame` | control | yes | none | Your code | `pipeline` (task finishes), filters let it through |
| `CancelFrame` | system | n/a | `Reason string` | `pipeline` (`PipelineTask.Cancel`, context cancel, fatal error, idle timeout) | `pipeline` (task finishes); services and transports stop immediately |
| `EndTaskFrame` | control | yes | `Reason string` | `transports`, `transports/webrtc` (session timeout), `processors/rtvi`, your code | `pipeline`: the task turns it into an `EndFrame` at the source. Push it upstream. |
| `CancelTaskFrame` | system | n/a | `Reason string` | `transports`, `transports/webrtc` (client disconnected) | `pipeline`: the task calls `Cancel()`. Push it upstream. |
| `InterruptionFrame` | system | n/a | none | `pipeline` (from `InterruptionTaskFrame`), `processors/aggregators` (user turn start, DTMF), `transports` (from the wire), `processors/rtvi`, `serializers` | Every `BaseProcessor` drops its interruptible queue; LLM, TTS, transports, aggregators and realtime services reset |
| `InterruptionTaskFrame` | control | no | none | `processors` (`BaseProcessor.PushInterruptionTaskFrame`) | `pipeline`: the task queues an `InterruptionFrame` at the source. Push it upstream. |
| `ErrorFrame` | system | n/a | `Error error`, `Fatal bool`, `Processor string`, `Category ErrorCategory` (empty unless the reporter set it; `NewCategorizedErrorFrame` sets it) | `processors` (`PushError`, `PushErrorFrame`, handler panics); the LLM, STT and TTS bases set `Category` | `pipeline` (task `OnError`, cancels on fatal, applies `ProcessorUnusablePolicy` on a permanent category: `authentication`, `authorization`, `quota` or `configuration`), `ServiceSwitcher` (failover, marks the service unusable on a permanent category). See [errors](errors.md#error-categories). |
| `HeartbeatFrame` | control | no | none | `pipeline` (when `HeartbeatInterval > 0`) | `pipeline` (task records arrival) |
| `PipelineFlushFrame` | control | yes | `Event chan struct{}`, `Returning bool`, `Origin string`, `FlushID uint64` | Your code | None built in |
| `FrameProcessorPauseFrame` | control | no | `Processor ProcessorRef` | Your code | None built in |
| `FrameProcessorPauseUrgentFrame` | system | n/a | `Processor ProcessorRef` | Your code | None built in |
| `FrameProcessorResumeFrame` | system | n/a | `Processor ProcessorRef` | Your code | None built in |
| `FrameProcessorResumeUrgentFrame` | system | n/a | `Processor ProcessorRef` | Your code | None built in |

`NewStartFrame()` builds a `StartFrame` with interruptions off. `NewStartFrameWithConfig(allowInterruptions bool, strategies turns.UserTurnStrategies)` sets both. You rarely build one yourself: the task does it from [`PipelineTaskConfig`](pipeline.md#pipelinetaskconfig).

The most common lifecycle frame you send yourself is `EndTaskFrame`, pushed upstream from inside a processor to end the call gracefully:

```go
func hangUp(p *processors.BaseProcessor) error {
	return p.PushFrame(frames.NewEndTaskFrame("caller said goodbye"), frames.Upstream)
}
```

## Audio

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `AudioFrame` | data | yes | `Data []byte`, `SampleRate int`, `Channels int` | `serializers` (caller audio), `transports`, `transports/webrtc`, `processors/rtvi` | `services/stt`, `audio/vad`, `audio/classify`, `processors/audiobuffer`, realtime services, `audio` (converter) |
| `TTSAudioFrame` | data | no | `Data []byte`, `SampleRate int`, `Channels int`, `ContextID string` | `services/tts`, `services/realtime` | `processors` (`AudioGuard`), `transports`, `transports/webrtc`, `serializers`, `processors/audiobuffer`, `processors/rtvi` |
| `STTMuteFrame` | system | n/a | `Mute bool` | Your code | `services/stt` (drops audio while muted) |
| `VADParamsUpdateFrame` | control | no | `Params VADParams` | `extensions/ivr` | `audio/vad` (`VADInputProcessor`) |
| `SpeechControlParamsFrame` | system | n/a | `VADParams *VADParams`, `TurnParams any` | `audio/vad` | None built in |
| `VolumeFrame` | control | yes | `Volume float64` | Your code | None built in |
| `VolumeGainFrame` | control | yes | `Gain float64` | Your code | None built in |

`NewAudioFrame(data, sampleRate, channels)` and `NewTTSAudioFrame(data, sampleRate, channels)` build audio frames. Audio frames carry a `"codec"` metadata value (`"linear16"`, `"mulaw"` or `"alaw"`); a frame with no metadata is treated as 16-bit PCM.

This builds 20 ms of 16 kHz mono silence, 320 samples of 2 bytes each, as caller audio:

```go
func silence20ms() *frames.AudioFrame {
	const rate = 16000
	return frames.NewAudioFrame(make([]byte, rate/50*2), rate, 1)
}
```

## Text

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `TextFrame` | data | no | `Text string`, `SkipTTS bool` | `processors/aggregators` (`SentenceAggregator`), `extensions/ivr`, `serializers` (protobuf), your code | `services/tts`, `processors/aggregators`, `processors` (`TextTransformProcessor`), `serializers` |
| `LLMTextFrame` | data | no | `Text string`, `SkipTTS bool` | `services/llmservice`, `services/realtime`, `processors/aggregators` (`UserTurnCompletionProcessor`), `extensions/ivr`, `serializers` (RTVI client) | `services/tts`, `processors/aggregators`, `processors/transcript`, `processors/rtvi`, `pipeline/observers`, `tracing` |
| `AggregatedTextFrame` | data | no | `Text string`, `SkipTTS bool`, `AppendToContext bool`, `TextType TextType`, `ContextID string`, `RawText string`, `WillBeSpoken bool` | Your code | `services/tts` (speaks it as one unit), `processors/aggregators`, `processors/rtvi` |
| `AggregatedTextProgressFrame` | data | no | `SegmentID int`, `ContextID string`, `Text string`, `TextType TextType`, `AccumulatedText string`, `RemainingText string`, `Delivery []string` | Your code | `processors/rtvi` |
| `TTSTextFrame` | data | no | `Text string`, `SkipTTS bool`, `AppendToContext bool`, `TextType TextType`, `ContextID string`, `RawText string`, `WillBeSpoken bool`, `SegmentID *int`, `Delivery []string` | `services/tts` (the text that was spoken), `services/realtime`, `serializers` | `processors/aggregators` (assistant context), `processors/transcript`, `processors/rtvi`, `tracing` |
| `VisionTextFrame` | data | no | `Text string`, `SkipTTS bool` | Your code | `processors` (`TextTransformProcessor`) |
| `InputTextRawFrame` | system | n/a | `Text string` | Your code | `processors` (`TextTransformProcessor`) |

`TextType` values: `TextTypeBackchannel`, `TextTypeExpressionSound`, `TextTypeExpressionPause`, `TextTypeSentence`, `TextTypeToken`, `TextTypeWord`. `NewAggregatedTextFrame` and `NewTTSTextFrame` set `AppendToContext` to true.

`processors.TextTransformProcessor` rewrites the text of `TextFrame`, `LLMTextFrame`, `TranscriptionFrame`, `AggregatedTextFrame`, `TTSTextFrame`, `VisionTextFrame`, `InterimTranscriptionFrame`, `EagerTranscriptionFrame`, `TranslationFrame` and `InputTextRawFrame`.

A processor that cares about text usually switches on the concrete types. This reads the text from the three most common text frames:

```go
func textOf(f frames.Frame) (string, bool) {
	switch t := f.(type) {
	case *frames.LLMTextFrame:
		return t.Text, true
	case *frames.TTSTextFrame:
		return t.Text, true
	case *frames.TextFrame:
		return t.Text, true
	}
	return "", false
}
```

## Transcription

**STT** (speech-to-text) turns caller audio into text. These frames carry its output.

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `TranscriptionFrame` | data | final only | `Text string`, `IsFinal bool`, `Language string`, `Timestamp time.Time` | `services/stt`, realtime services, `processors/aggregators` (`DTMFAggregator`), `serializers` | `processors/aggregators` (user turn text), `processors/transcript`, `processors/rtvi`, `processors/filters`, `extensions/voicemail`, `tracing`, observers |
| `InterimTranscriptionFrame` | data | no | `Text string`, `UserID string`, `Timestamp time.Time`, `Language string`, `Result any`, `Speaker *TranscriptionSpeaker`, `Perception []PerceptionSignal` | `services/stt`, `services/openai_realtime`, `serializers` | Turn strategies, `processors/rtvi`, `tracing`, observers |
| `EagerTranscriptionFrame` | data | no | same fields as `InterimTranscriptionFrame` | `services/deepgram` (Flux, only with `EnableEagerEndOfTurn`) | `turns/user_stop` (`EagerUserTurnStopStrategy` starts a speculative reply), `processors` (`TextTransformProcessor`); never added to the turn text |
| `TranslationFrame` | data | no | `Text string`, `UserID string`, `Timestamp time.Time`, `Language string` | `services/gladia` | `processors` (`TextTransformProcessor`); the user aggregator ignores it |
| `UserBackchannelFrame` | data | no | `Text string`, `UserID string`, `Timestamp time.Time` | Your code | `processors/rtvi` |
| `STTMetadataFrame` | data | no | `Provider string`, `TTFSP99Latency time.Duration` | `services/stt` (after `StartFrame`, when `Options.Provider` is set) | `turns/user_stop` (`SpeechTimeoutUserTurnStopStrategy` uses the P99 latency) |
| `PerceptionFrame` | system | n/a | `Perception []PerceptionSignal` | Your code | None built in |

`NewTranscriptionFrame(text, isFinal)` makes a final transcription survive interruption and an interim one droppable. `TranscriptionSpeaker` has `Label string` and `Number int`. `PerceptionSignal` has `Name string`, `Labels []PerceptionLabel` and `Value *float64`; `PerceptionLabel` has `Label string` and `Confidence float64`.

Partial and final results arrive as different types, so a type switch tells them apart:

```go
func logTranscript(f frames.Frame) {
	switch t := f.(type) {
	case *frames.InterimTranscriptionFrame:
		log.Printf("partial: %s", t.Text)
	case *frames.TranscriptionFrame:
		log.Printf("final (%s): %s", t.Language, t.Text)
	}
}
```

## LLM

An **LLM** (large language model) produces the bot's replies from the conversation **context**, the list of messages so far.

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `LLMContextFrame` | control | no | `Context *llm.Context`, `Speculation bool` | `processors/aggregators` (user aggregator at end of turn, or with `Speculation` and a throwaway context on an eager end of turn; assistant aggregator upstream to re-run the LLM) | `services/llmservice` (runs inference on downstream frames; holds a speculative reply until it is confirmed), realtime services, user aggregator (re-sends upstream frames downstream), `processors/rtvi` (skips speculative frames), `tracing`, observers |
| `LLMRunFrame` | data | no | none | `flows`, your code | `processors/aggregators` (push the context and run the LLM) |
| `LLMMessagesAppendFrame` | control | no | `Messages []llm.Message`, `RunLLM bool` | `flows`, `processors/rtvi`, `processors/aggregators` (`UserTurnCompletionProcessor` re-prompt), your code | `processors/aggregators` (appends; runs the LLM when `RunLLM`), realtime services |
| `LLMMessagesUpdateFrame` | control | no | `Messages []llm.Message`, `RunLLM bool` | `flows`, `extensions/ivr` | `processors/aggregators` (replaces all messages) |
| `LLMMessagesTransformFrame` | data | no | `Transform func([]llm.Message) []llm.Message`, `RunLLM *bool` | Your code | `processors/aggregators` (rewrites messages in place) |
| `LLMSetToolsFrame` | data | no | `Tools llm.Tools` | `flows` | `processors/aggregators`, realtime services |
| `LLMSetToolChoiceFrame` | data | no | `ToolChoice llm.ToolChoice` | Your code | `processors/aggregators`, `services/openai_realtime` |
| `LLMFullResponseStartFrame` | control | no | none | `services/llmservice`, `services/realtime`, `serializers` | `processors/aggregators`, `services/tts` (opens a TTS context), `processors/rtvi`, `tracing`, observers |
| `LLMFullResponseEndFrame` | control | no | none | same as start | `processors/aggregators` (commits the assistant turn), `services/tts` (flushes), `processors/transcript`, `processors/rtvi`, `tracing` |
| `LLMThoughtStartFrame` | control | no | `AppendToContext bool`, `LLM string` | `services/llmservice` | `processors/aggregators` |
| `LLMThoughtTextFrame` | data | no | `Text string` | `services/llmservice` | `processors/aggregators` |
| `LLMThoughtEndFrame` | control | no | `Signature string` | `services/llmservice` | `processors/aggregators` (stores the thought in context when `AppendToContext`) |
| `LLMSummarizeContextFrame` | control | no | none | Your code | `processors/aggregators` (forces a summary when a `Summarize` function is set) |
| `LLMContextSummaryRequestFrame` | control | no | `RequestID string`, `Context *llm.Context`, `MinMessagesToKeep int`, `TargetContextTokens int`, `SummarizationPrompt string`, `SummarizationTimeout time.Duration` | Your code | None built in |
| `LLMContextSummaryResultFrame` | control | yes | `RequestID string`, `Summary string`, `LastSummarizedIndex int`, `Error string` | Your code | None built in |
| `LLMAssistantPushAggregationFrame` | control | no | none | Your code | `processors/aggregators` (commits the assistant text now) |
| `LLMContextUserTurnFrame` | data | no | `Text string`, `Timestamp time.Time`, `Perception []PerceptionSignal`, `Speculation bool` | `processors/aggregators` (broadcast when a user turn is committed, and with `Speculation` ahead of a speculative reply; a processor that keeps the conversation should skip speculative ones) | None built in |
| `LLMContextAssistantTurnFrame` | data | no | `Text string`, `Timestamp time.Time` | `processors/aggregators` (broadcast when an assistant turn is committed) | None built in |
| `LLMContextAssistantTimestampFrame` | data | no | `Timestamp time.Time` | `processors/aggregators` | None built in |
| `LLMMarkerFrame` | data | no | `Marker string`, `AppendToContextImmediately bool` | Your code | `pipeline/observers` |
| `LLMMarkerResponseFrame` | data | no | `Raw string`, `Marker string`, `Kind string`, `Markers []string` | `serializers` (RTVI client) | `processors/rtvi` |
| `LLMEnablePromptCachingFrame` | data | no | `Enable bool` | Your code | None built in |
| `LLMConfigureOutputFrame` | data | no | `SkipTTS bool` | `processors/rtvi` | `processors/rtvi` |
| `LLMServiceMetadataFrame` | system | n/a | `ServiceName string`, `UserTurnStrategies *turns.UserTurnStrategies`, `IsRealtimeService bool` | `services/llmservice` (false), `services/openai_realtime` and `services/gemini` `LiveService` (true): upstream at start and when a `ServiceSwitcherRequestMetadataFrame` names the service | `pipeline` (`ServiceSwitcher` filters inactive services), `processors/aggregators` (empty-turn recovery off while the active LLM is realtime) |
| `VisionFullResponseStartFrame` | control | no | none | Your code | None built in |
| `VisionFullResponseEndFrame` | control | no | none | Your code | None built in |

Your code changes the conversation by queueing these frames on the task. This adds an instruction without making the bot answer (`runLLM` is false):

```go
func addInstruction(task *pipeline.PipelineTask) error {
	msgs := []llm.Message{llm.Developer("The caller is a premium customer.")}
	return task.QueueFrame(frames.NewLLMMessagesAppendFrame(msgs, false))
}
```

## Function calls

**Function calling** lets the LLM ask your Go code to run a named function and use its result.

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `FunctionCallsStartedFrame` | system | n/a | `FunctionCalls []FunctionCallInfo` | `services/llmservice`, `services/realtime` (broadcast both ways) | `processors/aggregators`, `turns/user_mute`, `processors/rtvi`, `tracing` |
| `FunctionCallInProgressFrame` | control | yes | `ToolCallID string`, `FunctionName string`, `Arguments map[string]interface{}`, `CancelOnInterruption bool`, `GroupID string`, `Signature string` | `services/llmservice`, `services/realtime`, `serializers` | `processors/aggregators` (adds the call to context), `flows`, `processors` (`UserIdleController`), `processors/rtvi`, observers |
| `FunctionCallResultFrame` | control | yes | `ToolCallID string`, `FunctionName string`, `Result interface{}`, `RunLLM *bool`, `Arguments map[string]interface{}`, `Properties *FunctionCallResultProperties`, `Error string`, `Delivery llm.ResultDelivery` (when the assistant aggregator runs the LLM: default, interrupt, when idle, silent) | `services/llmservice`, `services/realtime`, `processors/rtvi`, `serializers` | `processors/aggregators` (writes the result, re-runs the LLM), `flows`, `turns/user_mute`, realtime services, observers |
| `FunctionCallCancelFrame` | system | n/a | `ToolCallID string`, `FunctionName string`, `RunLLM bool`, `Result string` (tool result to record; empty means `llm.ToolResultInterrupted`) | `services/llmservice` (call cancelled by interruption, `CancelFrame` or the model's cancel tool), `services/realtime`, `serializers` | `processors/aggregators` (records `Result`, or `llm.ToolResultInterrupted` when empty), `flows`, `turns/user_mute`, `processors/rtvi` |
| `ExternalFunctionCallsStartedFrame` | system | n/a | `FunctionCalls []ExternalFunctionCall` | Your code | `processors/rtvi` |
| `ExternalFunctionCallInProgressFrame` | system | n/a | `FunctionName string`, `ToolCallID string`, `Arguments map[string]any` | Your code | `processors/rtvi` |
| `ExternalFunctionCallResultFrame` | system | n/a | `FunctionName string`, `ToolCallID string`, `Arguments map[string]any`, `Result any`, `IsFinal bool` | Your code | `processors/rtvi` |
| `ExternalFunctionCallCancelFrame` | system | n/a | `FunctionName string`, `ToolCallID string` | Your code | `processors/rtvi` |

Helper types:

| Type | Fields |
| --- | --- |
| `FunctionCallInfo` | `ToolCallID string`, `FunctionName string` |
| `ExternalFunctionCall` | `FunctionName string`, `ToolCallID string` |
| `FunctionCallResultProperties` | `RunLLM *bool` (overrides the frame's `RunLLM`), `OnContextUpdated func()` (called in a goroutine after the result is in context), `IsFinal bool` (false records an intermediate result and keeps the call open). `NewFunctionCallResultProperties()` sets `IsFinal: true`. |

This logs each stage of a function call, for example from a processor's `HandleFrame`:

```go
func logFunctionCall(f frames.Frame) {
	switch t := f.(type) {
	case *frames.FunctionCallInProgressFrame:
		log.Printf("calling %s(%v) id=%s", t.FunctionName, t.Arguments, t.ToolCallID)
	case *frames.FunctionCallResultFrame:
		log.Printf("%s returned %v (error %q)", t.FunctionName, t.Result, t.Error)
	case *frames.FunctionCallCancelFrame:
		log.Printf("%s cancelled", t.FunctionName)
	}
}
```

## TTS and playback

**TTS** (text-to-speech) turns the bot's text into audio.

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `TTSSpeakFrame` | data | no | `Text string`, `AppendToContext bool`, `TextType TextType` | `flows` (`TTSSay`), `processors/aggregators` (resuming an interrupted reply), your code | `services/tts` (speaks the text as its own context) |
| `BotBackchannelFrame` | data | yes | `Text string`, `AppendToContext bool`, `TextType TextType` | Your code | None built in |
| `TTSStartedFrame` | control | no | `ContextID string` | `services/tts` (downstream before the audio, and upstream when a context opens), `services/realtime` | `processors` (`AudioGuard`), `processors/aggregators`, `processors/rtvi`, `extensions/voicemail`, `tracing` |
| `TTSStoppedFrame` | control | no | `ContextID string` | `services/tts` (both directions when a context ends or is interrupted), `services/realtime` | `transports`, `transports/webrtc` (finish bot speech), `processors/aggregators` (ends a resumed reply), `processors/rtvi`, `tracing` |
| `TTSQueuedTextFrame` | system | n/a | `Text string`, `TextType TextType`, `ContextID string`, `PlaybackTracked bool`, `Complete bool` | `services/tts` (downstream, when text is sent to the provider, and with `Complete` when a context is closed) | `processors/aggregators` (unheard remainder after an interruption) |
| `TTSMetadataFrame` | system | n/a | `ServiceName string`, `UserTurnStrategies *turns.UserTurnStrategies` | Your code | `pipeline` (`ServiceSwitcher`) |
| `PlaybackCompleteFrame` | control | no | none; the `"correlation_id"` metadata names the acknowledged mark, and is absent for Asterisk `QUEUE_DRAINED`, which carries the `frames.MetadataQueueDrained` metadata instead and acknowledges the current mark | `serializers` (Twilio, Telnyx, Exotel "mark"; Plivo "playedStream"; Vonage "websocket:notify"; Asterisk "MEDIA_MARK_PROCESSED", "QUEUE_DRAINED") | `transports` (playback acknowledgement) |

`NewTTSSpeakFrame` sets `AppendToContext: true` and `TextType: TextTypeSentence`.

`TTSSpeakFrame` makes the TTS speak fixed text without the LLM. These two sentences are spoken in order:

```go
func announce(task *pipeline.PipelineTask) error {
	return task.QueueFrames([]frames.Frame{
		frames.NewTTSSpeakFrame("This call may be recorded."),
		frames.NewTTSSpeakFrame("How can I help?"),
	})
}
```

## User and bot speaking

**VAD** (voice activity detection) decides from audio alone whether someone is talking. A **turn** is one stretch of the user or the bot talking.

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `VADUserStartedSpeakingFrame` | system | n/a | `StartSecs float64`, `Timestamp time.Time` | `audio/vad`, `transports` (`VADAnalyzerHook`) | Turn start strategies, `services/stt`, `processors/rtvi`, observers |
| `VADUserStoppedSpeakingFrame` | system | n/a | `StopSecs float64`, `Timestamp time.Time` | same as above | Turn stop strategies, `services/stt` (arms TTFB, finalizes), `processors/rtvi`, observers |
| `UserStartedSpeakingFrame` | system | n/a | none | `audio/vad`, realtime services, `services/sarvam` | Turn strategies, `services/stt`, `processors/aggregators`, `processors` (`UserIdleController`), `processors/audiobuffer`, `processors/rtvi`, `extensions/voicemail`, `tracing`, observers |
| `UserStoppedSpeakingFrame` | system | n/a | none | same as above | same as above |
| `UserSpeakingFrame` | system | n/a | none | `audio/vad` (every 200 ms of speech by default) | None built in |
| `ProposedUserStartedSpeakingFrame` | system | n/a | none | STT services with their own turn detection (`deepgram`, `gladia`, `openai`, `soniox`) | The same services |
| `ProposedUserStoppedSpeakingFrame` | control | no | none | same as above | same as above |
| `UserMuteStartedFrame` | system | n/a | none | `processors/aggregators` (a mute strategy turned on) | `processors/rtvi` |
| `UserMuteStoppedFrame` | system | n/a | none | `processors/aggregators` | `processors/audiobuffer`, `processors/rtvi` |
| `BotStartedSpeakingFrame` | system | n/a | none | `transports`, `transports/webrtc` (first audio chunk sent), `serializers` | Turn strategies, `processors/aggregators`, `processors` (`UserIdleController`), `pipeline` (activity), `processors/audiobuffer`, `processors/rtvi`, observers |
| `BotStoppedSpeakingFrame` | system | n/a | none | `transports`, `transports/webrtc` (audio drained or playback acknowledged), `serializers` | Same as started, plus `flows` and `processors/transcript` |
| `BotSpeakingFrame` | system | n/a | none | `transports`, `transports/webrtc` (every 200 ms while audio plays) | None built in |
| `UserIdleTimeoutFrame` | control | no | none | `processors` (`UserIdleController`), `processors/aggregators` (user aggregator with `UserIdleTimeout`) | None built in; catch it with `OnFrameReachedDownstream` |
| `UserIdleTimeoutUpdateFrame` | control | no | `Timeout time.Duration` | Your code | `processors` (`UserIdleController`), `processors/aggregators` |
| `UserTurnResponseCancelFrame` | system | n/a | none | `services/deepgram` (Flux `TurnResumed` after an eager end of turn), `processors/aggregators` (user aggregator withdraws a speculation), `services/llmservice` (a speculative reply wants a tool call or failed) | `services/llmservice` (discards the held speculative reply and stops generating), `turns/user_stop` (`EagerUserTurnStopStrategy` forgets the speculation), `processors/aggregators` |
| `UserTurnSpeculationConfirmedFrame` | system | n/a | `Context *llm.Context` (the speculative context), `Text string` (the committed user text) | `processors/aggregators` (user aggregator, broadcast when the committed turn matches the eager one) | `services/llmservice` (releases the held reply), `processors/rtvi` (reports the user text) |
| `UserTurnInferenceCompletedFrame` | system | n/a | none | Your code | None built in |
| `UserTurnCompletionFrame` | system | n/a | `Kind string`, `Marker string`, `ContextFrameID uint64` | Your code | None built in |

Turn strategies in `src/turns` read frames by type name, so they react to these frames wherever the user aggregator sees them. See [turns](turns.md).

This tracks who is talking, for example in a processor placed after the session output:

```go
type speakingState struct {
	userSpeaking bool
	botSpeaking  bool
}

func (s *speakingState) update(f frames.Frame) {
	switch f.(type) {
	case *frames.UserStartedSpeakingFrame:
		s.userSpeaking = true
	case *frames.UserStoppedSpeakingFrame:
		s.userSpeaking = false
	case *frames.BotStartedSpeakingFrame:
		s.botSpeaking = true
	case *frames.BotStoppedSpeakingFrame:
		s.botSpeaking = false
	}
}
```

## Settings updates and service switching

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `LLMUpdateSettingsFrame` | control | yes | `Delta llm.Settings`, `Service ProcessorRef`, `ReachInactiveServices bool` | `flows` (role message), `pipeline` (`ServiceSwitcher` copies), your code | `services/llmservice` (merges `Delta`), realtime services, `pipeline` (`ServiceSwitcher`) |
| `TTSUpdateSettingsFrame` | control | yes | `Delta TTSSettings`, `Service ProcessorRef`, `ReachInactiveServices bool` | `pipeline` (`ServiceSwitcher` copies), your code | `services/tts`, `pipeline` |
| `STTUpdateSettingsFrame` | control | yes | `Delta STTSettings`, `Service ProcessorRef`, `ReachInactiveServices bool` | `pipeline` (`ServiceSwitcher` copies), your code | `services/stt`, `pipeline` |
| `ServiceMetadataFrame` | system | n/a | `ServiceName string`, `UserTurnStrategies *turns.UserTurnStrategies` | `services/deepgram`, `services/gladia`, `services/openai`, `services/soniox` | `processors/aggregators` (adopts the service's turn strategies unless you set your own), `pipeline` |
| `ServiceSwitcherFrame` | control | no | none | Your code | `pipeline` (passed to the switcher strategy) |
| `ManuallySwitchServiceFrame` | control | no | `Service ProcessorRef` | Your code | `pipeline` (`ManualSwitchStrategy` activates `Service`) |
| `ServiceSwitcherRequestMetadataFrame` | control | no | `Service ProcessorRef` | `pipeline` (`ServiceSwitcher` on switch) | `pipeline` |

`Service` limits the update to the processor whose `Name()` matches; nil means "whoever receives it". `ReachInactiveServices` makes a `ServiceSwitcher` copy the update to its inactive services too.

Settings types:

| Type | Fields | `Merge` behaviour |
| --- | --- | --- |
| `TTSSettings` | `Model string`, `Voice string`, `Language string`, `Speed *float64`, `Extra map[string]any` | Non-empty strings and non-nil `Speed` replace; `Extra` keys are merged. |
| `STTSettings` | `Model string`, `Language string`, `Extra map[string]any` | Same rules. |
| `VADParams` | `Confidence float32`, `StartSecs float32`, `StopSecs float32`, `MinVolume float32`, `HoldConfidence float32` | See [turns](turns.md#vad-parameters). |

This changes the TTS voice in the middle of a call. Only the non-empty fields change:

```go
func switchVoice(task *pipeline.PipelineTask, voiceID string) error {
	return task.QueueFrame(frames.NewTTSUpdateSettingsFrame(frames.TTSSettings{Voice: voiceID}))
}
```

## Transport messages and connection

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `InputTransportMessageFrame` | system | n/a | `Message any` | `transports`, `transports/webrtc` (data channel), `serializers` (protobuf) | `processors/rtvi`, `transports` |
| `OutputTransportMessageFrame` | data | no | `Message any` | Your code | `transports`, `transports/webrtc`, `serializers` (sent in order with audio) |
| `OutputTransportMessageUrgentFrame` | system | n/a | `Message any` | `processors/rtvi`, `serializers` (Genesys protocol replies) | `transports`, `transports/webrtc`, `serializers` (sent at once) |
| `ClientConnectedFrame` | system | n/a | none | `transports`, `transports/webrtc` | `pipeline/observers` |
| `ClientDisconnectedFrame` | system | n/a | none | `transports`, `transports/webrtc` | None built in |
| `BotConnectedFrame` | system | n/a | none | Your code | None built in |
| `OutputTransportReadyFrame` | control | no | none | Your code | None built in |
| `InputTransportStartAudioStreamingFrame` | control | no | none | `processors/rtvi` | None built in |
| `InputFileRawFrame` | system | n/a | `File string`, `Type FileSourceType`, `Filename string`, `Format string` | Your code | None built in |
| `UserFileRawFrame` | system | n/a | `File string`, `Type FileSourceType`, `Filename string`, `Format string`, `UserID string`, `Text string`, `AppendToContext *bool`, `CustomOptions map[string]any`, `RunLLM *bool` | `processors/rtvi` | None built in |

A message value of type `string` is sent as a text WebSocket message, `[]byte` as binary, `json.RawMessage` as text; anything else is JSON-encoded. Telephony serializers skip RTVI messages (those labelled `"rtvi-ai"`). `FileSourceType` is `FileSourceBytes` or `FileSourceURL`.

This sends a JSON message to the connected client ahead of any queued audio:

```go
func notifyClient(task *pipeline.PipelineTask) error {
	msg := map[string]any{"type": "status", "value": "looking that up"}
	return task.QueueFrame(frames.NewOutputTransportMessageUrgentFrame(msg))
}
```

## DTMF

**DTMF** tones are the beeps a phone keypad sends.

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `InputDTMFFrame` | system | n/a | `Button KeypadEntry` | `serializers` (keypad events from every carrier), `processors/rtvi` | `processors/aggregators` (`DTMFAggregator`) |
| `OutputDTMFFrame` | data | no | `Buttons []KeypadEntry` | Your code | `transports` (sent in order after queued audio) |
| `OutputDTMFUrgentFrame` | system | n/a | `Buttons []KeypadEntry` | `extensions/ivr`, your code | `transports` (sent at once) |

`KeypadEntry` is a string type with constants `KeypadOne` … `KeypadNine`, `KeypadZero`, `KeypadPound` (`#`) and `KeypadStar` (`*`). `Valid()` checks a value. `ParseKeypadEntries(s string) ([]KeypadEntry, error)` and `KeypadEntriesString([]KeypadEntry) string` convert. `NewOutputDTMFFrameFromString` and `NewOutputDTMFUrgentFrameFromString` reject invalid characters. All three frames implement the `DTMFFrame` interface (`DTMFButtons() []KeypadEntry`).

This presses keys on the call, for example to dial an extension. The string is checked before anything is sent:

```go
func pressKeys(task *pipeline.PipelineTask, keys string) error {
	f, err := frames.NewOutputDTMFFrameFromString(keys)
	if err != nil {
		return err
	}
	return task.QueueFrame(f)
}
```

## Metrics

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `MetricsFrame` | system | n/a | `Data []MetricsData` | `services/llmservice`, `services/stt`, `services/tts`, `services/realtime`, `audio/vad`, `processors/aggregators` | `pipeline/observers`, `processors/rtvi`, `tracing` |
| `TurnMetricsFrame` | data | no | `ProcessorName string`, `StartTime time.Time`, `EndTime time.Time`, `Duration time.Duration` | `pipeline/observers` (`TurnMetricsData.ToFrame`), your code | `pipeline/observers` (`TurnMetricsObserver`) |

Metrics are only produced when `EnableMetrics` (and, for usage, `EnableUsageMetrics`) is set on the task config. Every `MetricsData` value embeds `MetricsBase{Processor string; Model string}`.

| `MetricsData` type | Extra fields | Produced by |
| --- | --- | --- |
| `TTFBMetricsData` | `Value time.Duration` (time to first byte) | LLM, STT, TTS |
| `TTFAMetricsData` | `TTFA`, `TTFB`, `LeadingSilence time.Duration` | TTS with 16-bit linear PCM output |
| `TTFATMetricsData` | `TTFAT`, `TTFB`, `ThinkingTime time.Duration` | LLM, when a response streams reasoning before its answer |
| `ProcessingMetricsData` | `Value time.Duration` | LLM, segmented STT, HTTP TTS |
| `LLMUsageMetricsData` | `Value llm.Usage` | LLM (usage metrics) |
| `LLMResponseMetricsData` | `ResponseID`, `RequestID`, `ServedModel`, `StopReason`, `RawStopReason string` | LLM |
| `STTUsageMetricsData` | `Value STTUsage` (`AudioSeconds float64`) | STT (usage metrics) |
| `TTSUsageMetricsData` | `Value int` (characters) | TTS (usage metrics) |
| `TextAggregationMetricsData` | `Value time.Duration` (time to complete a sentence) | TTS in sentence mode |
| `TurnMetricsData` | `IsComplete bool`, `Probability float64`, `E2EProcessingTime time.Duration` | Smart-turn analysis (`audio/vad`, user aggregator) |

This adds up LLM token usage from the metrics frames that reach the end of the pipeline:

```go
func countTokens(task *pipeline.PipelineTask, total *atomic.Int64) {
	task.OnFrameReachedDownstream(func(f frames.Frame) {
		for _, d := range f.(*frames.MetricsFrame).Data {
			if u, ok := d.(frames.LLMUsageMetricsData); ok {
				total.Add(int64(u.Value.TotalTokens))
			}
		}
	}, (*frames.MetricsFrame)(nil))
}
```

## Audio control and recording

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `FilterEnableFrame` | control | no | `Enable bool` | Your code | `transports` (input filter), `audio/filter` |
| `FilterUpdateSettingsFrame` | control | no | `Settings map[string]any` | Your code | `transports` (input filter), `audio/filter` |
| `FilterControlFrame` | control | no | none (base of the two above) | Your code | None built in |
| `MixerEnableFrame` | control | no | `Enable bool` | Your code | `transports`, `transports/webrtc` (output mixer), `audio/mixer` |
| `MixerUpdateSettingsFrame` | control | no | `Settings map[string]any` | Your code | `transports`, `transports/webrtc`, `audio/mixer` |
| `MixerControlFrame` | control | no | none (base of the two above) | Your code | None built in |
| `AudioBufferStartRecordingFrame` | control | yes | none | Your code | `processors/audiobuffer` |
| `AudioBufferStopRecordingFrame` | control | yes | none | Your code | `processors/audiobuffer` |

The setting keys each filter and mixer accepts are in [audio](audio.md).

This turns the output mixer off, for example to stop background ambience while the bot reads out a card number:

```go
func pauseAmbience(task *pipeline.PipelineTask) error {
	return task.QueueFrame(frames.NewMixerEnableFrame(false))
}
```

## Images

Strawgo has no video transport. These frames exist for parity and for your own processors.

| Frame | Cat. | Survives | Fields | Emitted by | Consumed by |
| --- | --- | --- | --- | --- | --- |
| `InputImageRawFrame` | system | n/a | `Data []byte`, `Width int`, `Height int`, `Format string` | Your code | None built in |
| `OutputImageRawFrame` | data | no | `Data []byte`, `Width int`, `Height int`, `Format string`, `SyncWithAudio bool` | Your code | None built in |
| `URLImageRawFrame` | data | no | `Data []byte`, `Width int`, `Height int`, `Format string`, `SyncWithAudio bool`, `URL string` | Your code | None built in |
| `UserImageRequestFrame` | system | n/a | `UserID string`, `Text string`, `AppendToContext *bool`, `VideoSource string`, `FunctionName string`, `ToolCallID string`, `ResultCallback any` | Your code | None built in |
| `UserImageRawFrame` | system | n/a | `Data []byte`, `Width int`, `Height int`, `Format string`, `UserID string`, `Text string`, `AppendToContext *bool`, `Request *UserImageRequestFrame`, `RunLLM *bool` | `processors/rtvi` | None built in |

This wraps a PNG image your code received as an input image frame for your own processors:

```go
func screenshot(png []byte, width, height int) *frames.InputImageRawFrame {
	return frames.NewInputImageRawFrame(png, width, height, "PNG")
}
```

## Writing your own frame type

Embed one of the three base types and build it with `NewBaseFrame` (dropped on interruption) or `NewUninterruptibleBaseFrame` (kept). Pick the category by how urgent the frame is:

1. Use `*frames.SystemFrame` for signals that must overtake queued audio and text.
2. Use `*frames.ControlFrame` for signals that must stay in order with the data around them.
3. Use `*frames.DataFrame` for content.

This example defines a control frame that carries a CRM record ID and survives interruptions.

```go
type CRMRecordFrame struct {
	*frames.ControlFrame
	RecordID string
}

func NewCRMRecordFrame(id string) *CRMRecordFrame {
	return &CRMRecordFrame{
		ControlFrame: &frames.ControlFrame{BaseFrame: frames.NewUninterruptibleBaseFrame("CRMRecordFrame")},
		RecordID:     id,
	}
}
```

Built-in processors pass unknown frames through, so your frame reaches every stage. Handle it in your own processor with a type switch, as shown in [processors](processors.md#writing-a-processor). If you want to send it in both directions with `BroadcastFrame`, keep the embedded pointer so that `SetBroadcastSiblingID` is promoted.

## Next steps

- [Processors reference](processors.md): how frames are queued, handled and forwarded.
- [Pipeline reference](pipeline.md): the task that creates `StartFrame` and finishes on `EndFrame`.
- [Errors reference](errors.md): `ErrorFrame`, error categories and fatal errors.
- [Core concepts](../getting-started/core-concepts.md): frames and interruptions explained with diagrams.
