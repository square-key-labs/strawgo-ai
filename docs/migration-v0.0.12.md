# Migrating from v0.0.12

This page lists the breaking changes between the old API (v0.0.12) and the rebuilt engine on `main`, so you can port an existing app.

> [!NOTE]
> Which version has which API:
>
> - "v0.0.12" is the version the `VERSION` file held before the rebuild, and the last `CHANGELOG.md` entry before `0.2.0`. There is no `v0.0.12` git tag.
> - The git tags `v0.1.0`, `v0.1.1`, `v0.1.2` and `v0.1.3` were cut later from the pre-rebuild code: v0.0.12 plus the `onnx-worker` sidecar and build and security fixes (their `VERSION` file still says `0.0.12`). They have the old API on the left of the tables below. > - The rebuilt API on the right is in `v0.2.0` and newer. Install it with `go get github.com/square-key-labs/strawgo-ai@latest`.
>
> Check the current tags with `git ls-remote --tags https://github.com/square-key-labs/strawgo-ai`.

Breaking changes between the v0.0.12 tree and the rebuilt engine, confirmed by comparing the old sources (`git show <v0.0.12 commit>:<path>`) with the current ones. The examples were rewritten for the new API: compare the deleted `examples/voice_call_*.go` files with [examples/twilio](../examples/twilio/main.go) and [examples/asterisk](../examples/asterisk/main.go).

## Transports and serializers

| v0.0.12 | Now |
| --- | --- |
| `transports.NewWebSocketTransport(transports.WebSocketConfig{Port, Path, Serializer, PlaybackAckTimeout})`, one transport and one pipeline per process | `transports.NewServer(transports.ServerConfig{Addr, Path, NewSerializer, Handler, Params, CheckOrigin})`; `Handler` is called once per connection with a `*transports.Session` and builds a pipeline for that call |
| `transport.Input()`, `transport.Output()` | `session.Input()`, `session.Output()` |
| `WebSocketTransport`, `WebSocketInputProcessor`, `WebSocketOutputProcessor`, `Pacer`, `NewPacer`, `PacerEvent`, `PlaybackClassifier`, `PlaybackKind` | removed; pacing and playback tracking are internal to the session output |
| `WebSocketConfig.PlaybackAckTimeout` | `TransportParams.PlaybackAck` and `PlaybackAckTimeout` |
| `serializers.NewTwilioFrameSerializer(streamSid, callSid string)` | `serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{...})`; stream and call ids come from the handshake (`session.CallInfo()`), `AccountSID`/`AuthToken` enable REST hang-up |
| `FrameSerializer{Type, Setup, Serialize(frame) (interface{}, error), Deserialize(interface{}), Cleanup}` | `FrameSerializer{SerializeAudio(dst, pcm, rate), Serialize(dst, frame), Deserialize(Message)}` with typed `serializers.Message`; `SerializerType` removed |
| `PlaybackAckSerializer.SerializePlaybackDoneAck(id) (interface{}, error)` | `SerializePlaybackAck(id) (Message, error)` |
| `GetStreamSid`, `GetCallSid`, `GetChannelID`, `GetCodec`, `GetSampleRate` on serializers | `session.CallInfo()` |
| `AsteriskSerializerConfig{ChannelID, Codec, SampleRate}` | adds `DisableAutoHangUp` |

There is no global listener port any more: `ListenAndServe(ctx)` serves until `ctx` is cancelled, and `Server` is also an `http.Handler`.

## Frames

| Change | Consequence |
| --- | --- |
| `EndFrame` was a system frame; it is now a control frame that survives interruptions, and has a `Reason` | it is processed in order after everything queued before it. Every processor must forward it (the task finishes only when it reaches the sink). Code that checked `frames.IsSystem` on it changes behaviour |
| New `StopFrame`, `EndTaskFrame(reason)`, `CancelTaskFrame(reason)` | push `EndTaskFrame`/`CancelTaskFrame` upstream to end or cancel the task from inside the pipeline |
| `FunctionCallsStartedFrame`, `FunctionCallCancelFrame`: control to system | they overtake queued data |
| `FunctionCallInProgressFrame`, `FunctionCallResultFrame`, input `AudioFrame` now survive interruptions | they are not dropped from queues on interruption |
| `LLMContextFrame.Context interface{}` | `*llm.Context`; `NewLLMContextFrame(*llm.Context)` |
| `LLMMessagesAppendFrame.Messages`, `LLMMessagesUpdateFrame.Messages`: `interface{}` | `[]llm.Message`; constructors take `[]llm.Message` |
| `StartFrame{AllowInterruptions, TurnStrategies}` | adds `AudioInSampleRate`, `AudioOutSampleRate`, `EnableMetrics`, `EnableUsageMetrics`, `ReportOnlyInitialTTFB`; only the first `StartFrame` a processor sees is applied |
| `ErrorFrame{Error}` | adds `Fatal` and `Processor`; a fatal error cancels the task unless `KeepRunningOnFatal` |
| no metrics frames besides `TurnMetricsFrame` | `frames.MetricsFrame` with typed `MetricsData` (durations as `time.Duration`) |

## Processors

`processors.NewBaseProcessor(name, handler)` still compiles (it now takes `...Option`), but behaves differently:

| v0.0.12 | Now |
| --- | --- |
| system and data handlers ran concurrently with no lock | `HandleFrame` calls are serialized by a per-processor mutex unless `WithConcurrentSystemFrames()` |
| bounded channels (100 system, 1000 data); `QueueFrame` blocked when full | growable queues; `QueueFrame` never blocks |
| handlers received the processor-lifetime context | data and control handlers receive an epoch context cancelled on interruption; interrupted frames are dropped from the queue. `p.Context()` is the lifetime context |
| a handler panic crashed the goroutine | recovered and reported as an `ErrorFrame` |

`processors.NewLegacyProcessor(name, handler)` restores the old concurrency and context behaviour for a processor that depends on it. See [architecture](architecture.md#processors-and-queues).

## Pipeline and task

| v0.0.12 | Now |
| --- | --- |
| `Pipeline.Initialize(task)` linked the chain | removed; `NewPipeline` links immediately and pipelines can be nested |
| `Pipeline.QueueFrame(frame)`, `Pipeline.QueueFrameUpstream(frame)` | `Pipeline.QueueFrame(frame, direction)` |
| `PipelineSource`, `PipelineSink` exported | internal |
| `PipelineTaskConfig{AllowInterruptions, TurnStrategies}` | adds sample rates, metrics switches, `IdleTimeout`, `CancelOnIdleTimeout`, `CancelTimeout`, `HeartbeatInterval`, `KeepRunningOnFatal`; `DefaultPipelineTaskConfig()` sets 16 kHz in, 24 kHz out, `CancelTimeout` 5 s and enables idle cancellation (with `IdleTimeout` 0, i.e. off) |
| cancelling the `ctx` passed to `Run` cancelled the processors directly | it sends a `CancelFrame` and waits up to `CancelTimeout`; processors run under `context.WithoutCancel(ctx)` |

New on the task: `StopWhenDone`, `Cancel`, `Done`, `HasFinished`, `QueueFrames`, `OnIdleTimeout`, `OnFrameReachedDownstream`, `OnFrameReachedUpstream`. New in the package: `Runner`, `ParallelPipeline`, `SyncParallelPipeline`, `ServiceSwitcher`, `LLMSwitcher`.

## Conversation context

| v0.0.12 | Now |
| --- | --- |
| `services.NewLLMContext(systemPrompt)`, `services.LLMContext`, `services.LLMMessage`, `services.Tool`, `services.ToolCall`, `services.FunctionCall`, `services.ToolFunction` | `llm.NewContext(messages...)`, `*llm.Context`, `llm.Message` (built with `llm.System`, `llm.User`, `llm.Assistant`, `llm.Developer`, `llm.ToolCalls`, `llm.ToolResult`), `llm.Tools{Functions: []llm.FunctionSchema{...}}` |
| `ctx.AddUserMessage`, `AddAssistantMessage`, `AddSystemMessage`, `AddToolMessage`, `AddMessageWithToolCalls`, `Clear`, `Clone`, `GetMessages`, `SetTools`, `SetToolChoice` | `Add`, `Set`, `Transform`, `Messages`, `Snapshot`, `SetTools`, `SetToolChoice`, `ReplaceToolResult` |
| system prompt stored in the context | pass `SystemPrompt` in the service config (it becomes `llm.Settings.SystemInstruction`), or add an `llm.System` message |
| LLM services appended assistant messages and tool results to the context | only the context aggregators write to the context (single-writer rule); while a pipeline runs, change the context with `LLMMessagesAppendFrame`, `LLMMessagesUpdateFrame`, `LLMSetToolsFrame` and similar frames |
| `aggregators.NewLLMUserAggregator(*services.LLMContext, strategies)`, `NewLLMAssistantAggregator(*services.LLMContext, params)` | same names taking `*llm.Context`; prefer `aggregators.NewContextAggregatorPair(ctx, params)` |
| `NewLLMContextSummarizer(config, services.LLMService)`, `SummarizeContext` | `NewLLMContextSummarizer(config, SummaryFunc)`; summaries are applied by the assistant aggregator. `services.LLMService` is removed |

## Services

LLM constructors now return `*llmservice.Service`, which embeds the processor and adds `RegisterFunction`, `UnregisterFunction`, `UpdateSettings`, `Settings` and `RunInference`.

Functions registered without options are not cancelled when the user interrupts (`llm.DefaultFunctionOptions()` has `CancelOnInterruption: false`, the opposite of Pipecat). Pass `llm.FunctionOptions{CancelOnInterruption: true}` for a function that is safe to abandon; a cancelled call is recorded in the context as `llm.ToolResultInterrupted`, not as a completed or failed call.

| v0.0.12 | Now |
| --- | --- |
| `openai.NewLLMService(openai.LLMConfig{APIKey, Model, SystemPrompt, Temperature}) *openai.LLMService` | `openai.NewLLMService(openai.LLMConfig) *llmservice.Service`; `LLMConfig` is `openaicompat.ServiceConfig{APIKey, BaseURL, Model, SystemPrompt, Settings, Headers, HTTPClient, ServiceOptions}`. `Temperature` moved to `Settings.Temperature` (`*float64`) |
| `groq.NewGroqLLMService(groq.GroqLLMConfig)` | `groq.NewLLMService(groq.LLMConfig)`, same `openaicompat.ServiceConfig` shape |
| `ollama.NewOllamaLLMService(ollama.OllamaLLMConfig)` | `ollama.NewLLMService(ollama.LLMConfig)`, same shape |
| `anthropic.NewLLMService(cfg) *anthropic.LLMService` | `anthropic.NewLLMService(cfg, opts...) *llmservice.Service` |
| `gemini.NewLLMService(cfg) *gemini.LLMService` | `gemini.NewLLMService(cfg, opts...) *llmservice.Service` |
| `vertex.NewLLMService(cfg) (*vertex.LLMService, error)` | `vertex.NewLLMService(cfg, opts...) (*llmservice.Service, error)` |
| `openai_realtime.LLMConfig.Tools []services.Tool`, `ToolChoice interface{}` | `llm.Tools`, `llm.ToolChoice` |
| `google.NewGoogleTTSService(cfg) *GoogleTTSService`, `TTSConfig.ServiceAccount` | `google.NewTTSService(cfg) *TTSService`; `ServiceAccount` removed |
| `whisper.NewWhisperSTTService(apiKey)` | `whisper.NewSTTService(whisper.STTConfig{APIKey, ...})` |
| `deepgram.STTConfig.KeepaliveTimeout`; `azure.STTConfig.Encoding`, `KeepaliveInterval`, `KeepaliveTimeout` | removed |
| `assemblyai.STTConfig.Domain`, `EndUtteranceSilenceThreshold` | removed; turn detection uses `MinTurnSilence`, `MaxTurnSilence`, `DisableFormatTurns`, `DisableVADEndpoint` |

STT and TTS services are rebuilt on `stt.Base` / `tts.Base`. Their constructors keep their names unless listed above, but the returned types embed the new bases.

Other removals: `audio.MulawToPCM`, `PCMToMulaw`, `AlawToPCM`, `PCMToAlaw` (use `audio/g711`: `UlawToPCM16`, `PCM16ToUlaw`, `AlawToPCM16`, `PCM16ToAlaw`), `vad.VADProcessor`/`NewVADProcessor` (use `vad.NewVADInputProcessor` or `TransportParams.AudioInAnalyzer`), `turn.BaseTurnAnalyzer.GetAudioSegment`.

The context, LLM and turn-strategy setup of the deleted `examples/voice_call_complete.go`, written against the current API:

```go
package main

import (
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

func buildConversation(apiKey string) (*llmservice.Service, *aggregators.ContextAggregatorPair) {
	temperature := 0.7
	svc := openai.NewLLMService(openai.LLMConfig{
		APIKey:       apiKey,
		Model:        "gpt-4.1-mini",
		SystemPrompt: "You are a helpful voice assistant.",
		Settings:     llm.Settings{Temperature: &temperature},
	})
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewVADUserTurnStartStrategy(true),
			user_start.NewMinWordsUserTurnStartStrategy(3, true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(user_stop.DefaultUserSpeechTimeout),
		},
		MuteStrategies: []user_mute.UserMuteStrategy{
			user_mute.NewMuteUntilFirstBotCompleteUserMuteStrategy(),
		},
	}
	pair := aggregators.NewContextAggregatorPair(llm.NewContext(), aggregators.ContextAggregatorPairParams{
		User: aggregators.UserAggregatorParams{Strategies: strategies},
	})
	return svc, pair
}

func main() {
	_, _ = buildConversation("key")
}
```

Audio conversion processors are no longer needed in telephony pipelines: the session input and output convert between the wire encoding and PCM at the pipeline rates.

## Turn strategies

The interfaces are unchanged (`UserTurnStartStrategy`, `UserTurnStopStrategy`, `UserMuteStrategy`); behaviour changed:

| Strategy | v0.0.12 | Now |
| --- | --- | --- |
| no strategies configured | no default | `turns.DefaultUserTurnStrategies()`: VAD start, transcription start, speech-timeout stop with `user_stop.DefaultUserSpeechTimeout` (600 ms). Strategies in `PipelineTaskConfig.TurnStrategies` apply only when the user aggregator was built without explicit ones |
| `NewSpeechTimeoutUserTurnStopStrategy(timeout, ...)` | timer of `timeout + STT p99` from VAD stop; a final transcript while the timer ran ended the turn at once | the argument is the user-speech timeout after VAD stop. The turn ends when that timer and the STT wait (p99 from `STTMetadataFrame`, or `SetTTFSP99Latency`) have both elapsed, a non-empty final transcript exists (disable with `SetWaitForTranscript(false)`), and the user is not speaking. A final transcript ends the STT wait early, not the turn |
| `NewMinWordsUserTurnStartStrategy(n, ...)` | counted words cumulatively and latched after the first trigger | counts per utterance; the `n`-word threshold applies only while the bot is speaking, otherwise one word starts a turn; can ask the aggregator to drop a short utterance (`AggregationResetter`) |
| `NewTranscriptionUserTurnStartStrategy` | any `TranscriptionFrame` | final or interim transcription with non-empty text |
| `NewVADUserTurnStartStrategy` | `UserStartedSpeakingFrame` | also `VADUserStartedSpeakingFrame` |
| `NewTurnAnalyzerUserTurnStopStrategy` | any `TranscriptionFrame` | final transcriptions only |
| `NewFirstSpeechUserMuteStrategy` | muted from the start until the first bot speech finished | muted only while the first bot speech is playing. The old behaviour is `user_mute.NewMuteUntilFirstBotCompleteUserMuteStrategy()`, which also releases on an error before the bot speaks |
| `NewFunctionCallUserMuteStrategy` | one flag set on start, cleared by any result | tracks each tool call id; stays muted until every call in progress has finished or been cancelled |
| interruptions | emitted on a user turn start while the bot was speaking | also emitted while a response is pending (the LLM has been asked but the bot has not started speaking) |

The user aggregator also ends a turn after `UserAggregatorParams.UserTurnStopTimeout` (default 5 s) without activity when no stop strategy has a pending deadline.

## Removed examples

`examples/advanced_flow.go`, `aggregators_with_interruptions.go`, `audio_interruption_example.go`, `debug_logging.go`, `interruption_strategies.go`, `voice_call_asterisk.go`, `voice_call_complete.go`, `voice_call_mulaw.go` and `voice_call_with_vad.go` were deleted. Replacements: `examples/twilio`, `examples/function_calling`, `examples/asterisk`, `examples/gemini_live`, `examples/text_flow.go`.

## Next steps

- [Install the rebuilt API from `main`](getting-started/installation.md#add-strawgo-to-your-module)
- [Core concepts](getting-started/core-concepts.md): the new engine in plain terms.
- [Architecture](architecture.md): how the rebuilt engine works.
- [Changelog](../CHANGELOG.md): the full list of changes.
