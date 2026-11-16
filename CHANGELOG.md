# Changelog

All notable changes to Strawgo will be documented in this file.

## [Unreleased]

## [0.3.0] - 2026-11-16

Engine and telephony features on top of 0.2.1. Two defaults change: a user turn with no transcript that interrupted the bot now triggers one LLM run, and failover no longer returns to a service that reported a permanent error. OpenAI Realtime and Gemini Live moved to the current vendor APIs with new default models. Nothing here has been run against a live vendor or carrier API.

### Added
- **`llmwithbackend.New`** (`src/pipeline/llmwithbackend`): a fast frontend LLM that delegates tool and reasoning work to a backend LLM through a cancellable `delegate` function, with transcript and explicit request strategies, a backend history limit, and the backend's answer held until the acknowledgement is under way. Ported from Pipecat's `LLMWithBackend` without its worker bus. `llmservice.Service` gains `AddBuiltinTool`, `BuiltinTools`, `AppendSystemInstruction`, `AddRequestObserver` and `LookupFunction`
- **Text for speech** (`src/utils/text/transforms`): transforms and a `VoiceFormatter` for TTS (phone numbers, OTP and id spelling, money including ranges, dates, times, percentages, units, acronyms, emails, numbers, replacements, pronunciations) with `en-US`, `en-IN` (lakh and crore, rupees, +91) and `hi-IN` locales. `tts.Base.AddFilters` is safe while the pipeline runs and gives each service its own copy of a formatter. Nothing is enabled by default
- **Call control**: `serializers.Transferer` for Twilio, Telnyx, Plivo (through `PlivoConfig.TransferURL`) and Genesys, where success means the carrier accepted the transfer; `frames.TransferCallFrame` (with `Cancel`, `Claim` and `Complete`) and `TransferCallResultFrame` handled by the WebSocket output transport; `src/extensions/callcontrol` with allow-listed `transfer_call`, `end_call` (with an optional `EndCallGuard`) and `send_dtmf` LLM tools. Not verified against live carriers
- **Answer detection** (`src/extensions/voicemail`): opt-in result classes (human, voicemail, IVR, network announcement, unknown) with English and Hindi carrier announcements, a VAD timing heuristic that only confirms the classifier, beep detection with a leave-the-message-after-the-tone signal (`src/audio/tone`), and `ReportCarrierResult` and `ReportAnsweredBy`. Tested on synthetic audio only
- **Backchannels**: `user_start.NewBackchannelUserTurnStartStrategy` and `turns.BackchannelAwareUserTurnStrategies`. While the bot speaks, a transcript made only of acknowledgement words (English and Hindi defaults, configurable) does not interrupt; the user aggregator emits `UserBackchannelFrame` and calls `UserAggregatorParams.OnUserBackchannel`. A one-word answer finalised within `BotStopWindow` before the bot stops starts a turn
- **Speculative replies** for Deepgram Flux: with `FluxConfig.EnableEagerEndOfTurn`, the LLM answers Flux's eager end of turn and the LLM service holds the reply until the turn is confirmed with the same text (`UserTurnSpeculationConfirmedFrame`). The reply is discarded if the text changes, the user keeps talking or an interruption arrives. Speculative tool calls never run before confirmation
- **Function calling**: `FunctionOptions.CancellableByLLM` offers the model a `cancel_<name>` tool while the call runs; result delivery per function or per result (`llm.ResultDelivery`: `Interrupt`, `WhenIdle`, `Silent`); `FunctionCallParams.ReportProgress` for progress updates that do not trigger speech; opt-in `AssistantAggregatorParams.AddToolChangeMessages`
- **Empty user turns** (`UserAggregatorParams.EmptyUserTurn`, `turns.EmptyUserTurnConfig`): a user turn with no transcript that interrupted the bot runs the LLM once so the bot asks the caller to repeat; the prompt is removed from the context when the next user message is written. `ResumeInterruptedSpeech` makes the bot continue its interrupted reply instead, for a reply that had finished generating, has no function result waiting to be answered, and whose playback the TTS tracks (HTTP providers, or streaming with word timestamps). Recovery is off while the active LLM is a realtime service. `tts.Base` emits `TTSQueuedTextFrame`; `tts.SplitSentences` is exported
- **Error categories**: `frames.ErrorCategory`, `ErrorFrame.Category`, `frames.ClassifyError` and `frames.ClassifyHTTPStatus`; the LLM, STT and TTS bases set the category, and provider error codes are classified for OpenAI-compatible services, Anthropic and Gemini (an unknown model is a permanent `configuration` error). `FailoverSwitchStrategy.SwitchOnlyWhenUnusable`, `ServiceSwitcher.ServiceUsable` and `SetServiceUsable`, `PipelineTaskConfig.ProcessorUnusablePolicy`, `reconnect.HandshakeError`
- **Saving a conversation**: `llm.MarshalContext`, `llm.MarshalSnapshot` and `llm.UnmarshalContext` write and read a context as versioned JSON; unfinished tool calls are saved as interrupted. `llm.OmitBinary()` leaves out image, audio and file bytes, and `llm.StripProviderState()` drops thoughts, signatures and provider parts for restoring into a different provider or model
- **`src/strawtest`**, a public testing package: `RunProcessors` for processor tests, and scripted text conversations (`Say`, `Queue`, `ExpectReply`, `ExpectCall`, `WaitForBot`, `ExpectJudge`, `Bot.OnTask`) against your own pipeline with a deterministic `FakeLLM` and a pluggable `Judge`
- `sarvam.NewRealtimeSTTService` for Sarvam's realtime STT endpoint (`saaras:v3-realtime`, `saaras:v4`), with interim and final transcripts and server-side speech detection
- The logger masks credentials in every line (query-string keys, `Authorization` and `Bearer` values, URL passwords, known key formats); `logger.AddRedactor` registers application rules and `logger.RedactError` masks an error while keeping `errors.Is` and `errors.As`
- OpenAI Realtime options for turn detection (including `semantic_vad`), noise reduction, transcription language and prompt, speed, reasoning effort and truncation; Gemini Live optional context-window compression
- CI job that builds and vets the WebRTC transport with the `opus` build tag against libopus

### Changed
- `ServiceSwitcher` no longer marks a replayed request as speculative
- LLM services built on `llmservice`, OpenAI Realtime and Gemini Live push `LLMServiceMetadataFrame` upstream at start and when a `ServiceSwitcher` activates them
- `FailoverSwitchStrategy` no longer switches to a service that reported an authentication, authorization, quota or configuration error. Strategies that embed `FailoverSwitchStrategy` get its skipping and replay
- A user turn that ends with no transcript after interrupting the bot now triggers one LLM run by default; set `EmptyUserTurnConfig.Disabled` for the old behaviour
- **OpenAI Realtime** uses the GA API: no `OpenAI-Beta` header, the nested `audio` session shape and GA event names. The default model is `gpt-realtime-2.1` and the default transcription model is `gpt-realtime-whisper` (was `gpt-4o-realtime-preview` and `gpt-4o-transcribe`). Beta event names are no longer handled.
- **Gemini Live** defaults to `gemini-3.8-live` (was `gemini-2.0-flash-exp`), sends `historyConfig`, and declares tools as blocking on 3.8 models

### Fixed
- A system frame waiting for a processor's handler now goes ahead of queued data frames; under load it could be delivered after all of them
- HTTP errors from the shared LLM retry helper no longer carry query-string credentials

## [0.2.1] - 2026-11-16

### Security
- Upgraded `golang.org/x/crypto` to v0.52.0, `golang.org/x/net` to v0.55.0 and OpenTelemetry to v1.47.0, and the `onnx-worker` build dependency `openssl` to 0.10.82, for published advisories. OpenTelemetry v1.47.0 raises the minimum Go version to 1.26.0.

### Added
- CI workflow that builds, vets and tests the Go module and builds `onnx-worker` from its lock file

### Fixed
- **LLM failover replay** (`FailoverSwitchStrategy.ReplayFailedTurn`): a request superseded by a newer request for the same context is no longer replayed, which could answer one turn twice after a manual switch; an `LLMFullResponseStartFrame` passing through the switcher from upstream no longer shifts request tracking; an interruption racing a new request no longer leaves tracking out of step (`src/pipeline/service_switcher.go`)
- **TTS metrics**: TTFB and TTFA are reported when the first audio arrives after `StopTimeout`; they were dropped. Leading silence in TTFA no longer depends on how the provider splits audio into chunks, including odd-length chunks (`src/services/tts/playback.go`)
- **Input filters**: the WebRTC and WebSocket inputs no longer restart the filter for audio that arrives after `EndFrame` or `CancelFrame`, and stop it when the session closes without either frame
- **Docs workflow**: the Pages steps run only in this repository, so a fork's `main` does not fail or publish; the build fails if the search index is missing

## [0.2.0] - 2026-11-16

Rebuild of the engine, service layer and transports. Breaking; see [docs/migration-v0.0.12.md](docs/migration-v0.0.12.md). No provider added or rewritten here has been run against its live vendor API; see the capability table in the README for test status.

### Added
- **Engine**: epoch-based interruptions (data handlers receive a context cancelled on interruption; non-surviving queued frames are dropped), per-processor handler serialization with `WithConcurrentSystemFrames`, `WithDirectMode` and `NewLegacyProcessor`, panic recovery into `ErrorFrame`, `StartConfig()` (`src/processors/processor.go`)
- **Frames**: `StopFrame`, `EndTaskFrame`, `CancelTaskFrame`, uninterruptible frames, typed `MetricsFrame`, DTMF, image, settings-update and transport-message frames (`src/frames/`)
- **Pipeline**: nested pipelines, `ParallelPipeline`, `SyncParallelPipeline`, `ServiceSwitcher`, `LLMSwitcher`, `Runner`; task idle timeout, heartbeats, cancel timeout, frame subscriptions (`src/pipeline/`)
- **Conversation model**: `llm.Context` (copy-on-write, snapshots), typed messages and parts, tools, function-call types (`src/llm/`)
- **LLM service base**: `llmservice.Service` with a function registry and parallel or sequential function execution (`src/services/llmservice/`)
- **LLM providers**: OpenAI Responses API, Azure OpenAI, Together, Cerebras, DeepSeek, Fireworks, Mistral, NVIDIA, OpenRouter, Perplexity, Qwen, SambaNova, Sarvam (`sarvam.NewLLMService`), xAI (via `src/services/openaicompat/`), AWS Bedrock; MCP client (`src/services/mcp/`)
- **STT base** (`stt.NewStreaming`, `stt.NewSegmented`) and providers: OpenAI, Groq, ElevenLabs (batch and realtime), Cartesia, Google, AWS Transcribe, Gladia, Soniox, fal, Deepgram Flux
- **TTS base** (`tts.NewStreaming`, `tts.NewHTTP`) and providers: OpenAI, Groq, AWS Polly, Rime, Neuphonic, Inworld, Sarvam, Async (HTTP only, `asyncai.NewHTTPTTSService`), Fish Audio, Resemble, Speechify, MiniMax, Piper, XTTS; HTTP variants for Cartesia, Deepgram and ElevenLabs
- **Shared WebSocket session** for streaming providers with keepalive, reconnect and bounded send queue (`src/services/reconnect/`); realtime helpers (`src/services/realtime/`)
- **Transports**: `transports.NewServer` with one `Session` per connection, `transports.Dial` client, WebRTC handler with SmallWebRTC-style signalling (`src/transports/webrtc/`, G.711 only unless built with the uncompiled `opus` tag)
- **Serializers**: Telnyx, Plivo, Exotel, Vonage, Genesys AudioHook, Jambonz, Protobuf, RTVI client; handshake, hang-up, DTMF and native-audio interfaces (`src/serializers/`)
- **Audio**: streaming resampler, G.711, PCM and WAV helpers, DTMF tones, input filters, output mixer (`src/audio/`); recording processor (`src/processors/audiobuffer/`)
- **Turn-taking**: `DefaultUserTurnStrategies`, `MuteUntilFirstBotCompleteUserMuteStrategy`, user turn stop timeout, DTMF aggregator, user idle controller
- **Flows**: node-based conversation flows with JSON loading and Mermaid export (`src/flows/`)
- **Extensions**: IVR navigator, voicemail detector (`src/extensions/`)
- **Observability**: turn-tracking and logging observers, OpenTelemetry tracing observer (`src/tracing/`), RTVI processor and observer (`src/processors/rtvi/`), transcript processor
- **Examples**: `examples/twilio`, `examples/function_calling`, `examples/asterisk`, `examples/gemini_live`, `examples/text_flow.go`
- **Docs**: architecture, provider, transports, flows and migration guides under `docs/`; ADR-0002
- **Context summarization controls**: `LLMContextSummaryConfig.Keep` (per-message predicate; kept messages stay in place and never lose their tool call or tool result), `LLMAutoContextSummarizationConfig.ShouldSummarize` (gate consulted before an automatic summarization starts; a manual `LLMSummarizeContextFrame` bypasses it) and `EstimateTokens` (replaces the built-in characters/4 estimate for `MaxContextTokens`). Messages matched by `Keep` do not count towards `MaxContextTokens` or `MaxUnsummarizedMessages`
- **Turn metrics from stop strategies**: `user_stop.TurnMetricsReporter`; the user aggregator pushes the smart-turn result of `TurnAnalyzerUserTurnStopStrategy` as a `MetricsFrame` carrying `TurnMetricsData`
- **Time to first audio**: the TTS base emits `TTFAMetricsData` once per utterance with 16-bit linear PCM output: `TTFB` plus the leading silence before the first 10 ms window above -40 dBFS RMS. Needs `EnableMetrics`; skipped for µ-law, A-law and compressed output, for all-silent utterances and for utterances interrupted before audible audio

### Changed
- **Breaking, and different from Pipecat**: `llm.DefaultFunctionOptions()` now has `CancelOnInterruption: false`. User speech no longer cancels a running function call; the result is recorded and the follow-up inference runs once the user stops speaking. Pipecat cancels by default. Register with `llm.FunctionOptions{CancelOnInterruption: true}` to opt in per function. `flows.Function.CancelOnInterruption` already defaulted to false
- A function call that is cancelled is recorded with the tool result `llm.ToolResultInterrupted` (outcome unknown) instead of `CANCELLED`, in the context and in the result sent to the OpenAI Realtime server. A result that arrives after a call was cancelled or already settled is ignored
- **Breaking**: `EndFrame` is an ordered, uninterruptible control frame; `FunctionCallsStartedFrame` and `FunctionCallCancelFrame` are system frames
- **Breaking**: `Pipeline.QueueFrame(frame)` is now `QueueFrame(frame, direction)`; cancelling the `Run` context sends a `CancelFrame` bounded by `CancelTimeout`
- **Breaking**: `LLMContextFrame.Context` is `*llm.Context`; message frames carry `[]llm.Message`; aggregators take `*llm.Context` (`NewContextAggregatorPair`)
- **Breaking**: LLM services return `*llmservice.Service`; LLM services no longer write to the conversation (single-writer rule); OpenAI-compatible configs are `openaicompat.ServiceConfig` with temperature in `Settings`
- **Breaking**: `groq.NewGroqLLMService`, `ollama.NewOllamaLLMService`, `google.NewGoogleTTSService`, `whisper.NewWhisperSTTService` renamed to `NewLLMService` / `NewTTSService` / `NewSTTService`
- **Breaking**: serializer interface uses typed messages and append-style output; `NewTwilioFrameSerializer` takes `TwilioConfig`
- **Breaking**: turn strategies: speech-timeout stop waits for both the user-speech timer and the STT p99 window; min-words threshold applies only while the bot speaks; first-speech mute applies only during the first bot speech; function-call mute tracks each call
- **Breaking**: STT services emit interim results as `InterimTranscriptionFrame`; `TranscriptionFrame` is final only
- Existing providers (Anthropic, Gemini, Vertex, OpenAI, Groq, Ollama, Deepgram, AssemblyAI, Azure, Cartesia, ElevenLabs, Google, Sarvam, Whisper, OpenAI Realtime, Gemini Live) rebuilt on the new bases

### Removed
- `transports.NewWebSocketTransport`, `WebSocketConfig`, `WebSocketInputProcessor`, `WebSocketOutputProcessor`, exported `Pacer`
- `Pipeline.Initialize`, `Pipeline.QueueFrameUpstream`, exported `PipelineSource` / `PipelineSink`
- `services.LLMContext`, `services.LLMMessage`, `services.NewLLMContext`, `services.LLMService`, `services.Tool` and related types
- `audio.MulawToPCM`, `PCMToMulaw`, `AlawToPCM`, `PCMToAlaw` (replaced by `src/audio/g711`), `vad.VADProcessor`
- Old examples (`examples/voice_call_*.go`, `advanced_flow.go`, `aggregators_with_interruptions.go`, `audio_interruption_example.go`, `debug_logging.go`, `interruption_strategies.go`)
- `docs/architecture-review/2026-05-07-deepening-candidates.md` (described the removed WebSocket transport)
- Daily transport (`src/transports/daily`) and its example (`examples/daily`); the module now builds without libopus
- AsyncAI WebSocket TTS service; `asyncai.NewHTTPTTSService` remains

### Fixed
- System and data handlers of one processor no longer race on shared state
- Blocking handlers (LLM streams, HTTP TTS, function calls) stop on interruption through their context
- A slow processor no longer blocks the processor upstream of it (queues no longer block); a slow observer no longer blocks the pipeline (per-observer bounded buffers drop events)
- Function-call mute no longer clears while sibling calls are still running
- User speaking-state frames are still forwarded while the user is muted, so a tool result that arrives after the user stopped speaking during a muted function call triggers the next inference
- A function call cancelled before its in-progress frame is recorded as interrupted (`llm.ToolResultInterrupted`) instead of staying `IN_PROGRESS`
- Final `TranscriptionFrame`s survive interruptions, so the user's own turn no longer drops its transcript
- Calling `Manager.SetNode` from inside a flows function handler transitions after the result instead of running a second inference
- Output transports flush the resampler tail at `TTSStoppedFrame` so the last milliseconds of each utterance are played
- Serializers return messages that own their bytes, so several messages can be serialized before any is written
- The RTVI observer's default `OutputSources` now lists both `WebSocketOutput` and `WebRTCOutput`, so `bot-output` and `bot-tts-text` reach WebRTC clients without configuration; a text frame that passes more than one listed output is reported once
- `webrtc.Params` gains `NewAudioInAnalyzer` and `NewOutputMixer`, called once per session and taking precedence over `AudioInAnalyzer` and `AudioOutMixer`, so concurrent WebRTC peers no longer share analyzer or mixer state
- Asterisk `QUEUE_DRAINED` now ends the wait for the pending playback mark; a drain received before the mark was sent is ignored

### Known gaps
- No Daily or LiveKit transport; no video
- No Speechmatics STT, no AWS Nova Sonic
- WebRTC Opus (`-tags opus`) never compiled
- Many new packages have no tests (see README)

## [0.0.12] - 2026-03-04

### Fixed
- **Smart Turn resampling**: Add 16kHz resampling for non-16kHz audio inputs in Smart Turn analyzer (`src/audio/turn/`)
- **TTS context cleanup**: Close TTS audio contexts on normal speech completion (LLMFullResponseEndFrame), not just on interruption (`src/services/elevenlabs/`, `src/services/cartesia/`, `src/services/deepgram/`)
- **InterimTranscriptionFrame leak**: Ensure InterimTranscriptionFrame is consumed in LLMUserAggregator and not leaked downstream (`src/processors/aggregators/user.go`)
- **Azure error propagation**: Propagate Azure STT/TTS API errors as ErrorFrame instead of silently swallowing them (`src/services/azure/`)
- **VAD timing**: Update VAD stop_secs default from 0.8s to 0.2s; add user_speech_timeout to SpeechTimeoutUserTurnStopStrategy (`src/audio/vad/`, `src/turns/user_stop/`)
- **Context ID reuse**: Reuse context_id within TTS turns for consistent tracking across multiple TTS invocations in a single LLM turn

### Added
- **BroadcastInterruption**: Add `BroadcastInterruption()` method to BaseProcessor with frame pairing via `BroadcastSiblingID`; deprecate `InterruptionTaskFrame` (kept for backward compat) (`src/frames/`, `src/processors/`, `src/pipeline/`)
- **Anthropic Claude LLM**: New Claude LLM service with streaming and tool calling (`src/services/anthropic/`)
- **Ollama LLM**: New Ollama LLM service for local model hosting (`src/services/ollama/`)
- **TextAggregationMode**: Add SENTENCE and TOKEN aggregation modes to sentence aggregator (`src/processors/aggregators/`)
- **UserIdleController**: New processor for detecting user idle timeout with configurable duration (`src/processors/idle.go`)
- **ClientConnectedFrame / BotConnectedFrame**: New system frames for transport lifecycle events (`src/frames/system.go`)
- **STTMetadataFrame**: New data frame carrying P99 latency for auto-tuning turn detection (`src/frames/data.go`)
- **STT Keepalive**: Add keepalive mechanism for WebSocket-based STT services (Deepgram, Azure) to prevent connection timeout (`src/services/`)
- **AudioContextManager**: Extract TTS audio context management to shared `AudioContextManager` base struct; ElevenLabs and Cartesia refactored to use it (`src/services/service.go`)
- **Observer/Metrics Foundation**: Add `Observer` interface, `TaskObserver` fan-out proxy, and `FrameProcessorMetrics` for non-blocking pipeline observability (`src/pipeline/observer.go`, `src/pipeline/metrics.go`)
- **AssemblyAI STT**: New real-time STT service with built-in turn detection (`src/services/assemblyai/`)
- **OpenAI Realtime STT**: New OpenAI Realtime API STT service with local and server VAD modes (`src/services/openai_realtime/stt.go`)
- **OpenAI Realtime LLM S2S**: New OpenAI Realtime API speech-to-speech service (fat processor: audio in → STT+LLM+TTS frames out) (`src/services/openai_realtime/llm.go`)
- **Metrics Observers**: Concrete observer implementations — `UserBotLatencyObserver` (STT/LLM/TTS TTFB breakdown), `StartupTimingObserver` (processor start times), `TurnMetricsObserver` (`src/pipeline/observers/`)
- **Context Summarization**: Full `LLMContextSummarizer` with auto-trigger (token/message thresholds), on-demand via `LLMSummarizeContextFrame`, dedicated LLM support with 120s timeout, function-call preservation, summary role `"user"` (`src/processors/aggregators/summarizer.go`)

### Improved
- Service count: 11 → 16 AI services
- All new services have comprehensive tests with race detection (`-race` flag)


## [0.0.11] - 2026-02-14

### Added
- **Turn System Refactor**: Replaced deprecated interruption system with composable turn strategies
  - Turn start strategies: VAD, Transcription, MinWords
  - Turn stop strategies: SpeechTimeout, TurnAnalyzer
  - Turn mute strategies: FirstSpeech, FunctionCall, Always
  - Location: `src/turns/`
- **Universal Context ID Tracking**: All TTS services now track context_id for interruption handling
  - ElevenLabs TTS: Context ID in metadata
  - Cartesia TTS: Context ID in metadata
  - Deepgram TTS: Context ID in metadata
  - Google TTS: Context ID in metadata
  - Azure TTS: Context ID in metadata
- **New Services**:
  - Groq LLM service (OpenAI-compatible, fast inference) - `src/services/groq/`
  - Whisper STT service (batch-mode transcription) - `src/services/whisper/`
  - Deepgram TTS service (WebSocket streaming) - `src/services/deepgram/tts.go`
  - Google Cloud TTS service (HTTP-based) - `src/services/google/`
  - Azure Speech STT + TTS services (REST-only) - `src/services/azure/`
  - Gemini Multimodal Live S2S service (speech-to-speech) - `src/services/gemini/live.go`
- **New Transport**:
  - Daily WebRTC transport (peer-to-peer audio with Daily.co) - `src/transports/daily/`
- **Examples**:
  - Daily WebRTC voice bot example - `examples/daily/main.go`
  - Gemini Live S2S example - `examples/gemini_live/main.go`

### Changed
- **Breaking**: Removed `src/interruptions/` package
- **Breaking**: `PipelineTaskConfig.InterruptionStrategies` → `PipelineTaskConfig.TurnStrategies`
- **Breaking**: `LLMUserAggregator` constructor signature changed to accept `UserTurnStrategies`
- Updated all existing services to new turn system
- Updated all examples to use new turn strategies

### Fixed
- Race conditions in turn strategy state management
- Context ID tracking edge cases
- Stale audio filtering after interruptions

### Improved
- Documentation: Added turn strategies guide to README
- Test coverage: All services have comprehensive tests with race detection

## [0.0.10] - 2025-12-17

### Added

- **Smart Turn Detection System**: ML-based end-of-turn detection for natural conversation flow
  - Local ONNX inference using Smart Turn v3.1 model (based on Whisper Tiny encoder)
  - HTTP API support for Fal.ai hosted endpoint (`NewFalSmartTurn`)
  - Custom HTTP endpoint support (`NewHTTPSmartTurn`) for any compatible API
  - Configurable parameters: `StopSecs`, `PreSpeechMs`, `MaxDurationSecs`
  - Model auto-discovery in standard locations (`./models/`, `$HOME/models/`)
  - Location: `src/audio/turn/`

- **Turn Analyzer Interface**: Extensible turn detection framework
  - `TurnAnalyzer` interface with `AppendAudio()`, `AnalyzeEndOfTurn()`, `SpeechTriggered()`
  - `BaseTurnAnalyzer` with audio buffering, timestamping, and state management
  - `EndOfTurnState` enum (TurnIncomplete, TurnComplete)
  - `TurnMetrics` for inference timing and probability reporting
  - Location: `src/audio/turn/turn_analyzer.go`

- **Whisper Feature Extraction**: Pure Go mel spectrogram extraction
  - Log mel spectrogram computation compatible with Whisper models
  - Hann windowing, STFT, mel filterbank with Slaney normalization
  - Whisper-style normalization: `(log10(clamp(mel, 1e-10)) + 4.0) / 4.0`
  - Center padding and frame extraction matching `transformers.WhisperFeatureExtractor`
  - Location: `src/audio/turn/whisper_features.go`

- **Sentence Aggregator Processor**: Buffer text and emit complete sentences
  - Multilingual sentence boundary detection (Latin, CJK, Indic, Arabic, etc.)
  - Handles abbreviations (Dr., Mr., etc.) and decimal numbers ($29.95)
  - Prevents mid-sentence interruptions for smoother conversations
  - Flushes remaining buffer on `LLMFullResponseEndFrame` or `EndFrame`
  - Location: `src/processors/aggregators/sentence.go`

- **Aggregation Type System**: Typed text aggregation
  - `Aggregation` struct with `Text`, `Type`, `Spoken`, `Metadata`
  - `AggregationType` enum: `sentence`, `word`, `token`, `user`, `assistant`, `custom`
  - `NewAggregation()`, `NewSpokenAggregation()` constructors
  - `GetAggregation()` method on LLMContextAggregator
  - Location: `src/processors/aggregators/base.go`

- **TextAggregator Interface**: Pluggable text aggregation strategies
  - `TextAggregator` interface with async channel-based `Aggregate()` method
  - `SimpleTextAggregator`: Sentence-based aggregation with sync/async modes
  - `WordTextAggregator`: Word-by-word aggregation
  - `Flush()` for end-of-stream, `HandleInterruption()` for cleanup
  - Helper functions: `CollectAggregations()`, `MergeAggregationChannels()`
  - Location: `src/processors/aggregators/base.go`

- **Bot Speaking Frames**: Track when bot is outputting audio
  - `BotStartedSpeakingFrame`: Emitted when first audio chunk is sent
  - `BotStoppedSpeakingFrame`: Emitted when audio stops or on interruption
  - Pushed upstream by WebSocketOutputProcessor for aggregator tracking
  - Location: `src/frames/system.go`

- **VAD + Smart Turn Integration**: Combine VAD with ML turn detection
  - `NewVADInputProcessorWithTurn()`: VAD processor with optional turn analyzer
  - VAD triggers `UserStartedSpeakingFrame` on confirmed speech (SPEAKING state)
  - Turn analyzer runs ML inference when VAD detects silence after speech
  - Dual-mode: Silence timeout OR ML prediction can complete turn
  - Location: `src/audio/vad/vad_processor.go`

- **Complete Asterisk Example with Smart Turn**: Production-ready example
  - Full pipeline: WebSocket → VAD → Smart Turn → STT → Aggregator → LLM → TTS
  - Environment variable configuration for all components
  - Support for local ONNX, Fal.ai hosted, and custom HTTP Smart Turn modes
  - Location: `examples/asterisk/main.go`

### Changed

- **Context-based Audio Filtering**: Prevent stale audio during interruptions
  - WebSocketOutputProcessor now tracks `currentContextID` from TTS frames
  - Audio from old contexts is blocked after interruption
  - New context accepted from first audio frame after `TTSStartedFrame`
  - Prevents "ghost audio" from previous responses bleeding through
  - Location: `src/transports/websocket.go`

- **Bot Speaking Detection**: WebSocketOutputProcessor now emits `BotStartedSpeakingFrame`/`BotStoppedSpeakingFrame` instead of `TTSStartedFrame`/`TTSStoppedFrame` for output-side tracking

- **VAD State Transition Logic**: `UserStartedSpeakingFrame` now only emitted on transition to `SPEAKING` state (not `STARTING`), preventing false triggers from brief voice blips

- **Reduced Log Noise**: Stale audio blocking logs now summarized instead of per-frame; chunk streaming logs moved to debug level

### Technical Details

- **Smart Turn v3.1 Model Architecture**:
  - Based on Whisper Tiny encoder (39M parameters)
  - Input: 80 mel bands × 800 frames (8 seconds at 16kHz)
  - Output: Binary classification (complete/incomplete) with probability
  - Inference time: ~20-50ms on CPU (1 thread)

- **Turn Detection Flow**:
  ```
  Audio Input → VAD (speech detection)
    ↓
  Speech ends → VAD state: SPEAKING → STOPPING → QUIET
    ↓
  Turn Analyzer: Buffer last 8s of audio
    ↓
  Extract Whisper mel features (pure Go)
    ↓
  ONNX inference → probability
    ↓
  probability > 0.5 → TurnComplete → UserStoppedSpeakingFrame
  ```

- **Context ID Flow for Interruption Handling**:
  ```
  TTSStartedFrame → Reset currentContextID to ""
    ↓
  First TTSAudioFrame → Set currentContextID from frame metadata
    ↓
  InterruptionFrame → Set interrupted=true, keep contextID
    ↓
  New TTSStartedFrame → Reset contextID, keep interrupted
    ↓
  New TTSAudioFrame (different contextID) → Clear interrupted, accept audio
  ```

### Dependencies

- ONNX Runtime (shared library required for local Smart Turn)
- Smart Turn v3.1 ONNX model (auto-downloaded on first use)

### Notes

- Local Smart Turn requires `smart-turn-v3.1-cpu.onnx` model file (~8.7MB)
- Fal.ai hosted Smart Turn requires API key (https://fal.ai)
- Smart Turn adds ~20-100ms latency per turn detection (varies by mode)
- VAD-only mode remains available for low-latency scenarios

---

## [0.0.9] - 2025-11-18

### Added
- **Cartesia TTS Service**: New text-to-speech provider using Cartesia Sonic API
  - WebSocket streaming with context management
  - Sonic-3 model support with generation config (volume, speed, emotion)
  - Auto codec detection from StartFrame (mulaw, alaw, linear16)
  - Sentence aggregation for better audio quality
  - Word timestamp tracking and audio context management
  - Support for multiple sample rates (8kHz, 16kHz, 22050Hz, 24kHz, 44100kHz)
  - Interruption handling with proper context cleanup
  - Auto-reconnection on timeout (5-minute inactivity)
  - `use_original_timestamps` support for accurate word timing
  - Location: `src/services/cartesia/tts.go`

### Fixed
- **CRITICAL**: Fixed ElevenLabs and Cartesia TTS context accumulation bugs
  - **Root cause**: InterruptionFrame only closed contexts when `wasSpeaking=true`
  - **Fix**: Now ALWAYS closes/cancels context on interruption (regardless of speaking state)
  - Prevents context leaks that caused "Maximum simultaneous contexts exceeded (5)" error
  - Lets contexts persist across response ends for efficiency

### Changed
- Both TTS services now use consistent context management:
  - Close/cancel context on EVERY interruption (not just when speaking)
  - Context persists after flush (more efficient, no repeated setup)
  - Added code documentation explaining context lifecycle

---

## [0.0.8] - 2025-11-17

### Added
- Audio-based interruption strategies (Volume, VAD-based)
- SileroVAD model auto-embedding with Go's `embed` directive
- Enhanced audio flow logging with tree-style formatting
- Parallel LLM + TTS processing (eliminates 200-500ms latency)

### Changed
- **BREAKING**: Switched to audio-based interruption only (removed text-based)
- VAD timing optimization (0.8s → 0.4s stop detection)
- TTS WebSocket now connects eagerly at startup

### Fixed
- Ghost audio on interruption (overlapping audio issue)
- Bot speaking detection timing (premature TTSStoppedFrame)

## [0.0.7] - 2025-11-15

### Added
- **SileroVAD Integration**: Voice Activity Detection using Silero ONNX model
  - VAD analyzer with state machine (QUIET, STARTING, SPEAKING, STOPPING)
  - Configurable parameters: confidence threshold, start/stop delays, min volume
  - Supports 8kHz and 16kHz sample rates
  - Automatic model state reset every 5 seconds (prevents memory growth)
  - Location: `src/audio/vad/`

- **VAD State Machine**:
  - `VADState` enum with 4 states (QUIET, STARTING, SPEAKING, STOPPING)
  - `VADParams` for configurable thresholds (confidence=0.7, start=0.2s, stop=0.8s, min_volume=0.6)
  - Frame counting with exponential volume smoothing (factor: 0.2)
  - Location: `src/audio/vad/vad_analyzer.go`

- **SileroVAD ONNX Model**:
  - Pre-trained Silero VAD model (silero_vad.onnx - 2.2MB)
  - ONNX Runtime Go bindings (github.com/yalue/onnxruntime_go v1.10.0)
  - Model inputs: audio (1, samples), state (2, 1, 128), sample_rate (scalar)
  - Model outputs: confidence (0.0-1.0), new_state (2, 1, 128)
  - Context windowing: 32 samples (8kHz), 64 samples (16kHz)
  - Location: `src/audio/vad/silero.go`, `src/audio/vad/data/silero_vad.onnx`

- **VAD Input Processor**:
  - Accumulates audio frames until enough samples for VAD analysis
  - Emits `UserStartedSpeakingFrame` when user starts speaking
  - Emits `UserStoppedSpeakingFrame` when user stops speaking
  - Passes through all audio to downstream processors (STT)
  - Location: `src/audio/vad/vad_processor.go`

- **Example with VAD**:
  - Complete Asterisk WebSocket example with SileroVAD integration
  - Demonstrates VAD parameter configuration
  - Shows proper cleanup of VAD resources
  - Location: `examples/voice_call_with_vad.go`

### Technical Details

- **VAD State Machine Logic**:
  ```
  QUIET → (confidence ≥ threshold) → STARTING
  STARTING → (sustained for start_secs) → SPEAKING
  SPEAKING → (confidence < threshold) → STOPPING
  STOPPING → (sustained for stop_secs) → QUIET
  STOPPING → (confidence ≥ threshold) → SPEAKING  # Quick recovery
  ```

- **Volume Smoothing**:
  - RMS (Root Mean Square) volume calculation from int16 audio
  - Exponential smoothing: `smoothed = 0.2 * current + 0.8 * previous`
  - Filters out audio below `min_volume` threshold

- **ONNX Model Integration**:
  - Thread-safe model wrapper with mutex protection
  - Single-threaded execution (intra_op=1, inter_op=1)
  - Dynamic session for variable-sized inputs
  - Tensor shapes validated before inference

- **Architecture**:
  ```
  Asterisk WebSocket → AudioFrame
    ↓
  VADInputProcessor: Accumulate audio
    ↓
  VADAnalyzer: Run Silero model (256/512 samples)
    ↓
  State Machine: Track QUIET/STARTING/SPEAKING/STOPPING
    ↓
  Emit: UserStartedSpeakingFrame / UserStoppedSpeakingFrame
    ↓
  Pass through: AudioFrame → STT Service
  ```

### Dependencies

- Added `github.com/yalue/onnxruntime_go v1.10.0` for ONNX Runtime support
- Requires ONNX Runtime C libraries (installation varies by platform)

### Notes

- **Performance**: Silero VAD adds ~1ms latency per 512 samples (16kHz)
- **Accuracy**: Pre-trained model with high voice detection accuracy
- **Memory**: Model state (512 bytes) + context buffer (~256 bytes)
- **Compatibility**: Works with all WebSocket transports (Asterisk, Twilio, etc.)

---

## [0.0.6] - 2025-11-15

### Fixed
- **CRITICAL**: Fixed audio cutout issue caused by overwhelming WebSocket/Asterisk buffer with large TTS responses
  - Implemented rate-limited audio delivery for smooth playback
  - Added chunk queue (buffered channel) to decouple chunking from sending
  - Created sender goroutine with ticker-based pacing for controlled delivery
  - Send interval calculated as: `(chunkSize / sampleRate) / 2`
    - For mulaw/alaw (160 bytes at 8kHz): 10ms per chunk
    - For linear16 (320 bytes at 16kHz): 10ms per chunk
  - Large audio frames now sent over appropriate duration instead of burst
    - Example: 1648 chunks (263KB) sent over ~16 seconds instead of <1ms
  - Prevents buffer overflow that caused permanent audio silence
  - Location: `src/transports/websocket.go:277-547`

### Technical Details
- **Problem**: Large TTS audio frames (263KB+) were chunked and sent in tight loop
  - All 1648 chunks sent in microseconds, overwhelming Asterisk buffer
  - Result: Audio played briefly then stopped permanently (no recovery)
  - Flow control (MEDIA_XOFF/MEDIA_XON) couldn't prevent initial burst

- **Solution (Rate-Limited Delivery)**:
  - **Layer 1 - Chunking**: Buffer audio and chunk into appropriate sizes
    - Chunks queued to buffered channel instead of immediate send
    - Pre-serialize chunks before queueing for efficiency

  - **Layer 2 - Rate-Limited Sender**:
    - Dedicated goroutine consumes from chunk queue
    - Uses `time.Ticker` for precise interval pacing
    - Sleeps between sends to simulate audio device timing
    - Dynamic ticker adjustment for codec changes

  - **Interruption Handling**:
    - Drains chunk queue on InterruptionFrame (removes pending chunks)
    - Clears audio buffer (existing behavior)
    - Sends FLUSH_MEDIA to server (existing behavior)

- **Architecture**:
  ```
  TTS → Large Audio Frame
    ↓
  handleAudioFrame: Buffer + Chunk
    ↓
  chunkQueue (channel) ← Non-blocking enqueue
    ↓
  Sender Goroutine: Dequeue with rate limiting
    ↓
  Sleep (10ms for telephony codecs)
    ↓
  WebSocket Send → Smooth delivery to Asterisk
  ```

### Added
- `audioChunk` struct for pre-serialized chunks with metadata
- `calculateSendInterval()` function for pacing calculation
- `startChunkSender()` goroutine for rate-limited delivery
- `Cleanup()` method for graceful shutdown of sender goroutine
- Comprehensive logging for chunk queueing and pacing

### Changed
- `handleAudioFrame()` now queues chunks instead of immediate send
- InterruptionFrame handler now drains chunk queue in addition to clearing buffer
- Added `time` import for ticker-based pacing

## [0.0.5] - 2025-11-14

### Fixed
- **CRITICAL**: Fixed audio buffer not being cleared during interruptions
  - WebSocketOutputProcessor now handles InterruptionFrame to clear buffered audio chunks
  - Prevents residual audio from playing after user interrupts the bot
  - Location: `src/transports/websocket.go:299-322`

### Added
- **Server-Side Buffer Flush Support**
  - Asterisk serializer now sends `FLUSH_MEDIA` command on InterruptionFrame
  - Twilio serializer sends `clear` event on InterruptionFrame
  - Follows standard pattern for telephony providers (Twilio, Telnyx, Plivo, Exotel)
  - Locations:
    - `src/serializers/asterisk.go:133-145`
    - `src/serializers/twilio.go:87-98`

### Technical Details
- **Two-Layer Interruption Handling**
  - **Layer 1 (Client-Side)**: WebSocketOutputProcessor clears local `audioBuffer`
    - Discards audio chunks waiting to be sent to server
    - Prevents buffered audio from being transmitted

  - **Layer 2 (Server-Side)**: Protocol-specific flush commands
    - Asterisk: `{"command":"FLUSH_MEDIA"}` (JSON format, per Asterisk WebSocket docs)
    - Twilio: `{"event":"clear","streamSid":"..."}` (Twilio Media Streams API)
    - Clears server-side audio buffers (~900 frames for Asterisk)
    - Stops any bulk audio transfer in progress

- **Interruption Flow**:
  ```
  User speaks → InterruptionFrame
  ├─ TTS Service: Stops generating new audio
  ├─ WebSocketOutput: Clears local buffer
  └─ Serializer: Sends flush command to server
  Result: Immediate audio stop (no lag)
  ```

### References
- Asterisk WebSocket Protocol: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/

## [0.0.4] - 2025-11-14

### Fixed
- **CRITICAL**: Fixed audio mixing bug when interruption conditions not met
  - UserAggregator now consumes TranscriptionFrames instead of passing them downstream
  - Prevents LLM from receiving transcriptions when interruption conditions are not met
  - Fixes audio mixing issue where new TTS audio played over existing audio
  - Location: `src/processors/aggregators/user.go:153-155`

### Changed
- **BREAKING**: LLM services now only accept LLMContextFrame (removed backward compatibility)
  - Removed TranscriptionFrame handler from Gemini LLM service
  - Removed TranscriptionFrame handler from OpenAI LLM service
  - All LLM services must now be used with aggregators (UserAggregator)
  - Enforces proper pipeline architecture: STT → UserAggregator → LLMContextFrame → LLM
  - Locations:
    - `src/services/gemini/llm.go:86-113`
    - `src/services/openai/llm.go:86-113`

### Removed
- Dead code cleanup after backward compatibility removal
  - Removed unused `generateResponse()` function from OpenAI LLM service (98 lines)
  - Function became unreachable after removing TranscriptionFrame handling
  - All functionality preserved via `generateResponseFromContext()` method
  - Location: `src/services/openai/llm.go`

### Technical Details
- **Root Cause**: TranscriptionFrames were being passed downstream even when input was discarded
  - UserAggregator logged "discarding input" but still pushed frames downstream
  - LLM service's backward compatibility mode processed these frames anyway
  - Result: New LLM responses generated and mixed with currently playing audio

- **Fix Strategy**: Enforce single responsibility principle
  - UserAggregator: Consumes TranscriptionFrames, emits LLMContextFrames
  - LLM Services: Only process LLMContextFrames (no direct transcription handling)
  - Pipeline integrity: Each processor has clear input/output contracts

- **Migration Impact**: Applications not using aggregators will break
  - Required: Use UserAggregator between STT and LLM services
  - See: `examples/voice_call_complete.go` for proper pipeline setup

## [0.0.3] - 2025-11-14

### Fixed
- **CRITICAL**: Fixed text duplication in LLMUserAggregator when handling interim and final transcriptions
  - Interim transcriptions are no longer appended to aggregation
  - Only final transcriptions are now added to the aggregation buffer
  - Prevents duplicate text being sent to LLM (e.g., "how are you doing how are you doing")
  - Removed unused `lastInterimText` field and complex replacement logic
  - Location: `src/processors/aggregators/user.go:115-151`

## [0.0.2] - 2025-11-14

### Fixed
- **CRITICAL**: Fixed TTSStartedFrame being emitted multiple times (once per text chunk)
  - Implemented boolean flag pattern with `isSpeaking` state tracking
  - Added `sync.Mutex` for thread-safe concurrent access
  - Location: `src/services/elevenlabs/tts.go`

- **CRITICAL**: Fixed frame direction for TTS state tracking frames
  - Changed TTSStartedFrame/TTSStoppedFrame to push UPSTREAM instead of downstream
  - UserAggregator can now correctly track `botSpeaking` state
  - Without this fix, interruptions would not work at all
  - Location: `src/services/elevenlabs/tts.go`

- Added InterruptionFrame handler to TTS service
  - Resets `isSpeaking` flag when interrupted
  - Emits TTSStoppedFrame upstream on interruption

- Added proper state reset on LLMFullResponseEndFrame
  - Non-streaming mode resets immediately
  - Streaming mode resets when `isFinal` received from ElevenLabs

### Changed
- Updated aggregators README to document critical bug fixes
- Updated IMPLEMENTATION_COMPLETE.md with detailed bug analysis

## [0.0.1] - 2025-11-14

### Added
- **LLM Context Aggregators System**
  - Base LLMContextAggregator with shared functionality
  - LLMUserAggregator for user input accumulation and interruption logic
  - LLMAssistantAggregator for LLM response tracking and context updates
  - Background aggregation task with timeout handling
  - Location: `src/processors/aggregators/`

- **Intelligent Interruption System**
  - InterruptionTaskFrame (upstream) → InterruptionFrame (downstream) flow
  - Strategy-based interruption decisions via InterruptionStrategy interface
  - MinWordsInterruptionStrategy (configurable word threshold)
  - Bot speaking state tracking via TTS frames
  - Automatic input discard when interruption conditions not met

- **Enhanced LLM Context**
  - Tools and ToolChoice support for function calling
  - ToolCall and FunctionCall structures
  - Multi-turn conversation state management
  - Location: `src/services/service.go`

- **New Frame Types**
  - LLMContextFrame - Carries conversation context to LLM
  - LLMMessagesAppendFrame - Append messages to context
  - LLMMessagesUpdateFrame - Replace entire message history
  - FunctionCallsStartedFrame - Function call initiated
  - FunctionCallInProgressFrame - Function call executing
  - FunctionCallResultFrame - Function call completed
  - FunctionCallCancelFrame - Function call cancelled
  - Location: `src/frames/control.go`

- **LLM Service Enhancements**
  - LLMContextFrame handler in OpenAI service
  - Function calling support (tools, tool_choice, function calls)
  - Backward compatibility with existing TextFrame handling
  - Location: `src/services/openai/llm.go`

- **TTS State Tracking**
  - TTSStartedFrame emission at synthesis start
  - TTSStoppedFrame emission at synthesis end
  - State tracking for interruption support
  - Location: `src/services/elevenlabs/tts.go`

- **Documentation**
  - Comprehensive aggregators guide (`docs/AGGREGATORS.md` - 523 lines)
  - Implementation plan (`AGGREGATOR_IMPLEMENTATION_PLAN.md` - 750+ lines)
  - Standalone interruption demo (`examples/aggregators_with_interruptions.go` - 254 lines)
  - Complete API reference and troubleshooting guide

### Changed
- Updated `examples/voice_call_complete.go` to use aggregators
  - Configured interruption support with MinWordsInterruptionStrategy
  - Integrated LLMUserAggregator and LLMAssistantAggregator
  - Enabled intelligent interruptions with 3-word threshold

### Technical Details
- Total lines of code added: ~964 lines
- New components: 3 aggregator files
- Enhanced components: 5 existing files
- Documentation: 2 comprehensive docs + 1 implementation plan
- Examples: 2 working examples
- Compilation: All code verified
- Test coverage: Complete flow verified
