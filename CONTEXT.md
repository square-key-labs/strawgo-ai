# StrawGo

A Go framework for building real-time conversational AI applications over voice transports (telephony WebSocket media streams such as Twilio and Asterisk, and WebRTC). Pipeline-based: discrete data units flow through a chain of processors, with strict separation between the network transport and the protocol serializer.

## Language

### Pipeline primitives

**Frame**:
A discrete unit of data flowing through the pipeline. Carries audio, text, or lifecycle signals.
_Avoid_: message, event, packet

**FrameProcessor**:
A node in the pipeline that consumes Frames, transforms them, and emits Frames upstream or downstream. Has a system queue and a data queue, each drained by its own goroutine; `HandleFrame` calls are serialized unless the processor opts into concurrent system frames.
_Avoid_: handler, stage, filter, middleware

**Pipeline**:
An ordered chain of FrameProcessors wired by `Link` / `SetPrev`. A Pipeline is itself a FrameProcessor, so pipelines nest.
_Avoid_: chain, graph, flow

**PipelineTask**:
Owns one Pipeline: sends the `StartFrame`, injects application frames, and finishes when an `EndFrame`, `StopFrame` or `CancelFrame` reaches its sink.

**Direction**:
Either `Downstream` (toward output transport) or `Upstream` (toward input/aggregators).

### Frame categories

**System Frame**:
High-priority Frame handled from the system queue, ahead of queued data.
Examples: `StartFrame`, `CancelFrame`, `InterruptionFrame`, `ErrorFrame`, `UserStartedSpeakingFrame`, `MetricsFrame`.
_Avoid_: urgent frame

**Control Frame**:
Lifecycle or instruction Frame that flows in order on the data queue. Some carry payload (`LLMContextFrame`).
Examples: `EndFrame`, `LLMContextFrame`, `TTSStartedFrame`, `LLMFullResponseEndFrame`, `FunctionCallResultFrame`.

**Data Frame**:
Carries payload. Examples: `AudioFrame`, `TTSAudioFrame`, `TextFrame`, `LLMTextFrame`, `TranscriptionFrame`.

**Uninterruptible Frame**:
A control or data Frame whose `SurvivesInterruption()` is true: it is neither dropped from queues nor skipped when an Interruption arrives. Examples: `EndFrame`, `FunctionCallInProgressFrame`, `FunctionCallResultFrame`, settings-update frames, input `AudioFrame`. Category is independent of this property.

### Conversational lifecycle

**Turn**:
One user-bot exchange. Bounded by `UserTurnStartStrategy` and `UserTurnStopStrategy`.

**Generation**:
One TTS synthesis pass within a Turn. Identified by a UUID stamped as `ContextID` on `TTSStartedFrame` and `TTSAudioFrame`. A Turn may contain multiple Generations. TTS services use it to address upstream vendor contexts (send, flush, cancel); stale-audio removal no longer depends on it (see Epoch).
_Avoid_: context (clashes with `context.Context`, `llm.Context` and Epoch context), utterance, response

**Interruption**:
The user starts a turn while the bot is speaking or a response is pending. Triggers an `InterruptionFrame` broadcast both directions, which advances every processor's Epoch.

**Epoch**:
A per-FrameProcessor counter incremented when an `InterruptionFrame` is queued on that processor. Queued Frames from an older Epoch that are not Uninterruptible are dropped or skipped.

**Epoch context**:
The `context.Context` passed to a FrameProcessor's data-queue handler. Cancelled when the Epoch advances, so a handler blocked on I/O (LLM stream, HTTP TTS request) stops by honouring its `ctx`. Distinct from the processor-lifetime context (`p.Context()`), which lasts until the processor stops.
_Avoid_: request context, turn context

**Mute**:
A period during which user input is suppressed and interruptions are not raised. Configured via `UserMuteStrategy`.

### Conversation state

**Conversation context**:
The `*llm.Context`: messages, tools and tool choice for one conversation. Copy-on-write; `Snapshot()` gives an immutable view to providers.
_Avoid_: history, transcript, memory

**Single-writer rule**:
Only the context aggregators (user and assistant) modify the Conversation context while a pipeline runs. LLM services read snapshots and never write; application code changes the context by queueing frames (`LLMMessagesAppendFrame`, `LLMSetToolsFrame`, ...).

### Services

**Service**:
The FrameProcessor shared by all vendors of one kind: `llmservice.Service`, `stt.Base`, `tts.Base`. Owns frame handling, metrics, settings updates, function execution and interruption behaviour.

**Provider**:
The vendor-specific part plugged into a Service: `llm.Provider` (`Stream`), `stt.StreamingProvider` / `stt.SegmentedProvider`, `tts.StreamingProvider` / `tts.HTTPProvider`. Does I/O only.
_Avoid_: adapter, driver, backend

### Output-side modules

**Pacer**:
Part of the session output processor. Chunks audio into wire-sized pieces, emits at real-time playback rate, tracks bot speech (`BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame`) and playback acknowledgements. Owns the chunk queue and the sender goroutine.

**Playback ack**:
Client confirmation that queued audio finished playing (Twilio mark echo, Asterisk queue drained). When enabled and the Serializer is a `PlaybackAckSerializer`, the Pacer waits for it before declaring `BotStoppedSpeakingFrame`, with a timeout fallback.

**AudioGuard**:
A processor that drops `TTSAudioFrame`s whose `ContextID` does not match the active Generation. Still in the tree (`src/processors/audio_guard.go`) but not used by any transport or example; the Epoch mechanism removes stale audio.

### Transport seam

**Transport**:
The network layer. Owns connection lifecycle. Examples: the WebSocket `transports.Server`, the WebRTC `webrtc.Handler`.

**Session**:
One connection (call or peer) accepted by a Transport. Exposes `Input()` and `Output()` processors and call information; the application builds one Pipeline per Session in a session handler.
_Avoid_: connection, call (when the object is meant)

**Serializer**:
The protocol layer. Converts between Frames and wire messages. Created per Session and injected into the Transport. Examples: `TwilioFrameSerializer`, `AsteriskFrameSerializer`.
_Avoid_: codec, encoder, adapter

## Relationships

- A **Pipeline** contains many **FrameProcessors** linked by direction; a **PipelineTask** runs one **Pipeline**
- A **Transport** accepts many **Sessions**; each **Session** owns one **Serializer** and feeds one **Pipeline**
- A **Turn** contains zero or more **Generations**
- An **Interruption** advances the **Epoch** of every **FrameProcessor** it reaches and cancels their **Epoch contexts**
- A **Service** drives exactly one **Provider**
- The **Conversation context** has one writer per side: the user aggregator and the assistant aggregator
- A **Pacer** waits for a **Playback ack** only when its **Session**'s **Serializer** is a `PlaybackAckSerializer`

## Example dialogue

> **Dev:** "If the user interrupts mid-sentence, what stops the rest of the TTS audio from reaching the phone?"
> **Architect:** "The **InterruptionFrame** advances the **Epoch** of the TTS service and the session output. Queued text and audio from the old **Epoch** are dropped, the TTS **Provider** is told to cancel its open **Generations**, and the **Pacer** throws away its chunk queue."
> **Dev:** "And the phone — does it stop playing whatever it had buffered?"
> **Architect:** "That's the **Serializer**'s job. On `InterruptionFrame`, it produces a wire-level flush command (Twilio `clear`, Asterisk equivalent) and the **Session** sends it. The next **Generation** begins on the next `TTSStartedFrame`."

## Flagged ambiguities

- **"context"** means four different things: Go's `context.Context`, the **Epoch context** (a specific `context.Context`), the **Conversation context** (`llm.Context`), and the TTS generation ID. Resolved: the TTS ID is a **Generation**; the others are always qualified.
- **"audio context"** in upstream TTS APIs (Cartesia, ElevenLabs) refers to a single in-flight **Generation** the service is tracking against the upstream API. TTS services mint `ContextID`s, stamp outgoing **TTSAudioFrames**, and use them when calling upstream cancel APIs.
- **"bot stopped speaking"** has two meanings: (a) server stopped sending audio, (b) phone finished playing audio. With **Playback ack** enabled, `BotStoppedSpeakingFrame` means (b).
- **"provider"** also names the telephony vendor in `serializers.CallInfo.Provider`. In service code, **Provider** means the vendor plug-in of a **Service**.
