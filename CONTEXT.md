# StrawGo

A Go framework for building real-time conversational AI applications over voice transports (Twilio, Asterisk, Daily). Pipeline-based: discrete data units flow through a chain of processors, with strict separation between the network transport and the protocol serializer.

## Language

### Pipeline primitives

**Frame**:
A discrete unit of data flowing through the pipeline. Carries audio, text, or lifecycle signals.
_Avoid_: message, event, packet

**FrameProcessor**:
A node in the pipeline that consumes Frames, transforms them, and emits Frames upstream or downstream. Has its own goroutines and queue.
_Avoid_: handler, stage, filter, middleware

**Pipeline**:
An ordered chain of FrameProcessors wired by `Link` / `SetPrev`.
_Avoid_: chain, graph, flow

**Direction**:
Either `Downstream` (toward output transport) or `Upstream` (toward input/aggregators).

### Frame categories

**System Frame**:
High-priority Frame that preempts the data channel. Routed via a separate goroutine.
Examples: `InterruptionFrame`, `StartFrame`, `EndFrame`.
_Avoid_: control frame, urgent frame

**Control Frame**:
Lifecycle marker carrying no payload. Flows on the data channel.
Examples: `TTSStartedFrame`, `LLMFullResponseEndFrame`, `BotStartedSpeakingFrame`.

**Data Frame**:
Carries payload. Examples: `AudioFrame`, `TTSAudioFrame`, `TextFrame`.

### Conversational lifecycle

**Turn**:
One user-bot exchange. Bounded by `UserTurnStartStrategy` and `UserTurnStopStrategy`.

**Generation**:
One TTS synthesis pass within a Turn. Identified by a UUID stamped as `context_id` metadata on `TTSAudioFrame` and as `ContextID` on `TTSStartedFrame`. A Turn may contain multiple Generations as the LLM streams sentences. On interruption, the active Generation is invalidated; subsequent audio matching the invalidated Generation is dropped.
_Avoid_: context (clashes with `context.Context` and `LLMContext`), utterance, response

**Interruption**:
The user starts speaking while the bot is speaking. Triggers an `InterruptionFrame` broadcast both directions. Invalidates the active Generation and drains in-flight audio.

**Mute**:
A period during which interruptions are suppressed. Configured via `UserMuteStrategy`.

### Output-side modules

**AudioGuard**:
The pipeline processor that enforces "TTSAudioFrame reaching the wire belongs to the active Generation." Single source of truth for stale-audio filtering. Sits between TTS service and output transport. See [src/processors/audio_guard.go].

**Pacer**:
Submodule of the output transport. Chunks `TTSAudioFrame` into wire-sized pieces and emits at real-time playback rate. Owns the chunk queue and the sender goroutine.

**BotSpeechTracker**:
Submodule of the output transport. Emits `BotStartedSpeakingFrame` on first chunk and `BotStoppedSpeakingFrame` when audio flow ends and the LLM has finished generating.

**PlaybackAckTracker**:
Telephony-specific submodule. Waits for the client to confirm playback complete (Twilio mark echo, Asterisk QUEUE_DRAINED) before declaring `BotStoppedSpeakingFrame`. Falls back to a 10s deadman timer if no ack arrives. Only active when the serializer implements `PlaybackAckSerializer`.

### Transport seam

**Transport**:
The network layer. Owns connection lifecycle. Examples: `WebSocketTransport`, Daily WebRTC transport.

**Serializer**:
The protocol layer. Converts between Frame and wire bytes. Injected into Transport. Examples: `TwilioFrameSerializer`, `AsteriskFrameSerializer`.
_Avoid_: codec, encoder, adapter

## Relationships

- A **Pipeline** contains many **FrameProcessors** linked by direction
- A **Turn** contains zero or more **Generations**
- A **Generation** is created on `TTSStartedFrame` and ends on `TTSStoppedFrame` or `InterruptionFrame`
- An **Interruption** invalidates the active **Generation**
- An **AudioGuard** observes **TTSStartedFrame** to learn the active **Generation**, and drops **TTSAudioFrame** whose `context_id` doesn't match
- A **Transport** owns one or more **PlaybackAckTrackers** only when its **Serializer** is a `PlaybackAckSerializer`
- A **WebSocketOutputProcessor** is a **FrameProcessor** that wires together a **Pacer**, a **BotSpeechTracker**, and a **PlaybackAckTracker**

## Example dialogue

> **Dev:** "If the user interrupts mid-sentence, what stops the rest of the TTS audio from reaching the phone?"
> **Architect:** "The **InterruptionFrame** invalidates the active **Generation** in the **AudioGuard**. Any **TTSAudioFrame** still in flight whose `context_id` matches the invalidated **Generation** gets dropped. The **Pacer** also gets restarted, throwing away anything queued in the chunk queue."
> **Dev:** "And the phone — does it stop playing whatever it had buffered?"
> **Architect:** "That's the **Serializer**'s job. On `InterruptionFrame`, the serializer emits a wire-level flush command (Twilio `clear`, Asterisk equivalent). The transport sends it. Once the user starts speaking, the next **Generation** begins on the next **TTSStartedFrame**."

## Flagged ambiguities

- **"context"** was used in the codebase to mean three different things: Go's `context.Context`, the LLM conversation history (`LLMContext`), and the TTS generation ID. Resolved: only the third is renamed to **Generation**. The other two keep their names.
- **"audio context"** in upstream TTS APIs (Cartesia, ElevenLabs) refers to a single in-flight **Generation** the service is tracking against the upstream API. Stale-audio filtering by `context_id` no longer happens at the TTS-service layer; it lives in the **AudioGuard** processor (see ADR-0001). TTS services still mint `context_id`s, stamp outgoing **TTSAudioFrames**, and use them when calling upstream cancel APIs.
- **"bot stopped speaking"** has two meanings: (a) server stopped sending audio, (b) phone finished playing audio. The **PlaybackAckTracker** distinguishes them. The `BotStoppedSpeakingFrame` always means (b) when the serializer supports playback acks.
