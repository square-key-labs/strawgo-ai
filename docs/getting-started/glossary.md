# Glossary

This page defines every term used across the Strawgo docs, in alphabetical order, in one or two sentences each. Where a term is also a Go type, the type is named so you can find it in the code. The project's internal domain language is in [CONTEXT.md](../../CONTEXT.md); the definitions here agree with it.

## A

### Aggregator

A processor that collects frames over time and turns them into something larger. The context aggregators (`aggregators.NewContextAggregatorPair`) are the main ones: the user aggregator gathers the caller's words for one turn, and the assistant aggregator records what the bot said.

## B

### Barge-in

Another name for an [interruption](#interruption): the caller talks over the bot and the bot stops.

### Bot speaking

The period between `BotStartedSpeakingFrame` and `BotStoppedSpeakingFrame`, emitted by the session output. With playback acknowledgements on, "stopped" means the caller's phone finished playing the audio, not just that the server finished sending it.

## C

### Context

Means different things depending on the word in front of it. The conversation context is the LLM's message history (`*llm.Context`); the epoch context is a Go `context.Context` cancelled on interruption; plain "ctx" in code is Go's `context.Context`.

### Control frame

A frame for lifecycle steps and instructions that must stay in order with the data, such as `EndFrame` or `LLMContextFrame`. It travels on the data queue.

### Conversation context

The `*llm.Context` that holds the messages, tools and tool choice for one conversation. While a pipeline runs, only the context aggregators write to it (the single-writer rule).

## D

### Data frame

A frame that carries payload, such as audio (`AudioFrame`, `TTSAudioFrame`) or text (`TextFrame`, `LLMTextFrame`, `TranscriptionFrame`). It travels on the data queue, in order.

### Direction

Which way a frame travels. Downstream (`frames.Downstream`) goes from input toward output; upstream (`frames.Upstream`) goes back toward the input.

### DTMF

Dual-tone multi-frequency: the tones a phone sends when the caller presses a key. Serializers turn them into DTMF frames, `NewDTMFAggregator` collects digits, and `audio/dtmf` can generate the tones.

## E

### Epoch

A counter each processor keeps and increments when an `InterruptionFrame` reaches it. Queued frames from an older epoch are dropped unless they are uninterruptible.

### Epoch context

The `context.Context` passed to a processor's `HandleFrame` for control and data frames. It is cancelled when the epoch advances, so a handler blocked on network I/O stops as soon as the caller interrupts. It differs from the processor-lifetime context, `p.Context()`, which lasts until the processor stops.

## F

### Flow

A conversation structured as a graph of nodes (`src/flows`). Each node sets the instructions and tools for one stage of the conversation, and function calls move the conversation from node to node.

### Frame

One unit of data moving through a pipeline: a chunk of audio, a piece of text, a transcription, or a signal such as "the user started speaking". Every frame is a system, control or data frame.

### Function call

The LLM asking your program to run a function you declared as a tool. You register a Go handler with `RegisterFunction`; the result goes back to the LLM, which usually answers again with it. Also called a tool call.

## G

### G.711

The standard telephone audio codecs, µ-law and A-law, at 8 kHz. Most telephony serializers use µ-law, and WebRTC in Strawgo uses G.711 by default.

### Generation

One TTS synthesis pass, identified by a UUID stamped as `ContextID` on `TTSStartedFrame` and `TTSAudioFrame`. TTS services use it to address, flush and cancel work at the vendor.

## I

### Idle timeout

A limit on how long a task may go without activity (speech, transcription, LLM text, TTS audio). Set it with `PipelineTaskConfig.IdleTimeout`; by default the task is cancelled when it expires.

### Interruption

The caller starts a new turn while the bot is speaking or about to answer. An `InterruptionFrame` goes through the whole pipeline; queued bot text and audio are dropped and the bot stops talking.

### IVR

Interactive voice response: an automated phone menu ("press 1 for sales"). `src/extensions/ivr` lets a bot navigate one.

## L

### LLM

Large language model: the service that reads the conversation and writes the bot's replies. In Strawgo every LLM is an `*llmservice.Service` driven by a provider.

## M

### MCP

Model Context Protocol: a standard way for a server to offer tools to an LLM. `src/services/mcp` connects to an MCP server over stdio, SSE or streamable HTTP and registers its tools.

### Metrics frame

A system frame (`MetricsFrame`) that carries typed measurements, such as TTFB, processing time and token usage. Services emit them when `EnableMetrics` or `EnableUsageMetrics` is set on the task.

### µ-law

A G.711 audio encoding that squeezes each 16-bit sample into 8 bits. Twilio and many other carriers send and expect 8 kHz µ-law. Also written mu-law or mulaw.

### Mute strategy

A rule that suppresses user input and interruptions for a while, for example until the bot's first reply has finished or while a function call is running. Configured through `UserMuteStrategy`.

## N

### Node

One stage of a flow (`flows.Node`). It sets the role and task messages and the functions the LLM can call while the conversation is in that stage.

## O

### Observer

An object that watches frames as they move between processors without being part of the pipeline. Observers measure latency, track turns, log and export traces; attach them with `task.SetObserver`.

### onnx-worker

A Rust sidecar program that runs the Silero VAD and Smart Turn models. Your Go process talks to it over a Unix socket.

### Opus

A compressed audio codec common in WebRTC. In Strawgo it is available only with the `opus` build tag, which has never been compiled or verified.

## P

### Pacer

The part of the session output that sends audio at real-time speed, in wire-sized chunks, and decides when the bot started and stopped speaking.

### Pipeline

An ordered chain of processors (`pipeline.NewPipeline`). A pipeline is itself a processor, so pipelines can nest.

### Playback acknowledgement

A message from the client confirming that queued audio finished playing, such as a Twilio `mark` echo. When enabled, the session output waits for it before declaring that the bot stopped speaking.

### Processor

A node in the pipeline that receives frames, acts on them and pushes frames on. Every processor embeds `*processors.BaseProcessor`, which gives it a system queue and a data queue.

### Provider

The vendor-specific part plugged into a service, such as the OpenAI provider inside an LLM service. It only does I/O with the vendor. In telephony contexts "provider" can also mean the carrier, as in `CallInfo.Provider`.

## R

### Resampling

Converting audio from one sample rate to another, for example from Twilio's 8 kHz to the pipeline's 16 kHz. Serializers and services resample for you; `audio/resample` does it directly.

### RTVI

A client-server message protocol used by Pipecat's web and mobile client SDKs to exchange transcripts, bot state and actions. `processors/rtvi` implements the server side.

## S

### Serializer

The protocol layer of a transport. It converts between frames and wire messages for one vendor, for example `TwilioFrameSerializer`, and is created once per session.

### Service

The processor shared by all vendors of one kind: `llmservice.Service`, `stt.Base` and `tts.Base`. It handles frames, metrics, settings and interruptions, and drives one provider.

### Session

One connection (one call or one browser peer) accepted by a transport. It exposes `Input()` and `Output()` processors and the call information; you build one pipeline per session.

### Single-writer rule

While a pipeline runs, only the context aggregators change the conversation context. Your code changes it by queueing frames such as `LLMMessagesAppendFrame`.

### Smart turn

A model that listens to the end of the caller's speech and predicts whether they have finished their turn or are only pausing. Strawgo runs Smart Turn through `onnx-worker`, an HTTP endpoint or fal.

### Speech-to-speech

A model that takes audio in and gives audio out directly, without separate STT and TTS, such as OpenAI Realtime or Gemini Live.

### STT

Speech-to-text: the service that turns the caller's audio into text (transcriptions). Streaming STT sends audio continuously; segmented STT sends one recorded segment per utterance.

### System frame

A high-priority frame, such as `StartFrame`, `CancelFrame`, `InterruptionFrame` or `ErrorFrame`, handled on the system queue ahead of queued data.

## T

### Task

The object that runs one pipeline (`*pipeline.PipelineTask`): it sends the start frame, accepts frames from your code with `QueueFrame`, and finishes on `EndFrame`, `StopFrame` or `CancelFrame`.

### Tool

A function you describe to the LLM with a name, description and JSON schema (`llm.FunctionSchema`), so it can ask to call it. See function call.

### Transport

The network layer that carries audio to and from the caller and owns the connection lifecycle. Examples are the WebSocket server (`transports.NewServer`) and the WebRTC handler.

### TTFB

Time to first byte: how long a service takes from receiving a request to producing its first output, such as the first LLM token or first TTS audio. Reported in metrics frames.

### TTS

Text-to-speech: the service that turns the bot's reply text into audio.

### Turn

One stretch of speech from one side of the conversation. A user turn starts and stops according to the turn strategies; the bot answers after the user turn stops.

### Turn strategy

A rule that decides when a user turn starts (VAD, transcription, a minimum number of words) or stops (a speech timeout, a turn analyzer such as Smart Turn). The defaults start on VAD or transcription and stop after a short silence.

### TwiML

Twilio's XML instructions for a call. `<Connect><Stream url="wss://..."/></Connect>` connects a call to your bot's WebSocket.

## U

### Uninterruptible frame

A control or data frame whose `SurvivesInterruption()` returns true, so an interruption does not drop it. `EndFrame`, function-call result frames, settings updates and caller audio are uninterruptible.

## V

### VAD

Voice activity detection: deciding from the audio whether someone is speaking. Strawgo uses Silero VAD through `onnx-worker`.

### Voicemail detection

Telling whether an outbound call was answered by a person or an answering machine. `src/extensions/voicemail` uses an LLM to classify the first words of the call and can hold back the bot's speech until it knows.

## W

### WebRTC

The browser standard for real-time audio and video. Strawgo's WebRTC handler (`transports/webrtc`, built on Pion) accepts browser calls, audio only.

## Next steps

- [Learn the core concepts](core-concepts.md)
- [Read the introduction](introduction.md)
- [Browse all docs](../README.md)
