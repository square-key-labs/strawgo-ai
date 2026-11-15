# Strawgo documentation

Strawgo is a Go framework for real-time voice agents: programs that listen to a caller, understand what they say, and answer out loud. This page is the map of the documentation. Start with the "choose your path" table, or browse the full table of contents below it.

## Choose your path

| I want to... | Start here | Then read |
| --- | --- | --- |
| Understand what Strawgo is and whether it fits | [Introduction](getting-started/introduction.md) | [Core concepts](getting-started/core-concepts.md) |
| See a bot answer me in five minutes, with no phone or microphone | [Quickstart](getting-started/quickstart.md) | [Function calling](guides/function-calling.md) |
| Build a phone bot | [Your first phone bot](getting-started/first-phone-bot.md) | [Twilio guide](guides/telephony-twilio.md), [Turn-taking](guides/turn-taking.md) |
| Use a carrier other than Twilio (Telnyx, Plivo, Vonage, Asterisk, ...) | [Telephony providers](guides/telephony-providers.md) | [Transports overview](guides/transports-overview.md) |
| Build a browser voice bot | [WebRTC guide](guides/webrtc.md) | [Browser voice widget recipe](cookbook/browser-voice-widget.md), [RTVI](guides/rtvi.md) |
| Let the bot call my APIs or look things up | [Function calling](guides/function-calling.md) | [MCP tools](guides/mcp-tools.md), [Customer support bot recipe](cookbook/customer-support-bot.md) |
| Script a structured conversation (booking, intake, triage) | [Flows](guides/flows.md) | [Appointment booking recipe](cookbook/appointment-booking-flow.md) |
| Use a speech-to-speech model (OpenAI Realtime, Gemini Live) | [Speech-to-speech](guides/speech-to-speech.md) | [Speech-to-speech providers](providers/speech-to-speech.md) |
| Pick or compare vendors | [Provider matrix](providers/README.md) | [LLM](providers/llm.md), [STT](providers/stt.md), [TTS](providers/tts.md) |
| Add a provider that Strawgo does not have | [Writing a custom provider](providers/custom.md) | [STT and TTS bases](reference/stt-tts-bases.md), [LLM reference](reference/llm.md) |
| Run it in production | [Deployment](guides/deployment.md) | [Observability](guides/observability.md), [Troubleshooting](guides/troubleshooting.md) |
| Fix something that is not working | [Troubleshooting](guides/troubleshooting.md) | [Errors reference](reference/errors.md) |
| Migrate an app written against v0.0.12 or the `v0.1.x` tags | [Migrating from v0.0.12](migration-v0.0.12.md) | [Architecture](architecture.md) |
| Understand the engine internals | [Architecture](architecture.md) | [Architecture decision records](adr/) |

> [!NOTE]
> Provider integrations in Strawgo are tested against mock servers at best, some not at all; none against a live vendor API. The [provider matrix](providers/README.md) lists the status of each one.

> [!WARNING]
> These docs describe the rebuilt API on the `main` branch, which has not been tagged. The tagged releases `v0.1.0` to `v0.1.3` have the older API (the code the `VERSION` file calls `0.0.12`). Install from `main`; see [Installation](getting-started/installation.md#add-strawgo-to-your-module) and [Migrating from v0.0.12](migration-v0.0.12.md).

## Table of contents

### Getting started

- [Introduction](getting-started/introduction.md): what a voice agent is, how a call flows through Strawgo, and when to pick it.
- [Installation](getting-started/installation.md): Go version, adding the module, the optional `onnx-worker` sidecar, API keys.
- [Quickstart](getting-started/quickstart.md): a text chat bot in your terminal, then a tool it can call.
- [Your first phone bot](getting-started/first-phone-bot.md): a Twilio bot end to end, from code to a ringing phone.
- [Core concepts](getting-started/core-concepts.md): frames, processors, pipelines, tasks, context, interruptions.
- [Glossary](getting-started/glossary.md): every term used in these docs, in one or two sentences.

### Guides

- [Function calling](guides/function-calling.md): give the LLM tools and run your Go code when it calls them.
- [Conversation context](guides/conversation-context.md): messages, the system prompt, and summarization.
- [Turn-taking](guides/turn-taking.md): VAD, smart turn, start/stop/mute strategies, interruptions, idle detection.
- [Flows](guides/flows.md): node-based conversations with typed state.
- [Speech-to-speech](guides/speech-to-speech.md): OpenAI Realtime and Gemini Live.
- [MCP tools](guides/mcp-tools.md): expose tools from a Model Context Protocol server to the LLM.
- [Voicemail detection](guides/voicemail-detection.md): tell a person from an answering machine on outbound calls.
- [IVR navigation](guides/ivr-navigation.md): let the bot work its way through a phone menu.
- [Switching services](guides/switching-services.md): swap or fail over between providers at runtime.
- [Structured output and refusals](guides/structured-output-and-refusals.md): constrain model output and handle refusals.
- [Twilio](guides/telephony-twilio.md): Media Streams, playback acknowledgements, hang-up.
- [Telephony providers](guides/telephony-providers.md): Telnyx, Plivo, Exotel, Vonage, Asterisk, Genesys, Jambonz, DTMF, auto hang-up.
- [WebRTC](guides/webrtc.md): browser calls with the Pion-based handler.
- [RTVI](guides/rtvi.md): the RTVI client protocol for web and mobile front ends.
- [WebSocket client](guides/websocket-client.md): dial out to a WebSocket media server.
- [Audio](guides/audio.md): sample rates, resampling, mixers, filters, DTMF tones.
- [Recording](guides/recording.md): record calls and per-turn audio.
- [Observability](guides/observability.md): metrics frames, observers, latency, OpenTelemetry tracing.
- [Deployment](guides/deployment.md): many calls per process, graceful shutdown, the `onnx-worker` sidecar, Docker, security.
- [Troubleshooting](guides/troubleshooting.md): symptom, cause, fix.
- [Transports overview](guides/transports-overview.md): how transports, sessions and serializers fit together.

### Reference

- [Reference index](reference/README.md): what each reference page covers.
- [Frames](reference/frames.md): every frame type, its fields, and who emits or consumes it.
- [Processors](reference/processors.md): the `BaseProcessor` API, options, and built-in processors.
- [Pipeline](reference/pipeline.md): `Pipeline`, `ParallelPipeline`, `SyncParallelPipeline`, `PipelineTask`, `Runner`.
- [LLM](reference/llm.md): the `llm` package and `llmservice.Service`.
- [STT and TTS bases](reference/stt-tts-bases.md): the shared speech service bases.
- [Transports](reference/transports.md): `ServerConfig`, `TransportParams`, `Session`, `webrtc.Config`.
- [Serializers](reference/serializers.md): every wire protocol serializer.
- [Aggregators](reference/aggregators.md): the context aggregators and friends.
- [Turns](reference/turns.md): start, stop and mute strategies.
- [Audio](reference/audio.md): the `audio` packages.
- [Observers](reference/observers.md): the observer interface and the built-in observers.
- [Flows](reference/flows.md): the `flows` package API.
- [Errors](reference/errors.md): `ErrorFrame`, fatal versus non-fatal errors, refusals.

### Providers

- [Provider matrix](providers/README.md): every provider with package, constructor and test status.
- [LLM providers](providers/llm.md)
- [Speech-to-text providers](providers/stt.md)
- [Text-to-speech providers](providers/tts.md)
- [Speech-to-speech providers](providers/speech-to-speech.md)
- [Writing a custom provider](providers/custom.md)

### Cookbook

- [Cookbook index](cookbook/README.md)
- [Customer support bot](cookbook/customer-support-bot.md): function calling and hanging up.
- [Appointment booking flow](cookbook/appointment-booking-flow.md): a flow with typed state.
- [Outbound call with voicemail detection](cookbook/outbound-call-with-voicemail.md)
- [IVR navigator](cookbook/ivr-navigator.md)
- [Call recording and transcript](cookbook/call-recording-and-transcript.md)
- [Multilingual bot](cookbook/multilingual-bot.md)
- [LLM failover](cookbook/llm-failover.md)
- [Browser voice widget](cookbook/browser-voice-widget.md): WebRTC plus a minimal HTML page.
- [Background ambience](cookbook/background-ambience.md): mix a sound file under the bot.
- [Metrics dashboard](cookbook/metrics-dashboard.md): send observer data to your metrics system.

### Project

- [Architecture](architecture.md): how the engine works, for contributors and the curious.
- [Migrating from v0.0.12](migration-v0.0.12.md): breaking API changes between the tagged releases (old API) and `main`.
- [Architecture decision records](adr/)
- [Domain language](../CONTEXT.md): the project's own glossary of internal terms.
- [Changelog](../CHANGELOG.md)
- [Repository README](../README.md)

## Next steps

- [Read the introduction](getting-started/introduction.md)
- [Install Strawgo](getting-started/installation.md)
- [Run the quickstart](getting-started/quickstart.md)
