# Reference

The reference section is the lookup part of the documentation: every type, field, default and frame, derived from the source. Use it when you know what you want and need the exact name or value. For explanations and walkthroughs, start with the [guides](../README.md#guides).

A **frame** is one typed message in the pipeline; a **processor** is one stage; a **pipeline** is the chain of stages; a **task** runs one pipeline for one call.

| Page | What it covers |
| --- | --- |
| [Frames](frames.md) | Every frame type, grouped by topic, with category, interruption behaviour, fields, and who emits and consumes it. Writing your own frame. |
| [Processors](processors.md) | `FrameProcessor`, `BaseProcessor`, concurrency modes, rules for a correct processor, built-in processors. |
| [Pipeline](pipeline.md) | `Pipeline`, parallel pipelines, service switchers, `PipelineTask` config, events and shutdown, `Runner`. |
| [LLM](llm.md) | Messages, context, tools, settings, the `Provider` interface and events, `llmservice.Service`, `openaicompat`. |
| [STT and TTS bases](stt-tts-bases.md) | The speech-to-text and text-to-speech base packages and the reconnecting WebSocket session. |
| [Transports](transports.md) | WebSocket server and client, `TransportParams`, sessions, the WebRTC handler. |
| [Serializers](serializers.md) | The serializer interfaces and every carrier's config and wire behaviour. |
| [Aggregators](aggregators.md) | User and assistant context aggregators, summarization, idle detection, DTMF, sentences, turn completion. |
| [Turns](turns.md) | Start, stop and mute strategies, VAD, smart turn. |
| [Audio](audio.md) | Codecs, resampling, G.711, PCM, WAV, DTMF tones, mixers, filters, classifiers, the converter. |
| [Observers](observers.md) | Observer interfaces, `TaskObserver`, built-in observers, OpenTelemetry tracing. |
| [Flows](flows.md) | Nodes, functions, actions, state, the manager and processor, JSON flows. |
| [Errors](errors.md) | `ErrorFrame`, fatal and non-fatal errors, typed errors, handling patterns. |

Go code on these pages comes in two kinds: complete programs that start with `package main`, and fragments (functions and types) that compile once you add the imports. Plain `text` blocks list signatures and are not meant to compile.

## Next steps

- [Core concepts](../getting-started/core-concepts.md): the ideas behind these APIs.
- [Quickstart](../getting-started/quickstart.md): a working bot in a few minutes.
- [Provider matrix](../providers/README.md): every built-in vendor integration.
