# Interruptions cancel an epoch context; services split into shared bases and providers

The frame engine was rebuilt around three decisions. Each **FrameProcessor** keeps a system queue and a data queue in growable ring buffers, drained by one goroutine each, with `HandleFrame` calls serialized by a per-processor mutex unless the processor opts out (`WithConcurrentSystemFrames`). An **Interruption** is enforced in the engine rather than in each processor: queueing an `InterruptionFrame` increments the processor's epoch, cancels the context handed to data handlers of the previous epoch, and drops queued frames that do not survive interruption. And LLM, STT and TTS services are each one shared implementation (`llmservice.Service`, `stt.Base`, `tts.Base`) driven by a small vendor **Provider** interface, with the conversation held in a copy-on-write `llm.Context` that only the context aggregators write.

## Considered Options

- **Keep bounded channels with no handler lock (v0.0.12).** Rejected: system and data handlers raced on shared state, every processor re-implemented interruption filtering, and blocking `QueueFrame` on a full channel could stall the processor upstream of a slow one.
- **Per-processor interruption handling with `context_id` filters (the **Generation** approach, ADR-0001).** Rejected as the primary mechanism: it only covers TTS audio, and each blocking handler (LLM stream, HTTP TTS, function-call wait) still needed its own cancellation path. The epoch context gives every handler cancellation by honouring `ctx`. `AudioGuard` and TTS context ids remain, but no longer carry correctness.
- **Bounded queues with back-pressure.** Not adopted: blocking a producer on a full queue reintroduces cross-processor stalls during interruptions. The cost is that a processor that cannot keep up accumulates a backlog.
- **Each vendor service implements the full processor (v0.0.12).** Rejected: frame handling, metrics, settings updates, function execution and interruption behaviour were duplicated and drifted between vendors.

## Consequences

- `EndFrame` became an ordered, uninterruptible control frame that every processor must forward; `CancelFrame` stays a system frame.
- A handler that blocks must honour its `ctx`; long-lived work uses the processor-lifetime context (`p.Context()`). `NewLegacyProcessor` keeps the old behaviour for processors that cannot be converted.
- LLM services never write to the conversation. Assistant text, tool calls and tool results are written by the assistant aggregator from the frames it observes; application code changes the context through frames while a pipeline runs.
- A new vendor integration is a provider of a few methods; the shared base owns frames, metrics, settings and interruption handling.
- Queues are unbounded: memory, not latency, is the failure mode when a processor falls behind.
- Transports moved to one session per connection (`transports.NewServer` and a session handler), so each call owns its pipeline.
