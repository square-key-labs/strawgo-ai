# 02 — Extract Pacer submodule from WebSocketOutputProcessor

Status: needs-triage

## Parent

[ADR-0001 — Stale-audio filtering lives in a dedicated AudioGuard processor](../../../docs/adr/0001-audio-guard-as-dedicated-processor.md)

Phase 3 of the deepening plan (split the god-object output processor).

## What to build

Extract the chunk queue, real-time pacing logic, and sender goroutine from `WebSocketOutputProcessor` into a new `Pacer` submodule under `src/transports/pacer.go`. Behavior-preserving refactor — no semantic change. The processor delegates audio chunking and rate-limited send to the Pacer; the Pacer surfaces events the BotSpeechTracker and PlaybackAckTracker will subscribe to in the next slice.

The Pacer owns:

- `chunkQueue` channel
- Sender goroutine (`startChunkSender`, `calculateSendInterval`)
- `audioBuffer` and chunking logic from `handleAudioFrame`
- `cleanupOnce` / sender cancellation

The Pacer exposes:

- `Submit(chunk)` for the data path
- `Drain()` for the existing interruption path (kill-task-recreate is a separate slice)
- Event subscriptions for "first chunk sent" and "audio idle for N" (used by trackers in #03)
- `Cleanup()`

After this slice, `WebSocketOutputProcessor.handleAudioFrame` shrinks to: chunk → `pacer.Submit(...)`. The 200-line method becomes ~30 lines.

## Acceptance criteria

- [ ] New file [src/transports/pacer.go](../../../src/transports/pacer.go) containing the Pacer struct and methods.
- [ ] New file [src/transports/pacer_test.go](../../../src/transports/pacer_test.go) with unit tests covering chunking, rate-limiting (mock clock), drain, and event emission.
- [ ] `WebSocketOutputProcessor` constructs a `Pacer` and delegates audio handling to it.
- [ ] `chunkQueue`, `senderCtx`, `senderCancel`, `senderWg`, `audioBuffer`, `chunkSize` fields removed from `WebSocketOutputProcessor` ([src/transports/websocket.go:306-315](../../../src/transports/websocket.go)).
- [ ] `startChunkSender` and `calculateSendInterval` move to the Pacer ([src/transports/websocket.go:361-581](../../../src/transports/websocket.go)).
- [ ] `handleAudioFrame` reduces to chunk-and-submit; bot-speech and ack logic stays in the processor for now (it moves out in #03).
- [ ] Existing transport tests still pass.
- [ ] `go test ./...` clean.

## Blocked by

- [01-make-audioguard-authoritative.md](01-make-audioguard-authoritative.md)

The transport gate must be gone before extracting Pacer; otherwise the extraction has to carry the gate state across the new boundary.

## Comments
