# 04 — Adopt kill-task-recreate pattern in Pacer for interruption

Status: needs-triage

## Parent

[ADR-0001 — Stale-audio filtering lives in a dedicated AudioGuard processor](../../../docs/adr/0001-audio-guard-as-dedicated-processor.md)

Phase 3 cleanup. Behavior change isolated from the Pacer extraction (#02) so it can be reverted independently if it misbehaves under rapid back-to-back interrupts.

## What to build

Replace the queue-drain + flag pattern in the Pacer's `InterruptionFrame` handling with full sender goroutine restart and chunkQueue replacement. This is the pattern Pipecat's `BaseOutputTransport.MediaSender` uses on `handle_interruptions`.

Today (after #02 lands), interrupt handling drains the chunkQueue and continues using the same sender goroutine. There's a residual one-frame race where audio admitted by AudioGuard at moment T₁ can arrive at the Pacer just as `InterruptionFrame` fires — the audio gets queued *after* the drain completes and gets sent.

After this slice:

```go
func (p *Pacer) Restart() {
    p.senderCancel()
    p.senderWg.Wait()
    close(p.chunkQueue)
    p.chunkQueue = make(chan *audioChunk, 1000)
    p.senderCtx, p.senderCancel = context.WithCancel(context.Background())
    p.startChunkSender()
}
```

The new chunkQueue is fresh; any frame submission that races with the restart lands in the new queue (which is empty and current). The dead sender goroutine is fully drained before the new one starts. The race window is gone.

## Acceptance criteria

- [ ] `Pacer.Restart()` method implemented per the design above.
- [ ] `WebSocketOutputProcessor.HandleFrame` `InterruptionFrame` branch calls `pacer.Restart()` instead of `pacer.Drain()`.
- [ ] `Pacer.Drain()` either deleted or kept as a no-op alias for `Restart()` for backward compatibility.
- [ ] New unit test: rapid back-to-back interruptions (5 within 50ms) should leave the Pacer in a coherent state with a single live sender goroutine and an empty queue.
- [ ] New unit test: a `Submit()` racing with a `Restart()` does not panic on send-to-closed-channel; the submission either lands in the new queue or returns a benign error.
- [ ] Existing transport tests still pass.
- [ ] `go test ./... -race` clean (race detector explicitly enabled because this slice is concurrency-sensitive).
- [ ] Real voice call with rapid interruptions (user interrupts, immediately interrupts again) does not produce stale audio or a wedged sender.

## Blocked by

- [02-extract-pacer-submodule.md](02-extract-pacer-submodule.md)

Needs the Pacer to exist as a discrete module.

## Comments
