# 01 — Make AudioGuard authoritative and delete duplicate transport gate

Status: needs-triage

## Parent

[ADR-0001 — Stale-audio filtering lives in a dedicated AudioGuard processor](../../../docs/adr/0001-audio-guard-as-dedicated-processor.md)

Phase 1 (shadow-mode AudioGuard) shipped in PR #41.

## What to build

Flip the AudioGuard from shadow mode to authoritative, and delete the duplicate stale-audio filter that lives in `WebSocketOutputProcessor`. After this slice, AudioGuard is the only place in the pipeline enforcing "audio reaching the wire belongs to the active Generation."

End-to-end behavior:

- An interrupted user-bot exchange produces zero stale audio chunks at the wire.
- The new `AudioGuard would-drop` log lines from Phase 1 become `AudioGuard drop` (real drops).
- `WebSocketOutputProcessor` no longer carries the gate state fields (`interrupted`, `currentContextID`, `expectedContextID`, `interruptionMu`, `staleAudioBlockedCount`, `lastStaleContextID`).

## Acceptance criteria

- [ ] `AudioGuard.Active` is set to `true` at construction or via a constructor option used in all example pipelines.
- [ ] Gate state fields removed from `WebSocketOutputProcessor` ([src/transports/websocket.go:321-340](../../../src/transports/websocket.go)).
- [ ] `TTSStartedFrame` handler in `WebSocketOutputProcessor.HandleFrame` collapses to draining the playback-done channel and resetting `llmResponseEnded` ([src/transports/websocket.go:644-684](../../../src/transports/websocket.go)).
- [ ] Filter block in `handleAudioFrame` removed ([src/transports/websocket.go:820-905](../../../src/transports/websocket.go)).
- [ ] Mid-stream interrupted-flag checks in the sender goroutine and chunker removed ([src/transports/websocket.go:436-444](../../../src/transports/websocket.go), [src/transports/websocket.go:939-948](../../../src/transports/websocket.go)).
- [ ] Tests pinned to private gate fields in [src/transports/websocket_test.go](../../../src/transports/websocket_test.go) deleted; behaviour now covered by AudioGuard tests.
- [ ] AudioGuard inserted between TTS and `transport.Output()` in every example pipeline that uses TTS, not just `audio_interruption_example.go`.
- [ ] `go test ./...` clean.
- [ ] Real voice call with a mid-sentence interruption produces no stale audio at the wire.

## Blocked by

Phase 1 validation (manual): a real voice call against `audio_interruption_example.go` from PR #41 must show that AudioGuard's `would-drop` log lines agree with the existing transport gate's `Blocked stale audio` log lines. If they diverge, the divergence must be understood and resolved before this slice.

## Comments
