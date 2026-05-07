# 05 — Delete now-redundant context_id filters from TTS services

Status: needs-triage

## Parent

[ADR-0001 — Stale-audio filtering lives in a dedicated AudioGuard processor](../../../docs/adr/0001-audio-guard-as-dedicated-processor.md)

Phase 4 of the deepening plan (cleanup). Independent of the #02–#04 chain.

## What to build

Once AudioGuard is the authoritative gate (after #01 lands), the per-service `context_id` filter blocks inside each TTS service implementation are non-load-bearing for correctness. They remain only as a performance optimization (skip allocating frames that would later be dropped). For most production loads the optimization is negligible compared to the maintenance cost of keeping the same filter logic synchronized across N services.

Delete the filter from each TTS service that has one. Keep the per-service upstream cancellation logic (e.g., Cartesia's `cancel` message to the Cartesia API on `InterruptionFrame`) — that's API-specific, not pipeline filtering.

Known sites (verify all current TTS services in `src/services/`):

- [src/services/cartesia/tts.go:836-855](../../../src/services/cartesia/tts.go) — the `IGNORING audio from OLD context` block
- [src/services/elevenlabs/tts.go](../../../src/services/elevenlabs/tts.go) — equivalent filter block
- [src/services/deepgram/tts.go](../../../src/services/deepgram/tts.go) — equivalent filter block
- Any other TTS service in `src/services/` that performs context-id filtering on incoming audio chunks

The associated `audioContexts map[string]*AudioContext` and `audioContextAvailable` / `appendToAudioContext` / `removeAudioContext` helpers can also be deleted from each service if they exist solely to feed the filter.

`AudioContextManager` in [src/services/service.go](../../../src/services/service.go) shrinks: keep `GenerateContextID()` and the per-turn ID minting, drop the per-stream coordination state if it's no longer used. Audit each service to confirm.

## Acceptance criteria

- [ ] For each TTS service that has a `context_id` filter on its incoming audio loop, the filter block is deleted.
- [ ] Per-service `audioContexts` maps and helpers deleted if they're no longer referenced.
- [ ] `AudioContextManager` audited; unused fields removed; remaining surface documented.
- [ ] Existing TTS service tests updated to drop assertions on the deleted filter behavior; correctness assertions move to AudioGuard tests where appropriate.
- [ ] `go test ./...` clean.
- [ ] Real voice call with mid-sentence interruption produces no stale audio at the wire (AudioGuard catches it at the pipeline level).

## Blocked by

- [01-make-audioguard-authoritative.md](01-make-audioguard-authoritative.md)

Once AudioGuard is the sole authoritative gate, the TTS-level filters become deletable. This slice can run in parallel with #02, #03, #04.

## Comments
