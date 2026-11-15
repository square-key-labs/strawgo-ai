# Stale-audio filtering lives in a dedicated `AudioGuard` processor

Stale-audio filtering — dropping `TTSAudioFrame`s whose `context_id` no longer matches the active **Generation** after an interruption — is owned by a dedicated `AudioGuard` `FrameProcessor` placed between the TTS service and the output transport. The filter does not live in the output transport (where it accumulated tangled state alongside pacing, bot-speech tracking, and playback-ack tracking) and does not live in each TTS service (where it was duplicated across Cartesia, ElevenLabs, Deepgram, and others as belt-and-suspenders against the output transport). One **Generation** lifecycle, one place to enforce it.

## Considered Options

- **Keep the gate in the output transport (status quo).** Rejected: the transport is already a 1000-line god-object multiplexing pacing, bot-speech tracking, ack tracking, and gate state across system + data + sender goroutines. Adding modules around the gate without removing it leaves the duplicate filter in 5+ TTS services forever.
- **Push the gate into TTS services, Pipecat-style.** Rejected: makes correctness a per-service responsibility. Every new TTS vendor must remember to filter, and a missed filter causes silent stale-audio leakage to the wire. Pipecat ships this pattern over WebRTC (low-buffer); Strawgo ships over telephony where stale-audio bleed is more visible.

## Consequences

- TTS service-level filters (e.g., [src/services/cartesia/tts.go:836](../../src/services/cartesia/tts.go#L836)) become non-load-bearing for correctness — they remain only as performance optimizations to skip frame allocation.
- The output transport's `interrupted` flag, `currentContextID`, and `expectedContextID` fields are deleted; the system/data goroutine race they were guarding is eliminated architecturally.
- Tests that pinned to private gate state in `WebSocketOutputProcessor` are replaced by direct unit tests on `AudioGuard`, with no transport involvement.
