# 03 — Extract BotSpeechTracker and PlaybackAckTracker submodules

Status: needs-triage

## Parent

[ADR-0001 — Stale-audio filtering lives in a dedicated AudioGuard processor](../../../docs/adr/0001-audio-guard-as-dedicated-processor.md)

Phase 3 of the deepening plan (split the god-object output processor).

## What to build

Pull the bot-speech state machine and the telephony playback-ack state machine out of `WebSocketOutputProcessor` into two dedicated submodules. Both subscribe to events from the Pacer (extracted in #02) and emit lifecycle frames upstream.

`BotSpeechTracker` owns:

- `BotStartedSpeakingFrame` emission on first chunk from the Pacer
- VAD-stop timer (`BOT_VAD_STOP_SECS`) and the LLM-end gate (`llmResponseEnded`)
- `BotStoppedSpeakingFrame` emission when (no audio for VAD-stop) AND (LLM ended) AND no playback ack expected

`PlaybackAckTracker` owns:

- The ack-request side: when the serializer is a `PlaybackAckSerializer`, request a playback-done ack via `serializer.SerializePlaybackDoneAck()` after audio idle + LLM end
- The ack-receive side: handle `PlaybackCompleteFrame` and emit `BotStoppedSpeakingFrame`
- The fallback timer (10s deadman switch)

`WebSocketOutputProcessor` becomes the glue: routes frames, owns the serializer and connection lifecycle, wires Pacer → BotSpeechTracker and Pacer → PlaybackAckTracker.

## Acceptance criteria

- [ ] New file [src/transports/bot_speech_tracker.go](../../../src/transports/bot_speech_tracker.go) with `BotSpeechTracker` struct and tests.
- [ ] New file [src/transports/playback_ack_tracker.go](../../../src/transports/playback_ack_tracker.go) with `PlaybackAckTracker` struct and tests.
- [ ] `WebSocketOutputProcessor` no longer owns: `llmResponseEnded`, `llmMu`, `playbackDoneChan`, the VAD-stop timer logic, the fallback-timer logic, or the playback-ack send logic.
- [ ] `WebSocketOutputProcessor` shrinks to a thin glue layer (~250 lines, down from ~830 after slice #02 lands).
- [ ] Trackers receive Pacer events via Pacer's existing subscription API from #02; no shared mutex between Pacer and trackers.
- [ ] Existing transport tests still pass; new tracker tests run with no transport / no WebSocket.
- [ ] `go test ./...` clean.
- [ ] Real voice call with non-telephony serializer (Daily) still emits `BotStoppedSpeakingFrame` via the VAD-stop timer path.
- [ ] Real voice call with telephony serializer (Twilio or Asterisk) still emits `BotStoppedSpeakingFrame` only after the playback-ack arrives, with the 10s fallback intact.

## Blocked by

- [02-extract-pacer-submodule.md](02-extract-pacer-submodule.md)

The Pacer's event-emission API is the seam these trackers subscribe to.

## Comments
