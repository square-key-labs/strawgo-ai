# Architecture Review — 2026-05-07

Output of an architecture review using the `improve-codebase-architecture` skill. Goal: surface deepening opportunities — refactors that turn shallow modules into deep ones, in the [Ousterhout](https://web.stanford.edu/~ouster/cgi-bin/aposd.php) sense.

Vocabulary: see [CONTEXT.md](../../CONTEXT.md) for domain terms; the architecture vocabulary (Module / Interface / Implementation / Depth / Seam / Adapter / Leverage / Locality / deletion test) follows the skill.

## Status

| # | Candidate | Status |
|---|---|---|
| 1 | Transport seam leaks frame semantics | open |
| 2 | `WebSocketOutputProcessor` god-object | **in progress** — see [ADR-0001](../adr/0001-audio-guard-as-dedicated-processor.md) and the 4-phase plan |
| 3 | `AudioContextManager` is leaky transport state in TTS services | partially absorbed by #2 (Phase 4 cleanup) |
| 4 | `LLMContextAggregator` base type is a buffer struct | open |
| 5 | Vendor service streaming WebSocket duplication | open |
| 6 | `services/reconnect/wrap.go` is a one-of-a-kind wrapper | open |
| 7 | Turn strategies orchestrate via implicit pipeline reset choreography | open |
| 8 | 41 frame types, many zero-payload wrappers | open |

---

## Candidate 1 — Transport seam leaks frame semantics into transport

**Files:** [src/transports/websocket.go:159-218](../../src/transports/websocket.go), [src/serializers/serializer.go](../../src/serializers/serializer.go), [src/serializers/twilio.go](../../src/serializers/twilio.go), [src/serializers/asterisk.go](../../src/serializers/asterisk.go)

**Problem.** Two seams pretend to exist but only one really does. `FrameSerializer.Deserialize` returns `frames.Frame` (good), but transport then `switch f := frame.(type)` over `*AudioFrame / *StartFrame / *EndFrame / default` (websocket.go:193-218) before pushing. Output side sends raw `interface{}` (`[]byte` or `string`) and the transport type-switches *again* (websocket.go:236-246) to pick `BinaryMessage` vs `TextMessage`. Serializer's `Type() SerializerType` exists but nobody uses it for that decision. Result: transport knows about specific frame types and serializer knows about wire-message types, both leak through `interface{}`.

**Deletion test.** Delete the `FrameSerializer` interface. Twilio/Asterisk wire formats reappear as logic somewhere — *but* they currently *only have one transport that drives them*, so the seam is **one-adapter hypothetical**, not real.

**Benefit of deepening.** A serializer that owns *both* deserialize and "how to put bytes on the wire" (returning a typed `WireMessage` enum, not `interface{}`) lets the transport stop knowing frame types entirely. Locality: protocol quirks land in one file. Tests: Twilio/Asterisk get tested without a WebSocket at all.

---

## Candidate 2 — `WebSocketOutputProcessor` is a god-object state machine

**Status:** in progress. Decision recorded in [ADR-0001](../adr/0001-audio-guard-as-dedicated-processor.md). 4-phase plan agreed.

**Files:** [src/transports/websocket.go:302-999](../../src/transports/websocket.go), [src/transports/websocket_test.go:66-313](../../src/transports/websocket_test.go)

**Problem.** One processor multiplexes four concerns: (a) audio rate-limiting / pacing, (b) `BotStarted/StoppedSpeakingFrame` emission with VAD timer + LLM-end gate + playback-ack fallback, (c) interruption filtering by `currentContextID`/`expectedContextID`, (d) WebSocket cleanup. Eight sync primitives. Tests reach into `processor.interruptionMu`/`expectedContextID`/`currentContextID` directly — the **interface is the test surface**, and these tests prove the current interface doesn't expose the behavior under test.

**Solution.** Extract `AudioGuard` as a dedicated `FrameProcessor` between TTS service and output transport. Owns Generation lifecycle and stale-audio filtering. The remaining transport concerns (Pacer, BotSpeechTracker, PlaybackAckTracker) split out as submodules in a later phase.

**Benefit.** Single source of truth for stale-audio filtering. Goroutine race architecturally eliminated. Tests stop pinning private fields and assert on emitted frames.

---

## Candidate 3 — `AudioContextManager` is leaky transport state living inside every TTS service

**Status:** partially absorbed by Candidate 2. Phase 4 of that plan deletes the now-redundant filter from each TTS service.

**Files:** [src/services/service.go:229-336](../../src/services/service.go), [src/services/cartesia/tts.go:55](../../src/services/cartesia/tts.go), [src/services/elevenlabs/tts.go:59](../../src/services/elevenlabs/tts.go), and ~5 other TTS services embed `*AudioContextManager`.

**Problem.** Why does a TTS service track UUIDs? Because `WebSocketOutputProcessor` filters stale audio by `context_id` after interruption (#2). So every TTS vendor has been forced to embed `*AudioContextManager`, call `GetOrCreateContextID()` per chunk, and mint context IDs on `TTSStartedFrame`. The concern is *transport-level audio invalidation*, not synthesis.

**Solution.** After Candidate 2 ships, the TTS-side filter becomes non-load-bearing. `AudioContextManager` shrinks to just `GenerateContextID()`. The per-TTS filter blocks become deletable.

---

## Candidate 4 — `LLMContextAggregator` "base" is a buffer struct, not a deep module

**Files:** [src/processors/aggregators/base.go:100-200](../../src/processors/aggregators/base.go), [src/processors/aggregators/user.go](../../src/processors/aggregators/user.go), [src/processors/aggregators/assistant.go](../../src/processors/aggregators/assistant.go), [src/processors/aggregators/sentence.go](../../src/processors/aggregators/sentence.go)

**Problem.** `LLMContextAggregator` exposes 9 methods (`AggregationString`, `GetAggregation`, `AppendToAggregation`, `Reset`, `HasAggregation`, `SetAddSpaces`, `SetAggregationType`, `GetAggregationType`, `PushContextFrame`) — all wrappers around a `[]string` and a role string. Subclasses don't reuse the base's logic in any sophisticated way. `AggregationString` is even marked `Deprecated`. Two interfaces also exist for the *same concept*: `Aggregation` (struct with metadata) and `TextAggregator` (channel-based interface) — the base aggregator uses neither.

**Deletion test.** Delete the base. Each subclass needs ~10 lines of `[]string` + `strings.Join`. Complexity does **not** concentrate. Pass-through-with-bookkeeping.

**Benefit.** Replace the base with the existing `TextAggregator` interface. Each role-specific aggregator becomes an *adapter* that owns its frame routing. Tests for sentence/word logic stop touching `LLMContext`.

---

## Candidate 5 — Vendor service packages have no shared streaming-WebSocket module

**Files:** [src/services/cartesia/tts.go](../../src/services/cartesia/tts.go), [src/services/elevenlabs/tts.go](../../src/services/elevenlabs/tts.go), [src/services/deepgram/tts.go](../../src/services/deepgram/tts.go) (~700-1000 lines each)

**Problem.** Streaming TTS over WebSocket is one problem; each vendor reimplements: connect, ping/pong, reconnect on close, audio buffering, context-id bookkeeping, `TTSStartedFrame`/`TTSAudioFrame`/`TTSStoppedFrame` emission. The `AIService` interface is just `FrameProcessor + Initialize/Cleanup` — nothing about streaming. So the streaming pattern is duplicated, not abstracted.

**Deletion test.** Delete one TTS service. The others remain, each a copy-paste of the missing logic. Complexity is *spread*, not concentrated.

**Benefit.** Extract a `StreamingTTSAdapter` (vendor implements only: build request, parse audio chunk, parse end marker). Locality: reconnect / context tracking / pacing live in one place. **5+ adapters means a real seam.**

**Why this is the highest-leverage open candidate.** Adding a 6th TTS vendor today copies ~700 lines. After deepening, it's ~150.

---

## Candidate 6 — `services/reconnect/wrap.go` is a one-of-a-kind wrapper masquerading as a pattern

**Files:** [src/services/reconnect/wrap.go](../../src/services/reconnect/wrap.go)

**Problem.** Wraps STT only. No `WrapTTS` / `WrapLLM`. The wrapper invents a `frameBridge` to intercept upstream `ErrorFrame` and re-`Initialize` the inner service. **One-adapter rule: hypothetical seam.** Either TTS/LLM also need this (and the abstraction should be generalized to `FrameProcessor`-level reconnect) or only one provider is flaky and this is a special case poorly named.

**Deletion test.** Delete `reconnect.WrapSTT`. Reconnect logic moves into the one STT service that needs it (~50 lines). Complexity *moves but doesn't grow*.

**Benefit.** Either commit to it (deepen across all 3 service types) or absorb it into the one provider that needs it. Either way, less misleading API surface.

---

## Candidate 7 — Turn strategies orchestrate via implicit pipeline reset choreography

**Files:** [src/turns/strategies.go](../../src/turns/strategies.go), [src/turns/user_start/min_words.go](../../src/turns/user_start/min_words.go), [src/turns/user_stop/speech_timeout.go](../../src/turns/user_stop/speech_timeout.go), [src/turns/user_mute/first_speech.go](../../src/turns/user_mute/first_speech.go)

**Problem.** Three orthogonal-looking strategy lists (`Start`/`Stop`/`Mute`) with private mutable state (`triggered`, `wordCount`, `stopDeadline`). `Reset()` is called somewhere by `BaseProcessor` — but who is the source of truth for "a new turn just started, reset all start-strategies"? Strategies don't communicate; the pipeline must know the choreography. Cross-strategy invariants (e.g., MinWords + VAD: which one's "start" wins?) are not encoded.

**Deletion test.** Delete the 3-list config. Each pipeline would inline 3-5 lines. Complexity stays similar — but the *coordination contract* would now be visible at the call site.

**Benefit.** A `TurnController` that *owns* the lifecycle (start/stop/mute decisions, reset on each new turn) and accepts strategy *predicates*. Tests would simulate turns instead of poking individual strategies.

---

## Candidate 8 — 41 frame types, many are zero-payload wrappers

**Files:** [src/frames/system.go](../../src/frames/system.go), [src/frames/control.go](../../src/frames/control.go), [src/frames/data.go](../../src/frames/data.go)

**Problem.** 41 concrete frame structs, many are `*BaseFrame` only (e.g., `HeartbeatFrame`, `UserStartedSpeakingFrame`, `PlaybackCompleteFrame`). Identity-by-Go-type means every router does `switch frame.(type)` over a long list. The frame interface's "interface" (in the deep sense — what callers must know) is now "all 41 types and what they mean."

**Deletion test.** Borderline. Type-based routing is genuinely useful in Go and the frames *are* the protocol. Killing them would just push the discriminant elsewhere.

**Benefit (smaller).** Group by category with a tag (`SystemFrameKind`, `ControlFrameKind`) for the empty wrappers. Lets routers handle "any control frame" cleanly without exhaustive switches.

---

## Friction worth knowing about but not yet ready to deepen

- VAD vs Turn-Analyzer ONNX socket clients duplicated ([src/audio/vad/onnx_client.go](../../src/audio/vad/onnx_client.go) + [src/audio/turn/onnx_turn_client.go](../../src/audio/turn/onnx_turn_client.go)) — minor; per-client state differs slightly.
- Per-vendor `Config` structs are not unified — known anti-pattern; the differences are real.

---

## Recommended next pickup

When the next deepening session opens, **Candidate 5** is the highest-leverage open item. Adding any new TTS vendor without it copies ~700 lines.

## Re-validation note

This file is a snapshot from 2026-05-07. The codebase moves; before grilling any candidate, re-read its cited files. Friction may have resolved, or new friction may have emerged.
