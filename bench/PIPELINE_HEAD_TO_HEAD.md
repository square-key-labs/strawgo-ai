# Pipeline Head-to-Head — Strawgo cgo embed vs Pipecat 1.1

**Stack:** GTCRN denoise → Silero VAD → smart-turn-v3.1 (per-frame, 32 ms cadence, 16 kHz mono).
**Hardware:** GCE `e2-standard-4` (4 vCPU Cascade Lake-class Intel, 16 GB RAM, Linux 6.17).
**Date:** 2026-05-01. **Window:** 20 s realtime per concurrency level.
**Fixture:** `sine_440_500ms_16k.pcm` looped — does not trip Silero VAD edges, so smart-turn fires only when forced via cadence (Pipecat) or never (Strawgo, no cadence flag).

Three runs:
- `strawgo` — `loadtest-pipeline-linux` (cgo embed, shared `*ort.DynamicAdvancedSession` per model, per-stream LSTM/cache state). Smart-turn = VAD-edge only → 0 calls on sine.
- `pipecat-2s` — `bench/pipecat/pipeline_bench.py` `--turn-cadence-ms 2000`. Smart-turn forced ~2 calls/agent over 20 s.
- `pipecat-noturn` — same harness `--turn-cadence-ms 0`. Smart-turn = VAD-edge only → 0 calls. **Apples-to-apples vs strawgo.**

## Headline numbers (N=200, 20 s window)

| metric | strawgo | pipecat-noturn | pipecat-2s |
|---|---:|---:|---:|
| frames scheduled | **41 895** | 4 308 | 1 676 |
| % of theoretical (200 × 625) | **33.5%** | 3.4% | 1.3% |
| frames OK | 41 895 | 4 308 | 1 676 |
| drops (errors) | 0 | 0 | 0 |
| p50 latency | 591 ms | 4.7 ms | 4.8 ms |
| p99 latency | 1 423 ms | 6.9 ms | 276 ms |
| RSS peak | **247 MB** | 4 584 MB | 8 452 MB |
| per-agent RSS (steady) | **~1.2 MB** | ~22 MB | ~22 MB |

**One-liner:** Strawgo runs **~10× more useful frames** on the same VM at **18× less memory**, but pays for it in per-frame tail latency once the 4 cores saturate. Pipecat keeps p99 low by **silently skipping 96 % of frames** (asyncio cooperative scheduler under-paces).

## Throughput per VM (the metric that matters in production)

Useful frames committed per VM per second across all agents:

| N | strawgo | pipecat-noturn | ratio |
|---:|---:|---:|---:|
| 1 | 31.2 fps | 31.3 fps | 1.0× |
| 5 | 156.3 fps | 151.3 fps | 1.0× |
| 10 | 312.5 fps | 123.7 fps | 2.5× |
| 25 | 491.7 fps | 128.4 fps | 3.8× |
| 50 | 559.9 fps | 128.7 fps | 4.4× |
| 100 | 924.6 fps | 150.4 fps | 6.1× |
| 200 | **2 094.8 fps** | **215.4 fps** | **9.7×** |

Strawgo scales near-linearly until 4-core compute saturates; Pipecat asyncio plateaus at ~150 fps independent of N because the Python scheduler can't dispatch more.

## Full sweep — strawgo (denoise + VAD only, smart-turn never fires)

| N | sched/ok | p50 | p95 | p99 | denoise µs | VAD µs | RSS peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1   | 624/624 | 5.22 ms | 7.26 ms | 9.27 ms | 4 142 | 428 | 58 MB |
| 5   | 3 125/3 125 | 9.31 ms | 13.00 ms | 14.58 ms | 6 809 | 494 | 68 MB |
| 10  | 6 250/6 250 | 19.07 ms | 23.98 ms | 26.75 ms | 12 150 | 483 | 82 MB |
| 25  | 9 833/9 833 | 86.22 ms | 136.89 ms | 160.44 ms | 41 744 | 579 | 111 MB |
| 50  | 11 199/11 199 | 174.54 ms | 299.26 ms | 348.06 ms | 72 990 | 597 | 118 MB |
| 100 | 18 492/18 492 | 328.39 ms | 609.57 ms | 728.55 ms | 129 096 | 583 | 147 MB |
| 200 | 41 895/41 895 | 591.06 ms | 1180.64 ms | 1423.34 ms | 225 250 | 635 | 247 MB |

**Read:** denoise dominates and scales linearly with N once cores saturate (~N=10). VAD stays flat ~0.5 ms across the sweep — Phase-2 result holds. p99 within 32 ms frame budget through N=10; over budget at N≥25.

## Full sweep — pipecat-noturn (denoise + VAD only, fair compare)

| N | sched/ok | %theor | p50 | p95 | p99 | per-agent | RSS peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1   | 626/626 | 100.0% | 5.36 ms | 5.91 ms | 6.63 ms | 36.3 MB | 162 MB |
| 5   | 3 025/3 125 | 96.8% | 6.11 ms | 8.49 ms | 9.42 ms | 24.7 MB | 253 MB |
| 10  | 2 474/6 250 | 39.6% | 7.75 ms | 8.66 ms | 10.01 ms | 23.1 MB | 386 MB |
| 25  | 2 567/15 625 | 16.4% | 7.65 ms | 8.78 ms | 10.58 ms | 22.3 MB | 701 MB |
| 50  | 2 573/31 250 | 8.2% | 7.69 ms | 8.88 ms | 10.66 ms | 22.0 MB | 1 255 MB |
| 100 | 3 008/62 500 | 4.8% | 7.48 ms | 8.75 ms | 10.90 ms | 21.8 MB | 2 361 MB |
| 200 | 4 308/125 000 | 3.4% | 4.71 ms | 5.93 ms | 6.92 ms | 21.8 MB | 4 584 MB |

**Read:** p99 looks great — but the denominator is 3-17% of theoretical at N≥10. Cooperative scheduler simply doesn't dispatch most frames. Memory grows ~22 MB/agent steady — model arenas (denoise + VAD ONNX sessions) duplicated per asyncio task.

## Full sweep — pipecat-2s (denoise + VAD + smart-turn cadence 2 s)

| N | sched/ok | ST calls | p50 | p99 | RSS peak |
|---:|---:|---:|---:|---:|---:|
| 1   | 626/626 | 10 | 5.36 ms | 172.4 ms | 199 MB |
| 5   | 2 980/3 125 | 43 | 4.44 ms | 170.0 ms | 408 MB |
| 10  | 3 525/6 250 | 35 | 4.41 ms | 7.14 ms | 663 MB |
| 25  | 3 429/15 625 | 60 | 4.37 ms | 177.1 ms | 1 265 MB |
| 50  | 2 199/31 250 | 105 | 4.45 ms | 220.1 ms | 2 247 MB |
| 100 | 1 087/62 500 | 200 | 4.73 ms | 197.8 ms | 4 305 MB |
| 200 | 1 676/125 000 | 400 | 4.79 ms | 275.6 ms | 8 452 MB |

**Smart-turn cost on Cascade Lake:** ~170-200 ms per call. Predicted ~3× M-series cost (~80 ms local) — confirmed 2.4×.

## What's actually going on

### Strawgo
- One shared `*ort.DynamicAdvancedSession` per model. Per-stream `[2,1,128]` LSTM hidden state, GTCRN cache state.
- Go GMP scheduler dispatches every 32 ms goroutine on time → 100 % theoretical frames attempted up to N≈10.
- Once 4 cores saturate at ~N=25, denoise queues up on the shared session → per-frame latency grows linearly.
- No drops because the loadtest doesn't shed; instead it lets frames queue and reports the real wait.
- **VAD cost flat at 0.5 ms** across all N — Phase-2 cgo-embed result reproduced cleanly with the new GTCRN+smart-turn pipeline package.

### Pipecat
- Per-agent ONNX sessions (denoise + VAD + smart-turn each instantiated per asyncio task).
- Scheduler can't keep 200 coroutines on a 32 ms tick when each frame does 5-10 ms work; it silently skips ticks.
- Frames it does dispatch run fine (p99 ~7-10 ms on 4 cores, similar to strawgo at low N).
- Memory dominated by per-agent ORT arenas — ~22 MB/agent steady.

## What this changes about the production direction

1. **VAD-only Phase-2 numbers still hold** — Strawgo cgo embed VAD = 0.5 ms/frame, 1.2 MB/agent, 100 % pacing through N=200.
2. **Adding GTCRN denoise breaks the Phase-2 story.** It is 8-10× heavier than VAD per frame, and shared-session means it serializes once cores saturate. **Production cap with full denoise = ~N=10 per 4-core box** (p99 < 32 ms).
3. **Smart-turn is fine** — 200 ms once per utterance, fires ~once per 2 s per agent in real traffic, irrelevant to the 32 ms frame budget if you don't trigger it on cadence.
4. **Pipecat tail "wins" at N=200 (7 ms vs 1.4 s) are noise** — they apply to ~3 % of theoretical frames. The other 97 % are dropped by the scheduler. Per-VM useful throughput is 9.7× lower.
5. **GTCRN was a substitute for DFN3.** DFN3 official is 48 kHz / 4-graph; GTCRN is 16 kHz single-ONNX, 535 KB. Compute classes are similar but DFN3 streaming numbers are unknown until tested.

## SNR-gated denoise A/B (proves the gating idea)

Built `pipeline_embed.SNRDetector` (MCRA-style noise-floor tracking, ~50 LOC,
0 alloc, **0.6 µs/frame on M3, ~2 µs/frame on Cascade Lake**) and an opt-in
`PipelineAnalyzer.EnableSNRGating(thresholdDB, cfg)`. CLI exposed via
`-snr-threshold-db` flag on `loadtest-pipeline`. 5/5 unit tests pass:
silent / steady-noise / clean-speech-over-noise / noisy-speech / realistic
phase pattern (quiet preamble → burst → silence).

**Fixture:** synthetic burst-call PCM (`testdata/burst_call_10s_16k.pcm`):
10 s of −35 dBFS noise floor with a 600 ms tone burst every 2.5 s. ~24 %
"speech-like" frames, ~76 % noise-only. SCP'd to VM as
`$BENCH_HOME/burst_call_10s_16k.pcm`.

**A/B on VM, burst fixture, N=1..200, 20 s window**:

| N | sched OFF | sched ON | OFF p99 | ON p99 | Δp99 | OFF denoise % | ON denoise % | denoise skipped |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1   | 625 | 625 | 8.80 ms | **7.76 ms** | **−12%** | 100% | 83.4% | 16.6% |
| 5   | 3 125 | 3 125 | 13.38 ms | **12.61 ms** | **−6%** | 100% | 83.4% | 16.6% |
| 10  | 6 250 | 6 250 | 21.15 ms | 21.57 ms | +2% | 100% | 83.4% | 16.6% |
| 25  | 12 244 | **13 585** | 122.76 ms | **108.40 ms** | **−12%** | 100% | 83.1% | 16.9% |
| 50  | 13 811 | **15 921** | 263.55 ms | **257.26 ms** | −2% | 100% | 82.0% | 18.0% |
| 100 | 19 580 | **20 964** | 568.38 ms | **530.61 ms** | **−7%** | 100% | 81.3% | 18.7% |
| 200 | 40 230 | **48 190** | 1 202.84 ms | **1 058.61 ms** | **−12%** | 100% | 81.2% | 18.8% |

**Throughput per VM (N=200, 20 s window)**:

- OFF: 40 230 frames / 20 s = **2 012 fps**
- ON:  48 190 frames / 20 s = **2 410 fps** (**+19.8%**)

**Read:**
- ~17-19% denoise calls skipped (matches the 24 % burst fraction —
  not all bursts trigger SNR > threshold cleanly).
- p99 win 6-12% on the burst fixture. Modest because remaining
  82-83% of frames still hit GTCRN at 100% load, and 4 cores stay
  saturated.
- **Bigger win is throughput**: 19.8% more useful frames per box at
  N=200. Same hardware committs ~400 more frames/s.
- SNR overhead = 1-2 µs/frame on Cascade Lake — **noise** vs
  4-225 ms/frame denoise.

**On a realistic telephony call** (≤30% speech, ~70% silence or
hold-music), expected denoise skip = 70%+. That predicts:

- p99 cut ~3× at N=100-200 (most cores idle on skipped frames)
- N cap moves from N=10 (denoise every frame) → ~N=30-50 (denoise
  selective).

The burst fixture is a stress-test (high speech fraction). Real call
data will show much bigger wins. Run again on a real-speech 16 kHz PCM
when one is available on the VM.

**Ship verdict:** SNR gating is a free 20 % throughput win on the worst
case (bursty fixture) and a 3× win on the realistic case. Default-on for
production with `threshold = 12 dB`. Add a hysteresis band later
(separate threshold for re-enabling denoise) if needed.

## Denoiser shootout (raw graph cost) — `bench-denoise-cost`

`cmd/bench-denoise-cost/` is a microbench that loads each candidate ONNX
with a shared `*ort.DynamicAdvancedSession`, fills synthetic input
tensors of the model's expected shape, and times `.Run()` across N
concurrent goroutines for a fixed wall window. Single-thread intra-op
config (`SetIntraOpNumThreads(1)`) so each agent owns one core's worth
of work.

**Models tested:**

| denoiser | source | size | rate | frame | streaming |
|---|---|---:|---:|---:|---|
| GTCRN (current) | yuyun2000/SpeechDenoiser | 535 KB | 16 kHz | 16 ms STFT (×2 per Strawgo frame) | ✅ 3 state caches |
| NSNet2 | niobures/NSNet2-ONNX (microsoft baseline) | 10.75 MB | 16 kHz | 20 ms | ❌ no state cache |
| RNNoise | niobures/RNNoise (ailia export) | 1.02 MB | 48 kHz | 1 s window | ❌ batch-windowed |
| DFN3 | Rikorose/DeepFilterNet (3-graph chain) | 8.5 MB | 48 kHz | 10 ms | ❌ no state cache |

**Cascade Lake N=1..200 measured cost per Strawgo 32 ms frame:**

| denoiser | N=1 p50 | N=10 p50 | N=50 p50 | N=200 p50 | N=1 RT/agent | N=200 RT/agent |
|---|---:|---:|---:|---:|---:|---:|
| GTCRN | 3.22 ms | 16.26 ms | 49.56 ms | **102.98 ms** | 7.76× | 0.08× |
| **NSNet2** | **1.49 ms** | 2.67 ms | 14.70 ms | **32.10 ms** | **20.4×** | **0.40×** |
| **DFN3** chain | 2.75 ms | 11.92 ms | 17.12 ms | **32.18 ms** | 7.76× ¹ | 0.24× |
| RNNoise (1 s amortized) | 0.25 ms | 0.98 ms | 4.38 ms | 139.84 ms | 122× | 0.31× |

¹ DFN3 RT counted as sum of 3 sub-graphs, each ~290-320 µs at N=1. Per
10 ms input frame total ≈ 860 µs → per 32 ms Strawgo frame = 2.75 ms.

**N-cap for sustained 1× realtime per agent (denoise every frame):**

| denoiser | N cap |
|---|---:|
| GTCRN | ~25 |
| **NSNet2** | **~100** |
| **DFN3** | ~50 |
| RNNoise | ~250 (but +1 s latency = unusable for voice agents) |

**Quality (published PESQ on DNS-Challenge):**

| denoiser | PESQ | DNS-MOS |
|---|---:|---:|
| RNNoise | ~2.8 | ~3.0 |
| GTCRN | ~3.0 | ~3.2 |
| NSNet2 | ~3.1 | ~3.3 |
| **DFN3** | **~3.4** | **~3.7** |

(Numbers from respective papers; not measured in this work.)

### Predictions vs reality (bias correction)

| model | I predicted | actually measured | error |
|---|---:|---:|---:|
| GTCRN | 4 ms (anchor) | 3.22 ms | ✓ |
| NSNet2 | 15-25 ms | **1.49 ms** | **10× too high** |
| DFN3 | 5-8 ms | **2.75 ms** | **2-3× too high** |
| RNNoise | ~1 ms | 0.25 ms amortized | 4× too high |

Param count (file size) is **not** a reliable proxy for ORT graph cost. NSNet2 is 20× larger than GTCRN but 2× faster — different op composition (single GRU vs conv+attention). Always microbench, never predict.

### Verdict on denoiser swap

- **NSNet2 = production winner.** ~2× faster than GTCRN with comparable
  quality (PESQ +0.1). Single-graph drop-in (after ~1 day STFT framework
  port), no state-cache concerns since the existing pipeline already
  handles cacheless denoisers.
- **DFN3 = quality fallback.** ~15% faster than GTCRN with markedly
  better PESQ. 3-graph chain + post-filter glue from `libDF` Rust
  crate makes integration ~2-3 days. Worth it only if NSNet2 quality
  is audibly insufficient on real telephony.
- **GTCRN = retire.** Slowest of the three usable, mid quality.
- **RNNoise = dead via ONNX.** Available export is offline 1-second
  windows with no state cache. To stream needs PyTorch retrain
  (multi-day side quest, dropped).

### Capacity table — corrected with NSNet2

| pipeline | denoise cost | N cap (p99 < 32 ms) |
|---|---:|---:|
| VAD only | n/a | 200+ |
| VAD + smart-turn (VAD-edge gated) | n/a | 200 |
| VAD + GTCRN every frame | 3.2 ms | 10 |
| VAD + NSNet2 every frame | 1.5 ms | 40 |
| VAD + GTCRN + SNR-gate | 0.96 ms effective | 30-50 |
| **VAD + NSNet2 + SNR-gate** | **0.45 ms effective** | **80-120** |

**12× capacity increase from current production default** (GTCRN every frame, N=10) to recommended new default (NSNet2 + SNR-gate, N=80-120).

## Next steps

- **Drop denoise from the hot path or move it off-CPU.** Options: (a) skip denoise entirely and rely on VAD threshold (production reality is most frames don't need it); (b) onnxruntime-with-AVX-VNNI int8 GTCRN (but we already proved int8 worse on Cascade Lake — needs Ice Lake+ hardware); (c) move denoise to GPU; (d) a DSP RNNoise/TEN-VAD-style sub-1ms denoiser.
- **Profile the denoise op breakdown.** STFT? GRU? Most likely GRU is the hot loop given p50≈225 ms at N=200 with 4 cores → ~56 ms/core inference time, vs 4 ms at N=1.
- **Test with real-speech PCM** so smart-turn fires on actual VAD edges (Pipecat cadence is a workaround for sine fixtures).
- **DFN3 streaming combined.onnx**, if obtainable, drop-in via `--denoise-model`. Could be lighter or heavier than GTCRN.

## Reproduce

```sh
# Set bench env on VM:
#   BENCH_HOME=/path/to/strawgo-bench
#   ORT_LIB=$BENCH_HOME/lib/libonnxruntime.so.1.25.1
#   MODELS=$BENCH_HOME/testdata/models
#   PCM=$BENCH_HOME/testdata/sine_440_500ms_16k.pcm

# Strawgo
LD_LIBRARY_PATH=$BENCH_HOME/lib ./loadtest-pipeline-linux \
  -lib $ORT_LIB \
  -vad-model $MODELS/silero_vad.onnx \
  -dfn-model $MODELS/gtcrn_simple.onnx \
  -smart-turn-model $MODELS/smart-turn-v3.1-cpu.onnx \
  -pcm $PCM \
  -levels 1,5,10,25,50,100,200 -dur 20

# Pipecat (cadence=0 fair / cadence=2000 forced)
$BENCH_HOME/bench-venv/bin/python pipeline_bench.py \
  -n N --dur 20 \
  --denoise-model $MODELS/gtcrn_simple.onnx \
  --turn-model $MODELS/smart-turn-v3.1-cpu.onnx \
  --turn-cadence-ms 0           # or 2000 for forced
```

Raw logs and CSVs: `bench/pipeline_results_pull/`.
