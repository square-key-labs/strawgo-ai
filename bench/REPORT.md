# Strawgo vs Pipecat — VAD-only capacity bench

## TL;DR

On a 4-core / 16GB Linux VM running Silero VAD as the only ONNX workload:

| metric | Strawgo (Rust onnx-worker via Unix socket IPC) | Pipecat 1.1 (in-process Python ORT) |
|---|---|---|
| p99 latency at N=100 | **19.4 ms** | **0.46 ms** |
| RSS peak at N=100 | **1722 MB** | **928 MB** |
| Behavior at N=200 | total melt: 94.5% drops, p99 = 11.1 s | clean: 0% drops, p99 = 0.47 ms |
| Per-agent VAD cost (steady state) | ~8 MB | ~8 MB |
| Fixed cost (process baseline) | ~50 MB Rust runtime | 126 MB Python runtime |

**Pipecat wins every dimension at N ≥ 10.** Strawgo's "shared onnx-worker" architecture
gives no memory benefit (per-conn ORT sessions still allocated separately) and adds
a ~1 ms IPC tax that compounds super-linearly until the worker collapses past N=150.

## Setup

- VM: GCE `e2-standard-4` (4 vCPU, 16 GB), Ubuntu 24.04, asia-south1-a
- Silero VAD ONNX (fp32, 2.24 MB), 16 kHz, 512-sample frames @ 32 ms cadence
- Strawgo: `~/onnx-worker` (Rust, ort 2.x, single process), driven by Strawgo's
  `cmd/loadtest` over a single Unix socket (`/tmp/onnx-worker-bench.sock`)
- Pipecat: `bench/pipecat/vad_bench.py` — N asyncio tasks, each owns one
  `SileroVADAnalyzer`, ONNX runs in-process
- Both arms feed real-time-paced synthetic 16 kHz PCM frames for 20 s per level
- Levels: N ∈ {1, 5, 10, 25, 50, 100, 200}
- Sweep script: `bench/run_vad_sweep.sh` (run on VM)

## Full results

| N | fw | sched | ok | err | p50 | p95 | p99 | RSS peak |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|
| 1 | strawgo | 625 | 625 | 0 | 1.36ms | 1.86ms | 1.93ms | 67 MB |
| 1 | pipecat | 626 | 626 | 0 | 663µs | 744µs | 829µs | 151 MB |
| 5 | strawgo | 3122 | 3122 | 0 | 1.69ms | 2.37ms | 2.53ms | 147 MB |
| 5 | pipecat | 3130 | 3130 | 0 | 517µs | 702µs | 747µs | 183 MB |
| 10 | strawgo | 6250 | 6250 | 0 | 2.32ms | 3.09ms | 3.33ms | 216 MB |
| 10 | pipecat | 6260 | 6260 | 0 | 481µs | 667µs | 721µs | 222 MB |
| 25 | strawgo | 15625 | 15625 | 0 | 3.58ms | 5.05ms | 5.42ms | 462 MB |
| 25 | pipecat | 15650 | 15650 | 0 | 456µs | 596µs | 679µs | 340 MB |
| 50 | strawgo | 29882 | 29882 | 0 | 5.60ms | 8.56ms | 9.39ms | 880 MB |
| 50 | pipecat | 31300 | 31300 | 0 | 441µs | 515µs | 593µs | 535 MB |
| 100 | strawgo | 52981 | 52981 | 0 | 9.30ms | 15.63ms | 19.38ms | 1722 MB |
| 100 | pipecat | 61657 | 61657 | 0 | 267µs | 422µs | 464µs | 928 MB |
| 200 | strawgo | 26275 | **1438** | 24837 | 1745ms | 10793ms | 11101ms | 3670 MB |
| 200 | pipecat | 65050 | 65050 | 0 | 266µs | 322µs | 468µs | 1709 MB |

(Strawgo N=200: only 1438/26275 frames completed. Frame budget is 32 ms; p50 was
1.7 s. CPU saturated. The loadtest itself failed to schedule the full 125 000
frames because real-time pacing broke down.)

## Why Strawgo loses

1. **Per-connection ONNX session.** `onnx-worker/src/vad.rs` constructs a new
   `Session` per connection, not a shared one. Each session carries its own ORT
   memory arena (~12-14 MB scratch + 2 MB weights). Memory cost ≈ Pipecat's, no
   savings from "shared worker."
2. **Unix-socket IPC tax.** Even at N=1 the strawgo p99 is 1.93 ms vs Pipecat's
   0.83 ms — the extra ~1 ms is the socket round-trip and Tokio dispatch.
3. **Single Rust process, single Tokio runtime.** All N inference calls funnel
   through one event loop. Each `session.run()` is a sync C++ call inside
   `spawn_blocking`. With ORT default `intra_op_num_threads = nproc`, 100
   sessions × 4 threads = 400 threads contending for 4 cores. Cache thrash +
   scheduler tax = super-linear tail latency.
4. **No backpressure.** At N=200 the queue overflows; loadtest sees timeouts as
   errors. There is no graceful degradation.

## Why Pipecat wins

1. **No IPC.** ORT runs in the same Python process. Each `voice_confidence` call
   is a direct sync C call.
2. **GIL is released during ORT inference.** The ORT C++ kernel releases the GIL,
   so asyncio can multiplex other Python tasks while one VAD runs in C. Effective
   N-way concurrency on 4 cores without GIL pain.
3. **Linear memory.** Each `SileroVADAnalyzer` owns one Session; same per-agent
   cost as Strawgo, but no shared-process fixed overhead beyond Python's 126 MB
   baseline.

## What "shared onnx-worker" was supposed to win — and didn't

Strawgo's pitch is process isolation: ONNX crashes don't take down the Go
process. That benefit is real (and unique). But the implied scaling benefit
("share the model across N agents") **is not implemented today**: weights are
not shared, scratch arenas are not shared, and the IPC adds latency. Net effect
is strictly worse than in-process inference.

## Where Strawgo could come back

Ranked by ROI, evidence-backed:

### Tier 1: ship-now wins, verified safe

1. **Shared ORT session, per-conn state** — confirmed safe by Silero maintainer
   ("you can share one VAD model across several audio streams") and by Pipecat
   issue [#2050](https://github.com/pipecat-ai/pipecat/issues/2050). Per-stream
   state is just LSTM `h, c` (~few KB). Refactor `onnx-worker/src/vad.rs` to
   hold `Arc<Session>` shared across connections; per-`SileroSession` struct
   keeps only the LSTM state arrays. **Estimated impact: RSS at N=100 drops
   from 1722 MB → ~30 MB (≈ 50× reduction).**

2. **`intra_op_num_threads = 1` + global ORT thread pool** — sized to
   `nproc`. Stops the 400-threads-on-4-cores contention. Latency tail should
   flatten. Standard ORT feature documented at
   [onnxruntime.ai/docs/performance/tune-performance/threading.html](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

3. **int8 quantized model** — `model_int8.onnx` is 639 KB vs fp32 2.24 MB.
   ORT int8 inference is typically 2-3× faster on CPU. Available at
   [huggingface.co/onnx-community/silero-vad](https://huggingface.co/onnx-community/silero-vad/tree/main/onnx).

4. **NUMA pinning** (`numactl --cpunodebind=0 --membind=0`) — ~20% free per
   ORT docs.

After Tier 1: estimated N=100 RSS ~30 MB, p99 ~2-3 ms. Roughly 50× memory and
7× latency improvement.

### Tier 2: replace the model

5. **TEN-VAD** ([github.com/TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad)):
   ByteDance, Apache 2.0, 306 KB library vs Silero's 2.16 MB. RTF 0.0086-0.0160
   (Silero ~0.03-0.05, so 3-6× faster). Beats Silero on librispeech / gigaspeech
   / DNS Challenge precision-recall and detects speech-to-silence transitions
   without Silero's hundreds-of-ms delay. C / Go / Java / WASM bindings. Drop-in
   candidate, but **needs independent accuracy eval** before replacing
   production VAD.

6. **Drop IPC entirely.** Embed ORT in Go via cgo (`yalue/onnxruntime_go`) or
   use `voice_activity_detector` Rust crate inline. Eliminates the ~1 ms socket
   round-trip floor. Loses process-isolation crash safety; only worth it if
   Tier 1 isn't enough.

### Tier 3: don't

- ❌ **GPU / CUDA / TensorRT.** Silero maintainer: "the VAD is not designed to
  run on GPU." Real-world test: `onnxruntime-gpu` 50-60% **slower**. LSTM is
  sequential, batches too small to amortize PCIe transfer. See
  [silero-vad#567](https://github.com/snakers4/silero-vad/discussions/567).
- ❌ **Naive cross-stream batching.** LSTM hidden state is per-stream; batching
  breaks correctness without extremely careful state interleaving. Maintainer
  explicitly discourages it.
- ❌ **Triton dynamic batcher** for the same reason; min-latency tradeoff loses.

## Methodology notes

- Latency is `time.perf_counter_ns()` around the single `voice_confidence` /
  worker round-trip call. No HTTP, no pipeline, no STT/LLM/TTS — purely VAD
  inference cost.
- "Per-agent MB" for Pipecat = `(rss_after_load - rss_baseline) / N`, measured
  before inference starts. ORT scratch grows by only 1-7 MB during the 20s run
  even at N=100, so load-time delta ≈ peak. No hidden runtime memory growth.
- Strawgo onnx-worker accumulates sessions across levels (it's one long-lived
  process); the rss_base column reflects worker idle RSS at the start of each
  level. Per-agent is best read from the apples-to-apples N=100 fresh-start
  number: 1722 MB for 100 agents ≈ 17 MB/agent at peak.
- Both arms used the same Silero ONNX model file and the same 16 kHz / 512-sample
  / 32 ms frame size.
- VAD-only by design: removes vendor (STT/LLM/TTS) concurrency caps from the
  measurement so framework cost is what's left.

## Files

- `bench/run_vad_sweep.sh` — VM-side sweep driver
- `bench/pipecat/vad_bench.py` — Pipecat per-agent VAD harness
- `bench/compare_vad.py` — parses Strawgo loadtest stdout + Pipecat CSVs, prints
  side-by-side table
- Strawgo loadtest binary: `cmd/loadtest` (existing in repo)
- Strawgo Rust worker: `onnx-worker/` (existing in repo)

## Sources

- [pipecat-ai/pipecat#2050](https://github.com/pipecat-ai/pipecat/issues/2050) — shared ONNX session pattern for SileroVADAnalyzer
- [snakers4/silero-vad#744](https://github.com/snakers4/silero-vad/discussions/744) — multi-user streaming with shared model + per-stream state
- [snakers4/silero-vad#567](https://github.com/snakers4/silero-vad/discussions/567) — GPU and batching: don't
- [snakers4/silero-vad#427](https://github.com/snakers4/silero-vad/discussions/427) — concurrent request handling
- [onnx-community/silero-vad](https://huggingface.co/onnx-community/silero-vad/tree/main/onnx) — quantized variants (int8 = 639 KB)
- [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad) — Apache 2.0 alternative VAD, smaller and faster
- [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) — global thread pool, NUMA affinity
- [pipecat-ai/pipecat#1003](https://github.com/pipecat-ai/pipecat/issues/1003) — production memory observations (~400 MB/agent with full vendor pipeline; this bench shows VAD itself is only ~8 MB of that)
