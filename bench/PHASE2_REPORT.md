# Phase-2 — Strawgo VAD: Three Architectures Head-to-Head

Same GCE `e2-standard-4` (4 vCPU / 16 GB) Ubuntu 24.04 VM that produced
`bench/REPORT.md` (the Phase-1 baseline). VAD-only, Silero / TEN-VAD model,
16 kHz 32 ms (512-sample) frames, real-time pacing, 20 s per concurrency level.

## What changed

Phase-1 baseline measured **Strawgo (per-conn ORT session in Rust subprocess
+ Unix socket IPC)** vs **Pipecat (in-process Python ORT)**. Pipecat won
every dimension at N ≥ 10. Phase-2 ships three new architectures on the
Strawgo side:

1. **Rust Tier 1** — same Rust onnx-worker, but with the four
   `bench/RUST_TIER1_REPORT.md` fixes: shared `Session`, `intra_op = 1`,
   global ORT thread pool, optional int8 model. Branch
   `perf/onnx-worker-tier1`.
2. **cgo embed** — Strawgo running ORT directly in-process via
   `github.com/yalue/onnxruntime_go`, one shared session, per-instance
   LSTM state. Eliminates the Unix-socket IPC. Same branch.
3. **TEN-VAD** — replace Silero+ORT entirely with TEN-VAD's 306 KB native
   library, called via cgo. Branch `bench/tenvad-go-path`.

Strawgo Phase-1 (per-conn worker) and Pipecat Phase-1 numbers are the
reference points.

## Headline

| metric @ N=100 | Phase-1 Strawgo | Phase-1 Pipecat | **cgo embed** | **TEN-VAD** | Rust Tier 1 |
|---|---:|---:|---:|---:|---:|
| p99 | 19.4 ms | **0.46 ms** | 9.85 ms | 14.27 ms | 131.07 ms |
| RSS | 1722 MB | 928 MB | **44 MB** | 73 MB | 1149 MB |
| frames ok | 52981 | 61657 | **62500** | 62500 | 45257 |
| drops | 0 | 0 | 0 | 0 | 0 |

| metric @ N=200 | Phase-1 Strawgo | Phase-1 Pipecat | **cgo embed** | **TEN-VAD** | Rust Tier 1 |
|---|---:|---:|---:|---:|---:|
| p99 | 11.1 s ❌ | **0.47 ms** | 17.75 ms | 28.33 ms | 10.5 s ❌ |
| RSS | 3670 MB | 1709 MB | **47 MB** | 134 MB | 7173 MB |
| frames ok | 1438 | 65050 | **125000** | 125236 | 20846 |
| sched/expected | 26275/125k (21%) | 65050/125k (52%) | **125000/125k (100%)** | 125236/125k (100%) | 30132/125k (24%) |
| drop rate | 94.5% | 0% (under-sched) | 0% | 0% | 30.8% |

**cgo embed** is the only architecture that delivers the full 125 000-frame
real-time load at N=200 with zero drops on a 4-core box, in **47 MB**.

## Full sweep, all five architectures

p99 latency (ms):

| N | strawgo (P1) | pipecat (P1) | rust-tier1 | **cgo embed** | TEN-VAD |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.93 | **0.83** | 2.23 | 3.05 | 1.76 |
| 5 | 2.53 | **0.75** | 4.71 | 3.27 | 2.25 |
| 10 | 3.33 | **0.72** | 7.16 | 2.44 | 2.80 |
| 25 | 5.42 | **0.68** | 12.96 | 3.53 | 4.69 |
| 50 | 9.39 | **0.59** | 27.54 | 5.47 | 8.74 |
| 100 | 19.38 | **0.46** | 131.07 | 9.85 | 14.27 |
| 200 | **11101** ❌ | **0.47** ⚠️ | 10455 ❌ | **17.75** | 28.33 |

⚠️ Pipecat held p99=0.47 ms but only managed 65 050 of 125 000 frames at
N=200 — asyncio on 4 cores couldn't pace the full real-time load.

RSS peak (MB):

| N | strawgo (P1) | pipecat (P1) | rust-tier1 | **cgo embed** | TEN-VAD |
|---:|---:|---:|---:|---:|---:|
| 1 | 67 | 151 | 67 | 37 | **9** |
| 5 | 147 | 183 | 125 | 38 | **12** |
| 10 | 216 | 222 | 171 | 39 | **15** |
| 25 | 462 | 340 | 332 | 40 | **25** |
| 50 | 880 | 535 | 607 | 42 | **41** |
| 100 | 1722 | 928 | 1149 | **44** | 73 |
| 200 | 3670 | 1709 | 7173 | **47** | 134 |

## Key findings

### 1. The Mutex bug in Rust Tier 1

`onnx-worker/src/vad.rs` wraps the shared `Session` in `Arc<Mutex<Session>>`
because `pykeio/ort 2.0-rc.12`'s public `Session::run` takes `&mut self`.
The Mutex **serializes every inference call across all connections** — at
high N the worker effectively becomes single-threaded.

Result:
- N=100 p99 = **131 ms** (Phase-1 baseline was 19.4 ms — 6.7× regression)
- N=200 p99 = **10.5 s**, 31% drop rate

Memory side did improve (1149 vs 1722 MB at N=100, 33% less) — shared
weights work — but the Mutex destroys the latency win.

**Fix**: ORT's C API guarantees `OrtSession::Run` is thread-safe on `&Session`.
The Rust binding's `&mut self` signature is overly conservative. Either
expose `run_unsafe(&self)` on the binding (upstream PR), or use a per-thread
Session with shared `Arc<EnvironmentInner>` (still amortizes the runtime,
keeps inference parallel). This work is **not done in this branch** — it's
a clear next step.

### 2. cgo embed is the production path

`yalue/onnxruntime_go` works correctly with one shared session and
per-instance LSTM state. No Mutex. Latency stays linear in N. Memory stays
flat at ~40-47 MB total for N=1 → 200 (shared model arena dominates;
per-instance tensor + state ≈ 0.05 MB).

What it gets right that everything else gets wrong:
- **Real-time throughput**: 125 000/125 000 frames at N=200. Pipecat
  managed 65 050. Strawgo Phase-1 managed 1 438.
- **Memory at scale**: 47 MB at N=200. Pipecat 1 709 MB. Strawgo Phase-1
  3 670 MB. **36× less than Pipecat, 78× less than Strawgo Phase-1.**
- **Frame budget**: p99 17.75 ms at N=200, well under 32 ms. Comfortable
  headroom for adding STT/LLM/TTS on the same box.

What's left on the table:
- p99 latency is still 38× behind Pipecat at N=200 (17.75 ms vs 0.47 ms).
  Suspect cause: yalue allocates a fresh `Tensor` per `Run()` call. Pipecat
  reuses internal buffers across calls. Estimated win from tensor reuse:
  3-5×, putting cgo embed under ~5 ms p99 at N=200.
- int8 quantized model not yet plumbed in this path. Estimated 2-3× more
  on top of tensor reuse.

### 3. TEN-VAD: smaller, but slower at scale on this hardware

TEN-VAD's value prop is "306 KB library, RTF 0.0086-0.0160, beats Silero
on detection". On the 4-core e2-standard-4 here:
- N=1 RSS 9 MB (vs cgo embed 37 MB — 4× less)
- N=200 RSS 134 MB (vs cgo embed 47 MB — **2.8× more**)
- p99 always 1.5-2× the cgo embed number

The per-instance C handle (TEN-VAD has no documented session-sharing API)
means RSS scales ~0.6 MB/agent. cgo embed's shared session means
~0.05 MB/agent. At N ≥ 50, cgo embed wins memory; TEN-VAD only wins at
small N where the fixed cost dominates.

Latency parity surprises me — Agent A measured 251 µs/call on M3 Pro.
On Intel e2-standard-4 the per-call cost is closer to 1-2 ms in our
binding. Possible causes: TEN-VAD's "two 256-sample hops per Strawgo
512-sample frame" doubles inference count; the Linux .so was compiled
against libc++ and we link with libstdc++ (cosmetic only after fix); the
e2 box doesn't have the AVX-512 / NEON paths TEN-VAD optimizes for.

There's also the **license** issue — Apache 2.0 with an Agora non-compete
clause. Engineering green, legal yellow.

### 4. Pipecat's real ceiling

Pipecat's p99 is gorgeous (sub-ms across all N) **but** at N=200 it
schedules only 52 % of the theoretical 125 000 frames. asyncio on 4 cores
can't drive 200 tasks at perfect 32 ms cadence; Pipecat masks frame skips
as "fewer scheduled," not as drops. Both Go variants schedule **100 %**
of the theoretical max. So:
- If you measure "how many frames did each agent see?" — Go variants
  win 1.9×.
- If you measure "of the frames we did process, how fast?" — Pipecat
  wins on tail.

For voice agents, what the user perceives is whether VAD reacts within
the 32 ms tick. Both Go variants stay under that budget. Pipecat is
"faster than necessary"; Go variants are "fast enough on a budget."

## Architectural takeaways

1. **The Phase-1 conclusion stands but with nuance.** The original
   "outsource ONNX over IPC is strictly worse" finding holds — Strawgo
   Phase-1 was indeed worse than Pipecat on every dimension. But the
   reverse is also true once you bring ORT in-process in Go: you can
   match or beat Pipecat on memory and real-time throughput, even with a
   slower per-call latency.

2. **Memory advantage of in-process Go is huge.** Pipecat's 126 MB Python
   baseline is a fixed tax. Go's runtime is ~5 MB. At N=200 this is the
   difference between 47 MB and 1 709 MB. On a 16 GB box that's "5 000
   agents per VM" vs "~600 agents per VM."

3. **Latency advantage of Pipecat is real but bounded.** It evaporates
   once asyncio can't keep up at high N. Go's GMP scheduler does not have
   that ceiling on this hardware.

4. **Mutex on a hot inference path is fatal.** The Rust Tier 1 lesson
   is general: any time you "share" something to save memory, verify
   that you didn't accidentally serialize the work it was doing.

## Recommended path

**Short term (1-2 days):**
- Land cgo embed as the production VAD path. Make it the default;
  keep the Rust onnx-worker behind an `STRAWGO_VAD=ipc` flag for users
  who need crash isolation.
- Tensor reuse in `silero_embedded.go`: pre-allocate input/state/sr
  tensors at construction, mutate in place each frame. Estimated 3-5×
  latency win (p99 ~3-5 ms at N=200).

**Medium term (1 week):**
- int8 quantized Silero in cgo embed path. Plumb the
  `--vad-model-int8` flag from Rust Tier 1 into Go. Estimated additional
  2-3× latency win.
- Fix `Arc<Mutex<Session>>` in Rust onnx-worker — file upstream PR
  against `pykeio/ort` for `run_unsafe(&self)`, or switch to per-thread
  sessions on a shared environment.

**Long term:**
- Eliminate the per-conn `SmartTurnSession` allocation in onnx-worker
  (Tier 1.5). Becomes the dominant memory cost once VAD is shared.
- Evaluate TEN-VAD on production audio for accuracy parity. Resolve
  the Agora license question with counsel.

## Numbers in one place

```
arm           N    p50      p95      p99       sched   ok      err   rss_peak
─────────────────────────────────────────────────────────────────────────────
strawgo p1    1    1.36ms   1.86ms   1.93ms    625     625     0     67
              5    1.69ms   2.37ms   2.53ms    3122    3122    0     147
              10   2.32ms   3.09ms   3.33ms    6250    6250    0     216
              25   3.58ms   5.05ms   5.42ms    15625   15625   0     462
              50   5.60ms   8.56ms   9.39ms    29882   29882   0     880
              100  9.30ms   15.63ms  19.38ms   52981   52981   0     1722
              200  1745ms   10793ms  11101ms   26275   1438    24837 3670

pipecat p1    1    663µs    744µs    829µs     626     626     0     151
              5    517µs    702µs    747µs     3130    3130    0     183
              10   481µs    667µs    721µs     6260    6260    0     222
              25   456µs    596µs    679µs     15650   15650   0     340
              50   441µs    515µs    593µs     31300   31300   0     535
              100  267µs    422µs    464µs     61657   61657   0     928
              200  266µs    322µs    468µs     65050   65050   0     1709

rust-tier1    1    1.46ms   1.99ms   2.23ms    625     625     0     67
              5    2.56ms   3.97ms   4.71ms    3121    3121    0     125
              10   3.72ms   6.03ms   7.16ms    6250    6250    0     171
              25   6.80ms   11.73ms  12.96ms   15625   15625   0     332
              50   12.50ms  22.61ms  27.54ms   30414   30414   0     607
              100  66.15ms  107.93ms 131.07ms  45257   45257   0     1149
              200  97.81ms  8899ms   10455ms   30132   20846   9286  7173

cgo-embed     1    1.33ms   2.58ms   3.05ms    625     625     0     37
              5    1.38ms   2.17ms   3.27ms    3124    3124    0     38
              10   1.56ms   2.25ms   2.44ms    6250    6250    0     39
              25   2.23ms   3.26ms   3.53ms    15625   15625   0     40
              50   3.26ms   5.08ms   5.47ms    31250   31250   0     42
              100  5.50ms   8.87ms   9.85ms    62500   62500   0     44
              200  9.98ms   16.57ms  17.75ms   125000  125000  0     47

tenvad        1    1.22ms   1.66ms   1.76ms    624     624     0     9
              5    1.49ms   2.05ms   2.25ms    3125    3125    0     12
              10   1.69ms   2.53ms   2.80ms    6250    6250    0     15
              25   2.72ms   4.36ms   4.69ms    15624   15624   0     25
              50   4.69ms   7.63ms   8.74ms    31250   31250   0     41
              100  8.07ms   13.63ms  14.27ms   62500   62500   0     73
              200  15.82ms  26.94ms  28.33ms   125236  125236  0     134
```

## Files

- Phase-1 baseline: `bench/REPORT.md`
- Per-arm impl reports: `bench/RUST_TIER1_REPORT.md`,
  `bench/CGO_EMBED_REPORT.md`, `bench/TEN_VAD_REPORT.md`
- Sweep driver: `bench/run_phase2_sweep.sh`
- Raw logs: `bench/phase2_results_pull/{rust-tier1,cgo-embed,tenvad}_n*.log`

## Methodology notes

Same VM as Phase-1 to keep numbers comparable. Each arm ran 7 levels (N ∈
{1,5,10,25,50,100,200}), 20 s of measurement per level, sleep 3 s between
levels. Synthetic 16 kHz int16 PCM frames at 32 ms cadence (512 samples).
RSS captured from `/proc/<pid>/status` (rust-tier1 worker) or
`getrusage(RUSAGE_SELF)` (cgo-embed, tenvad). Latency is wall time around
a single inference call (Unix-socket round-trip for rust-tier1, direct
cgo Run for cgo-embed and tenvad). The `cmd/loadtest` client is unchanged
across arms — the rust-tier1 arm uses the existing socket protocol; the
cgo-embed and tenvad arms use stand-alone harnesses (`cmd/loadtest-embed`,
`cmd/loadtest-tenvad`) that mirror its stdout format and pacing logic.
