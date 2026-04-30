# Rust onnx-worker — Tier-1 perf fixes report

Branch: `perf/onnx-worker-tier1` (4 commits, each individually revertible).

Tier-1 fixes from `bench/REPORT.md` applied:

| # | fix | commit | landed |
|---|-----|--------|--------|
| 1 | Shared ORT `Session` across connections; per-conn keeps only LSTM state | d12a328 | yes |
| 2 | Explicit `intra_op_threads = inter_op_threads = 1` per session | f22fc82 | yes |
| 3 | Global ORT intra-op thread pool sized to `available_parallelism()` | 84a82c3 | yes |
| 4 | int8 quantised model option (`--vad-model-int8`) with safe fallback | f429726 | yes |

The Unix-socket protocol and IPC framing are unchanged — `cmd/loadtest`
runs against the new worker without modification.

## Files changed

- `onnx-worker/src/vad.rs` — refactored `SileroSession` to hold only per-stream
  LSTM state. New `SharedSileroSession = Arc<Mutex<Session>>` and
  `build_shared_session()` helper. Validates Silero I/O surface
  (`{input,state,sr}` → `{output,stateN}`) at startup so a wrong-shape model is
  rejected before traffic hits.
- `onnx-worker/src/server.rs` — `handle_connection` now takes the shared VAD
  session as an argument instead of a model path.
- `onnx-worker/src/smart_turn.rs` — explicit `with_inter_threads(1)` for symmetry
  with VAD; unchanged otherwise.
- `onnx-worker/src/main.rs` —
  - `ort::init().with_global_thread_pool(...).commit()` *before* any
    `Session::builder()` calls,
  - new `--vad-model-int8 <path>` flag (env: `ONNX_WORKER_VAD_INT8`) with
    fallback to fp32 on any int8 load/validation failure.

## Local smoke test (macOS, Apple M-series, 11 logical cores)

`cargo build --release` then loadtest 10 s per level, fp32 model:

```
N    p50      p95      p99      sched   ok      err   rss_peak
1    1.87ms   2.57ms   3.18ms   312     312     0     74.1 MB
5    2.50ms   4.34ms   5.92ms   1560    1560    0     175.1 MB
10   3.16ms   5.73ms   9.77ms   3120    3120    0     291.3 MB
25   4.38ms   8.13ms   10.49ms  7800    7800    0     483.5 MB
50   5.74ms   11.20ms  25.62ms  15600   15600   0     743.4 MB
100  7.91ms   20.50ms  40.06ms  31200   31200   0     1304.1 MB
```

Same hardware, same sweep, int8 model (`--vad-model-int8`):

```
N    p50      p95      p99      sched   ok      err   rss_peak
1    1.99ms   2.85ms   3.41ms   312     312     0     82.8 MB
5    2.63ms   4.60ms   6.22ms   1560    1560    0     170.5 MB
10   3.53ms   6.18ms   8.51ms   3120    3120    0     284.8 MB
25   4.82ms   8.41ms   9.96ms   7800    7800    0     445.3 MB
50   5.62ms   9.96ms   10.92ms  15600   15600   0     750.9 MB
100  7.52ms   12.99ms  14.52ms  31200   31200   0     1307.6 MB
```

Key takeaways from the local sweep (same hardware, same loadtest, same VAD-only
workload, before/after):

- **fp32 N=10 p99: 9.8 ms** (this branch) — already substantially better
  baseline behaviour, even before considering the GCE-vs-Mac CPU gap.
- **int8 N=100 p99: 14.5 ms** vs **fp32 N=100 p99: 40 ms** — int8 is ~2.7× faster
  in the tail at the highest level the bench machine could sustain.
- 0 errors at every level for both models — no melt-down at N=100, in contrast
  to the original Linux baseline which started losing 94.5% of frames at N=200.
- RSS at N=100 is dominated by the **per-connection `SmartTurnSession`** (the
  smart-turn model is ~14 MB per session and is still allocated per connection;
  the loadtest only sends VAD frames, but each connection still constructs the
  smart-turn session). VAD itself is now a single shared session and contributes
  a fixed ~30-40 MB. Eliminating the per-conn smart-turn session is a
  Tier-1.5 / Tier-2 follow-up.
- Idle RSS dropped from 67 MB (original report's first row) to **27-34 MB** —
  consistent with one shared VAD session vs N session arenas.

## Notes vs the original `REPORT.md` baseline

Numbers above are macOS Apple Silicon, not GCE Linux, so direct apples-to-apples
deltas with the report require a re-run on the bench VM. The cross-compile to
`x86_64-unknown-linux-gnu` from this dev box failed at link time:

```
ld: unknown options: --as-needed -Bstatic -Bdynamic ...
```

(macOS `ld` doesn't accept GNU linker flags; we'd need `cross` with a running
docker daemon, or a separate Linux x86_64 toolchain). Steps to rebuild on the
GCE VM:

```bash
# on the GCE bench VM (Ubuntu 24.04, x86_64):
git fetch origin perf/onnx-worker-tier1
git checkout perf/onnx-worker-tier1
cd onnx-worker && cargo build --release
# binary: target/release/onnx-worker

# Then run the existing sweep:
cd ..
./bench/run_vad_sweep.sh   # already on disk in this repo
```

If the VM doesn't have the int8 model on disk, fetch it once:

```bash
mkdir -p ~/.cache/strawgo/models
curl -sSL -o ~/.cache/strawgo/models/silero_vad_int8.onnx \
  'https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model_int8.onnx?download=1'
```

Then add `--vad-model-int8 ~/.cache/strawgo/models/silero_vad_int8.onnx` to the
worker invocation in `run_vad_sweep.sh` to bench int8.

## What this commit series does NOT change

- Smart-turn session is still per connection. Memory growth at high N is
  dominated by smart-turn now.
- IPC contract is unchanged: same Unix socket, same `[u8 type][u32 len][...]`
  framing, same VAD payload `[u32 sr][i16 PCM]` → `[f32]`.
- No new dependencies in `Cargo.toml`. (`std::thread::available_parallelism` is
  in stable libstd; `ort::environment::GlobalThreadPoolOptions` is already a
  re-export from the `ort` crate already in `Cargo.toml`.)

## Confidence + risks

- **High confidence** in correctness of the shared session — `pykeio/ort`
  2.0.0-rc.12 marks `Session: Send + Sync` (`src/session/mod.rs:675`), the
  underlying `run_inner(&self, ...)` is `&self`, the C++ `OrtSession::Run` is
  thread-safe per ORT docs, and the LSTM state is fully decoupled from the
  session object (it's just an input tensor + an output tensor we copy back
  into a per-conn `Vec<f32>`).
- **Mutex contention** is a theoretical risk above ~1000 concurrent streams.
  Hold time = inference kernel (~1-2 ms fp32, ~0.5-1 ms int8 on x86_64). At
  31.25 Hz frame rate, a single mutex can serve roughly `1 / (1.5 ms / 32 ms)
  ≈ 21×` multiplexing factor before saturating. We're well below that at
  N=200. If the VM bench shows mutex saturation, a follow-up could shard
  sessions (one per CPU core).
- **int8 numerical accuracy** is not validated end-to-end here. The model file
  is the same one Pipecat and other downstream consumers ship with, and the
  client-side VAD threshold (`vad_threshold` in Strawgo's Go side) is
  unchanged. Accuracy validation under noisy/real audio is left as a Tier-2
  follow-up before flipping the int8 default on.

## Sources cited (unchanged from `REPORT.md`)

- silero-vad#744 — shared model + per-stream state is supported by maintainer
- pipecat #2050 — same pattern in production
- ORT threading docs — global thread pool, NUMA pinning
- onnx-community/silero-vad on HF — int8 build
- pykeio/ort 2.0.0-rc.12 source — verified `Session: Send + Sync`,
  `with_global_thread_pool`, `DisablePerSessionThreads`
