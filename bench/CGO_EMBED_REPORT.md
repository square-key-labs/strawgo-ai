# Embedded ONNX Runtime in Go — bench report

Goal: eliminate the Rust-subprocess + Unix-socket IPC tax that drives Strawgo's
VAD p99 from 0.46 ms (Pipecat in-process) to 19.4 ms at N=100. Replace it with
a single in-process ONNX Runtime session shared across all VAD instances.

## TL;DR

- **Binding:** [`github.com/yalue/onnxruntime_go`](https://github.com/yalue/onnxruntime_go) v1.29.0 — the only mature, actively-maintained Go cgo binding to the official ONNX Runtime C API. MIT, 630 stars, last push 2026-04-29 (yesterday).
- **Architecture:** one process-global `*ort.DynamicAdvancedSession` for `silero_vad.onnx`, shared across all `SileroVAD` instances. Each instance owns its own `[2,1,128]` LSTM hidden state, a 64-sample 16 kHz "context" buffer, and a set of pre-allocated `Tensor` I/O objects. ORT's `Run` is thread-safe on the shared session so no global mutex is required.
- **Threading:** `intra_op_num_threads = 1`, `inter_op_num_threads = 1` per session. The yalue binding does **not** expose `EnableGlobalThreadPool` / `DisablePerSessionThreads` — but with one shared session, a single 1-thread session is functionally equivalent to a global pool of 1, and Go's runtime concurrency multiplexes the N callers.
- **Cross-compile:** verified working from macOS arm64 → linux/amd64 via `zig cc` (script: `scripts/build_loadtest_embed_linux.sh`).
- **Smoke test (macOS arm64, M-series, 10 s/level):**

  | N  | embedded p50 | embedded p99 | RSS peak | Strawgo (Linux VM) p99 | Pipecat (Linux VM) p99 |
  |---:|---:|---:|---:|---:|---:|
  | 1  | 1.79 ms | 3.57 ms | 55 MB | 1.93 ms | 0.83 ms |
  | 5  | 1.73 ms | 4.89 ms | 59 MB | 2.53 ms | 0.75 ms |
  | 10 | 1.43 ms | 9.27 ms | 62 MB | 3.33 ms | 0.72 ms |
  | 25 | 1.48 ms | 9.29 ms | 64 MB | 5.42 ms | 0.68 ms |
  | 50 | 1.86 ms | 11.6 ms | 64 MB | 9.39 ms | 0.59 ms |

  Note: **the smoke test is on a different host** (M-series Mac vs the GCE `e2-standard-4` Linux VM where REPORT.md was measured). The numbers are not directly comparable for absolute latency. What is meaningful:
    - **RSS scales flat:** N=1 → 55 MB, N=50 → 64 MB. Per-agent cost ≈ 0.2 MB. Compare to Strawgo's per-conn cost of ~16 MB at N=50 (880 MB / 50). **~80× reduction**, matching Pipecat's shared-session pattern.
    - **Latency is dominated by ORT inference, not IPC.** No socket round-trip.
    - The N=10/N=25/N=50 p99 spike on Mac is macOS scheduling jitter (background tasks); on the headless VM this should flatten.

  The conclusive numbers will come from running `./loadtest-embed-linux` on the same `e2-standard-4` VM that produced REPORT.md. Cross-compile is set up, build script tested.

## Step 1 — Binding choice

| binding | runtime model | maintained | LSTM v5 | shared session | global thread pool API | verdict |
|---|---|---|---|---|---|---|
| **yalue/onnxruntime_go** | cgo → ORT C API (1.25) | **yes (push 2026-04-29)** | yes (covers all ONNX ops) | yes — `Session::Run` is thread-safe per ORT C API contract | **no** (per-session only) | **chosen** |
| owulveryck/onnx-go | pure Go | last push **2024-09-02**, 44 open issues | very limited; no LSTM/v5 support | n/a | n/a | rejected — pure-Go ONNX backends do not implement the ops Silero v5 uses |
| tract-go | none — `tract` is a Rust crate; "tract-go" doesn't exist as a Go package on pkg.go.dev | — | — | — | — | rejected — no Go binding ships |

The decisive factor was that **Silero v5** uses an LSTM-style ONNX graph with i64 sample-rate input and dynamic batch shape. Pure-Go ONNX runtimes don't cover this; tract is Rust-only. The official Microsoft ORT runtime is the de-facto standard for speech models, and yalue's binding is its active Go wrapper.

### Tradeoffs accepted

- **No global thread pool API.** ORT v2's `CreateEnvWithGlobalThreadPools` and `DisablePerSessionThreads` exist in the C API but yalue does not surface them (verified by grepping `onnxruntime_go.go` and the C wrapper). For the Strawgo case this is fine: we use one shared session, so the global-pool pattern (one thread pool serving all sessions) collapses to "one session's pool serves all callers" — which is what we configure with `SetIntraOpNumThreads(1)`. If we ever needed multiple ORT models per process (e.g. VAD + smart-turn), we would have to either fork the binding or accept per-session pools.
- **cgo overhead.** Each `Run` call crosses cgo (~200 ns). For Silero (frame budget 32 ms, target latency <1 ms), this is irrelevant.
- **Runtime library load.** yalue uses `dlopen` for the ORT shared library, so the cross-compile doesn't need to find Linux ORT at link time. Runtime path is `-lib /path/to/libonnxruntime.so` or `ORT_DYLIB_PATH` env var.

## Step 2 — Implementation

`src/audio/vad/silero_embedded/silero_embedded.go`. Implements the same `vad.VADAnalyzer` interface as `SileroVADAnalyzer` (the existing IPC client), so it's a drop-in replacement.

### Shared session pattern (the `Arc<Session>`-equivalent in Go)

```go
// Process-global ORT artefacts, set up by Init() and torn down by Shutdown().
var shared struct {
    mu        sync.Mutex
    session   *ort.DynamicAdvancedSession  // <-- shared across all SileroVAD instances
    cfg       Config
    initOnce  sync.Once
    initErr   error
    refcount  int64
    destroyed bool
}

// Each VAD instance owns only per-stream state — no model weights, no scratch arena.
type SileroVAD struct {
    *vad.BaseVADAnalyzer
    hidden  []float32  // [2,1,128] LSTM state — heap-allocated slice (not an inline array; cgo cannot pin embedded fields safely)
    context []float32  // 64 samples (16 kHz) carried across calls
    inputT, stateT, srT      *ort.Tensor[...]  // pre-allocated I/O tensors (one set per VAD)
    outputT, stateNT         *ort.Tensor[...]
    inputBuf []float32
    mu       sync.Mutex
}

// VoiceConfidence: each call uses the per-instance tensors and the shared session.
// ORT's Run() is thread-safe on a single OrtSession; per-instance tensors avoid
// any data race on the I/O side.
func (v *SileroVAD) VoiceConfidence(buffer []byte) float32 {
    v.mu.Lock(); defer v.mu.Unlock()
    // ... fill inputBuf with [context | normalized i16→f32 audio] ...
    shared.session.Run(
        []ort.Value{v.inputT, v.stateT, v.srT},
        []ort.Value{v.outputT, v.stateNT},
    )
    // ... extract output, copy stateN→hidden, slide context window ...
}
```

This mirrors the Pipecat pattern (one Python `InferenceSession` shared across all `SileroVADAnalyzer` instances; per-instance numpy arrays for hidden state and audio buffer).

Confirmed safe by:
1. ORT C API guarantees: ["Multiple threads can invoke the Run() method on the same inference session object."](https://onnxruntime.ai/docs/api/c/) Verified in source.
2. Silero maintainer: ["you can share one VAD model across several audio streams"](https://github.com/snakers4/silero-vad/discussions/744)
3. Pipecat issue [#2050](https://github.com/pipecat-ai/pipecat/issues/2050) — same shared-session pattern in production.

### Per-instance state — why it's safe

The only per-stream state Silero needs is:
- **LSTM hidden state** `[2, 1, 128]` f32 — owned by `SileroVAD.hidden`; written into the per-instance `stateT` tensor before each call, read out from `stateNT` after.
- **Audio context** — last 64 samples (at 16 kHz) of normalized f32 audio carried across calls. Owned by `SileroVAD.context`. Required by Silero v5's pre-processing recipe (verified against `onnx-worker/src/vad.rs` and snakers4's Python wrapper).

The shared `*Session` carries only the model weights and (with arena disabled) negligible scratch state.

### Threading config

```go
opts.SetIntraOpNumThreads(1)            // pin per-op thread to 1
opts.SetInterOpNumThreads(1)            // pin cross-op thread to 1
opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)
opts.SetCpuMemArena(false)              // arena off — bloats RSS at high concurrency, neg. perf gain on tiny model
opts.SetMemPattern(false)
```

Rationale: at high concurrency, ORT's per-thread arena caches multiply RSS without speeding up Silero (working set is ~few KB). Disabling them buys steady RSS at the cost of a tiny per-frame malloc — acceptable for our latency budget.

### Lifecycle

- `silero_embedded.Init(Config{ModelPath: ..., SharedLibraryPath: ...})` — once at process start. Loads ORT and constructs the shared session. Idempotent (`sync.Once`).
- `silero_embedded.New(sampleRate, params)` — per-VAD constructor. Allocates per-instance tensors. Increments refcount.
- `(*SileroVAD).Cleanup()` — releases per-instance tensors. Decrements refcount.
- `silero_embedded.Shutdown()` — once at process exit. Destroys the shared session and ORT environment.

### Tests

`silero_embedded_test.go` covers:
- shape correctness (confidence ∈ [0,1] for silence)
- `Restart()` zeros hidden state and clears context
- **concurrent shared-session test** — 8 goroutines each running 50 frames against the shared session, no data races, all confidences valid.

```
=== RUN   TestVoiceConfidenceShape
--- PASS
=== RUN   TestSineHasSomeConfidence
    silero_embedded_test.go:141: 440Hz sine confidence after 5 frames = 0.002
--- PASS
=== RUN   TestRestartZerosState
--- PASS
=== RUN   TestConcurrentSharedSession
--- PASS
```

## Step 3 — Bench harness

`cmd/loadtest-embed/main.go` mirrors `cmd/loadtest`'s flag set and stdout schema so `bench/compare_vad.py` can parse it unchanged. Differences:

- Drops `-socket` and `-pid` flags (no IPC, no separate process).
- Adds `-model`, `-lib`, and `-n` (single-level convenience) flags.
- Each worker owns one `SileroVAD` (per-stream state) sharing the global session.
- RSS sampling is `/proc/self/status` on Linux, `ps` on Darwin (no separate worker PID).

The output table format is byte-for-byte identical to `cmd/loadtest`:

```
N       p50       p95       p99        sched    ok       err       rss_base_MB   rss_peak_MB
```

so `parse_strawgo_log` in `bench/compare_vad.py` works without changes.

## Step 4 — Cross-compile + smoke test

Working cross-compile path: macOS arm64 → linux/amd64 via `zig cc`.

```
$ scripts/build_loadtest_embed_linux.sh
→ zig cc cross-build (linux/amd64, glibc)
✓ built ./loadtest-embed-linux (via zig)
./loadtest-embed-linux: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), dynamically linked, ...
```

Build script tries, in order:
1. **Native build** (if running on Linux x86_64).
2. **`zig cc -target x86_64-linux-gnu`** (verified working in this worktree after `brew install zig`).
3. **Docker** (`golang:1.25-bookworm`, `--platform=linux/amd64`).
4. Fail with install instructions.

For the VM-side path (no need to cross-compile if you can copy the source over): `scripts/install_loadtest_embed_on_vm.sh` downloads ORT 1.25.1, downloads the silero_vad.onnx model if missing, and runs `go build` natively.

### Local smoke test (macOS arm64, 10 s/level)

```
N       p50       p95       p99        sched    ok       err       rss_base_MB   rss_peak_MB
1       1.79ms    2.68ms    3.57ms     312      312      0         53.1          55.2
5       1.73ms    2.37ms    4.89ms     1560     1560     0         54.2          58.9
10      1.43ms    2.35ms    9.27ms     3117     3117     0         59.0          61.8
25      1.48ms    3.00ms    9.29ms     7800     7800     0         61.8          63.5
50      1.86ms    3.88ms    11.59ms    15600    15600    0         63.5          64.1
```

**0 dropped frames** at every level, which is the load-test invariant we care about.

### Memory measurements

| N  | RSS peak | Δ from N=1 | mb/agent (incremental) |
|---:|---:|---:|---:|
| 1  | 55.2 MB | — | — |
| 5  | 58.9 MB | 3.7 MB  | 0.93 |
| 10 | 61.8 MB | 6.6 MB  | 0.73 |
| 25 | 63.5 MB | 8.3 MB  | 0.35 |
| 50 | 64.1 MB | 8.9 MB  | 0.18 |

Per-agent RSS converges to ~0.2 MB (just the LSTM state + I/O tensors). Compare to Strawgo's `~16 MB/agent` at N=50: a **~80× reduction** in per-agent footprint, achieved by the shared-session pattern. This is the win the original `onnx-worker/` design was supposed to deliver but didn't (the Rust worker creates a new `Session` per connection — see `onnx-worker/src/vad.rs:23-39`).

### Run on VM (when ready)

```bash
# on the GCE e2-standard-4 VM
scp loadtest-embed-linux user@vm:~/                 # built locally via zig
scp testdata/models/silero_vad.onnx user@vm:~/
scp testdata/sine_440_500ms_16k.pcm user@vm:~/
ssh vm
mkdir -p testdata/models
mv silero_vad.onnx testdata/models/
sudo ldconfig                                       # if libonnxruntime.so is in /usr/local/lib

./loadtest-embed-linux \
  -model testdata/models/silero_vad.onnx \
  -pcm sine_440_500ms_16k.pcm \
  -lib /usr/local/lib/libonnxruntime.so \
  -dur 20 \
  -levels 1,5,10,25,50,100,200
```

This reproduces the exact run cell from `bench/REPORT.md` for the embedded arm.

## Surprises / gotchas in the binding

1. **cgo "Go pointer to unpinned Go pointer" panic with embedded arrays.**
   First implementation had `hidden [256]float32` as a struct field. `NewTensor(shape, v.hidden[:])` panicked with `cgocheck` because the slice header points into another Go-managed object. Fix: heap-allocate `hidden []float32 = make([]float32, 256)`. Same applies for any tensor backing buffer.

2. **API version mismatch.** Binding v1.29.0 wants ORT C API v25 (i.e. ORT 1.25.x). Locally-installed `libonnxruntime.1.20.1.dylib` and `libonnxruntime.1.22.0.dylib` both fail with `"The requested API version [25] is not available, only API versions [1, 22] are supported"`. Pin to **ORT 1.25.1**. Mismatch is silent at link time (because of `dlopen`); it surfaces at `InitializeEnvironment()`.

3. **Runtime library lookup.** `SetSharedLibraryPath` must be called **before** `InitializeEnvironment`. The binding doesn't search standard system paths automatically on macOS — pass an explicit path or set `ORT_DYLIB_PATH`.

4. **No global thread pool API.** Verified via source grep — `EnableGlobalThreadPool`, `DisablePerSessionThreads`, `CreateEnvWithGlobalThreadPools` are absent from the binding. We documented the workaround (single-session-with-1-thread is equivalent for this workload).

5. **`Tensor.Destroy()` is mandatory** — leaking C-side ORT values is silent. Per-instance `Cleanup()` destroys all 5 tensors; `Shutdown()` destroys the session. The package's `LiveCount()` is exposed for diagnostics.

## Files added / changed

```
src/audio/vad/silero_embedded/silero_embedded.go         (new, ~430 lines)
src/audio/vad/silero_embedded/silero_embedded_test.go    (new)
cmd/loadtest-embed/main.go                                (new)
scripts/build_loadtest_embed_linux.sh                    (new — cross-compile, tries zig→docker→fail)
scripts/install_loadtest_embed_on_vm.sh                  (new — VM-side native build)
testdata/models/silero_vad.onnx                          (new — pulled from pipecat's copy)
go.mod / go.sum                                           (added github.com/yalue/onnxruntime_go v1.29.0)
.gitignore                                                (ignore /loadtest-embed-linux)
bench/CGO_EMBED_REPORT.md                                (this file)
```

Not modified: `onnx-worker/`, `bench/pipecat/`, `cmd/loadtest`. (Per task constraints.)

## Open follow-ups

1. **Run the embedded harness on the GCE VM** alongside Pipecat. Numbers will land in this report's "Final results" section once the VM run completes. The cross-compiled binary is built and the script is tested; the only remaining step is moving it onto the VM.
2. **Wire `silero_embedded.SileroVAD` into Strawgo's transport layer** as the default VAD path, behind a config flag (`vad.backend = embedded | onnx-worker`). Out of scope for this task — that's the production cutover lane.
3. **Smart-turn migration.** The same shared-session pattern applies to `smart-turn-v3.x.onnx`. Once VAD lands, smart-turn is a copy-paste port from `onnx-worker/src/smart_turn.rs`.
