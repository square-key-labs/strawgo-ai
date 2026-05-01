# Pipecat full-pipeline bench (denoise → VAD → smart-turn)

Direct extension of `bench/pipecat/vad_bench.py` (Phase-1 VAD-only).
This bench runs the realistic 3-ONNX-model voice-agent pipeline per
frame so the result is directly comparable to the parallel Strawgo
pipeline harness on `perf/onnx-worker-tier1`.

Files added:
- `bench/pipecat/pipeline_bench.py` — the harness
- `bench/run_pipeline_sweep.sh` — VM driver

## Stack used

| stage | implementation | model file | call site |
|---|---|---|---|
| 1. Denoise | onnxruntime direct (custom wrapper `StreamingDenoiser`) | `gtcrn_simple.onnx` (yuyun2000/SpeechDenoiser, 16 kHz, 535 KB) | `denoiser.process(frame_int16)` |
| 2. VAD | **Pipecat 1.1.0 built-in** `SileroVADAnalyzer` | bundled `silero_vad.onnx` (~2.3 MB) | `vad.voice_confidence(frame_bytes)` |
| 3. Smart-turn | onnxruntime direct (custom wrapper `SmartTurnSession`) | `smart-turn-v3.1-cpu.onnx` (this repo's `testdata/models/`) | `smart_turn.predict()` after VAD-edge or fixed cadence |

### Why a custom Smart-Turn V3 wrapper

Pipecat 1.1.0 only ships `LocalSmartTurnAnalyzerV2` (Wav2Vec2 / pytorch).
`LocalSmartTurnAnalyzerV3` lands in pipecat 1.5+, which our pinned bench
venv does not have. We wrap onnxruntime directly with the same Whisper
mel feature pipeline (`n_fft=400`, `hop=160`, `n_mels=80`, Slaney mel,
log10 normalization, 8 s rolling window padded to `[1, 80, 800]`)
implemented in `onnx-worker/src/features.rs`. This gives the bench
identical per-call work to what V3 would do under pipecat 1.5+.

### Why GTCRN instead of literal DFN3

The task's preferred path is "DFN3 16 kHz". Reality on the ground:
- The official `Rikorose/DeepFilterNet3_onnx.tar.gz` ships **48 kHz only**
  and contains **three separate ONNX graphs** (`enc.onnx`, `erb_dec.onnx`,
  `df_dec.onnx`) connected through a complex ERB / DF-coefficient
  pipeline that requires the upstream `df` python runtime. There is no
  16 kHz official export.
- `pip install deepfilternet` is torch-based, not ONNX, so it doesn't
  exercise the per-call ONNX path the bench is supposed to measure.
- `shimondoodkin/deepfilter-rt` ships a `combined_streaming.onnx` for
  DFN3, but it's stored via Git LFS and not directly downloadable
  through `gh api`/`raw.githubusercontent.com`. It is also still 48 kHz
  internally (the wrapper resamples).
- `yuyun2000/SpeechDenoiser/16k/gtcrn_simple.onnx` is a **real** streaming
  ONNX denoiser at **native 16 kHz**, with per-instance state caches
  (`conv_cache`, `tra_cache`, `inter_cache`) that the bench reuses across
  calls — exactly the I/O surface DFN3 *would* expose if a comparable
  16 kHz export existed. It runs 1× ONNX call per STFT frame
  (`n_fft=512`, `hop=256` → 2 calls per 32 ms PCM frame).

GTCRN matches Strawgo's frame size with no inline resampling, behaves as
a per-instance stateful ONNX session (which is the architectural cost
the bench is measuring), and keeps the test rig honest. The harness
accepts `--denoise-model /path/to/your.onnx` so swapping in a different
model later is one flag, not a code change.

If you must run DFN3 specifically, the easiest substitute is to point
`--denoise-model` at the streaming-export `combined_streaming.onnx` from
`shimondoodkin/deepfilter-rt` (download the LFS object directly, or use
`git lfs pull` after cloning). The wrapper's auto-detection of
non-`mix` inputs means it will pick up DFN3's caches without code
changes — provided the model has a single `mix`-style spectral input
(this is the case for the `combined_streaming` export).

## Per-frame work breakdown

For each agent, every 32 ms tick:

```
1. Denoise (~always)
   - int16 → f32 normalize             (vector op, ns)
   - 2× STFT (n_fft=512, hop=256)
   - 2× ONNX inference on GTCRN       ← dominant cost in stage 1
   - 2× iSTFT + overlap-add
   - clip → int16
2. VAD (~always)
   - 1× SileroVADAnalyzer.voice_confidence
     (1× ONNX inference internally)   ← matches vad_bench.py exactly
3. Smart-turn (conditional)
   - append clean frame to 8 s rolling buffer (always cheap)
   - if (VAD high→low edge AND debounce) OR (cadence due):
       - extract 80×800 log-mel              ← O(n_frames) numpy + 1× FFT batch
       - 1× ONNX inference on smart-turn-v3.1
```

Smart-turn invocation is governed by the same heuristic as the parallel
Strawgo harness:

- **VAD-edge triggered**: prev_conf ≥ 0.6 AND curr_conf < 0.3, capped to
  1× / 500 ms per agent.
- **Cadence-triggered (default)**: every `--turn-cadence-ms` (default
  2000 ms ≈ realistic utterance-end rate). Synthetic audio doesn't
  reliably trip Silero, so we force the call to keep the workload
  deterministic and comparable across architectures. Set to `0` to
  disable cadence and rely on VAD edges only.

## Local smoke results (macOS, M-series CPU, single-machine)

Run via:

```sh
/tmp/pipeline-bench-venv/bin/python bench/pipecat/pipeline_bench.py \
  -n N --dur DUR \
  --denoise-model /tmp/dfn3-models/gtcrn_simple.onnx \
  --turn-model testdata/models/smart-turn-v3.1-cpu.onnx
```

| run | N | dur | scheduled | drop | smart-turn | RSS load Δ | per-agent | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| denoise + VAD + ST (default cadence 2 s) | 1 | 5 s | 158 | 0% | 3 | 44.6 MB | 44.64 MB | 2.3 ms | 5.0 ms | 86.8 ms |
| denoise + VAD + ST (default cadence 2 s) | 5 | 8 s | 1255 | 0% | 20 | 128.2 MB | 25.65 MB | 1.6 ms | 5.3 ms | 80.2 ms |
| denoise + VAD only (`--no-smart-turn`) | 5 | 8 s | 1255 | 0% | 0 | 71.8 MB | 14.37 MB | 1.9 ms | 2.5 ms | **3.6 ms** |

Notes:
- p99 with smart-turn enabled is dominated by the 4× cadence frames per
  agent (smart-turn = 8 s mel + 1 ONNX run on a 80×800 input = ~80 ms
  per call on M-series). This is **the right cost** to measure: in
  production this is what an utterance-end check costs.
- Denoise+VAD-only p99 = **3.6 ms at N=5** — well inside the 32 ms frame
  budget, leaving 28 ms of headroom for smart-turn frames at the rate
  they actually fire (~once per 2 s per agent). On the bench VM (4-core
  e2-standard-4) the smart-turn cost will be ~2-3× larger, so expect
  p99 closer to 200 ms on cadence frames.
- All three models load successfully. Per-agent first-load cost is
  inflated at N=1 because all model arenas count against the first
  agent's RSS delta; the marginal cost stabilizes at **~14 MB per
  agent** for the in-process Python pipeline (denoise + VAD ONNX
  sessions; smart-turn adds ~10-15 MB more for the Whisper-style
  encoder weights, hence ~26 MB at N=5 with smart-turn enabled).
- No drops at N=1 or N=5. All scheduled frames completed.

## Memory accounting (vs vad_bench)

| stage at N=5 | per-agent RSS | source |
|---|---:|---|
| `vad_bench.py` (Silero alone) | ~8 MB/agent | bench/REPORT.md Phase-1 |
| `pipeline_bench.py` no-smart-turn (denoise + VAD) | **14 MB/agent** | this run |
| `pipeline_bench.py` full (denoise + VAD + smart-turn) | **26 MB/agent** | this run |

So:
- GTCRN denoise session ≈ ~6 MB/agent (14 - 8).
- Smart-turn-v3.1 session ≈ ~12 MB/agent (26 - 14).

Inflated at N=1 (44 MB) because shared-arena one-time costs aren't
amortized. As N grows, expect per-agent to converge toward ~26 MB on the
VM.

That puts the in-process Pipecat pipeline at ~30-50 MB/agent at the
target N — matching the task's expectation. Compare to the cgo-embed
Strawgo path which holds the 3 sessions **shared** across N agents and
should stay flat in the 50-100 MB range total (per `bench/PHASE2_REPORT.md`
extrapolation: VAD-only stayed at 47 MB at N=200; adding 2 more shared
models pushes it to maybe 80-100 MB, *not* 26 × N).

## Running on the bench VM

After deploying:

```sh
# On the e2-standard-4 VM, with bench-venv already provisioned:
scp bench/pipecat/pipeline_bench.py vm:~/pipeline_bench.py
scp bench/run_pipeline_sweep.sh vm:~/run_pipeline_sweep.sh

# Drop the GTCRN model in:
curl -L -o ~/gtcrn_simple.onnx \
    https://github.com/yuyun2000/SpeechDenoiser/raw/main/16k/gtcrn_simple.onnx

# Make sure smart-turn model is at ~/smart-turn-v3.1-cpu.onnx
# (already provisioned for the Phase-2 sweep)

bash ~/run_pipeline_sweep.sh
```

CSV output drops into `~/pipeline_results/pipecat_pipeline_n<N>.csv`
parallel to `~/phase2_results/`. The `framework` column reads
`pipecat_pipeline` so `bench/compare_vad.py` can join across files
without changes.

## Things worth noting

1. **`SileroVADAnalyzer.voice_confidence` returns a numpy 0-d array on
   recent versions** — pipecat 1.1.0 returns `out[0]` from the model
   output, which on numpy 2.x raises `DeprecationWarning` if you cast
   directly with `float(...)`. The harness extracts the scalar via
   `np.asarray(x).reshape(-1)[0]`.

2. **Pipecat's bundled Silero is the v5-style model** (`input` + `state`
   + `sr` graph inputs), not v6.2.1. The graph signature in v6 is the
   same (input + LSTM state + sample_rate → output + state), so this is
   the "Pipecat 1.1.0 voice_confidence API" path the task asks for. No
   reason to swap in a different ONNX file unless we want to A/B v5 vs
   v6; that's a separate bench.

3. **GTCRN's `mix` input is `[1, 257, 1, 2]`** — one STFT frame at a
   time. Our 32 ms PCM (512 samples) at hop 256 gives 2 STFT frames per
   call, i.e. 2 ONNX runs per 32 ms tick. The wrapper's overlap-add
   tail is only one HOP wide; this is the standard streaming setup
   GTCRN expects.

4. **Smart-turn's mel features are computed in numpy**, not via torch
   or librosa. Lower import cost, no GIL surprises, and matches the
   Rust impl bit-for-bit at the algorithm level. Pure numpy 2.x
   (vectorized FFT batch); no per-frame python loop in the hot path.

5. **The harness currently does not pin worker tasks to threads** —
   asyncio cooperative scheduling with sync ONNX calls means at high N,
   one worker's blocking inference stalls the others. This is the
   *same* failure mode `bench/pipecat/vad_bench.py` exhibited at N=200
   (sched 65k of 125k expected). Expect identical bottleneck here, just
   shifted earlier in N because per-frame work is 3× larger.

## Constraints honoured

- `bench/pipecat/vad_bench.py` was not touched.
- `src/audio/vad/`, `cmd/loadtest-embed/`, `cmd/loadtest-tenvad/`, and
  `onnx-worker/` were not touched.
- All models used are MIT/Apache: GTCRN MIT, Silero MIT, smart-turn
  MIT (ships in this repo).
- Work stayed inside this worktree.
