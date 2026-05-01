"""Pipecat full-pipeline (denoise → VAD → smart-turn) capacity bench.

Direct extension of `vad_bench.py` (which measures VAD-only). This bench
runs the realistic 3-ONNX-model voice-agent pipeline per frame so the
result is directly comparable to the parallel Strawgo pipeline harness on
`perf/onnx-worker-tier1`.

Per-frame pipeline (real-time 32 ms / 512-sample cadence at 16 kHz):

  1. **Denoise** — ONNX, every frame.
       Default model: yuyun2000/SpeechDenoiser `gtcrn_simple.onnx` (16 kHz,
       streaming, 535 KB). The DeepFilterNet3 official ONNX export is
       48 kHz only with a 3-graph pipeline (enc / erb_dec / df_dec) that
       requires the full DFN python runtime. GTCRN is the same kind of
       streaming-ONNX denoiser at 16 kHz, runs 1× ONNX call per STFT frame
       (n_fft=512, hop=256 → 2 calls per 32 ms PCM frame), and exposes
       per-instance state caches the way DFN3 would. Pass
       `--denoise-model /path/to/your.onnx` to swap in a different model
       — the wrapper auto-detects model I/O surface (mix-only or
       mix+caches).
  2. **VAD** — Pipecat 1.1's `SileroVADAnalyzer.voice_confidence(frame)`.
       Same model and call site as `vad_bench.py`.
  3. **Smart-turn** — onnxruntime directly on `smart-turn-v3.1-cpu.onnx`.
       Pipecat 1.1 ships `LocalSmartTurnAnalyzerV2` (Wav2Vec2 / pytorch)
       and only adds V3 in 1.5+, so we wrap onnxruntime with the Whisper
       mel feature pipeline (80-bin, 16 kHz, n_fft=400, hop=160) that
       matches `onnx-worker/src/features.rs`. Run only when VAD confidence
       drops from high → low (utterance-end candidate), capped to once
       per 500 ms per agent.

Latency is measured **end-to-end across the 3-stage pipeline** (denoise +
VAD + maybe smart-turn) so it's directly comparable to the parallel
Strawgo pipeline harness.

Metrics:
  - RSS (RUSAGE_SELF) at peak
  - per-frame end-to-end latency p50/p95/p99
  - scheduled / succeeded / errors / drop_rate

CSV columns are identical to `vad_bench.py` so `bench/compare_vad.py`
parses it without changes — the only difference is the `framework`
column reads `pipecat_pipeline` instead of `pipecat`.

Usage:
  python pipeline_bench.py -n 50 --dur 20 --csv pipeline_n50.csv \\
      --turn-model $MODELS_DIR/smart-turn-v3.1-cpu.onnx \\
      --denoise-model $MODELS_DIR/gtcrn_simple.onnx
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import resource
import sys
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.audio.vad.vad_analyzer import VADParams  # noqa: E402

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # Silero requires exactly 512 at 16 kHz
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 LE
FRAME_INTERVAL = FRAME_SAMPLES / SAMPLE_RATE  # 0.032 s = 32 ms

# Smart-turn invocation thresholds — match the heuristic in the parallel
# Strawgo pipeline harness so per-frame work is comparable.
#
# Two ways to drive smart-turn:
#   1. VAD-edge triggered (real production behaviour): smart-turn fires
#      when VAD confidence drops from ≥ TURN_VAD_HIGH to < TURN_VAD_LOW,
#      capped to 1 / TURN_MIN_INTERVAL_S per agent.
#   2. Fixed cadence (`--turn-cadence-ms`): smart-turn fires every N ms
#      regardless of VAD output. Ensures deterministic smart-turn load
#      so latency numbers are comparable across architectures (synthetic
#      audio doesn't reliably trip Silero — real speech would).
TURN_VAD_HIGH = 0.6
TURN_VAD_LOW = 0.3
TURN_MIN_INTERVAL_S = 0.5


# ---------------------------------------------------------------------------
# Denoiser — onnxruntime wrapper for streaming ONNX denoisers (default GTCRN)
# ---------------------------------------------------------------------------

class StreamingDenoiser:
    """Per-instance streaming ONNX denoiser.

    Runs STFT → ONNX → iSTFT each call. Holds the model's recurrent state
    caches across calls. Auto-detects model I/O — works with GTCRN
    (`mix` + 3 caches) and falls back to mix-only models.

    Frame layout: input PCM is 32 ms (512 samples @ 16 kHz). With STFT
    n_fft=512 / hop=256, that's 2 STFT frames per call → 2 ONNX runs.
    """

    # GTCRN STFT params (n_fft=512, hop=256 at 16 kHz)
    N_FFT = 512
    HOP = 256
    N_FREQ = N_FFT // 2 + 1  # 257

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        in_names = [i.name for i in self._session.get_inputs()]
        self._in_names = in_names
        self._out_names = [o.name for o in self._session.get_outputs()]

        # Pre-build hann^0.5 analysis/synthesis window (matches GTCRN ref).
        win = np.hanning(self.N_FFT).astype(np.float32)
        self._win = np.sqrt(np.maximum(win, 0.0))

        # State caches (GTCRN). Will be reset on first call to match shapes.
        self._caches: dict[str, np.ndarray] = {}
        self._init_caches_from_model()

        # Overlap-add scratch — last HOP samples of the previous frame's
        # iSTFT tail, kept so 32 ms input → 32 ms output with continuous
        # phase across calls.
        self._ola_tail = np.zeros(self.N_FFT - self.HOP, dtype=np.float32)
        # And the pre-buffer: the leading HOP samples we need from the
        # previous call to form the first STFT window of this call.
        self._prev_pcm = np.zeros(self.N_FFT - self.HOP, dtype=np.float32)

    def _init_caches_from_model(self) -> None:
        """Pre-allocate state caches matching the model's declared shapes."""
        for inp in self._session.get_inputs():
            n = inp.name
            if n == "mix":
                continue
            shape = []
            for d in inp.shape:
                if isinstance(d, int) and d > 0:
                    shape.append(d)
                else:
                    # Symbolic / dynamic dim — fall back to 1 for batch.
                    shape.append(1)
            self._caches[n] = np.zeros(shape, dtype=np.float32)

    def _stft_frame(self, x: np.ndarray) -> np.ndarray:
        """Return one STFT frame as [1, 257, 1, 2] (real, imag)."""
        windowed = x * self._win
        spec = np.fft.rfft(windowed, n=self.N_FFT)  # complex [257]
        out = np.empty((1, self.N_FREQ, 1, 2), dtype=np.float32)
        out[0, :, 0, 0] = spec.real.astype(np.float32)
        out[0, :, 0, 1] = spec.imag.astype(np.float32)
        return out

    def _istft_frame(self, spec: np.ndarray) -> np.ndarray:
        """Inverse: spec [1, 257, 1, 2] → time-domain [N_FFT] frame."""
        re = spec[0, :, 0, 0]
        im = spec[0, :, 0, 1]
        complex_spec = re + 1j * im
        sig = np.fft.irfft(complex_spec, n=self.N_FFT).astype(np.float32)
        return sig * self._win

    def process(self, pcm_int16: np.ndarray) -> np.ndarray:
        """Denoise a 512-sample int16 frame; return 512-sample int16 frame.

        If the model has no `mix` input, this is a no-op (returns input)
        — bench treats the denoiser stage as having loaded successfully
        and counts the load-time cost only.
        """
        if "mix" not in self._in_names:
            return pcm_int16

        # int16 → float32 normalized
        pcm_f32 = pcm_int16.astype(np.float32) / 32768.0
        # Concatenate prev-tail + this frame so we have enough samples
        # for two HOP-spaced STFT frames inside one 32 ms call.
        buf = np.concatenate([self._prev_pcm, pcm_f32])
        # Save the new prev-tail (last N_FFT-HOP=256 samples of buf).
        self._prev_pcm = buf[-(self.N_FFT - self.HOP):].copy()

        # Two STFT frames per 32 ms PCM frame (hop 256, n_fft 512).
        n_stft = (len(buf) - self.N_FFT) // self.HOP + 1
        if n_stft < 1:
            return pcm_int16

        out_time = np.zeros(len(buf), dtype=np.float32)
        for k in range(n_stft):
            start = k * self.HOP
            seg = buf[start:start + self.N_FFT]
            mix = self._stft_frame(seg)

            feed: dict[str, np.ndarray] = {"mix": mix}
            for name, arr in self._caches.items():
                feed[name] = arr

            outs = self._session.run(self._out_names, feed)

            # First output is the enhanced spectrum.
            enh = outs[0]
            # Remaining outputs map back to the cache inputs by suffix
            # `_out` (GTCRN convention). If a model uses different names,
            # update the dict in order of declared cache inputs.
            cache_inputs = [n for n in self._in_names if n != "mix"]
            cache_outs = [n for n in self._out_names if n != self._out_names[0]]
            for in_name, out_arr in zip(cache_inputs, outs[1:]):
                # If # outputs lines up positionally, just copy.
                self._caches[in_name] = out_arr

            time_seg = self._istft_frame(enh)
            out_time[start:start + self.N_FFT] += time_seg

        # The portion of `out_time` corresponding to the *current* PCM
        # frame is buf[N_FFT-HOP:]. Take the first FRAME_SAMPLES = 512
        # samples of out_time aligned with that range.
        # buf layout: [prev_pcm (256) | pcm_f32 (512)] = 768 samples
        # We want the 512-sample window aligned with pcm_f32, applying
        # OLA across calls via self._ola_tail.
        offset = len(buf) - len(pcm_f32)  # 256
        out_window = out_time[offset:offset + len(pcm_f32)].copy()

        # Add the previous frame's tail at the start of this frame.
        tail_len = len(self._ola_tail)
        out_window[:tail_len] += self._ola_tail
        # Save the tail for next time: portion of out_time past the
        # current frame's right edge (only matters if hop config means
        # one of the windows extended past it).
        self._ola_tail = out_time[offset + len(pcm_f32):].copy()
        if len(self._ola_tail) < tail_len:
            # Pad to the fixed tail length so future adds don't drift.
            pad = np.zeros(tail_len - len(self._ola_tail), dtype=np.float32)
            self._ola_tail = np.concatenate([self._ola_tail, pad])

        # Re-clip to int16
        out_window = np.clip(out_window * 32768.0, -32768, 32767)
        return out_window.astype(np.int16)


# ---------------------------------------------------------------------------
# Smart-turn — onnxruntime wrapper for smart-turn-v3.1-cpu.onnx
# ---------------------------------------------------------------------------

class WhisperFeatureExtractor:
    """80-bin Whisper log-mel feature extractor at 16 kHz.

    Mirrors `onnx-worker/src/features.rs` (which itself mirrors the Go
    reference). n_fft=400, hop=160, n_mels=80, Slaney mel scale, Hann
    window, center reflect padding, log10 normalization to [-1, 1].

    Output shape is `[n_mels, n_frames]` flattened row-major; smart-turn
    v3.1 expects `[1, 80, 800]` so caller pads/truncates to 800 frames =
    8 s @ hop 160.
    """

    SAMPLE_RATE = 16000
    N_FFT = 400
    HOP = 160
    N_MELS = 80
    FFT_PAD = 512  # next pow2 above N_FFT

    # Slaney mel scale constants — must match Rust impl exactly.
    SLANEY_F_SP = 200.0 / 3.0
    SLANEY_MIN_LOG_HZ = 1000.0
    SLANEY_MIN_LOG_MEL = SLANEY_MIN_LOG_HZ / SLANEY_F_SP
    SLANEY_LOG_STEP = 0.06875177742094912

    def __init__(self) -> None:
        self._window = self._hann_window(self.N_FFT)
        self._mel_filters = self._build_mel_filterbank()

    @staticmethod
    def _hann_window(n: int) -> np.ndarray:
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)

    @classmethod
    def _hz_to_mel(cls, hz: float) -> float:
        if hz < cls.SLANEY_MIN_LOG_HZ:
            return hz / cls.SLANEY_F_SP
        return cls.SLANEY_MIN_LOG_MEL + math.log(hz / cls.SLANEY_MIN_LOG_HZ) / cls.SLANEY_LOG_STEP

    @classmethod
    def _mel_to_hz(cls, mel: float) -> float:
        if mel < cls.SLANEY_MIN_LOG_MEL:
            return mel * cls.SLANEY_F_SP
        return cls.SLANEY_MIN_LOG_HZ * math.exp(cls.SLANEY_LOG_STEP * (mel - cls.SLANEY_MIN_LOG_MEL))

    def _build_mel_filterbank(self) -> np.ndarray:
        n_freqs = self.N_FFT // 2 + 1  # 201
        f_min = 0.0
        f_max = self.SAMPLE_RATE / 2.0

        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        n_points = self.N_MELS + 2
        mel_points = [mel_min + i * (mel_max - mel_min) / (self.N_MELS + 1)
                      for i in range(n_points)]
        hz_points = [self._mel_to_hz(m) for m in mel_points]
        bin_points = [
            min(int(math.floor((self.N_FFT + 1) * hz / self.SAMPLE_RATE)), n_freqs - 1)
            for hz in hz_points
        ]

        filters = np.zeros((self.N_MELS, n_freqs), dtype=np.float32)
        for i in range(self.N_MELS):
            left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
            if center != left:
                for j in range(left, min(center, n_freqs)):
                    filters[i, j] = (j - left) / (center - left)
            if right != center:
                for j in range(center, min(right, n_freqs)):
                    filters[i, j] = (right - j) / (right - center)

            band = mel_points[i + 2] - mel_points[i]
            if band > 0:
                filters[i] *= 2.0 / band

        return filters

    def extract(self, audio: np.ndarray, max_samples: int) -> np.ndarray:
        """Compute log-mel spectrogram. audio in [-1, 1] f32."""
        # Pad/truncate as in Rust: keep LAST max_samples; zero-pad at
        # BEGINNING.
        padded = np.zeros(max_samples, dtype=np.float32)
        n = min(len(audio), max_samples)
        if len(audio) >= max_samples:
            padded[:] = audio[-max_samples:]
        else:
            padded[max_samples - n:] = audio

        # Center reflect-pad N_FFT/2 each side.
        half = self.N_FFT // 2
        # numpy's reflect pad without endpoint matches torch.stft's
        # default `center=True` (symmetric reflect, NO repeat).
        padded_full = np.pad(padded, (half, half), mode="reflect")

        n_frames = max(max_samples // self.HOP, 1)
        n_freqs = self.N_FFT // 2 + 1

        # Build all frames at once.
        starts = np.arange(n_frames) * self.HOP
        # Frames matrix [n_frames, N_FFT]
        frames = np.empty((n_frames, self.N_FFT), dtype=np.float32)
        for i, s in enumerate(starts):
            frames[i] = padded_full[s:s + self.N_FFT] * self._window

        # FFT padded to FFT_PAD=512 to match Rust scratch size.
        spec = np.fft.rfft(frames, n=self.FFT_PAD, axis=1)
        # Rust slices first 201 freqs from the 512-pt FFT output.
        spec = spec[:, :n_freqs]
        magnitudes = (spec.real * spec.real + spec.imag * spec.imag).T  # [n_freqs, n_frames]

        mel_spec = self._mel_filters @ magnitudes  # [n_mels, n_frames]

        # log10 with floor 1e-10
        log_mel = np.log10(np.maximum(mel_spec, 1e-10))
        max_val = float(log_mel.max())
        min_val = max_val - 8.0
        log_mel = np.maximum(log_mel, min_val)
        log_mel = (log_mel + 4.0) / 4.0

        return log_mel.astype(np.float32)


class SmartTurnSession:
    """ONNX-runtime wrapper for smart-turn-v3.1-cpu.onnx."""

    MAX_SAMPLES = 8 * 16000  # 8 s window
    MEL_BINS = 80
    MEL_FRAMES = 800

    def __init__(self, model_path: str) -> None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._fe = WhisperFeatureExtractor()
        # Rolling 8s utterance buffer (f32 in [-1, 1]).
        self._buf = np.zeros(0, dtype=np.float32)

    def append_audio(self, pcm_int16: np.ndarray) -> None:
        """Push a denoised frame into the rolling utterance buffer."""
        f32 = pcm_int16.astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, f32])
        if len(self._buf) > self.MAX_SAMPLES:
            self._buf = self._buf[-self.MAX_SAMPLES:]

    def predict(self) -> float:
        """Return turn-completion probability for the last 8 s of audio."""
        if len(self._buf) == 0:
            return 0.0
        mel = self._fe.extract(self._buf, self.MAX_SAMPLES)
        # Pad/truncate to [80, 800]
        mel_padded = np.zeros((self.MEL_BINS, self.MEL_FRAMES), dtype=np.float32)
        copy_frames = min(mel.shape[1], self.MEL_FRAMES)
        mel_padded[:, :copy_frames] = mel[:, :copy_frames]
        tensor = mel_padded.reshape(1, self.MEL_BINS, self.MEL_FRAMES)

        outs = self._session.run(None, {"input_features": tensor})
        return float(outs[0].flatten()[0])


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@dataclass
class WorkerStats:
    scheduled: int = 0
    succeeded: int = 0
    errors: int = 0
    smart_turn_calls: int = 0
    latencies_us: list[float] = None  # type: ignore[assignment]


def synth_pcm_frame(rng: np.random.Generator) -> bytes:
    """Speech-like noise at ~-12 dBFS, returned as int16 LE bytes."""
    sig = rng.standard_normal(FRAME_SAMPLES).astype(np.float32) * 8000
    return sig.astype(np.int16).tobytes()


def synth_speechy_frame(rng: np.random.Generator, t_idx: int, agent_idx: int) -> bytes:
    """Frame that toggles between speech-like and silence so VAD bounces.

    Pattern: ~640 ms of mid-band-noise "speech" alternating with ~320 ms
    of low-amplitude background. The exact frequency mix doesn't matter
    for the bench — what matters is that VAD confidence rises above
    TURN_VAD_HIGH and falls below TURN_VAD_LOW so smart-turn is exercised
    on the same frequency the parallel Strawgo harness would see in real
    traffic.
    """
    period = 30  # frames; 30 * 32 ms ≈ 960 ms (640 ms speech / 320 ms gap)
    in_speech = (t_idx % period) < 20
    if in_speech:
        # Sum of three sine waves in speech band + low-amp noise.
        n = np.arange(FRAME_SAMPLES, dtype=np.float32)
        # Phase varies per agent so all N agents don't sync.
        phase = (agent_idx * 0.37) % 1.0
        sig = (
            np.sin(2 * np.pi * (180 + phase * 40) * n / SAMPLE_RATE) * 6000
            + np.sin(2 * np.pi * 540 * n / SAMPLE_RATE) * 3000
            + np.sin(2 * np.pi * 1240 * n / SAMPLE_RATE) * 2000
        )
        sig += rng.standard_normal(FRAME_SAMPLES).astype(np.float32) * 800
    else:
        sig = rng.standard_normal(FRAME_SAMPLES).astype(np.float32) * 200
    return np.clip(sig, -32768, 32767).astype(np.int16).tobytes()


async def worker(
    idx: int,
    dur_s: float,
    ready: asyncio.Event,
    start: asyncio.Event,
    stats_out: list[WorkerStats],
    *,
    denoise_path: str,
    turn_path: str,
    enable_smart_turn: bool,
    turn_cadence_s: float,
) -> None:
    """One 'agent': owns 1× denoiser, 1× VAD, 1× smart-turn."""
    rng = np.random.default_rng(idx ^ 0xC0FFEE)

    denoiser = StreamingDenoiser(denoise_path)

    vad = SileroVADAnalyzer(
        sample_rate=SAMPLE_RATE,
        params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.6),
    )
    vad.set_sample_rate(SAMPLE_RATE)

    smart_turn = SmartTurnSession(turn_path) if enable_smart_turn else None

    ws = WorkerStats(latencies_us=[])
    prev_vad = 0.0
    last_turn_t = 0.0
    t_idx = 0

    ready.set()
    await start.wait()

    deadline = time.monotonic() + dur_s
    next_tick = time.monotonic()
    while time.monotonic() < deadline:
        sleep = next_tick - time.monotonic()
        if sleep > 0:
            await asyncio.sleep(sleep)
        next_tick += FRAME_INTERVAL

        ws.scheduled += 1
        frame_bytes = synth_speechy_frame(rng, t_idx, idx)
        t_idx += 1
        frame_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
        t0 = time.perf_counter_ns()
        try:
            # Stage 1: denoise
            clean_int16 = denoiser.process(frame_int16)
            clean_bytes = clean_int16.tobytes()

            # Stage 2: VAD
            vad_raw = vad.voice_confidence(clean_bytes)
            # voice_confidence returns numpy 0-d array on some pipecat versions;
            # extract scalar safely (numpy 2.x deprecated implicit conversion).
            vad_prob = float(np.asarray(vad_raw).reshape(-1)[0]) if hasattr(vad_raw, "shape") else float(vad_raw)

            # Stage 3: smart-turn — VAD-edge OR fixed cadence
            if smart_turn is not None:
                smart_turn.append_audio(clean_int16)
                now = time.monotonic()
                vad_edge = (
                    prev_vad >= TURN_VAD_HIGH
                    and vad_prob < TURN_VAD_LOW
                    and (now - last_turn_t) > TURN_MIN_INTERVAL_S
                )
                cadence_due = (
                    turn_cadence_s > 0
                    and (now - last_turn_t) >= turn_cadence_s
                )
                if vad_edge or cadence_due:
                    smart_turn.predict()
                    ws.smart_turn_calls += 1
                    last_turn_t = now

            prev_vad = vad_prob
            ws.succeeded += 1
        except Exception as e:  # noqa: BLE001
            ws.errors += 1
            if ws.errors <= 3:
                print(f"[worker {idx}] error: {e}", file=sys.stderr)
            continue
        ws.latencies_us.append((time.perf_counter_ns() - t0) / 1000.0)

    try:
        result = vad.cleanup()
        if asyncio.iscoroutine(result):
            await result
    except Exception:  # noqa: BLE001
        pass

    stats_out.append(ws)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def get_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[int(min(len(s) - 1, (len(s) - 1) * q / 100))]


async def main_async(args: argparse.Namespace) -> None:
    print(f"config: n={args.n} dur={args.dur}s sample_rate={SAMPLE_RATE} "
          f"frame_samples={FRAME_SAMPLES} frame_interval_ms={FRAME_INTERVAL*1000:.1f}")
    print(f"denoise_model: {args.denoise_model}")
    print(f"turn_model: {args.turn_model} "
          f"(enabled={'yes' if args.turn_model and not args.no_smart_turn else 'no'})")

    if not os.path.exists(args.denoise_model):
        print(f"ERROR: denoise model not found: {args.denoise_model}", file=sys.stderr)
        sys.exit(2)
    enable_smart_turn = bool(args.turn_model) and not args.no_smart_turn
    if enable_smart_turn and not os.path.exists(args.turn_model):
        print(f"ERROR: smart-turn model not found: {args.turn_model}", file=sys.stderr)
        sys.exit(2)

    rss_baseline_mb = get_rss_mb()
    print(f"RSS baseline (before agents): {rss_baseline_mb:.1f} MB")

    ready_events = [asyncio.Event() for _ in range(args.n)]
    start_event = asyncio.Event()
    stats_out: list[WorkerStats] = []

    tasks = [
        asyncio.create_task(worker(
            i, args.dur, ready_events[i], start_event, stats_out,
            denoise_path=args.denoise_model,
            turn_path=args.turn_model,
            enable_smart_turn=enable_smart_turn,
            turn_cadence_s=args.turn_cadence_ms / 1000.0,
        ))
        for i in range(args.n)
    ]

    await asyncio.wait([asyncio.create_task(e.wait()) for e in ready_events])
    rss_after_load_mb = get_rss_mb()
    print(f"RSS after {args.n} pipelines loaded: {rss_after_load_mb:.1f} MB "
          f"(Δ {(rss_after_load_mb - rss_baseline_mb):.1f} MB; "
          f"{(rss_after_load_mb - rss_baseline_mb) / max(args.n,1):.2f} MB/agent)")

    t_start = time.monotonic()
    start_event.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.monotonic() - t_start

    rss_peak_mb = get_rss_mb()

    total_sched = sum(s.scheduled for s in stats_out)
    total_ok = sum(s.succeeded for s in stats_out)
    total_err = sum(s.errors for s in stats_out)
    total_turn = sum(s.smart_turn_calls for s in stats_out)
    all_lats = [v for s in stats_out for v in s.latencies_us]

    drop_rate = 0.0 if total_sched == 0 else (total_sched - total_ok) / total_sched

    print()
    print("─" * 72)
    print(f"ELAPSED            {elapsed:.1f}s  (target {args.dur}s)")
    print(f"AGENTS             {args.n}")
    print(f"RSS (baseline → after load → peak)")
    print(f"                   {rss_baseline_mb:.0f} → {rss_after_load_mb:.0f} → {rss_peak_mb:.0f} MB")
    print(f"                   per-agent load cost: "
          f"{(rss_after_load_mb - rss_baseline_mb) / max(args.n,1):.2f} MB")
    print(f"FRAMES             scheduled={total_sched} ok={total_ok} err={total_err}  "
          f"drop_rate={drop_rate*100:.2f}%")
    print(f"SMART-TURN CALLS   {total_turn}  ({total_turn / max(args.n, 1):.1f} per agent)")
    if all_lats:
        print(f"FRAME LATENCY (µs) p50={pct(all_lats,50):.0f} "
              f"p95={pct(all_lats,95):.0f} p99={pct(all_lats,99):.0f} "
              f"(n={len(all_lats)})")
    print("─" * 72)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["framework", "n", "elapsed_s", "rss_baseline_mb",
                        "rss_after_load_mb", "rss_peak_mb", "per_agent_mb",
                        "scheduled", "ok", "err", "drop_rate",
                        "lat_p50_us", "lat_p95_us", "lat_p99_us"])
            w.writerow(["pipecat_pipeline", args.n, f"{elapsed:.2f}",
                        f"{rss_baseline_mb:.1f}",
                        f"{rss_after_load_mb:.1f}",
                        f"{rss_peak_mb:.1f}",
                        f"{(rss_after_load_mb - rss_baseline_mb) / max(args.n,1):.2f}",
                        total_sched, total_ok, total_err, f"{drop_rate:.4f}",
                        f"{pct(all_lats,50):.0f}",
                        f"{pct(all_lats,95):.0f}",
                        f"{pct(all_lats,99):.0f}"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1, help="concurrent pipeline agents")
    ap.add_argument("--dur", type=float, default=20.0, help="seconds per level")
    ap.add_argument("--csv", default="", help="write summary row")
    ap.add_argument(
        "--denoise-model",
        default=os.environ.get("PIPELINE_DENOISE_MODEL",
                               os.path.expanduser("~/gtcrn_simple.onnx")),
        help="path to streaming-ONNX denoiser (default GTCRN 16k from "
             "yuyun2000/SpeechDenoiser; pass DFN3 combined.onnx if you have one)",
    )
    ap.add_argument(
        "--turn-model",
        default=os.environ.get("PIPELINE_TURN_MODEL",
                               os.path.expanduser("~/smart-turn-v3.1-cpu.onnx")),
        help="path to smart-turn-v3.1-cpu.onnx",
    )
    ap.add_argument(
        "--no-smart-turn",
        action="store_true",
        help="skip stage 3 (smart-turn) entirely; for isolating denoise+VAD cost",
    )
    ap.add_argument(
        "--turn-cadence-ms",
        type=int,
        default=2000,
        help="force smart-turn invocation every N ms regardless of VAD "
             "(default 2000 ms ≈ realistic utterance-end rate). "
             "Set to 0 to fall back to VAD-edge-only triggering.",
    )
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
