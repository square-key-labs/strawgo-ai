#!/usr/bin/env python3
"""quality_test — VAD-edge agreement A/B for candidate denoisers (NSNet2,
GTCRN, no-denoise) on LibriSpeech-test-clean speech mixed with ESC-50 noise.

What it measures:

  Ground-truth voice mask  := Silero VAD on CLEAN speech (16 kHz, 32 ms hop)
  Candidate voice mask     := Silero VAD on cleaned PCM produced by:
    - no-denoise: pass NOISY through Silero unchanged
    - GTCRN:      apply GTCRN gain mask × noisy spec, iSTFT, then VAD
    - NSNet2:     apply NSNet2 gain mask × noisy spec, iSTFT, then VAD

  Metrics per (clean, noise, snr_db) tuple:
    - Jaccard(candidate_mask, ground_truth_mask)
    - VAD precision / recall vs ground truth
    - SNR estimate before/after (dB)
    - PESQ if `pesq` available on $PATH

  Aggregated across all pairs:
    mean Jaccard, mean PESQ, mean SNR improvement, win/loss counts.

Uses:
  - LibriSpeech test-clean as clean speech source (~30 wav files sampled)
  - ESC-50 as noise source (5 categories: keyboard, fan, traffic, dog, kid)
  - Silero VAD ONNX (the same one the production pipeline uses) for fair comparison
  - GTCRN ONNX (yuyun2000/SpeechDenoiser, 4-input streaming)
  - NSNet2 ONNX (Microsoft DNS baseline, 1-input 161-bin)

True 320-pt FFT for NSNet2 via scipy (the Go wrapper currently uses a
zero-padded 512-pt FFT — this Python harness uses the production-correct
size to give NSNet2 its best shot).

Usage:
  bench-venv/bin/python bench/quality_test.py \\
      --speech-dir   $DATA_DIR/LibriSpeech/test-clean \\
      --noise-dir    $DATA_DIR/ESC-50-master/audio \\
      --vad-model    testdata/models/silero_vad.onnx \\
      --gtcrn-model  testdata/models/gtcrn_simple.onnx \\
      --nsnet2-model testdata/models/nsnet2-20ms.onnx \\
      --out          bench/quality_results.csv \\
      --snr-db       5 \\
      --num-pairs    20

Output: CSV row per (speech, noise, snr) tuple + a summary table to stdout.

KNOWN HARNESS LIMITATIONS (codex review #40, must fix before re-running):
  1. GTCRN denoise() writes cleaned_frame at out[t:t+512] but the analysis
     buf at hop t represents audio[t-256:t+256], so output is shifted
     forward by ~256 samples (~16 ms). Systematically hurts GTCRN Jaccard
     by up to one 32 ms VAD frame on each speech edge. Fix: write to
     out[t-256:t+256] with proper boundary handling, or post-shift the
     output by 256 samples.
  2. NSNet2 denoise() uses hop=320 with sqrt-Hann analysis × sqrt-Hann
     synthesis (= Hann) and ZERO overlap. This applies a Hann-shaped gain
     per 20 ms frame with amplitude nulls at every frame boundary. Causes
     real artifacts independent of the model. Fix: use hop=160 (50 %
     overlap) so analysis × synthesis sums to COLA, or use rectangular
     synthesis only.
  3. SileroVAD runs on [1, 512] input shape but Strawgo production wraps
     each frame with a 64-sample context prefix (vad.go, [1, 576]).
     RELATIVE Jaccard between candidates and noisy baseline is still
     internally consistent (same Silero call applied to all), but
     ABSOLUTE Jaccard numbers do NOT directly translate to what the
     production VAD would emit. Frame report conclusions accordingly.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from scipy.signal import stft, istft, get_window


SR = 16_000
FRAME_SIZE = 512    # Strawgo's 32 ms frame at 16 kHz
HOP = 256           # GTCRN/Silero hop


def load_wav_16k_mono(path: str) -> np.ndarray:
    """Load WAV → float32 [-1, 1], resampled to 16 kHz mono."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        # Lazy resample. For a quality test on speech this is fine.
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g).astype(np.float32)
    return audio


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Pad/loop noise to clean length, scale to target SNR, mix."""
    if len(noise) < len(clean):
        n_repeat = (len(clean) // len(noise)) + 1
        noise = np.tile(noise, n_repeat)
    noise = noise[: len(clean)]

    p_clean = (clean ** 2).mean()
    p_noise = (noise ** 2).mean()
    if p_noise <= 1e-12:
        return clean.copy()
    target_noise_p = p_clean / (10 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_p / p_noise))
    return (clean + scale * noise).astype(np.float32)


# ── GTCRN denoiser ──────────────────────────────────────────────────────────
class GTCRNDenoiser:
    """Streaming GTCRN with 256-sample hop, 512 FFT, 3 cache states."""

    def __init__(self, model_path: str):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        # Cache shapes from gtcrn_simple.onnx:
        self.conv_cache = np.zeros((2, 1, 16, 16, 33), dtype=np.float32)
        self.tra_cache = np.zeros((2, 3, 1, 1, 16), dtype=np.float32)
        self.inter_cache = np.zeros((2, 1, 33, 16), dtype=np.float32)
        self.window = np.sqrt(np.hanning(512)).astype(np.float32)

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """Run streaming GTCRN over audio, return cleaned PCM at same length."""
        n = len(audio)
        # Pad to whole hop boundary.
        pad = (HOP - n % HOP) % HOP
        if pad:
            audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])
        n_padded = len(audio)

        # Overlap-add buffers.
        out = np.zeros(n_padded + 512, dtype=np.float32)
        # Sliding STFT: window over [t-512:t] every HOP samples.
        prev_window = np.zeros(512, dtype=np.float32)
        for t in range(0, n_padded - HOP + 1, HOP):
            buf = np.zeros(512, dtype=np.float32)
            buf[:256] = prev_window[256:]
            buf[256:] = audio[t : t + 256] if t + 256 <= n_padded else audio[t:]
            prev_window = buf

            windowed = buf * self.window
            spec = np.fft.rfft(windowed, n=512)  # → 257 bins
            mix = np.zeros((1, 257, 1, 2), dtype=np.float32)
            mix[0, :, 0, 0] = spec.real
            mix[0, :, 0, 1] = spec.imag

            outs = self.sess.run(
                self.output_names,
                {
                    self.input_names[0]: mix,
                    self.input_names[1]: self.conv_cache,
                    self.input_names[2]: self.tra_cache,
                    self.input_names[3]: self.inter_cache,
                },
            )
            enh, conv_o, tra_o, inter_o = outs
            self.conv_cache = conv_o
            self.tra_cache = tra_o
            self.inter_cache = inter_o

            cleaned_spec = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
            cleaned_frame = np.fft.irfft(cleaned_spec, n=512).astype(np.float32) * self.window
            out[t : t + 512] += cleaned_frame

        return out[:n].astype(np.float32)


# ── NSNet2 denoiser (true 320-pt FFT) ───────────────────────────────────────
class NSNet2Denoiser:
    """NSNet2 baseline. 320-sample (20 ms) hop, 161 magnitude bins.
    No state cache — GRU resets per call; we feed each 20 ms frame standalone.
    """

    def __init__(self, model_path: str):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.window = np.sqrt(np.hanning(320)).astype(np.float32)

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        n_orig = len(audio)
        # Pad to whole 320-sample frame (no overlap).
        pad = (320 - n_orig % 320) % 320
        if pad:
            audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])

        out = np.zeros_like(audio)
        for t in range(0, len(audio) - 320 + 1, 320):
            frame = audio[t : t + 320] * self.window
            spec = np.fft.rfft(frame, n=320)  # 161 bins
            mag = np.abs(spec).astype(np.float32)
            mag_in = mag.reshape(1, 1, 161)

            mask = self.sess.run(None, {"input": mag_in})[0]  # [1,1,161]
            mask = mask.reshape(161)

            cleaned_spec = spec * mask
            cleaned = np.fft.irfft(cleaned_spec, n=320).astype(np.float32) * self.window
            out[t : t + 320] += cleaned

        return out[:n_orig].astype(np.float32)


# ── Silero VAD wrapper ──────────────────────────────────────────────────────
class SileroVAD:
    def __init__(self, model_path: str):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        # Silero v5 takes [1, 512] input (32 ms at 16 kHz) + state + sr.
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def edges(self, audio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return per-frame voice mask (bool array, length = len(audio)//512)."""
        self.reset()
        n_frames = len(audio) // 512
        scores = np.zeros(n_frames, dtype=np.float32)
        sr_t = np.array([16000], dtype=np.int64)
        for i in range(n_frames):
            chunk = audio[i * 512 : (i + 1) * 512].astype(np.float32).reshape(1, 512)
            outs = self.sess.run(
                None,
                {"input": chunk, "state": self.state, "sr": sr_t},
            )
            scores[i] = float(np.asarray(outs[0]).ravel()[0])
            self.state = outs[1]
        return scores >= threshold


# ── Metrics ─────────────────────────────────────────────────────────────────
def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 1.0


def precision_recall(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return p, r


def snr_db(clean: np.ndarray, mix: np.ndarray) -> float:
    noise = mix - clean
    p_c = (clean ** 2).mean()
    p_n = (noise ** 2).mean()
    if p_n <= 1e-12 or p_c <= 1e-12:
        return 99.0
    return 10 * float(np.log10(p_c / p_n))


def try_pesq(clean: np.ndarray, candidate: np.ndarray) -> Optional[float]:
    """If `pesq` Python package is available, compute wideband PESQ."""
    try:
        from pesq import pesq as pesq_score  # type: ignore
    except ImportError:
        return None
    try:
        return float(pesq_score(SR, clean, candidate, "wb"))
    except Exception:
        return None


# ── Main A/B loop ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-dir", required=True)
    ap.add_argument("--noise-dir", required=True)
    ap.add_argument("--vad-model", required=True)
    ap.add_argument("--gtcrn-model", required=True)
    ap.add_argument("--nsnet2-model", required=True)
    ap.add_argument("--out", default="quality_results.csv")
    ap.add_argument("--snr-db", type=float, default=5.0,
                    help="target SNR for synthetic mix (default 5 dB = noisy)")
    ap.add_argument("--num-pairs", type=int, default=20,
                    help="number of (speech, noise) pairs to sample")
    ap.add_argument("--max-len-s", type=float, default=10.0,
                    help="trim each speech clip to this many seconds")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    speech_files = sorted(glob.glob(os.path.join(args.speech_dir, "**/*.flac"), recursive=True))
    if not speech_files:
        speech_files = sorted(glob.glob(os.path.join(args.speech_dir, "**/*.wav"), recursive=True))
    noise_files = sorted(glob.glob(os.path.join(args.noise_dir, "*.wav")))
    if not speech_files or not noise_files:
        print(f"ERROR: no audio. speech={len(speech_files)} noise={len(noise_files)}", file=sys.stderr)
        sys.exit(2)

    print(f"speech files: {len(speech_files)}  noise files: {len(noise_files)}")
    rng.shuffle(speech_files)
    rng.shuffle(noise_files)
    speech_files = speech_files[: args.num_pairs]

    print("loading models...")
    t0 = time.time()
    vad = SileroVAD(args.vad_model)
    gtcrn = GTCRNDenoiser(args.gtcrn_model)
    nsnet2 = NSNet2Denoiser(args.nsnet2_model)
    print(f"  loaded in {time.time() - t0:.1f}s")

    rows = []
    csv_fields = [
        "pair", "speech_file", "noise_file", "snr_in_db",
        "snr_gtcrn_db", "snr_nsnet2_db",
        "jacc_noisy", "jacc_gtcrn", "jacc_nsnet2",
        "prec_noisy", "rec_noisy",
        "prec_gtcrn", "rec_gtcrn",
        "prec_nsnet2", "rec_nsnet2",
        "pesq_noisy", "pesq_gtcrn", "pesq_nsnet2",
        "n_frames",
    ]

    for pair_idx, sp_path in enumerate(speech_files):
        try:
            clean = load_wav_16k_mono(sp_path)
        except Exception as e:
            print(f"  skip {sp_path}: {e}", file=sys.stderr)
            continue
        if len(clean) < SR:
            continue
        clean = clean[: int(args.max_len_s * SR)]

        noise_path = noise_files[pair_idx % len(noise_files)]
        try:
            noise = load_wav_16k_mono(noise_path)
        except Exception as e:
            print(f"  skip noise {noise_path}: {e}", file=sys.stderr)
            continue

        # Normalize clean to ~-25 dBFS to avoid scale sensitivity in PESQ.
        peak = float(np.abs(clean).max())
        if peak > 0:
            clean = clean / peak * 0.5

        noisy = mix_at_snr(clean, noise, args.snr_db)
        snr_in = snr_db(clean, noisy)

        # Ground-truth VAD on clean.
        gt = vad.edges(clean)
        # Candidates.
        m_noisy = vad.edges(noisy)
        cleaned_gtcrn = gtcrn.denoise(noisy.copy())
        m_gtcrn = vad.edges(cleaned_gtcrn)
        cleaned_nsnet2 = nsnet2.denoise(noisy.copy())
        m_nsnet2 = vad.edges(cleaned_nsnet2)

        # Trim to common length.
        n = min(len(gt), len(m_noisy), len(m_gtcrn), len(m_nsnet2))
        gt, m_noisy, m_gtcrn, m_nsnet2 = gt[:n], m_noisy[:n], m_gtcrn[:n], m_nsnet2[:n]

        snr_g = snr_db(clean, cleaned_gtcrn)
        snr_n = snr_db(clean, cleaned_nsnet2)

        j_no = jaccard(m_noisy, gt)
        j_g  = jaccard(m_gtcrn, gt)
        j_n  = jaccard(m_nsnet2, gt)
        p_no, r_no = precision_recall(m_noisy, gt)
        p_g,  r_g  = precision_recall(m_gtcrn, gt)
        p_n,  r_n  = precision_recall(m_nsnet2, gt)

        pesq_no = try_pesq(clean, noisy)
        pesq_g  = try_pesq(clean, cleaned_gtcrn)
        pesq_n  = try_pesq(clean, cleaned_nsnet2)

        row = {
            "pair": pair_idx,
            "speech_file": Path(sp_path).name,
            "noise_file": Path(noise_path).name,
            "snr_in_db": round(snr_in, 2),
            "snr_gtcrn_db": round(snr_g, 2),
            "snr_nsnet2_db": round(snr_n, 2),
            "jacc_noisy": round(j_no, 4),
            "jacc_gtcrn": round(j_g, 4),
            "jacc_nsnet2": round(j_n, 4),
            "prec_noisy": round(p_no, 4),
            "rec_noisy": round(r_no, 4),
            "prec_gtcrn": round(p_g, 4),
            "rec_gtcrn": round(r_g, 4),
            "prec_nsnet2": round(p_n, 4),
            "rec_nsnet2": round(r_n, 4),
            "pesq_noisy": round(pesq_no, 3) if pesq_no is not None else "",
            "pesq_gtcrn": round(pesq_g, 3) if pesq_g is not None else "",
            "pesq_nsnet2": round(pesq_n, 3) if pesq_n is not None else "",
            "n_frames": n,
        }
        rows.append(row)
        print(f"[{pair_idx+1}/{len(speech_files)}] {Path(sp_path).name[:30]:30s} × "
              f"{Path(noise_path).stem[:18]:18s} "
              f"jacc no={j_no:.3f} gtcrn={j_g:.3f} nsnet2={j_n:.3f}  "
              f"snrΔ g={snr_g - snr_in:+.1f} n={snr_n - snr_in:+.1f}")

    if not rows:
        print("no results", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.out} ({len(rows)} rows)")

    # Summary.
    def mean(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n══ SUMMARY ══")
    print(f"  pairs: {len(rows)}, target SNR_in: {args.snr_db} dB")
    print(f"  Jaccard vs ground-truth VAD on clean speech:")
    print(f"    noisy (no denoise):   {mean('jacc_noisy'):.4f}")
    print(f"    GTCRN denoised:        {mean('jacc_gtcrn'):.4f}")
    print(f"    NSNet2 denoised:       {mean('jacc_nsnet2'):.4f}")
    print(f"  SNR improvement (dB):")
    print(f"    GTCRN:  {mean('snr_gtcrn_db') - mean('snr_in_db'):+.2f}")
    print(f"    NSNet2: {mean('snr_nsnet2_db') - mean('snr_in_db'):+.2f}")
    pesq_mean_g = [r["pesq_gtcrn"] for r in rows if isinstance(r.get("pesq_gtcrn"), (int, float))]
    pesq_mean_n = [r["pesq_nsnet2"] for r in rows if isinstance(r.get("pesq_nsnet2"), (int, float))]
    if pesq_mean_g and pesq_mean_n:
        print(f"  PESQ (wideband):")
        print(f"    GTCRN:  {sum(pesq_mean_g)/len(pesq_mean_g):.3f}")
        print(f"    NSNet2: {sum(pesq_mean_n)/len(pesq_mean_n):.3f}")
    print(f"\n  ship verdict:")
    if mean("jacc_nsnet2") >= 0.95 * mean("jacc_gtcrn"):
        print(f"    ✓ NSNet2 within 5% of GTCRN on Jaccard — quality OK to flip default")
    elif mean("jacc_nsnet2") >= 0.90 * mean("jacc_gtcrn"):
        print(f"    ⚠ NSNet2 within 10% — borderline, consider DFN3")
    else:
        print(f"    ✗ NSNet2 regresses >10% — keep GTCRN default, escalate to DFN3")


if __name__ == "__main__":
    main()
