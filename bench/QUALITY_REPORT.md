# Real-speech quality A/B — NSNet2 vs GTCRN vs no-denoise

**Question**: Does the cgo-embed pipeline's denoiser actually improve VAD-edge quality on real speech with realistic noise — and does NSNet2 (PR-2 candidate) preserve that improvement?

**Answer**: **No**. Both denoisers regress VAD agreement at typical telephony SNR (10 dB). NSNet2 is catastrophically worse than GTCRN. **The right move is keeping GTCRN as opt-in default and tuning the SNR gate to skip denoise on most frames**, not flipping the denoiser.

This report is the negative result that prevents shipping NSNet2 to production without quality regression. PR-2 (#39) stays opt-in and explicitly cost-only.

---

## Method

`bench/quality_test.py` — pure Python, scipy + onnxruntime, no Go pipeline involvement. Measures **what would happen if Strawgo's pipeline produced cleaned audio and fed it to Silero VAD**.

Per pair `(speech, noise, snr_db)`:

1. Load clean speech (LibriSpeech `test-clean`, 16 kHz mono).
2. Load noise (ESC-50 single-shot environmental sounds).
3. Mix at target SNR — produces `noisy = clean + scale·noise`.
4. Run Silero VAD on **clean** signal → ground-truth voice mask (per 32 ms frame).
5. Run Silero VAD on **noisy** signal directly → "no-denoise" candidate.
6. Run GTCRN denoise on noisy → cleaned PCM → Silero → "GTCRN" candidate.
7. Run NSNet2 denoise on noisy → cleaned PCM → Silero → "NSNet2" candidate.
8. Compute Jaccard, precision, recall, PESQ wideband, SNR delta.

Sources:
- LibriSpeech test-clean: 2 620 .flac files, openslr.org/resources/12
- ESC-50: 2 000 .wav files (50 categories of environmental sound), karoldvl/ESC-50 GitHub
- All audio resampled to 16 kHz mono float32 in [-1, 1].

NSNet2 here uses a **true 320-pt FFT** via numpy (the Go wrapper's zero-pad-to-512 approximation isn't used). This gives NSNet2 its best shot — bin frequencies match what the model was trained on.

GTCRN runs streaming with state caches (the way it was designed).

Both denoisers apply the predicted gain mask to the original complex spectrum and iSTFT back to PCM. **This is the production-correct flow**, not the cost-only path used in `pipeline_embed`.

## Results — 5 dB SNR (heavy noise)

30 random LibriSpeech × ESC-50 pairs.

| pipeline | mean Jaccard | mean PESQ-WB | mean SNR Δ (dB) |
|---|---:|---:|---:|
| noisy baseline (no denoise) | 0.539 | — | 0 |
| **GTCRN denoised** | **0.608** | **1.817** | −7.72 |
| NSNet2 denoised | 0.402 | 1.147 | −3.71 |

### Verdict at 5 dB
- GTCRN improves Jaccard **+12.9%** over no-denoise. PESQ 1.82 (poor but better than nothing — the input is genuinely noisy).
- **NSNet2 makes Jaccard worse than no-denoise (−25%)**. PESQ 1.15 — close to the floor.
- SNR delta is misleading (both negative due to phase artifacts in iSTFT vs the original clean signal); the operationally-meaningful metric is Jaccard on the downstream VAD.

## Results — 10 dB SNR (typical telephony)

30 random pairs, different seed.

| pipeline | mean Jaccard | mean PESQ-WB | mean SNR Δ (dB) |
|---|---:|---:|---:|
| **noisy baseline (no denoise)** | **0.692** | — | 0 |
| GTCRN denoised | 0.570 | 2.300 | −12.93 |
| NSNet2 denoised | 0.374 | 1.164 | −8.63 |

### Verdict at 10 dB — the surprise
- **GTCRN is now WORSE than no-denoise** (Jaccard 0.570 vs 0.692, **−17.6%**).
- **NSNet2 catastrophically worse** (0.374 vs 0.692, **−46%**).
- PESQ tells a different story (GTCRN 2.30, "fair" speech quality), but VAD doesn't care about audible quality — it cares about whether the spectral envelope matches what Silero learned to detect speech in.
- Denoiser-induced phase distortion + spectral smoothing actively confuses Silero's VAD edges when the input was already moderately clean.

## What's actually happening

These denoisers were trained on heavily-noisy mixtures (DNS-Challenge corpus typically 0-15 dB SNR with reverb). Training distribution = noisy. At inference time on cleaner inputs they still apply gain modifications, smearing the spectral envelope.

Silero VAD is a small CRNN trained on clean speech-detection. It learned a tight mapping from "clean spectral patterns → voice probability". A denoiser that subtly modifies the spectrum (even in helpful ways audibly) shifts the input out of Silero's training distribution.

**Result: at moderate SNR, the denoiser harms more than helps. At low SNR it (sometimes) helps.**

## Real-world implication for Strawgo

### Production SNR distribution unknown

We don't have telemetry on real-call SNR. Telephony research suggests median ~10-15 dB on a typical call (mobile mic + ambient noise + codec degradation). If that's true:
- GTCRN-everywhere = active VAD regression
- NSNet2-everywhere = catastrophic VAD regression
- **No-denoise (rely on Silero alone)** = best Jaccard at typical SNR

### SNR gate is the right idea, threshold needs tuning

PR-1's `SNRDetector` skips denoise when SNR ≥ threshold. At threshold = 12 dB:
- Frames at SNR ~ 5-12 dB: gate keeps denoise ON → at 5 dB this helps, at 10-12 dB this HURTS
- Frames at SNR > 12 dB: gate skips denoise → no harm done

**Recommendation: drop SNR threshold to ~6-8 dB.** Only denoise frames that are genuinely noisy (where GTCRN's training distribution applies). Leave moderate-noise frames alone. Need real-call data to confirm the optimal threshold.

### NSNet2 is dead for this stack

Two independent failures:
1. **No state cache** in the available ONNX export. GRU resets every 20 ms call, can't track stationary noise floor.
2. **Quality regression at all tested SNRs.** Worse than no-denoise at 5 dB. Catastrophic at 10 dB.

NSNet2 stays opt-in (PR-2) **for benchmark / cost-comparison purposes only**. Production deploy = forbidden until either:
- A streaming NSNet2 export with state cache becomes available, OR
- We retrain NSNet2 on telephony-band-limited audio matching our SNR distribution.

Neither is a 1-week task. Drop NSNet2.

### DFN3 as quality fallback — still untested

DFN3 (PR-5 placeholder) has higher published PESQ (~3.4 vs GTCRN 3.0) and is closer to a 16 kHz streaming design. **Worth testing through this same harness** before any production swap. ~2 day work to wire the 3-graph chain through Python with iSTFT.

### What to ship now (minimal change)

1. **Keep GTCRN as the only opt-in denoiser** in `pipeline_embed`.
2. **Tune SNR gate threshold to 6 dB** (down from 12). Only denoise truly noisy frames.
3. **Default `pipeline_embed.SNRConfig` recommends gate ON, threshold = 6 dB** — production guidance.
4. **NSNet2 stays opt-in research-only**. Add comment in `nsnet2_denoiser.go` package doc citing this report.
5. **Run real-call SNR distribution telemetry** — only way to know if 6 dB threshold is right for production.

## Per-pair raw results

CSVs: `bench/quality_results_30pair_5db.csv`, `bench/quality_results_30pair_10db.csv`.

Logs: `bench/q5db.log`, `bench/q10db.log`.

## Reproduce

```sh
# On VM with bench-venv:
$BENCH_VENV/bin/python bench/quality_test.py \
  --speech-dir   $DATA_DIR/LibriSpeech/test-clean \
  --noise-dir    $DATA_DIR/ESC-50-master/audio \
  --vad-model    $MODELS/silero_vad.onnx \
  --gtcrn-model  $MODELS/gtcrn_simple.onnx \
  --nsnet2-model $MODELS/nsnet2-20ms.onnx \
  --out          quality_results.csv \
  --snr-db 5 \
  --num-pairs 30
```

Pre-reqs: `pip install soundfile pesq onnxruntime scipy numpy`

LibriSpeech test-clean: `https://www.openslr.org/resources/12/test-clean.tar.gz` (340 MB)

ESC-50: `https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip` (645 MB)

## Caveats

1. **iSTFT phase**: this Python harness does basic Hann-windowed iSTFT with overlap-add. Production Go integration would need the same, plus careful gain compensation. Phase artifacts likely contribute to the SNR-Δ being negative even where Jaccard improves.
2. **VAD reference choice**: Silero is the de-facto VAD for cgo-embed pipeline; results would differ with WebRTC VAD or Porcupine. Silero is a tight learner — easier to break with input distribution shift.
3. **Synthetic noise**: ESC-50 is single-source environmental noise (clean recording). Real telephony has codec artifacts, packet loss, line hiss — different distribution. Numbers should hold qualitatively but may shift quantitatively.
4. **PESQ scale**: wideband PESQ in [1.0, 4.5]. 1.5-2.0 = poor, 2.0-3.0 = fair, 3.0-3.5 = good. We're measuring at the bottom of the scale because input mixes are noisier than typical PESQ test conditions.
5. **No real-call data**: this whole analysis is on synthetic mixes. **Production SNR distribution telemetry is the missing input** before the SNR-gate threshold can be tuned with confidence.

## Status

- **NSNet2 default-flip rejected** (PR-2 stays opt-in)
- **GTCRN tuning in PR-3** (drop SNR threshold to 6 dB) — 1 line change
- **DFN3 wiring queued** (PR-5) — only if real-call telemetry shows median SNR < 8 dB AND DFN3 quality holds in this same harness
- **Production telemetry asked of operator** — before tuning anything further
