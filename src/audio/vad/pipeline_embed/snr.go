package pipeline_embed

import "math"

// SNRDetector estimates per-frame signal-to-noise ratio using
// minima-controlled recursive averaging (MCRA-style noise floor tracking) on
// short-term energy.
//
// Algorithm:
//
//   1. Compute frame RMS energy in dBFS.
//   2. Maintain a rolling minimum over a search window (default 1.0 s ≈ 31
//      frames at 32 ms cadence) — that is the noise floor estimate.
//   3. Update floor with exponential decay so it slowly adapts upward when
//      noise rises after a quiet stretch.
//   4. SNR = current_rms_dB - noise_floor_dB.
//
// Cost per frame: one pass over 512 samples for sum-of-squares (~5-10 µs on
// modern hardware), no allocations, no FFT, no cgo.
//
// Tuning:
//
//   - WindowFrames: longer window adapts slower but is more stable. 30 frames
//     (~1 s) is the WebRTC NS default.
//   - DecayPerFrame: how fast the floor rises when no quiet frame is seen.
//     0.05 dB/frame ≈ 1.5 dB/s upward drift.
//   - InitFloorDB: starting estimate before first window fills. -60 dBFS is
//     "very quiet" so the first few frames will read high SNR until the
//     floor catches up. Tune higher (e.g. -40) if your input is known to be
//     consistently noisy.
//
// Threshold guidance:
//
//   - **Recommended production default: 6 dB.** Empirically derived from
//     bench/QUALITY_REPORT.md (LibriSpeech + ESC-50 quality A/B): GTCRN
//     denoise *helps* VAD-edge agreement at 5 dB SNR (+13 % Jaccard) but
//     *hurts* at 10 dB SNR (-18 % Jaccard) due to phase / spectral
//     artifacts that confuse Silero. A 6 dB gate skips denoise on
//     moderately-noisy frames where it would harm and runs it only when
//     the input is genuinely below the model's training distribution.
//   - 12-15 dB was the original cost-only threshold (saved compute by
//     skipping clean frames). It's correct for *throughput* but wrong for
//     *quality* — at 12 dB many frames still get denoised that would have
//     been better untouched. Use this value only on synthetic fixtures
//     or when quality is not the goal.
//   - Below 0 dB: gating disabled (denoise every frame). Legacy.
//   - Above 20 dB: signal is clean, gate skips denoise unconditionally.
type SNRDetector struct {
	windowFrames  int
	decayPerFrame float64

	// Rolling minimum of recent dBFS values. Implemented as a fixed-size
	// circular buffer; we recompute the min each Update because window is
	// small (~30) and the cost is negligible.
	dbWindow []float64
	cursor   int
	filled   int

	// Last computed values, useful for logs/metrics without re-computing.
	LastRMSDB    float64
	LastFloorDB  float64
	LastSNRDB    float64
}

// SNRConfig defaults tuned for 16 kHz / 32 ms / int16 PCM.
type SNRConfig struct {
	WindowFrames  int     // default 30 (~1 s)
	DecayPerFrame float64 // default 0.05 dB/frame
	InitFloorDB   float64 // default -60 dBFS
}

// NewSNRDetector builds a fresh detector. Zero-value cfg uses defaults above.
func NewSNRDetector(cfg SNRConfig) *SNRDetector {
	if cfg.WindowFrames <= 0 {
		cfg.WindowFrames = 30
	}
	if cfg.DecayPerFrame <= 0 {
		cfg.DecayPerFrame = 0.05
	}
	if cfg.InitFloorDB == 0 {
		cfg.InitFloorDB = -60
	}
	w := make([]float64, cfg.WindowFrames)
	for i := range w {
		w[i] = cfg.InitFloorDB
	}
	return &SNRDetector{
		windowFrames:  cfg.WindowFrames,
		decayPerFrame: cfg.DecayPerFrame,
		dbWindow:      w,
		LastFloorDB:   cfg.InitFloorDB,
	}
}

// Update consumes one frame of int16 PCM and returns the current estimated
// SNR in dB. Caller decides whether to gate downstream processing.
func (s *SNRDetector) Update(frame []int16) float64 {
	rmsDB := rmsDBFS(frame)

	// Stuff into circular buffer.
	s.dbWindow[s.cursor] = rmsDB
	s.cursor = (s.cursor + 1) % s.windowFrames
	if s.filled < s.windowFrames {
		s.filled++
	}

	// Floor = min of window. Once the window is filled, cap upward drift at
	// the configured decay rate so a sudden loud burst doesn't yank the floor
	// up. During warm-up we accept the window-min directly so cold-start
	// converges quickly to the actual noise level.
	minDB := s.dbWindow[0]
	for i := 1; i < s.filled; i++ {
		if s.dbWindow[i] < minDB {
			minDB = s.dbWindow[i]
		}
	}
	floor := minDB
	if s.filled >= s.windowFrames && floor > s.LastFloorDB+s.decayPerFrame {
		// Cap upward drift only after warm-up.
		floor = s.LastFloorDB + s.decayPerFrame
	}
	s.LastFloorDB = floor

	snr := rmsDB - floor
	s.LastRMSDB = rmsDB
	s.LastSNRDB = snr
	return snr
}

// ShouldDenoise is the gating decision: true ⇒ run denoiser on this frame.
// thresholdDB ≤ 0 disables gating (always denoise).
func (s *SNRDetector) ShouldDenoise(thresholdDB float64) bool {
	if thresholdDB <= 0 {
		return true
	}
	return s.LastSNRDB < thresholdDB
}

// rmsDBFS returns frame energy in dBFS (0 dB = full-scale int16). Silent
// frames (all zero) return -120 to avoid -Inf.
func rmsDBFS(frame []int16) float64 {
	if len(frame) == 0 {
		return -120
	}
	var sumSq float64
	for _, s := range frame {
		v := float64(s)
		sumSq += v * v
	}
	mean := sumSq / float64(len(frame))
	if mean <= 0 {
		return -120
	}
	rms := math.Sqrt(mean)
	// 32768 = int16 full scale.
	dbfs := 20 * math.Log10(rms/32768.0)
	if dbfs < -120 {
		dbfs = -120
	}
	return dbfs
}
