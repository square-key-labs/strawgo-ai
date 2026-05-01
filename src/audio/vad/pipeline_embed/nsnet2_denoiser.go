// nsnet2_denoiser — Microsoft NSNet2 baseline (DNS-Challenge) wired as an
// alternative to GTCRN.
//
// Model:
//
//	source     : niobures/NSNet2-ONNX (mirror of microsoft/DNS-Challenge baseline)
//	graph      : 1 ONNX, opset 11
//	I/O        : input  f32 [1, S, 161]  (STFT magnitude bins)
//	             output f32 [1, S, 161]  (gain mask)
//	rate       : 16 kHz, 20 ms hop = 320 samples
//	state      : NONE — GRU hidden state is internal and resets every call.
//
// Streaming on Strawgo's 32 ms / 512-sample frames:
//
//	NSNet2's natural hop is 20 ms / 320 samples. We accumulate audio in a
//	320-sample ring; each time it fills we run one inference. Strawgo's
//	32 ms frame fills the ring 1.6× on average, so ProcessFrame runs the
//	model 1 or 2 times per call.
//
// FFT size: 320 is not a power of two. This wrapper zero-pads to 512 and
// reuses the existing fft512 from mel.go, then takes the first 161 bins.
//
// **THIS IS A BENCHMARK APPROXIMATION, NOT A PRODUCTION-VALID DENOISER.**
//
// The first 161 bins of a 512-pt FFT span 0–~5 kHz at 31.25 Hz/bin. A true
// 320-pt FFT spans 0–8 kHz at 50 Hz/bin (Nyquist for 16 kHz audio). NSNet2
// was trained on the latter — voiced-speech harmonics fall in different
// bins than the model expects, so the gain mask **does not predict the
// real model's output** even though the per-call ORT cost is representative.
//
// Use this wrapper for:
//   - per-frame ORT graph cost benchmarking (Cascade Lake / M-series numbers)
//   - integration smoke testing
//   - throughput sweeps where denoise output is discarded downstream
//
// Do NOT use this wrapper for:
//   - shipping NSNet2 to production
//   - quality A/B (PESQ, MOS, VAD-edge agreement)
//   - any test that consumes the gain mask or reconstructed audio
//
// Production NSNet2 integration MUST swap fft512 for a true 320-pt FFT
// (mixed-radix; gonum/dsp/fourier handles this, or cgo'd FFTW). PR-3
// blocking item.
//
// Cost note (microbenched, no STFT framework overhead):
//
//	M3 Pro:        162 µs per inference → ~260 µs per 32 ms frame
//	Cascade Lake:  931 µs per inference → ~1.49 ms per 32 ms frame
//
// vs GTCRN's ~3.22 ms per 32 ms frame on the same VM. ~2× faster.
//
// Quality caveat: this NSNet2 ONNX export has no exposed state cache, so the
// GRU resets every inference. On per-frame streaming the gain mask quality is
// degraded vs windowed (multi-frame) inference. For raw cost benchmarking
// that doesn't matter; for production deploy you must measure VAD-edge
// agreement vs GTCRN/no-denoise on real telephony PCM before locking it in.

package pipeline_embed

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	// NSNet2 native: 320-pt STFT at 16 kHz → 161 positive-frequency mag bins.
	nsnet2NFFT     = 320
	nsnet2NFreq    = nsnet2NFFT/2 + 1 // 161
	// We reuse fft512 from mel.go; pad 320-sample window with zeros to 512.
	nsnet2FFTPad   = 512
)

var nsnet2Shared struct {
	mu       sync.Mutex
	session  *ort.DynamicAdvancedSession
	refcount int64
	window   []float32 // sqrt-Hann, length 320
}

// NSNet2Config initializes the shared NSNet2 ORT session. Must be called
// before the first NewNSNet2Denoiser. SharedLibraryPath should be a
// libonnxruntime path; if empty, ORT must already be initialized.
type NSNet2Config struct {
	ModelPath         string
	SharedLibraryPath string
	IntraOpNumThreads int
	InterOpNumThreads int
}

// InitNSNet2 brings up the shared NSNet2 session. Idempotent.
func InitNSNet2(cfg NSNet2Config) error {
	nsnet2Shared.mu.Lock()
	defer nsnet2Shared.mu.Unlock()
	if nsnet2Shared.session != nil {
		return nil
	}
	if cfg.ModelPath == "" {
		return errors.New("nsnet2: ModelPath required")
	}
	if cfg.SharedLibraryPath != "" {
		ort.SetSharedLibraryPath(cfg.SharedLibraryPath)
	}
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return fmt.Errorf("nsnet2: InitializeEnvironment: %w", err)
		}
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("nsnet2: NewSessionOptions: %w", err)
	}
	defer opts.Destroy()
	if cfg.IntraOpNumThreads > 0 {
		_ = opts.SetIntraOpNumThreads(cfg.IntraOpNumThreads)
	}
	if cfg.InterOpNumThreads > 0 {
		_ = opts.SetInterOpNumThreads(cfg.InterOpNumThreads)
	}

	sess, err := ort.NewDynamicAdvancedSession(
		cfg.ModelPath,
		[]string{"input"},
		[]string{"output"},
		opts,
	)
	if err != nil {
		return fmt.Errorf("nsnet2: NewDynamicAdvancedSession: %w", err)
	}
	nsnet2Shared.session = sess
	nsnet2Shared.window = sqrtHannWindow(nsnet2NFFT)
	return nil
}

// ShutdownNSNet2 destroys the shared session if no streams hold it.
func ShutdownNSNet2() error {
	nsnet2Shared.mu.Lock()
	defer nsnet2Shared.mu.Unlock()
	if nsnet2Shared.session == nil {
		return nil
	}
	if atomic.LoadInt64(&nsnet2Shared.refcount) > 0 {
		return errors.New("nsnet2: streams still active")
	}
	nsnet2Shared.session.Destroy()
	nsnet2Shared.session = nil
	return nil
}

// NSNet2Denoiser is one stream's NSNet2 state. Implements DenoiserImpl
// (defined in pipeline.go).
type NSNet2Denoiser struct {
	mu sync.Mutex

	// 320-sample audio ring; we accumulate until we have a full STFT frame.
	ringBuf []float32
	filled  int

	// Pre-allocated tensors. Input/output shape [1, 1, 161].
	magBuf  []float32
	gainBuf []float32
	magT    *ort.Tensor[float32]
	gainT   *ort.Tensor[float32]

	closed atomic.Bool
}

// NewNSNet2Denoiser creates a stream denoiser bound to the shared session.
// Reads the session pointer under nsnet2Shared.mu so init publication and
// shutdown teardown are observed consistently.
func NewNSNet2Denoiser() (*NSNet2Denoiser, error) {
	nsnet2Shared.mu.Lock()
	if nsnet2Shared.session == nil {
		nsnet2Shared.mu.Unlock()
		return nil, errors.New("pipeline_embed: InitNSNet2() not called")
	}
	atomic.AddInt64(&nsnet2Shared.refcount, 1)
	nsnet2Shared.mu.Unlock()

	d := &NSNet2Denoiser{
		ringBuf: make([]float32, nsnet2NFFT),
		magBuf:  make([]float32, nsnet2NFreq),
		gainBuf: make([]float32, nsnet2NFreq),
	}

	magT, err := ort.NewTensor(ort.NewShape(1, 1, int64(nsnet2NFreq)), d.magBuf)
	if err != nil {
		atomic.AddInt64(&nsnet2Shared.refcount, -1)
		return nil, fmt.Errorf("nsnet2: alloc input: %w", err)
	}
	d.magT = magT

	gainT, err := ort.NewTensor(ort.NewShape(1, 1, int64(nsnet2NFreq)), d.gainBuf)
	if err != nil {
		_ = magT.Destroy()
		atomic.AddInt64(&nsnet2Shared.refcount, -1)
		return nil, fmt.Errorf("nsnet2: alloc output: %w", err)
	}
	d.gainT = gainT

	return d, nil
}

// ProcessFrame consumes one 512-sample / 32 ms frame at 16 kHz. Internally
// runs NSNet2 inference once per 320 accumulated samples (i.e., 1-2 times per
// ProcessFrame call, averaging 1.6×).
//
// Like the GTCRN wrapper, this discards the gain mask for benchmarking —
// Silero VAD downstream gets the original int16 PCM. To use NSNet2 in
// production you'd reconstruct the cleaned audio (apply mask × complex
// spectrum, iSTFT back) and pass that downstream. That adds ~30 µs of FFT
// bookkeeping per frame.
func (d *NSNet2Denoiser) ProcessFrame(frame []int16) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed.Load() {
		return errors.New("nsnet2: closed")
	}
	if len(frame) != 512 {
		return fmt.Errorf("nsnet2: frame size %d != 512", len(frame))
	}

	// Append all 512 samples to ring; run model whenever ≥ 320 accumulated.
	for i := range len(frame) {
		d.ringBuf[d.filled] = float32(frame[i]) / 32768.0
		d.filled++
		if d.filled == nsnet2NFFT {
			if err := d.runOneSTFT(); err != nil {
				return err
			}
			// Reset ring (no overlap, hop = window).
			d.filled = 0
		}
	}
	return nil
}

// runOneSTFT computes a 320-pt windowed magnitude spectrum from ringBuf
// (zero-padded to 512 for power-of-2 FFT) and runs the NSNet2 graph. Output
// gain mask is held in d.gainBuf; production code would multiply it into the
// complex spectrum and iSTFT back. We skip iSTFT here for cost parity with
// GTCRN's cost-only path.
func (d *NSNet2Denoiser) runOneSTFT() error {
	// BENCHMARK APPROXIMATION — see package doc.
	//
	// Build windowed input: 320 samples × Hann; pad to 512 with zeros, run
	// the existing 512-pt FFT, then take the first 161 bins. This is NOT a
	// substitute for a true 320-pt FFT — the bin frequencies don't match
	// what NSNet2 was trained on. Cost is representative; gain mask is not.
	scratch := make([]complex128, nsnet2FFTPad)
	for i := range nsnet2NFFT {
		scratch[i] = complex(float64(d.ringBuf[i]*nsnet2Shared.window[i]), 0)
	}
	// scratch[320:512] already zero. Run shared 512-pt FFT.
	fft512(scratch)

	// Take first 161 bins; magnitude.
	for f := range nsnet2NFreq {
		c := scratch[f]
		mag := math.Sqrt(real(c)*real(c) + imag(c)*imag(c))
		d.magBuf[f] = float32(mag)
	}

	if err := nsnet2Shared.session.Run(
		[]ort.Value{d.magT},
		[]ort.Value{d.gainT},
	); err != nil {
		return fmt.Errorf("nsnet2: ORT Run: %w", err)
	}
	// gainBuf holds the per-bin gain mask. Discard for cost-only bench;
	// production would multiply mask × original complex bins and iSTFT.
	return nil
}

// Cleanup releases per-stream tensors.
func (d *NSNet2Denoiser) Cleanup() error {
	if !d.closed.CompareAndSwap(false, true) {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.magT != nil {
		_ = d.magT.Destroy()
	}
	if d.gainT != nil {
		_ = d.gainT.Destroy()
	}
	atomic.AddInt64(&nsnet2Shared.refcount, -1)
	return nil
}
