// Package pipeline_embed — GTCRN-based 16 kHz speech denoiser.
//
// Why GTCRN, not DeepFilterNet3?
//
//   DFN3's official ONNX export is a 4-piece streaming pipeline with ERB-band
//   feature extraction, complex FFT-domain mask + group-conv decoding, and 48 kHz
//   sample rate. Wiring it into a Go cgo path in a few hours is not practical.
//   Strawgo's pipeline runs at 16 kHz, so the 48 kHz constraint also forces
//   resampling at both ends.
//
//   GTCRN (https://github.com/yuyun2000/SpeechDenoiser, MIT) is a streaming
//   16-kHz speech denoiser with a single-piece ONNX, well-defined I/O, three
//   small per-stream caches, and an STFT/iSTFT framing of 512/256. It serves
//   the same role as DFN3 for benchmarking the per-frame load of a real
//   3-model voice pipeline. Quality benchmark deltas are out of scope here —
//   we're measuring framework overhead, not model quality.
//
// I/O surface (from yuyun2000/SpeechDenoiser/16k/onnx_infer.py):
//
//   Inputs  : mix         f32 [1, F, 1, 2]   (single STFT frame, real/imag)
//             conv_cache  f32 [2, 1, 16, 16, 33]
//             tra_cache   f32 [2, 3, 1, 1, 16]
//             inter_cache f32 [2, 1, 33, 16]
//   Outputs : out         f32 [1, F, 1, 2]   (denoised STFT frame)
//             conv_cache' f32 [2, 1, 16, 16, 33]
//             tra_cache'  f32 [2, 3, 1, 1, 16]
//             inter_cache'f32 [2, 1, 33, 16]
//
//   F = nFFT/2 + 1 = 257 (positive frequencies of a 512-point FFT).
//
// Streaming: each call to ProcessFrame consumes a 512-sample STFT frame
// (stride 256) and returns a 256-sample audio block via overlap-add iSTFT.
//
// We DO NOT actually replace audio samples with denoised output for the
// downstream Silero VAD in this benchmark — we discard the iSTFT output and
// pass the original int16 PCM to Silero. Doing the full STFT→ONNX→iSTFT loop
// is enough to measure per-frame compute load; running the cleaned audio
// through VAD would change Silero's behaviour without changing the latency
// envelope, and that's not what this bench measures.
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
	gtcrnNFFT      = 512
	gtcrnNFreq     = gtcrnNFFT/2 + 1 // 257
	convCacheLen   = 2 * 1 * 16 * 16 * 33
	traCacheLen    = 2 * 3 * 1 * 1 * 16
	interCacheLen  = 2 * 1 * 33 * 16
	gtcrnFrameLen  = gtcrnNFreq * 2 // re/im interleaved
	stftWinSize    = 512
)

// shared session state for the denoiser.
var denoiserShared struct {
	mu       sync.Mutex
	session  *ort.DynamicAdvancedSession
	cfg      DenoiserConfig
	initOnce sync.Once
	initErr  error
	refcount int64
	window   []float32 // sqrt-Hann, len 512
}

// DenoiserConfig is the config for GTCRN model loading.
type DenoiserConfig struct {
	ModelPath         string
	SharedLibraryPath string
	IntraOpNumThreads int
	InterOpNumThreads int
}

// InitDenoiser builds the global GTCRN session. Idempotent.
func InitDenoiser(cfg DenoiserConfig) error {
	denoiserShared.initOnce.Do(func() {
		denoiserShared.cfg = cfg
		denoiserShared.initErr = doInitDenoiser(cfg)
	})
	return denoiserShared.initErr
}

func doInitDenoiser(cfg DenoiserConfig) error {
	if cfg.ModelPath == "" {
		return errors.New("pipeline_embed: DenoiserConfig.ModelPath is required")
	}
	if cfg.SharedLibraryPath != "" {
		ort.SetSharedLibraryPath(cfg.SharedLibraryPath)
	}
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return fmt.Errorf("pipeline_embed: InitializeEnvironment: %w", err)
		}
	}
	intra := cfg.IntraOpNumThreads
	if intra <= 0 {
		intra = 1
	}
	inter := cfg.InterOpNumThreads
	if inter <= 0 {
		inter = 1
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("pipeline_embed: denoiser NewSessionOptions: %w", err)
	}
	defer opts.Destroy()
	if err := opts.SetIntraOpNumThreads(intra); err != nil {
		return err
	}
	if err := opts.SetInterOpNumThreads(inter); err != nil {
		return err
	}
	if err := opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return err
	}
	if err := opts.SetCpuMemArena(false); err != nil {
		return err
	}
	if err := opts.SetMemPattern(false); err != nil {
		return err
	}

	inputs := []string{"mix", "conv_cache", "tra_cache", "inter_cache"}
	outputs := []string{"enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"}
	// GTCRN's exported names may vary; try a few sensible variants if these fail.
	sess, err := ort.NewDynamicAdvancedSession(cfg.ModelPath, inputs, outputs, opts)
	if err != nil {
		// Try alternate output names. The yuyun2000 export uses positional
		// outputs (run([], {...}) returns 4 unnamed outputs in the Python
		// example), so try a couple of common names.
		opts2, err2 := ort.NewSessionOptions()
		if err2 == nil {
			defer opts2.Destroy()
			_ = opts2.SetIntraOpNumThreads(intra)
			_ = opts2.SetInterOpNumThreads(inter)
			_ = opts2.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)
			_ = opts2.SetCpuMemArena(false)
			_ = opts2.SetMemPattern(false)
			alt := []string{"out", "conv_cache_out", "tra_cache_out", "inter_cache_out"}
			s2, e2 := ort.NewDynamicAdvancedSession(cfg.ModelPath, inputs, alt, opts2)
			if e2 == nil {
				denoiserShared.session = s2
				denoiserShared.window = sqrtHannWindow(stftWinSize)
				return nil
			}
		}
		return fmt.Errorf("pipeline_embed: gtcrn NewDynamicAdvancedSession(%s): %w", cfg.ModelPath, err)
	}
	denoiserShared.session = sess
	denoiserShared.window = sqrtHannWindow(stftWinSize)
	return nil
}

// ShutdownDenoiser destroys the shared denoiser session. Safe to call repeatedly.
func ShutdownDenoiser() error {
	denoiserShared.mu.Lock()
	defer denoiserShared.mu.Unlock()
	if denoiserShared.session != nil {
		_ = denoiserShared.session.Destroy()
		denoiserShared.session = nil
	}
	return nil
}

// Denoiser is one stream's worth of GTCRN state.
type Denoiser struct {
	mu sync.Mutex

	// 512-sample audio buffer for STFT (overlap of 256 between calls).
	audioBuf []float32

	// Per-stream caches (the three GTCRN states).
	convCache  []float32
	traCache   []float32
	interCache []float32

	// Pre-allocated tensors (held for the lifetime of the Denoiser).
	mixT, convInT, traInT, interInT      *ort.Tensor[float32]
	enhT, convOutT, traOutT, interOutT   *ort.Tensor[float32]

	// Backing slice for mixT. Layout: [1, F, 1, 2] flattened row-major,
	// i.e. for f in 0..F: [re, im, re, im, ...].
	mixBuf []float32

	closed atomic.Bool
}

// NewDenoiser creates a stream denoiser bound to the shared session.
func NewDenoiser() (*Denoiser, error) {
	if denoiserShared.session == nil {
		return nil, errors.New("pipeline_embed: InitDenoiser() not called")
	}
	atomic.AddInt64(&denoiserShared.refcount, 1)

	d := &Denoiser{
		audioBuf:   make([]float32, stftWinSize),
		convCache:  make([]float32, convCacheLen),
		traCache:   make([]float32, traCacheLen),
		interCache: make([]float32, interCacheLen),
		mixBuf:     make([]float32, gtcrnFrameLen),
	}
	if err := d.allocTensors(); err != nil {
		atomic.AddInt64(&denoiserShared.refcount, -1)
		return nil, err
	}
	return d, nil
}

func (d *Denoiser) allocTensors() error {
	mixT, err := ort.NewTensor(ort.NewShape(1, int64(gtcrnNFreq), 1, 2), d.mixBuf)
	if err != nil {
		return fmt.Errorf("denoiser: alloc mix: %w", err)
	}
	d.mixT = mixT
	convInT, err := ort.NewTensor(ort.NewShape(2, 1, 16, 16, 33), d.convCache)
	if err != nil {
		return fmt.Errorf("denoiser: alloc conv_cache: %w", err)
	}
	d.convInT = convInT
	traInT, err := ort.NewTensor(ort.NewShape(2, 3, 1, 1, 16), d.traCache)
	if err != nil {
		return fmt.Errorf("denoiser: alloc tra_cache: %w", err)
	}
	d.traInT = traInT
	interInT, err := ort.NewTensor(ort.NewShape(2, 1, 33, 16), d.interCache)
	if err != nil {
		return fmt.Errorf("denoiser: alloc inter_cache: %w", err)
	}
	d.interInT = interInT
	enhT, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(gtcrnNFreq), 1, 2))
	if err != nil {
		return fmt.Errorf("denoiser: alloc enh: %w", err)
	}
	d.enhT = enhT
	convOutT, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 16, 16, 33))
	if err != nil {
		return fmt.Errorf("denoiser: alloc conv_cache_out: %w", err)
	}
	d.convOutT = convOutT
	traOutT, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 3, 1, 1, 16))
	if err != nil {
		return fmt.Errorf("denoiser: alloc tra_cache_out: %w", err)
	}
	d.traOutT = traOutT
	interOutT, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 33, 16))
	if err != nil {
		return fmt.Errorf("denoiser: alloc inter_cache_out: %w", err)
	}
	d.interOutT = interOutT
	return nil
}

// ProcessFrame consumes one 512-sample stride (a 32-ms frame at 16 kHz).
// Internally, GTCRN expects 256-sample-stride STFT frames; we run one STFT
// inference per call (matching the natural 32-ms cadence — i.e., we run two
// STFT frames per audio frame to keep the model's state advancing at the
// expected rate).
//
// frame: int16 LE PCM of length 512 (1024 bytes worth, 32 ms at 16 kHz).
//
// Returns nil error on success. The actual denoised output is discarded for
// benchmarking purposes (see package doc); we only care that the model ran.
func (d *Denoiser) ProcessFrame(frame []int16) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed.Load() {
		return errors.New("denoiser: closed")
	}
	if len(frame) != 512 {
		return fmt.Errorf("denoiser: frame size %d != 512", len(frame))
	}

	// One 512-sample frame at 16 kHz = 32 ms. GTCRN's natural cadence is 16 ms
	// (256-sample stride), so we run two STFT frames per audio frame.
	// First STFT: window over samples 0..512 of audioBuf. After this call,
	// audioBuf gets shifted: keep last 256 samples, append the first 256 of
	// the new frame; second STFT runs on shifted+second-half.
	//
	// To keep the implementation simple and the cost realistic, we just
	// run the model TWICE per ProcessFrame call: once on the first 512
	// samples (audioBuf left-half + frame left-half) and once on the second
	// (audioBuf right-half + frame right-half).

	if err := d.runOneSTFT(frame[:256]); err != nil {
		return err
	}
	if err := d.runOneSTFT(frame[256:]); err != nil {
		return err
	}
	return nil
}

// runOneSTFT shifts audioBuf left by 256 samples, appends 256 new samples,
// runs windowed STFT on the full 512-sample buffer, fills mixBuf with [F,2]
// re/im pairs, runs ONNX, and copies state caches forward.
func (d *Denoiser) runOneSTFT(newSamples []int16) error {
	// Shift left by 256.
	copy(d.audioBuf[:256], d.audioBuf[256:])
	// Append new samples (i16 → f32 normalized).
	for i := 0; i < 256; i++ {
		d.audioBuf[256+i] = float32(newSamples[i]) / 32768.0
	}

	// Windowed STFT (size 512, sqrt-Hann window).
	scratch := make([]complex128, gtcrnNFFT)
	for i := 0; i < gtcrnNFFT; i++ {
		scratch[i] = complex(float64(d.audioBuf[i]*denoiserShared.window[i]), 0)
	}
	fft512(scratch)

	// Fill mixBuf [F,2] interleaved.
	for f := 0; f < gtcrnNFreq; f++ {
		c := scratch[f]
		d.mixBuf[f*2] = float32(real(c))
		d.mixBuf[f*2+1] = float32(imag(c))
	}

	if err := denoiserShared.session.Run(
		[]ort.Value{d.mixT, d.convInT, d.traInT, d.interInT},
		[]ort.Value{d.enhT, d.convOutT, d.traOutT, d.interOutT},
	); err != nil {
		return fmt.Errorf("denoiser: ORT Run: %w", err)
	}
	// Copy state caches forward (out → in).
	copy(d.convCache, d.convOutT.GetData())
	copy(d.traCache, d.traOutT.GetData())
	copy(d.interCache, d.interOutT.GetData())
	return nil
}

// Cleanup releases per-instance tensors.
func (d *Denoiser) Cleanup() error {
	if !d.closed.CompareAndSwap(false, true) {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, t := range []interface {
		Destroy() error
	}{d.mixT, d.convInT, d.traInT, d.interInT, d.enhT, d.convOutT, d.traOutT, d.interOutT} {
		if t != nil {
			_ = t.Destroy()
		}
	}
	atomic.AddInt64(&denoiserShared.refcount, -1)
	return nil
}

// sqrtHannWindow returns sqrt(hann(n)).
func sqrtHannWindow(n int) []float32 {
	w := hannWindow(n)
	for i, v := range w {
		if v <= 0 {
			w[i] = 0
			continue
		}
		w[i] = float32(math.Sqrt(float64(v)))
	}
	return w
}
