// Package silero_embedded runs Silero VAD ONNX inference in-process via cgo,
// eliminating the Unix-socket IPC tax of the Rust onnx-worker.
//
// Architecture:
//
//   - One global ONNX Runtime environment (Initialize once per process).
//   - One shared *DynamicAdvancedSession across all VAD instances. The Silero
//     maintainer confirms a single VAD model can be shared across many audio
//     streams (https://github.com/snakers4/silero-vad/discussions/744). ORT's
//     OrtSession::Run is thread-safe; only the per-call I/O tensors need to be
//     per-goroutine.
//   - Per-instance LSTM hidden state (the [2,1,128] f32 tensor), a 64-sample
//     16-kHz "context" buffer carried across calls (matches the official
//     pre-processing recipe used by silero-vad's Python wrapper), and a set
//     of preallocated input/output tensors owned by that instance.
//
// Threading:
//
//   - Per-session: intra_op_num_threads=1, inter_op_num_threads=1. With one
//     shared session this prevents the N×nproc oversubscription that
//     destroyed the Rust worker at N≥100.
//   - The yalue/onnxruntime_go binding does NOT expose
//     CreateEnvWithGlobalThreadPools / DisablePerSessionThreads. Because we
//     use a single shared session anyway, this is fine: one session pinned to
//     one intra-op thread is functionally equivalent to a global pool sized
//     to 1 — Go's runtime handles N concurrent goroutines calling Run on
//     the shared session.
//
// Wire compatibility with the existing VADAnalyzer interface:
//
//   - VoiceConfidence(buf []byte, sampleRate int) float32
//   - SetSampleRate / NumFramesRequired / Restart / AnalyzeAudio
//   - Cleanup() — releases per-instance tensors (NOT the shared session).
//
// Lifecycle:
//
//	if err := silero_embedded.Init(silero_embedded.Config{ModelPath: "..."}); err != nil { ... }
//	defer silero_embedded.Shutdown()
//
//	v, err := silero_embedded.New(16000, vad.DefaultVADParams())
//	defer v.Cleanup()
package silero_embedded

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// Config controls process-global ORT setup. Must be passed to Init() exactly
// once before any New() call.
type Config struct {
	// ModelPath is the path to silero_vad.onnx (v5 model with [2,1,128] state).
	ModelPath string

	// SharedLibraryPath optionally points to libonnxruntime.{so,dylib,dll}.
	// If empty, the binding uses platform defaults / DYLD/LD_LIBRARY_PATH /
	// rpath. Recommended on macOS: "/usr/local/lib/libonnxruntime.dylib".
	SharedLibraryPath string

	// IntraOpNumThreads pins the per-op thread count on the shared session.
	// Defaults to 1 — correct for a tiny model like Silero where multi-thread
	// overhead exceeds gains and thread oversubscription wrecks tail latency.
	IntraOpNumThreads int

	// InterOpNumThreads pins the cross-op thread count. Defaults to 1.
	InterOpNumThreads int
}

const (
	// stateSize is the LSTM hidden-state tensor element count: [2, 1, 128].
	stateSize = 2 * 1 * 128

	// ctxSize16k is the number of context samples carried across calls at
	// 16 kHz. Must match the official Silero pre-processing recipe.
	ctxSize16k = 64
	// ctxSize8k is the same at 8 kHz.
	ctxSize8k = 32

	// expSamples16k is the per-call audio frame size at 16 kHz.
	expSamples16k = 512
	// expSamples8k is the per-call audio frame size at 8 kHz.
	expSamples8k = 256
)

// shared holds the process-global ORT artefacts.
var shared struct {
	mu        sync.Mutex
	session   *ort.DynamicAdvancedSession
	cfg       Config
	initOnce  sync.Once
	initErr   error
	refcount  int64 // # of live SileroVAD instances; informational only
	destroyed bool
}

// Init prepares the global ORT environment and constructs the single shared
// session. Safe to call multiple times — only the first call wins; subsequent
// calls return the original error (or nil).
func Init(cfg Config) error {
	shared.initOnce.Do(func() {
		shared.cfg = cfg
		shared.initErr = doInit(cfg)
	})
	return shared.initErr
}

func doInit(cfg Config) error {
	if cfg.ModelPath == "" {
		return errors.New("silero_embedded: ModelPath is required")
	}
	if cfg.SharedLibraryPath != "" {
		ort.SetSharedLibraryPath(cfg.SharedLibraryPath)
	}
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return fmt.Errorf("silero_embedded: InitializeEnvironment: %w", err)
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
		return fmt.Errorf("silero_embedded: NewSessionOptions: %w", err)
	}
	defer opts.Destroy()
	if err := opts.SetIntraOpNumThreads(intra); err != nil {
		return fmt.Errorf("silero_embedded: SetIntraOpNumThreads: %w", err)
	}
	if err := opts.SetInterOpNumThreads(inter); err != nil {
		return fmt.Errorf("silero_embedded: SetInterOpNumThreads: %w", err)
	}
	if err := opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return fmt.Errorf("silero_embedded: SetGraphOptimizationLevel: %w", err)
	}
	// Memory arena off: with N goroutines all calling Run on the shared
	// session, the arena's per-thread caches just bloat RSS. Silero's working
	// set is tiny, so the malloc cost per call is irrelevant.
	if err := opts.SetCpuMemArena(false); err != nil {
		return fmt.Errorf("silero_embedded: SetCpuMemArena: %w", err)
	}
	if err := opts.SetMemPattern(false); err != nil {
		return fmt.Errorf("silero_embedded: SetMemPattern: %w", err)
	}

	// Silero v5 ONNX inputs/outputs (verified against onnx-worker/src/vad.rs):
	//   inputs : [input (f32 [1,N]), state (f32 [2,1,128]), sr (i64 [1])]
	//   outputs: [output (f32 [1,1]), stateN (f32 [2,1,128])]
	inputNames := []string{"input", "state", "sr"}
	outputNames := []string{"output", "stateN"}

	sess, err := ort.NewDynamicAdvancedSession(cfg.ModelPath, inputNames, outputNames, opts)
	if err != nil {
		return fmt.Errorf("silero_embedded: NewDynamicAdvancedSession(%s): %w", cfg.ModelPath, err)
	}
	shared.session = sess
	logger.Info("[silero_embedded] Initialized (model=%s intra=%d inter=%d)",
		cfg.ModelPath, intra, inter)
	return nil
}

// Shutdown releases the shared session and ORT environment. Call once at
// process exit. After Shutdown, calling New() returns an error.
func Shutdown() error {
	shared.mu.Lock()
	defer shared.mu.Unlock()
	if shared.destroyed {
		return nil
	}
	shared.destroyed = true
	if shared.session != nil {
		if err := shared.session.Destroy(); err != nil {
			logger.Warn("[silero_embedded] session.Destroy: %v", err)
		}
		shared.session = nil
	}
	if ort.IsInitialized() {
		if err := ort.DestroyEnvironment(); err != nil {
			return fmt.Errorf("silero_embedded: DestroyEnvironment: %w", err)
		}
	}
	return nil
}

// SileroVAD is one stream's worth of Silero VAD state. Owns its LSTM hidden
// state and a set of pre-allocated input/output tensors. Not safe for
// concurrent use within a single instance — wrap calls in your own
// serialization if needed (the per-conn pattern in strawgo guarantees
// single-threaded use).
type SileroVAD struct {
	*vad.BaseVADAnalyzer

	// hidden is the LSTM state, mirrored on Go side; rewritten after every
	// inference. Shape [2, 1, 128] = 256 f32 values. Stored as a heap-allocated
	// slice (not an inline array) because cgo refuses pointers to fields of
	// a Go-managed object — the slice's own backing array sits in its own
	// heap allocation, which is what ORT can safely write to.
	hidden []float32

	// context is the last ctxSize samples (in f32-normalized form) carried
	// to the next call. nil-len until first call.
	context []float32

	// preallocated tensors (one set per instance). Reusing these across
	// Run() calls avoids per-frame ORT allocation churn.
	inputT  *ort.Tensor[float32]
	stateT  *ort.Tensor[float32]
	srT     *ort.Tensor[int64]
	outputT *ort.Tensor[float32]
	stateNT *ort.Tensor[float32]

	inputBuf []float32 // backing slice for inputT, length ctxSize+expSamples

	mu sync.Mutex // guards everything above

	logEveryNFrames int
	frameCount      int
	closed          atomic.Bool
}

// New constructs a SileroVAD bound to the shared global session. Init() must
// have been called.
func New(sampleRate int, params vad.VADParams) (*SileroVAD, error) {
	shared.mu.Lock()
	if shared.destroyed {
		shared.mu.Unlock()
		return nil, errors.New("silero_embedded: package shut down")
	}
	if shared.session == nil {
		shared.mu.Unlock()
		return nil, errors.New("silero_embedded: Init() not called")
	}
	atomic.AddInt64(&shared.refcount, 1)
	shared.mu.Unlock()

	if sampleRate != 8000 && sampleRate != 16000 {
		atomic.AddInt64(&shared.refcount, -1)
		return nil, fmt.Errorf("silero_embedded: sample rate %d unsupported (need 8000 or 16000)", sampleRate)
	}

	v := &SileroVAD{
		BaseVADAnalyzer: vad.NewBaseVADAnalyzer(sampleRate, params),
		logEveryNFrames: 0, // disabled; loadtest will spam otherwise
	}
	if err := v.allocTensors(sampleRate); err != nil {
		atomic.AddInt64(&shared.refcount, -1)
		return nil, err
	}
	return v, nil
}

// allocTensors allocates the per-instance Tensor objects sized for the given
// sample rate. Called from New() and from SetSampleRate when rate changes.
// Caller must hold v.mu (or be in the constructor before any sharing).
func (v *SileroVAD) allocTensors(sampleRate int) error {
	ctxSize, expSamples := layoutFor(sampleRate)

	// Heap-allocate hidden state. Must NOT be a field-embedded array because
	// cgo cannot safely take pointers into another Go-managed object.
	if v.hidden == nil {
		v.hidden = make([]float32, stateSize)
	} else {
		for i := range v.hidden {
			v.hidden[i] = 0
		}
	}

	// input tensor: [1, ctxSize+expSamples]
	v.inputBuf = make([]float32, ctxSize+expSamples)
	inT, err := ort.NewTensor(ort.NewShape(1, int64(ctxSize+expSamples)), v.inputBuf)
	if err != nil {
		return fmt.Errorf("silero_embedded: alloc input tensor: %w", err)
	}
	v.inputT = inT

	// state tensor: [2, 1, 128]
	stT, err := ort.NewTensor(ort.NewShape(2, 1, 128), v.hidden)
	if err != nil {
		_ = inT.Destroy()
		return fmt.Errorf("silero_embedded: alloc state tensor: %w", err)
	}
	v.stateT = stT

	// sr tensor: [1] of i64
	srData := []int64{int64(sampleRate)}
	srT, err := ort.NewTensor(ort.NewShape(1), srData)
	if err != nil {
		_ = inT.Destroy()
		_ = stT.Destroy()
		return fmt.Errorf("silero_embedded: alloc sr tensor: %w", err)
	}
	v.srT = srT

	// output tensor: [1, 1]
	outT, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		_ = inT.Destroy()
		_ = stT.Destroy()
		_ = srT.Destroy()
		return fmt.Errorf("silero_embedded: alloc output tensor: %w", err)
	}
	v.outputT = outT

	// stateN tensor: [2, 1, 128]
	stnT, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128))
	if err != nil {
		_ = inT.Destroy()
		_ = stT.Destroy()
		_ = srT.Destroy()
		_ = outT.Destroy()
		return fmt.Errorf("silero_embedded: alloc stateN tensor: %w", err)
	}
	v.stateNT = stnT

	return nil
}

func layoutFor(sampleRate int) (ctxSize, expSamples int) {
	if sampleRate == 16000 {
		return ctxSize16k, expSamples16k
	}
	return ctxSize8k, expSamples8k
}

// SetSampleRate resets state and re-allocates tensors if the rate changed.
func (v *SileroVAD) SetSampleRate(sampleRate int) error {
	if sampleRate != 8000 && sampleRate != 16000 {
		return fmt.Errorf("silero_embedded: sample rate %d unsupported (need 8000 or 16000)", sampleRate)
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.GetSampleRate() == sampleRate {
		return v.BaseVADAnalyzer.SetSampleRate(sampleRate)
	}
	v.destroyTensorsLocked()
	if err := v.BaseVADAnalyzer.SetSampleRate(sampleRate); err != nil {
		return err
	}
	// Reset state: rate change invalidates hidden state and context.
	v.hidden = nil
	v.context = nil
	return v.allocTensors(sampleRate)
}

// NumFramesRequired returns 512 at 16k, 256 at 8k.
func (v *SileroVAD) NumFramesRequired() int {
	if v.GetSampleRate() == 16000 {
		return expSamples16k
	}
	return expSamples8k
}

// VoiceConfidence runs ONNX inference and returns voice probability in [0, 1].
// Uses the shared session; per-instance state and tensors guarantee no data
// race with other VAD instances.
func (v *SileroVAD) VoiceConfidence(buffer []byte) float32 {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.closed.Load() {
		return 0
	}
	sr := v.GetSampleRate()
	ctxSize, expSamples := layoutFor(sr)

	// Decode int16 LE PCM → f32 normalized into the second half of inputBuf.
	if len(buffer) != expSamples*2 {
		// Wrong frame size — Silero requires exact frame counts. Return 0.
		return 0
	}
	if v.context == nil {
		v.context = make([]float32, ctxSize)
		// First call: zero context. (Matches snakers4/silero-vad init.)
	}
	// inputBuf layout: [context (ctxSize) | audio (expSamples)]
	copy(v.inputBuf[:ctxSize], v.context)
	for i := 0; i < expSamples; i++ {
		// Little-endian int16
		s := int16(buffer[i*2]) | int16(buffer[i*2+1])<<8
		v.inputBuf[ctxSize+i] = float32(s) / 32768.0
	}

	// stateT and inputT are zero-copy views of v.hidden / v.inputBuf — no
	// extra copy needed here. Just run.
	if err := shared.session.Run(
		[]ort.Value{v.inputT, v.stateT, v.srT},
		[]ort.Value{v.outputT, v.stateNT},
	); err != nil {
		logger.Error("[silero_embedded] Run: %v", err)
		return 0
	}

	// Read out probability and update hidden state.
	confidence := v.outputT.GetData()[0]
	newState := v.stateNT.GetData()
	if len(newState) == stateSize {
		copy(v.hidden, newState)
	}

	// Carry context: last ctxSize samples of the input (i.e., the tail of the
	// audio frame, since the full input is [context | audio]).
	if cap(v.context) < ctxSize {
		v.context = make([]float32, ctxSize)
	} else {
		v.context = v.context[:ctxSize]
	}
	copy(v.context, v.inputBuf[ctxSize+expSamples-ctxSize:])

	return confidence
}

// AnalyzeAudio is the standard interface.
func (v *SileroVAD) AnalyzeAudio(buffer []byte) (vad.VADState, error) {
	confidence := v.VoiceConfidence(buffer)
	state, err := v.ProcessAudio(buffer, confidence, v.NumFramesRequired())
	if err != nil {
		return vad.VADStateQuiet, err
	}
	v.frameCount++
	if v.logEveryNFrames > 0 && v.frameCount%v.logEveryNFrames == 0 {
		logger.Debug("[silero_embedded] conf=%.3f state=%s", confidence, state.String())
	}
	return state, nil
}

// Restart zeros LSTM hidden state and clears context. Does NOT reallocate
// tensors — the existing stateT keeps pointing at v.hidden (we just zero it
// in place).
func (v *SileroVAD) Restart() {
	v.mu.Lock()
	for i := range v.hidden {
		v.hidden[i] = 0
	}
	v.context = nil
	v.mu.Unlock()
	v.BaseVADAnalyzer.Restart()
}

// Cleanup destroys per-instance tensors. The shared session is NOT touched.
// Must be called when the VAD is no longer needed; otherwise C-side memory
// leaks until process exit.
func (v *SileroVAD) Cleanup() error {
	if !v.closed.CompareAndSwap(false, true) {
		return nil
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	v.destroyTensorsLocked()
	atomic.AddInt64(&shared.refcount, -1)
	return nil
}

func (v *SileroVAD) destroyTensorsLocked() {
	for _, t := range []interface {
		Destroy() error
	}{v.inputT, v.stateT, v.srT, v.outputT, v.stateNT} {
		if t == nil {
			continue
		}
		if err := t.Destroy(); err != nil {
			logger.Warn("[silero_embedded] tensor.Destroy: %v", err)
		}
	}
	v.inputT = nil
	v.stateT = nil
	v.srT = nil
	v.outputT = nil
	v.stateNT = nil
}

// LiveCount returns the number of SileroVAD instances currently allocated.
// Useful for diagnostics.
func LiveCount() int64 {
	return atomic.LoadInt64(&shared.refcount)
}
