// Package pipeline_embed — Silero VAD wrapper.
//
// Mirrors src/audio/vad/silero_embedded but with its own shared session so
// the pipeline can co-exist with the existing single-VAD package without
// double-initialising ORT or sharing model handles across packages.
package pipeline_embed

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	vadStateSize     = 2 * 1 * 128
	vadCtxSize16k    = 64
	vadExpSamples16k = 512
)

var vadShared struct {
	mu       sync.Mutex
	session  *ort.DynamicAdvancedSession
	cfg      VADConfig
	initOnce sync.Once
	initErr  error
	refcount int64
}

// VADConfig configures the Silero VAD shared session.
type VADConfig struct {
	ModelPath         string
	SharedLibraryPath string
	IntraOpNumThreads int
	InterOpNumThreads int
}

// InitVAD constructs the global Silero session. Idempotent.
//
// Note: the SharedLibraryPath is also picked up by the denoiser/smart-turn
// packages — ORT is process-global. Set it on whichever Init is called first.
func InitVAD(cfg VADConfig) error {
	vadShared.initOnce.Do(func() {
		vadShared.cfg = cfg
		vadShared.initErr = doInitVAD(cfg)
	})
	return vadShared.initErr
}

func doInitVAD(cfg VADConfig) error {
	if cfg.ModelPath == "" {
		return errors.New("pipeline_embed: VADConfig.ModelPath is required")
	}
	if cfg.SharedLibraryPath != "" {
		ort.SetSharedLibraryPath(cfg.SharedLibraryPath)
	}
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return fmt.Errorf("pipeline_embed: InitializeEnvironment (vad): %w", err)
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
		return err
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

	inputs := []string{"input", "state", "sr"}
	outputs := []string{"output", "stateN"}
	sess, err := ort.NewDynamicAdvancedSession(cfg.ModelPath, inputs, outputs, opts)
	if err != nil {
		return fmt.Errorf("pipeline_embed: silero NewDynamicAdvancedSession(%s): %w", cfg.ModelPath, err)
	}
	vadShared.session = sess
	return nil
}

// ShutdownVAD destroys the shared VAD session.
func ShutdownVAD() error {
	vadShared.mu.Lock()
	defer vadShared.mu.Unlock()
	if vadShared.session != nil {
		_ = vadShared.session.Destroy()
		vadShared.session = nil
	}
	return nil
}

// VAD is one stream's Silero VAD state. 16 kHz only — pipeline is fixed-rate.
type VAD struct {
	mu      sync.Mutex
	hidden  []float32
	context []float32

	inputT  *ort.Tensor[float32]
	stateT  *ort.Tensor[float32]
	srT     *ort.Tensor[int64]
	outputT *ort.Tensor[float32]
	stateNT *ort.Tensor[float32]

	inputBuf []float32

	closed atomic.Bool
}

// NewVAD allocates one VAD instance bound to the shared session.
func NewVAD() (*VAD, error) {
	if vadShared.session == nil {
		return nil, errors.New("pipeline_embed: InitVAD() not called")
	}
	atomic.AddInt64(&vadShared.refcount, 1)

	v := &VAD{
		hidden:   make([]float32, vadStateSize),
		inputBuf: make([]float32, vadCtxSize16k+vadExpSamples16k),
	}

	inT, err := ort.NewTensor(ort.NewShape(1, int64(vadCtxSize16k+vadExpSamples16k)), v.inputBuf)
	if err != nil {
		atomic.AddInt64(&vadShared.refcount, -1)
		return nil, fmt.Errorf("vad: alloc input: %w", err)
	}
	v.inputT = inT
	stT, err := ort.NewTensor(ort.NewShape(2, 1, 128), v.hidden)
	if err != nil {
		_ = inT.Destroy()
		atomic.AddInt64(&vadShared.refcount, -1)
		return nil, fmt.Errorf("vad: alloc state: %w", err)
	}
	v.stateT = stT
	srData := []int64{16000}
	srT, err := ort.NewTensor(ort.NewShape(1), srData)
	if err != nil {
		_ = inT.Destroy()
		_ = stT.Destroy()
		atomic.AddInt64(&vadShared.refcount, -1)
		return nil, fmt.Errorf("vad: alloc sr: %w", err)
	}
	v.srT = srT
	outT, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		_ = inT.Destroy()
		_ = stT.Destroy()
		_ = srT.Destroy()
		atomic.AddInt64(&vadShared.refcount, -1)
		return nil, fmt.Errorf("vad: alloc output: %w", err)
	}
	v.outputT = outT
	stnT, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128))
	if err != nil {
		_ = inT.Destroy()
		_ = stT.Destroy()
		_ = srT.Destroy()
		_ = outT.Destroy()
		atomic.AddInt64(&vadShared.refcount, -1)
		return nil, fmt.Errorf("vad: alloc stateN: %w", err)
	}
	v.stateNT = stnT
	return v, nil
}

// VoiceConfidence runs Silero on a 512-sample int16 PCM frame at 16 kHz.
func (v *VAD) VoiceConfidence(frame []int16) float32 {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.closed.Load() {
		return 0
	}
	if len(frame) != vadExpSamples16k {
		return 0
	}
	if v.context == nil {
		v.context = make([]float32, vadCtxSize16k)
	}
	copy(v.inputBuf[:vadCtxSize16k], v.context)
	for i := 0; i < vadExpSamples16k; i++ {
		v.inputBuf[vadCtxSize16k+i] = float32(frame[i]) / 32768.0
	}
	if err := vadShared.session.Run(
		[]ort.Value{v.inputT, v.stateT, v.srT},
		[]ort.Value{v.outputT, v.stateNT},
	); err != nil {
		return 0
	}
	confidence := v.outputT.GetData()[0]
	newState := v.stateNT.GetData()
	if len(newState) == vadStateSize {
		copy(v.hidden, newState)
	}
	copy(v.context, v.inputBuf[vadCtxSize16k+vadExpSamples16k-vadCtxSize16k:])
	return confidence
}

// Cleanup releases per-instance tensors.
func (v *VAD) Cleanup() error {
	if !v.closed.CompareAndSwap(false, true) {
		return nil
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	for _, t := range []interface {
		Destroy() error
	}{v.inputT, v.stateT, v.srT, v.outputT, v.stateNT} {
		if t != nil {
			_ = t.Destroy()
		}
	}
	atomic.AddInt64(&vadShared.refcount, -1)
	return nil
}
