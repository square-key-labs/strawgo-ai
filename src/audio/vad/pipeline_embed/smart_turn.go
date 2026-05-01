// Package pipeline_embed — smart-turn v3.1 wrapper.
//
// Mirrors onnx-worker/src/smart_turn.rs:
//   - input  : "input_features" f32 [1, 80, 800] (log-mel @ 16 kHz, 8 s window)
//   - output : "logits" f32 [1] (turn-end probability)
//
// Per stream: rolling 8-second i16 buffer. On PredictEnd, we copy that buffer,
// extract mel features, and run inference. The shared session is global; each
// stream has its own buffer + per-call mel scratch + pre-allocated I/O tensors.
package pipeline_embed

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	smartTurnMaxSamples  = 8 * 16000 // 128_000
	smartTurnFeatureSize = 80 * 800  // 64_000 floats
)

var smartTurnShared struct {
	mu       sync.Mutex
	session  *ort.DynamicAdvancedSession
	cfg      SmartTurnConfig
	initOnce sync.Once
	initErr  error
	refcount int64
}

// SmartTurnConfig configures the smart-turn shared session.
type SmartTurnConfig struct {
	ModelPath         string
	SharedLibraryPath string
	IntraOpNumThreads int
	InterOpNumThreads int
}

// InitSmartTurn constructs the global smart-turn session. Holds
// smartTurnShared.mu so the session pointer is published under the same lock
// that ShutdownSmartTurn/NewSmartTurn observe.
func InitSmartTurn(cfg SmartTurnConfig) error {
	smartTurnShared.initOnce.Do(func() {
		smartTurnShared.mu.Lock()
		defer smartTurnShared.mu.Unlock()
		smartTurnShared.cfg = cfg
		smartTurnShared.initErr = doInitSmartTurn(cfg)
	})
	return smartTurnShared.initErr
}

func doInitSmartTurn(cfg SmartTurnConfig) error {
	if cfg.ModelPath == "" {
		return errors.New("pipeline_embed: SmartTurnConfig.ModelPath is required")
	}
	if cfg.SharedLibraryPath != "" {
		ort.SetSharedLibraryPath(cfg.SharedLibraryPath)
	}
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return fmt.Errorf("pipeline_embed: InitializeEnvironment (smart-turn): %w", err)
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
	inputs := []string{"input_features"}
	outputs := []string{"logits"}
	sess, err := ort.NewDynamicAdvancedSession(cfg.ModelPath, inputs, outputs, opts)
	if err != nil {
		return fmt.Errorf("pipeline_embed: smart_turn NewDynamicAdvancedSession(%s): %w", cfg.ModelPath, err)
	}
	smartTurnShared.session = sess
	return nil
}

// ShutdownSmartTurn destroys the shared smart-turn session. Refuses while any
// per-stream SmartTurn is alive (refcount > 0). Safe to call repeatedly once
// refcount is zero.
func ShutdownSmartTurn() error {
	smartTurnShared.mu.Lock()
	defer smartTurnShared.mu.Unlock()
	if smartTurnShared.session == nil {
		return nil
	}
	if atomic.LoadInt64(&smartTurnShared.refcount) > 0 {
		return errors.New("pipeline_embed: ShutdownSmartTurn: streams still active")
	}
	_ = smartTurnShared.session.Destroy()
	smartTurnShared.session = nil
	return nil
}

// SmartTurn is one stream's smart-turn state.
type SmartTurn struct {
	mu sync.Mutex

	// Rolling i16 buffer of recent audio (length up to smartTurnMaxSamples).
	buf      []int16
	bufStart int // ring-buffer head; len(buf) == smartTurnMaxSamples after warm-up

	mel    *MelExtractor
	melBuf []float32 // smartTurnFeatureSize, reused

	inputT  *ort.Tensor[float32]
	outputT *ort.Tensor[float32]

	closed atomic.Bool
}

// NewSmartTurn allocates one stream-bound smart-turn instance. Reads the
// session pointer under smartTurnShared.mu to synchronize with init/shutdown.
func NewSmartTurn() (*SmartTurn, error) {
	smartTurnShared.mu.Lock()
	if smartTurnShared.session == nil {
		smartTurnShared.mu.Unlock()
		return nil, errors.New("pipeline_embed: InitSmartTurn() not called")
	}
	atomic.AddInt64(&smartTurnShared.refcount, 1)
	smartTurnShared.mu.Unlock()

	st := &SmartTurn{
		buf:    make([]int16, 0, smartTurnMaxSamples),
		mel:    NewMelExtractor(),
		melBuf: make([]float32, smartTurnFeatureSize),
	}

	inT, err := ort.NewTensor(ort.NewShape(1, 80, 800), st.melBuf)
	if err != nil {
		atomic.AddInt64(&smartTurnShared.refcount, -1)
		return nil, fmt.Errorf("smart_turn: alloc input: %w", err)
	}
	st.inputT = inT
	// Smart-turn v3.1 outputs logits with shape [1, 1].
	outT, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		_ = inT.Destroy()
		atomic.AddInt64(&smartTurnShared.refcount, -1)
		return nil, fmt.Errorf("smart_turn: alloc output: %w", err)
	}
	st.outputT = outT
	return st, nil
}

// AppendAudio adds a frame's worth of audio to the rolling buffer. Drops the
// oldest samples if the buffer would exceed 8 seconds.
func (s *SmartTurn) AppendAudio(frame []int16) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Always append; trim from the head if over capacity.
	s.buf = append(s.buf, frame...)
	if len(s.buf) > smartTurnMaxSamples {
		excess := len(s.buf) - smartTurnMaxSamples
		s.buf = s.buf[excess:]
	}
}

// PredictEnd extracts mel features from the rolling buffer, runs the model,
// and returns the turn-end probability in [0, 1].
//
// If the buffer has fewer than ~1 s of audio, returns 0 without inference.
func (s *SmartTurn) PredictEnd() (float32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed.Load() {
		return 0, errors.New("smart_turn: closed")
	}
	if len(s.buf) < 16000 {
		// Not enough audio yet — don't waste an inference.
		return 0, nil
	}

	// Copy buffer to f32 normalized.
	audio := make([]float32, len(s.buf))
	for i, v := range s.buf {
		audio[i] = float32(v) / 32768.0
	}
	mel := s.mel.Extract(audio, smartTurnMaxSamples)
	// Zero melBuf, then copy mel (size = whisperNMels * whisperNFrames =
	// 80 * (smartTurnMaxSamples/whisperHopLength) = 80 * 800 = 64000).
	for i := range s.melBuf {
		s.melBuf[i] = 0
	}
	n := len(mel)
	if n > smartTurnFeatureSize {
		n = smartTurnFeatureSize
	}
	copy(s.melBuf[:n], mel[:n])

	if err := smartTurnShared.session.Run(
		[]ort.Value{s.inputT},
		[]ort.Value{s.outputT},
	); err != nil {
		return 0, fmt.Errorf("smart_turn: ORT Run: %w", err)
	}
	return s.outputT.GetData()[0], nil
}

// Cleanup releases per-instance tensors.
func (s *SmartTurn) Cleanup() error {
	if !s.closed.CompareAndSwap(false, true) {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inputT != nil {
		_ = s.inputT.Destroy()
	}
	if s.outputT != nil {
		_ = s.outputT.Destroy()
	}
	atomic.AddInt64(&smartTurnShared.refcount, -1)
	return nil
}
