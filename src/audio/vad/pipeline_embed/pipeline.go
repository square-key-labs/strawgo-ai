// Package pipeline_embed runs a 3-model voice pipeline in-process via cgo:
//
//   GTCRN denoiser → Silero VAD → smart-turn
//
// All three models share the global ORT environment configured by the first
// Init* call. Each *PipelineAnalyzer owns one stream's worth of state for
// every sub-model; ORT sessions themselves are global, one per model.
//
// Smart-turn is run on a heuristic: when VAD confidence drops from
// "speaking" (>= 0.5) to "silent" (< 0.5), we treat that frame as a putative
// utterance end and run smart-turn — capped to once per 500 ms per stream so
// a chatty signal doesn't spam smart-turn inferences.
package pipeline_embed

import (
	"errors"
	"sync"
	"time"
)

// Config bundles all three model paths plus the shared ORT settings.
type Config struct {
	VADModelPath       string
	DenoiserModelPath  string
	SmartTurnModelPath string

	SharedLibraryPath string
	IntraOpNumThreads int
	InterOpNumThreads int
}

// Init brings up all three shared ORT sessions. Idempotent — only the first
// call wins. Returns the first error encountered.
func Init(cfg Config) error {
	if cfg.VADModelPath == "" {
		return errors.New("pipeline_embed: VADModelPath required")
	}
	if cfg.DenoiserModelPath == "" {
		return errors.New("pipeline_embed: DenoiserModelPath required")
	}
	if cfg.SmartTurnModelPath == "" {
		return errors.New("pipeline_embed: SmartTurnModelPath required")
	}
	// VAD first because that's the historically-validated path.
	if err := InitVAD(VADConfig{
		ModelPath:         cfg.VADModelPath,
		SharedLibraryPath: cfg.SharedLibraryPath,
		IntraOpNumThreads: cfg.IntraOpNumThreads,
		InterOpNumThreads: cfg.InterOpNumThreads,
	}); err != nil {
		return err
	}
	if err := InitDenoiser(DenoiserConfig{
		ModelPath:         cfg.DenoiserModelPath,
		SharedLibraryPath: cfg.SharedLibraryPath,
		IntraOpNumThreads: cfg.IntraOpNumThreads,
		InterOpNumThreads: cfg.InterOpNumThreads,
	}); err != nil {
		return err
	}
	if err := InitSmartTurn(SmartTurnConfig{
		ModelPath:         cfg.SmartTurnModelPath,
		SharedLibraryPath: cfg.SharedLibraryPath,
		IntraOpNumThreads: cfg.IntraOpNumThreads,
		InterOpNumThreads: cfg.InterOpNumThreads,
	}); err != nil {
		return err
	}
	return nil
}

// Shutdown destroys every shared session. Call at process exit.
func Shutdown() {
	_ = ShutdownSmartTurn()
	_ = ShutdownDenoiser()
	_ = ShutdownVAD()
}

// PipelineAnalyzer is one stream's worth of state for the 3-model pipeline.
type PipelineAnalyzer struct {
	denoiser  *Denoiser
	vad       *VAD
	smartTurn *SmartTurn

	// Smart-turn rate-limit + edge detection.
	smartTurnMinInterval time.Duration
	lastSmartTurn        time.Time
	prevVADProb          float32

	// SNR-gated denoise. If snr is nil OR snrThresholdDB <= 0, denoise runs
	// every frame (legacy behavior). Otherwise denoise runs only when
	// detector reports SNR < threshold.
	snr             *SNRDetector
	snrThresholdDB  float64

	mu sync.Mutex

	// Per-frame timing breakdown (last call). Reset on each ProcessFrame.
	LastDenoiseNS   int64
	LastVADNS       int64
	LastSmartTurnNS int64
	LastSNRNS       int64
	LastDenoiseRan  bool
	LastSNRDB       float64
}

// NewPipelineAnalyzer builds one stream's pipeline state.
func NewPipelineAnalyzer() (*PipelineAnalyzer, error) {
	d, err := NewDenoiser()
	if err != nil {
		return nil, err
	}
	v, err := NewVAD()
	if err != nil {
		_ = d.Cleanup()
		return nil, err
	}
	st, err := NewSmartTurn()
	if err != nil {
		_ = d.Cleanup()
		_ = v.Cleanup()
		return nil, err
	}
	return &PipelineAnalyzer{
		denoiser:             d,
		vad:                  v,
		smartTurn:            st,
		smartTurnMinInterval: 500 * time.Millisecond,
	}, nil
}

// EnableSNRGating turns on SNR-based denoise gating. thresholdDB <= 0
// disables gating (every frame is denoised, legacy default). Typical value:
// 12-15 dB for telephony.
func (p *PipelineAnalyzer) EnableSNRGating(thresholdDB float64, cfg SNRConfig) {
	p.snr = NewSNRDetector(cfg)
	p.snrThresholdDB = thresholdDB
}

// ProcessFrame runs the 3-model pipeline on one 32 ms / 512-sample frame at
// 16 kHz. It returns:
//
//   - vadProb: Silero confidence in [0, 1]
//   - turnEnd: smart-turn decision (true ⇒ probability > 0.5), or false if
//     smart-turn was not run on this frame
//   - smartTurnRan: whether the smart-turn ONNX call was actually executed
//
// frame must be 1024 bytes (512 × int16 LE) at 16 kHz.
func (p *PipelineAnalyzer) ProcessFrame(frame []byte) (vadProb float32, turnEnd, smartTurnRan bool, err error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(frame) != 1024 {
		return 0, false, false, errors.New("pipeline_embed: frame must be 1024 bytes (512 samples × int16)")
	}

	// Decode to int16 once; both denoiser and smart-turn buffer want it.
	samples := make([]int16, 512)
	for i := 0; i < 512; i++ {
		samples[i] = int16(frame[i*2]) | int16(frame[i*2+1])<<8
	}

	// 0. SNR gate (if enabled).
	p.LastSNRNS = 0
	p.LastSNRDB = 0
	runDenoise := true
	if p.snr != nil && p.snrThresholdDB > 0 {
		t := time.Now()
		_ = p.snr.Update(samples)
		p.LastSNRNS = time.Since(t).Nanoseconds()
		p.LastSNRDB = p.snr.LastSNRDB
		runDenoise = p.snr.ShouldDenoise(p.snrThresholdDB)
	}

	// 1. Denoiser (gated).
	p.LastDenoiseNS = 0
	p.LastDenoiseRan = false
	if runDenoise {
		t0 := time.Now()
		if err := p.denoiser.ProcessFrame(samples); err != nil {
			return 0, false, false, err
		}
		p.LastDenoiseNS = time.Since(t0).Nanoseconds()
		p.LastDenoiseRan = true
	}

	// 2. VAD on the same int16 frame.
	t1 := time.Now()
	vadProb = p.vad.VoiceConfidence(samples)
	p.LastVADNS = time.Since(t1).Nanoseconds()

	// 3. Append to smart-turn rolling buffer.
	p.smartTurn.AppendAudio(samples)

	// 4. Smart-turn heuristic: high → low VAD edge.
	const speakingThreshold = 0.5
	now := time.Now()
	wasSpeaking := p.prevVADProb >= speakingThreshold
	nowSilent := vadProb < speakingThreshold
	canRun := now.Sub(p.lastSmartTurn) >= p.smartTurnMinInterval
	p.prevVADProb = vadProb

	p.LastSmartTurnNS = 0
	if wasSpeaking && nowSilent && canRun {
		t2 := time.Now()
		prob, stErr := p.smartTurn.PredictEnd()
		p.LastSmartTurnNS = time.Since(t2).Nanoseconds()
		if stErr != nil {
			return vadProb, false, true, stErr
		}
		smartTurnRan = true
		turnEnd = prob > 0.5
		p.lastSmartTurn = now
	}

	return vadProb, turnEnd, smartTurnRan, nil
}

// Cleanup releases all per-instance tensors. Idempotent.
func (p *PipelineAnalyzer) Cleanup() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	_ = p.denoiser.Cleanup()
	_ = p.vad.Cleanup()
	_ = p.smartTurn.Cleanup()
	return nil
}
