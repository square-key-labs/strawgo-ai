package vad

import (
	"context"
	"fmt"
	"math"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// VADState represents the current state of voice activity detection
type VADState int

const (
	VADStateQuiet VADState = iota + 1
	VADStateStarting
	VADStateSpeaking
	VADStateStopping
)

func (s VADState) String() string {
	switch s {
	case VADStateQuiet:
		return "quiet"
	case VADStateStarting:
		return "starting"
	case VADStateSpeaking:
		return "speaking"
	case VADStateStopping:
		return "stopping"
	default:
		return "unknown"
	}
}

// VADParams holds configuration parameters for voice activity detection
type VADParams struct {
	// Confidence threshold for voice detection (0.0 to 1.0)
	// Higher values are more strict (default: 0.7)
	Confidence float32

	// StartSecs: Duration in seconds that voice must be detected before
	// transitioning from QUIET to SPEAKING (default: 0.2)
	StartSecs float32

	// StopSecs: Duration in seconds that silence must be detected before
	// transitioning from SPEAKING to QUIET (default: 0.8)
	StopSecs float32

	// MinVolume: Minimum audio volume threshold (0.0 to 1.0)
	// Audio below this volume is ignored (default: 0.6)
	MinVolume float32
}

// DefaultVADParams returns the default VAD parameters
func DefaultVADParams() VADParams {
	return VADParams{
		Confidence: 0.7,
		StartSecs:  0.2,
		StopSecs:   0.8,
		MinVolume:  0.6,
	}
}

// VADAnalyzer is the interface for voice activity detection implementations
type VADAnalyzer interface {
	// SetSampleRate configures the sample rate for audio processing
	SetSampleRate(sampleRate int) error

	// NumFramesRequired returns the number of audio frames required for analysis
	NumFramesRequired() int

	// VoiceConfidence calculates voice activity confidence for the given audio buffer
	// Returns a value between 0.0 (no voice) and 1.0 (definitely voice)
	VoiceConfidence(buffer []byte) float32

	// AnalyzeAudio processes audio and returns the current VAD state
	AnalyzeAudio(buffer []byte) (VADState, error)

	// Restart resets the VAD analyzer state
	Restart()
}

// BaseVADAnalyzer provides common functionality for VAD implementations
type BaseVADAnalyzer struct {
	params     VADParams
	sampleRate int

	// State machine
	state           VADState
	startFrames     int
	stopFrames      int
	startThreshold  int
	stopThreshold   int
	prevSampleCount int

	// Volume tracking
	smoothedVolume float32

	// Thread safety
	mu sync.RWMutex
}

// NewBaseVADAnalyzer creates a new base VAD analyzer
func NewBaseVADAnalyzer(sampleRate int, params VADParams) *BaseVADAnalyzer {
	return &BaseVADAnalyzer{
		params:         params,
		sampleRate:     sampleRate,
		state:          VADStateQuiet,
		smoothedVolume: 0.0,
	}
}

// SetSampleRate configures the sample rate and recalculates frame thresholds
func (v *BaseVADAnalyzer) SetSampleRate(sampleRate int) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.sampleRate = sampleRate
	return nil
}

// GetSampleRate returns the current sample rate
func (v *BaseVADAnalyzer) GetSampleRate() int {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.sampleRate
}

// GetParams returns the current VAD parameters
func (v *BaseVADAnalyzer) GetParams() VADParams {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.params
}

// GetState returns the current VAD state
func (v *BaseVADAnalyzer) GetState() VADState {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.state
}

// Restart resets the VAD analyzer state
func (v *BaseVADAnalyzer) Restart() {
	v.mu.Lock()
	defer v.mu.Unlock()

	logger.Debug("[VADAnalyzer] Restarting VAD state machine")
	v.state = VADStateQuiet
	v.startFrames = 0
	v.stopFrames = 0
	v.smoothedVolume = 0.0
}

// ProcessAudio implements the VAD state machine logic
// This should be called by subclasses after computing voice confidence
func (v *BaseVADAnalyzer) ProcessAudio(buffer []byte, voiceConfidence float32, numFramesRequired int) (VADState, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Calculate volume from audio buffer (int16 samples)
	volume := v.calculateVolume(buffer)

	// Smooth volume with exponential averaging (factor: 0.2)
	const smoothingFactor = 0.2
	v.smoothedVolume = smoothingFactor*volume + (1.0-smoothingFactor)*v.smoothedVolume

	// Recalculate thresholds if sample rate changed
	sampleCount := len(buffer) / 2 // int16 = 2 bytes per sample
	if sampleCount != v.prevSampleCount {
		v.prevSampleCount = sampleCount
		frameTime := float32(numFramesRequired) / float32(v.sampleRate)
		v.startThreshold = int(v.params.StartSecs / frameTime)
		v.stopThreshold = int(v.params.StopSecs / frameTime)
		logger.Debug("[VADAnalyzer] Thresholds updated: start=%d frames (%.2fs), stop=%d frames (%.2fs)",
			v.startThreshold, v.params.StartSecs, v.stopThreshold, v.params.StopSecs)
	}

	// Check if audio meets minimum volume threshold
	if v.smoothedVolume < v.params.MinVolume {
		voiceConfidence = 0.0
	}

	// State machine logic
	oldState := v.state

	switch v.state {
	case VADStateQuiet:
		if voiceConfidence >= v.params.Confidence {
			v.startFrames++
			if v.startFrames >= v.startThreshold {
				v.state = VADStateSpeaking
				v.startFrames = 0
				logger.Debug("[VADAnalyzer] QUIET → SPEAKING (confidence=%.3f, volume=%.3f)",
					voiceConfidence, v.smoothedVolume)
			} else {
				v.state = VADStateStarting
			}
		}

	case VADStateStarting:
		if voiceConfidence >= v.params.Confidence {
			v.startFrames++
			if v.startFrames >= v.startThreshold {
				v.state = VADStateSpeaking
				v.startFrames = 0
				logger.Debug("[VADAnalyzer] STARTING → SPEAKING (confidence=%.3f, volume=%.3f)",
					voiceConfidence, v.smoothedVolume)
			}
		} else {
			v.state = VADStateQuiet
			v.startFrames = 0
		}

	case VADStateSpeaking:
		if voiceConfidence < v.params.Confidence {
			v.stopFrames++
			if v.stopFrames >= v.stopThreshold {
				v.state = VADStateQuiet
				v.stopFrames = 0
				logger.Debug("[VADAnalyzer] SPEAKING → QUIET (confidence=%.3f, volume=%.3f)",
					voiceConfidence, v.smoothedVolume)
			} else {
				v.state = VADStateStopping
			}
		} else {
			v.stopFrames = 0
		}

	case VADStateStopping:
		if voiceConfidence < v.params.Confidence {
			v.stopFrames++
			if v.stopFrames >= v.stopThreshold {
				v.state = VADStateQuiet
				v.stopFrames = 0
				logger.Debug("[VADAnalyzer] STOPPING → QUIET (confidence=%.3f, volume=%.3f)",
					voiceConfidence, v.smoothedVolume)
			}
		} else {
			v.state = VADStateSpeaking
			v.stopFrames = 0
			logger.Debug("[VADAnalyzer] STOPPING → SPEAKING (confidence=%.3f, volume=%.3f)",
				voiceConfidence, v.smoothedVolume)
		}
	}

	if oldState != v.state {
		logger.Info("[VADAnalyzer] State transition: %s → %s", oldState, v.state)
	}

	return v.state, nil
}

// calculateVolume computes RMS volume from int16 audio buffer
func (v *BaseVADAnalyzer) calculateVolume(buffer []byte) float32 {
	if len(buffer) < 2 {
		return 0.0
	}

	// Convert bytes to int16 samples
	numSamples := len(buffer) / 2
	var sumSquares float64

	for i := 0; i < numSamples; i++ {
		// Read little-endian int16
		sample := int16(buffer[i*2]) | int16(buffer[i*2+1])<<8
		// Normalize to [-1.0, 1.0]
		normalized := float64(sample) / 32768.0
		sumSquares += normalized * normalized
	}

	// RMS (Root Mean Square)
	rms := math.Sqrt(sumSquares / float64(numSamples))
	return float32(rms)
}

// VADProcessor is a frame processor that uses VAD to detect user speech
type VADProcessor struct {
	analyzer       VADAnalyzer
	onSpeakingFunc func()
	onQuietFunc    func()

	mu     sync.Mutex
	ctx    context.Context
	cancel context.CancelFunc
}

// NewVADProcessor creates a new VAD processor
func NewVADProcessor(analyzer VADAnalyzer, onSpeaking, onQuiet func()) *VADProcessor {
	return &VADProcessor{
		analyzer:       analyzer,
		onSpeakingFunc: onSpeaking,
		onQuietFunc:    onQuiet,
	}
}

// Start initializes the VAD processor
func (p *VADProcessor) Start(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.ctx, p.cancel = context.WithCancel(ctx)
	logger.Info("[VADProcessor] Started")
	return nil
}

// Stop stops the VAD processor
func (p *VADProcessor) Stop() error {
	p.mu.Lock()
	if p.cancel != nil {
		p.cancel()
	}
	p.mu.Unlock()

	logger.Info("[VADProcessor] Stopped")
	return nil
}

// ProcessAudio processes audio through the VAD analyzer
func (p *VADProcessor) ProcessAudio(buffer []byte) error {
	state, err := p.analyzer.AnalyzeAudio(buffer)
	if err != nil {
		return fmt.Errorf("VAD analysis error: %w", err)
	}

	// Trigger callbacks on state transitions
	switch state {
	case VADStateSpeaking:
		if p.onSpeakingFunc != nil {
			p.onSpeakingFunc()
		}
	case VADStateQuiet:
		if p.onQuietFunc != nil {
			p.onQuietFunc()
		}
	}

	return nil
}
