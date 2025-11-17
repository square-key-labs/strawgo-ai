package interruptions

import (
	"sync"
	"time"
)

// VADBasedInterruptionStrategy detects user interruption based on Voice Activity Detection
// This strategy triggers interruption when it detects sustained voice activity
// indicating the user is actively speaking (not just noise)
type VADBasedInterruptionStrategy struct {
	BaseInterruptionStrategy

	// Configuration
	minDuration      time.Duration // Minimum speech duration to trigger (e.g., 300ms)
	energyThreshold  float64       // Energy threshold for voice detection
	zeroCrossRate    float64       // Zero-crossing rate threshold

	// State
	speechStartTime  time.Time // When speech was first detected
	isSpeaking       bool      // Currently detecting speech
	lastAudioTime    time.Time // Last time audio was received
	mu               sync.Mutex
}

// VADBasedInterruptionStrategyParams holds configuration for VAD-based interruption
type VADBasedInterruptionStrategyParams struct {
	MinDuration     time.Duration // Minimum speech duration (default: 300ms)
	EnergyThreshold float64       // Energy threshold (default: 0.02)
	ZeroCrossRate   float64       // ZCR threshold (default: 0.1)
}

// NewVADBasedInterruptionStrategy creates a new VAD-based interruption strategy
func NewVADBasedInterruptionStrategy(params *VADBasedInterruptionStrategyParams) *VADBasedInterruptionStrategy {
	if params == nil {
		params = &VADBasedInterruptionStrategyParams{
			MinDuration:     300 * time.Millisecond,
			EnergyThreshold: 0.02,
			ZeroCrossRate:   0.1,
		}
	}

	return &VADBasedInterruptionStrategy{
		minDuration:     params.MinDuration,
		energyThreshold: params.EnergyThreshold,
		zeroCrossRate:   params.ZeroCrossRate,
		isSpeaking:      false,
	}
}

// AppendAudio analyzes the audio frame for voice activity
func (v *VADBasedInterruptionStrategy) AppendAudio(audio []byte, sampleRate int) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.lastAudioTime = time.Now()

	// Calculate audio features
	energy := calculateEnergy(audio)
	zcr := calculateZeroCrossingRate(audio)

	// Detect voice activity (simple heuristic)
	hasVoice := energy > v.energyThreshold && zcr > v.zeroCrossRate

	if hasVoice {
		if !v.isSpeaking {
			// Speech just started
			v.isSpeaking = true
			v.speechStartTime = time.Now()
		}
	} else {
		// No voice detected, reset
		v.isSpeaking = false
	}

	return nil
}

// ShouldInterrupt determines if sustained speech has been detected
func (v *VADBasedInterruptionStrategy) ShouldInterrupt() (bool, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if !v.isSpeaking {
		return false, nil
	}

	// Check if speech has been sustained long enough
	duration := time.Since(v.speechStartTime)
	return duration >= v.minDuration, nil
}

// Reset clears the speech detection state
func (v *VADBasedInterruptionStrategy) Reset() error {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.isSpeaking = false
	v.speechStartTime = time.Time{}

	return nil
}

// calculateEnergy computes the short-term energy of audio samples
func calculateEnergy(audio []byte) float64 {
	return calculateRMS(audio) // Reuse RMS calculation
}

// calculateZeroCrossingRate computes the zero-crossing rate
// ZCR measures how often the signal changes sign (crosses zero)
// Speech typically has different ZCR patterns than noise
func calculateZeroCrossingRate(audio []byte) float64 {
	if len(audio) < 4 {
		return 0.0
	}

	zeroCrossings := 0
	prevSign := false

	for i := 0; i+1 < len(audio); i += 2 {
		// Read 16-bit sample
		sample := int16(uint16(audio[i]) | uint16(audio[i+1])<<8)

		currentSign := sample >= 0
		if i > 0 && currentSign != prevSign {
			zeroCrossings++
		}
		prevSign = currentSign
	}

	numSamples := len(audio) / 2
	if numSamples == 0 {
		return 0.0
	}

	return float64(zeroCrossings) / float64(numSamples)
}
