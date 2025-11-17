package interruptions

import (
	"encoding/binary"
	"math"
	"sync"
)

// VolumeInterruptionStrategy detects user interruption based on audio volume
// This is an audio-based strategy that analyzes the RMS (Root Mean Square) volume
// of incoming audio frames to detect when the user starts speaking
type VolumeInterruptionStrategy struct {
	BaseInterruptionStrategy

	// Configuration
	threshold    float64 // RMS volume threshold (0.0 - 1.0)
	windowSize   int     // Number of audio frames to analyze
	minFrames    int     // Minimum frames above threshold to trigger

	// State
	volumes      []float64 // Recent RMS volumes
	framesAbove  int       // Count of frames above threshold
	mu           sync.Mutex
}

// VolumeInterruptionStrategyParams holds configuration for volume-based interruption
type VolumeInterruptionStrategyParams struct {
	Threshold  float64 // RMS volume threshold (default: 0.02)
	WindowSize int     // Frames to analyze (default: 10)
	MinFrames  int     // Min frames above threshold (default: 3)
}

// NewVolumeInterruptionStrategy creates a new volume-based interruption strategy
func NewVolumeInterruptionStrategy(params *VolumeInterruptionStrategyParams) *VolumeInterruptionStrategy {
	if params == nil {
		params = &VolumeInterruptionStrategyParams{
			Threshold:  0.02,  // Low threshold for sensitive detection
			WindowSize: 10,    // Analyze last 10 frames (~200ms at 20ms/frame)
			MinFrames:  3,     // Need 3+ frames above threshold
		}
	}

	return &VolumeInterruptionStrategy{
		threshold:   params.Threshold,
		windowSize:  params.WindowSize,
		minFrames:   params.MinFrames,
		volumes:     make([]float64, 0, params.WindowSize),
		framesAbove: 0,
	}
}

// AppendAudio analyzes the audio frame and updates volume statistics
func (v *VolumeInterruptionStrategy) AppendAudio(audio []byte, sampleRate int) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Calculate RMS volume from audio samples
	rms := calculateRMS(audio)

	// Add to rolling window
	v.volumes = append(v.volumes, rms)
	if len(v.volumes) > v.windowSize {
		v.volumes = v.volumes[1:]
	}

	// Count frames above threshold in current window
	v.framesAbove = 0
	for _, vol := range v.volumes {
		if vol > v.threshold {
			v.framesAbove++
		}
	}

	return nil
}

// ShouldInterrupt determines if the user is speaking based on volume analysis
func (v *VolumeInterruptionStrategy) ShouldInterrupt() (bool, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Need enough frames in window
	if len(v.volumes) < v.minFrames {
		return false, nil
	}

	// Check if enough frames are above threshold
	shouldInterrupt := v.framesAbove >= v.minFrames

	return shouldInterrupt, nil
}

// Reset clears the volume history
func (v *VolumeInterruptionStrategy) Reset() error {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.volumes = make([]float64, 0, v.windowSize)
	v.framesAbove = 0

	return nil
}

// calculateRMS computes the Root Mean Square (RMS) volume of audio samples
// RMS is a measure of the average "power" or "loudness" of the audio
func calculateRMS(audio []byte) float64 {
	if len(audio) == 0 {
		return 0.0
	}

	var sumSquares float64
	numSamples := 0

	// Assume 16-bit linear PCM (most common)
	// Each sample is 2 bytes (int16)
	for i := 0; i+1 < len(audio); i += 2 {
		// Read 16-bit sample (little-endian)
		sample := int16(binary.LittleEndian.Uint16(audio[i : i+2]))

		// Normalize to -1.0 to 1.0
		normalized := float64(sample) / 32768.0

		// Square and accumulate
		sumSquares += normalized * normalized
		numSamples++
	}

	if numSamples == 0 {
		return 0.0
	}

	// Calculate mean of squares
	meanSquare := sumSquares / float64(numSamples)

	// Return square root (RMS)
	return math.Sqrt(meanSquare)
}
