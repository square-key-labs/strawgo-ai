package vad

import (
	"fmt"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// SileroVADAnalyzer implements VAD using the Rust onnx-worker via Unix socket.
// Each instance maintains a persistent connection to the worker; the worker
// creates a new SileroSession (independent hidden state) per connection.
type SileroVADAnalyzer struct {
	*BaseVADAnalyzer
	client   *OnnxVADClient
	sockPath string
	mu       sync.Mutex

	// Debug logging — log every N frames to avoid spam
	frameCount      int
	logEveryNFrames int
}

// NewSileroVADAnalyzer creates a new Silero VAD analyzer backed by the Rust
// onnx-worker reachable at sockPath (Unix socket path).
func NewSileroVADAnalyzer(sampleRate int, params VADParams, sockPath string) (*SileroVADAnalyzer, error) {
	client, err := NewOnnxVADClient(sockPath)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to onnx-worker at %s: %w", sockPath, err)
	}

	base := NewBaseVADAnalyzer(sampleRate, params)

	return &SileroVADAnalyzer{
		BaseVADAnalyzer: base,
		client:          client,
		sockPath:        sockPath,
		logEveryNFrames: 50,
	}, nil
}

// SetSampleRate validates and sets the audio sample rate.
func (v *SileroVADAnalyzer) SetSampleRate(sampleRate int) error {
	if sampleRate != 8000 && sampleRate != 16000 {
		return fmt.Errorf("Silero VAD requires 8000 or 16000 Hz (got %d)", sampleRate)
	}
	return v.BaseVADAnalyzer.SetSampleRate(sampleRate)
}

// NumFramesRequired returns the number of audio frames required per analysis window.
func (v *SileroVADAnalyzer) NumFramesRequired() int {
	if v.GetSampleRate() == 16000 {
		return 512
	}
	return 256
}

// VoiceConfidence sends the audio buffer to the onnx-worker and returns the
// voice confidence score in [0.0, 1.0].
func (v *SileroVADAnalyzer) VoiceConfidence(buffer []byte) float32 {
	v.mu.Lock()
	defer v.mu.Unlock()

	confidence, err := v.client.VoiceConfidence(buffer, v.GetSampleRate())
	if err != nil {
		logger.Error("[SileroVAD] onnx-worker error: %v", err)
		return 0.0
	}
	return confidence
}

// AnalyzeAudio processes audio and returns the current VAD state.
func (v *SileroVADAnalyzer) AnalyzeAudio(buffer []byte) (VADState, error) {
	confidence := v.VoiceConfidence(buffer)
	numFramesRequired := v.NumFramesRequired()

	state, err := v.ProcessAudio(buffer, confidence, numFramesRequired)
	if err != nil {
		return VADStateQuiet, err
	}

	v.frameCount++
	if v.logEveryNFrames > 0 && v.frameCount%v.logEveryNFrames == 0 {
		params := v.GetParams()
		logger.Debug("[SileroVAD] Stats: confidence=%.3f (thresh=%.2f), state=%s",
			confidence, params.Confidence, state.String())
	}

	return state, nil
}

// Restart closes the current connection (resetting the Rust-side hidden state)
// and opens a fresh one, then resets the base state machine.
func (v *SileroVADAnalyzer) Restart() {
	v.mu.Lock()
	if v.client != nil {
		v.client.Close()
	}
	client, err := NewOnnxVADClient(v.sockPath)
	if err != nil {
		logger.Error("[SileroVAD] Restart: failed to reconnect: %v", err)
	} else {
		v.client = client
	}
	v.mu.Unlock()

	v.BaseVADAnalyzer.Restart()
	logger.Info("[SileroVAD] Analyzer restarted (new connection = fresh hidden state)")
}

// Cleanup releases the underlying Unix socket connection.
func (v *SileroVADAnalyzer) Cleanup() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.client != nil {
		return v.client.Close()
	}
	return nil
}
