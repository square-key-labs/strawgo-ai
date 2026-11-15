package vad

import (
	"fmt"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type SileroVADAnalyzer struct {
	*BaseVADAnalyzer
	client   *OnnxVADClient
	sockPath string
	mu       sync.Mutex
}

func NewSileroVADAnalyzer(sampleRate int, params VADParams, sockPath string) (*SileroVADAnalyzer, error) {
	client, err := NewOnnxVADClient(sockPath)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to onnx-worker at %s: %w", sockPath, err)
	}
	return &SileroVADAnalyzer{
		BaseVADAnalyzer: NewBaseVADAnalyzer(sampleRate, params),
		client:          client,
		sockPath:        sockPath,
	}, nil
}

func (v *SileroVADAnalyzer) SetSampleRate(sampleRate int) error {
	if sampleRate != 8000 && sampleRate != 16000 {
		return fmt.Errorf("silero VAD requires 8000 or 16000 Hz (got %d)", sampleRate)
	}
	return v.BaseVADAnalyzer.SetSampleRate(sampleRate)
}

func (v *SileroVADAnalyzer) NumFramesRequired() int {
	if v.GetSampleRate() == 16000 {
		return 512
	}
	return 256
}

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

func (v *SileroVADAnalyzer) AnalyzeAudio(buffer []byte) (VADState, error) {
	confidence := v.VoiceConfidence(buffer)
	return v.ProcessAudio(buffer, confidence, v.NumFramesRequired())
}

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
}

func (v *SileroVADAnalyzer) Cleanup() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.client != nil {
		return v.client.Close()
	}
	return nil
}
