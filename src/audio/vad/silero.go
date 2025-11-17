package vad

import (
	_ "embed"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/logger"
	ort "github.com/yalue/onnxruntime_go"
)

//go:embed data/silero_vad.onnx
var embeddedModelData []byte

const (
	// ModelResetStatesTime: How often to reset internal model state (5 seconds)
	// This prevents memory growth in the ONNX model's internal state
	ModelResetStatesTime = 5.0 * time.Second
)

// SileroOnnxModel wraps the ONNX runtime for Silero VAD model
type SileroOnnxModel struct {
	session *ort.DynamicAdvancedSession

	// Model state (2, 1, 128) - batch_size=1
	state []float32
	// Context buffer (1, context_size) where context_size = 32 for 8kHz, 64 for 16kHz
	context []float32

	lastSampleRate int
	lastResetTime  time.Time

	mu sync.Mutex
}

// NewSileroOnnxModel creates a new Silero ONNX model instance
// Automatically loads the embedded model from package data
func NewSileroOnnxModel() (*SileroOnnxModel, error) {
	// Set ONNX Runtime library path
	// Try project lib directory first, then system paths
	libPath := "lib/libonnxruntime.1.19.2.dylib"
	if _, err := os.Stat(libPath); err == nil {
		ort.SetSharedLibraryPath(libPath)
		logger.Debug("[SileroVAD] Using ONNX Runtime from: %s", libPath)
	} else {
		// Try common system paths
		systemPaths := []string{
			"/usr/local/lib/libonnxruntime.dylib",
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"libonnxruntime.dylib", // Let system find it
		}
		for _, path := range systemPaths {
			if _, err := os.Stat(path); err == nil {
				ort.SetSharedLibraryPath(path)
				logger.Debug("[SileroVAD] Using ONNX Runtime from: %s", path)
				break
			}
		}
	}

	// Initialize ONNX Runtime
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX environment: %w\n\nTo install ONNX Runtime:\n  macOS ARM64: curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-osx-arm64-1.19.2.tgz -o /tmp/onnx.tgz && tar -xzf /tmp/onnx.tgz -C /tmp && sudo cp /tmp/onnxruntime-osx-arm64-1.19.2/lib/libonnxruntime.*.dylib /usr/local/lib/\n  macOS x86_64: curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-osx-x86_64-1.19.2.tgz -o /tmp/onnx.tgz && tar -xzf /tmp/onnx.tgz -C /tmp && sudo cp /tmp/onnxruntime-osx-x86_64-1.19.2/lib/libonnxruntime.*.dylib /usr/local/lib/\n  Linux: sudo apt-get install libonnxruntime-dev", err)
	}

	// Write embedded model to temp file (ONNX Runtime requires file path)
	// Extract the model from package resources
	tmpDir := os.TempDir()
	modelPath := filepath.Join(tmpDir, "silero_vad.onnx")

	if err := os.WriteFile(modelPath, embeddedModelData, 0644); err != nil {
		return nil, fmt.Errorf("failed to write embedded model to temp file: %w", err)
	}

	logger.Debug("[SileroVAD] Extracted embedded model to: %s", modelPath)

	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Set thread options
	if err := options.SetIntraOpNumThreads(1); err != nil {
		return nil, fmt.Errorf("failed to set intra op threads: %w", err)
	}
	if err := options.SetInterOpNumThreads(1); err != nil {
		return nil, fmt.Errorf("failed to set inter op threads: %w", err)
	}

	// Create dynamic session (allows variable-sized inputs)
	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"input", "state", "sr"},
		[]string{"output", "stateN"},
		options)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	model := &SileroOnnxModel{
		session:       session,
		lastResetTime: time.Now(),
	}

	model.ResetStates()

	logger.Info("[SileroVAD] Model loaded successfully from embedded package data")
	return model, nil
}

// ResetStates resets the internal model state
func (m *SileroOnnxModel) ResetStates() {
	m.mu.Lock()
	defer m.mu.Unlock()

	// State shape: (2, 1, 128) - flattened to 256 elements
	m.state = make([]float32, 2*1*128)
	// Context initially empty
	m.context = []float32{}
	m.lastSampleRate = 0
	m.lastResetTime = time.Now()

	logger.Debug("[SileroVAD] Model states reset")
}

// validateInput validates and preprocesses input audio data
func (m *SileroOnnxModel) validateInput(audio []float32, sampleRate int) error {
	// Check sample rate
	if sampleRate != 8000 && sampleRate != 16000 {
		return fmt.Errorf("unsupported sample rate %d (must be 8000 or 16000)", sampleRate)
	}

	// Check number of samples
	numSamples := 512
	if sampleRate == 8000 {
		numSamples = 256
	}

	if len(audio) != numSamples {
		return fmt.Errorf("invalid number of samples %d (expected %d for %d Hz)", len(audio), numSamples, sampleRate)
	}

	// Check if audio chunk is too short
	if float32(sampleRate)/float32(len(audio)) > 31.25 {
		return fmt.Errorf("input audio chunk is too short")
	}

	return nil
}

// Run processes audio input through the VAD model
// Returns voice confidence score (0.0 to 1.0)
func (m *SileroOnnxModel) Run(audio []float32, sampleRate int) (float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Validate input
	if err := m.validateInput(audio, sampleRate); err != nil {
		return 0.0, err
	}

	numSamples := len(audio)
	contextSize := 32
	if sampleRate == 16000 {
		contextSize = 64
	}

	// Reset state if sample rate changed
	if m.lastSampleRate != 0 && m.lastSampleRate != sampleRate {
		logger.Debug("[SileroVAD] Sample rate changed (%d → %d), resetting states", m.lastSampleRate, sampleRate)
		m.state = make([]float32, 2*1*128)
		m.context = []float32{}
	}

	// Initialize context if empty
	if len(m.context) == 0 {
		m.context = make([]float32, contextSize)
	}

	// Concatenate context + audio
	inputAudio := make([]float32, contextSize+numSamples)
	copy(inputAudio, m.context)
	copy(inputAudio[contextSize:], audio)

	// Prepare input tensors
	inputShape := ort.NewShape(1, int64(len(inputAudio))) // (batch=1, samples)
	inputTensor, err := ort.NewTensor(inputShape, inputAudio)
	if err != nil {
		return 0.0, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	stateShape := ort.NewShape(2, 1, 128) // (2, batch=1, hidden=128)
	stateTensor, err := ort.NewTensor(stateShape, m.state)
	if err != nil {
		return 0.0, fmt.Errorf("failed to create state tensor: %w", err)
	}
	defer stateTensor.Destroy()

	// Sample rate tensor (scalar - shape with single dimension of size 1)
	srShape := ort.NewShape(1)
	srTensor, err := ort.NewTensor(srShape, []int64{int64(sampleRate)})
	if err != nil {
		return 0.0, fmt.Errorf("failed to create sr tensor: %w", err)
	}
	defer srTensor.Destroy()

	// Prepare output tensors (will be filled by inference)
	// Output shape is (batch_size, 1) = (1, 1) for single batch
	outputShape := ort.NewShape(1, 1)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return 0.0, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	stateOutShape := ort.NewShape(2, 1, 128) // Same as state input
	stateOutTensor, err := ort.NewEmptyTensor[float32](stateOutShape)
	if err != nil {
		return 0.0, fmt.Errorf("failed to create state output tensor: %w", err)
	}
	defer stateOutTensor.Destroy()

	// Run inference
	err = m.session.Run(
		[]ort.ArbitraryTensor{inputTensor, stateTensor, srTensor},
		[]ort.ArbitraryTensor{outputTensor, stateOutTensor},
	)
	if err != nil {
		return 0.0, fmt.Errorf("failed to run inference: %w", err)
	}

	// Extract output confidence
	outputData := outputTensor.GetData()
	confidence := outputData[0]

	// Extract new state
	stateOutData := stateOutTensor.GetData()
	copy(m.state, stateOutData)

	// Update context (last contextSize samples)
	m.context = inputAudio[len(inputAudio)-contextSize:]

	m.lastSampleRate = sampleRate

	return confidence, nil
}

// Cleanup releases model resources
func (m *SileroOnnxModel) Cleanup() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.session != nil {
		m.session.Destroy()
		m.session = nil
	}

	logger.Info("[SileroVAD] Model cleaned up")
	return nil
}

// SileroVADAnalyzer implements VAD using the Silero ONNX model
type SileroVADAnalyzer struct {
	*BaseVADAnalyzer
	model *SileroOnnxModel
	mu    sync.Mutex
}

// NewSileroVADAnalyzer creates a new Silero VAD analyzer
// Model is automatically loaded from embedded package data (no path required)
func NewSileroVADAnalyzer(sampleRate int, params VADParams) (*SileroVADAnalyzer, error) {
	// Validate sample rate
	if sampleRate != 8000 && sampleRate != 16000 {
		return nil, fmt.Errorf("Silero VAD requires sample rate of 8000 or 16000 Hz (got %d)", sampleRate)
	}

	// Load ONNX model from embedded data
	model, err := NewSileroOnnxModel()
	if err != nil {
		return nil, fmt.Errorf("failed to load Silero model: %w", err)
	}

	base := NewBaseVADAnalyzer(sampleRate, params)

	analyzer := &SileroVADAnalyzer{
		BaseVADAnalyzer: base,
		model:           model,
	}

	logger.Info("[SileroVAD] Analyzer initialized (rate=%d Hz, confidence=%.2f, start=%.2fs, stop=%.2fs)",
		sampleRate, params.Confidence, params.StartSecs, params.StopSecs)

	return analyzer, nil
}

// SetSampleRate sets the sample rate for audio processing
func (v *SileroVADAnalyzer) SetSampleRate(sampleRate int) error {
	if sampleRate != 8000 && sampleRate != 16000 {
		return fmt.Errorf("Silero VAD requires sample rate of 8000 or 16000 Hz (got %d)", sampleRate)
	}
	return v.BaseVADAnalyzer.SetSampleRate(sampleRate)
}

// NumFramesRequired returns the number of audio frames required for analysis
func (v *SileroVADAnalyzer) NumFramesRequired() int {
	sampleRate := v.GetSampleRate()
	if sampleRate == 16000 {
		return 512
	}
	return 256
}

// VoiceConfidence calculates voice activity confidence for the given audio buffer
func (v *SileroVADAnalyzer) VoiceConfidence(buffer []byte) float32 {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Convert int16 buffer to float32 normalized to [-1.0, 1.0]
	numSamples := len(buffer) / 2 // 2 bytes per int16 sample
	audioFloat32 := make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		// Read little-endian int16
		sample := int16(buffer[i*2]) | int16(buffer[i*2+1])<<8
		// Normalize: divide by 32768.0
		audioFloat32[i] = float32(sample) / 32768.0
	}

	// Run model inference
	sampleRate := v.GetSampleRate()
	confidence, err := v.model.Run(audioFloat32, sampleRate)
	if err != nil {
		logger.Error("[SileroVAD] Error running model inference: %v", err)
		return 0.0
	}

	// Periodic model state reset (every 5 seconds)
	// This prevents memory growth in ONNX model's internal state
	if time.Since(v.model.lastResetTime) >= ModelResetStatesTime {
		logger.Debug("[SileroVAD] Resetting model states (5s interval)")
		v.model.ResetStates()
	}

	return confidence
}

// AnalyzeAudio processes audio and returns the current VAD state
func (v *SileroVADAnalyzer) AnalyzeAudio(buffer []byte) (VADState, error) {
	// Get voice confidence from Silero model
	confidence := v.VoiceConfidence(buffer)

	// Run state machine logic
	numFramesRequired := v.NumFramesRequired()
	state, err := v.ProcessAudio(buffer, confidence, numFramesRequired)
	if err != nil {
		return VADStateQuiet, err
	}

	return state, nil
}

// Restart resets the VAD analyzer state and model
func (v *SileroVADAnalyzer) Restart() {
	v.mu.Lock()
	v.model.ResetStates()
	v.mu.Unlock()

	v.BaseVADAnalyzer.Restart()
	logger.Info("[SileroVAD] Analyzer restarted")
}

// Cleanup releases resources
func (v *SileroVADAnalyzer) Cleanup() error {
	return v.model.Cleanup()
}

// calculateVolumeFromFloat32 is a helper to compute RMS volume (for debugging)
func calculateVolumeFromFloat32(audio []float32) float32 {
	if len(audio) == 0 {
		return 0.0
	}

	var sumSquares float64
	for _, sample := range audio {
		sumSquares += float64(sample) * float64(sample)
	}

	rms := math.Sqrt(sumSquares / float64(len(audio)))
	return float32(rms)
}
