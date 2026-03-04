package turn

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// getModelPath returns the path to the Smart Turn model for testing.
// Uses auto-download via models.EnsureModel if not found locally.
func getModelPath(t *testing.T) string {
	// Try local paths first
	paths := []string{
		"./models/smart-turn-v3.1-cpu.onnx",
		"models/smart-turn-v3.1-cpu.onnx",
		"../../../models/smart-turn-v3.1-cpu.onnx",
	}

	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			abs, _ := filepath.Abs(path)
			t.Logf("Using model path: %s", abs)
			return path
		}
	}

	// Not found locally — return empty to trigger auto-download in NewLocalSmartTurn
	t.Log("Model not found locally, will auto-download from HuggingFace")
	return ""
}

// TestLocalSmartTurn_8kHzInput tests that 8kHz audio is properly resampled to 16kHz
// before feature extraction, preventing NaN or panic in the model
func TestLocalSmartTurn_8kHzInput(t *testing.T) {
	// Create analyzer with default config
	config := LocalSmartTurnConfig{
		ModelPath: getModelPath(t),
		Params:    DefaultSmartTurnParams(),
	}

	analyzer, err := NewLocalSmartTurn(config)
	if err != nil {
		t.Fatalf("Failed to create LocalSmartTurn: %v", err)
	}
	defer analyzer.Close()

	// Override sample rate to 8kHz
	analyzer.SetSampleRate(8000)

	// Create 8kHz audio data (1 second of silence)
	// 8000 samples at 8kHz = 1 second
	sampleRate := 8000
	duration := 1.0 // 1 second
	numSamples := int(float64(sampleRate) * duration)

	// Create int16 PCM data
	pcmData := make([]int16, numSamples)
	for i := 0; i < numSamples; i++ {
		pcmData[i] = int16(i % 100) // Small values to simulate audio
	}

	// Convert to bytes (little-endian)
	audioBytes := make([]byte, numSamples*2)
	for i := 0; i < numSamples; i++ {
		binary.LittleEndian.PutUint16(audioBytes[i*2:], uint16(pcmData[i]))
	}

	// Add audio frames to the analyzer
	// We need to add enough audio to trigger analysis
	for i := 0; i < 10; i++ {
		analyzer.AppendAudio(audioBytes, true) // true = speech detected
	}

	// Run analysis
	state, metrics, err := analyzer.AnalyzeEndOfTurn()

	// Verify no error occurred
	if err != nil {
		t.Fatalf("AnalyzeEndOfTurn failed: %v", err)
	}

	// Verify metrics are valid
	if metrics == nil {
		t.Fatalf("Expected metrics, got nil")
	}

	// Verify probability is valid (not NaN, not Inf, between 0 and 1)
	if math.IsNaN(metrics.Probability) {
		t.Fatalf("Probability is NaN - resampling may have failed")
	}
	if math.IsInf(metrics.Probability, 0) {
		t.Fatalf("Probability is Inf - resampling may have failed")
	}
	if metrics.Probability < 0.0 || metrics.Probability > 1.0 {
		t.Fatalf("Probability out of range [0, 1]: %f", metrics.Probability)
	}

	// Verify state is valid
	if state != TurnComplete && state != TurnIncomplete {
		t.Fatalf("Invalid turn state: %v", state)
	}

	// Verify timing metrics are reasonable
	if metrics.InferenceTimeMs < 0 {
		t.Fatalf("Negative inference time: %f", metrics.InferenceTimeMs)
	}
	if metrics.TotalTimeMs < 0 {
		t.Fatalf("Negative total time: %f", metrics.TotalTimeMs)
	}

	t.Logf("8kHz input test passed: state=%v, probability=%.3f, inference_time=%.1fms",
		state, metrics.Probability, metrics.InferenceTimeMs)
}

// TestLocalSmartTurn_16kHzInput tests that 16kHz audio (native sample rate) works correctly
func TestLocalSmartTurn_16kHzInput(t *testing.T) {
	// Create analyzer with default config
	config := LocalSmartTurnConfig{
		ModelPath: getModelPath(t),
		Params:    DefaultSmartTurnParams(),
	}

	analyzer, err := NewLocalSmartTurn(config)
	if err != nil {
		t.Fatalf("Failed to create LocalSmartTurn: %v", err)
	}
	defer analyzer.Close()

	// Sample rate is already 16kHz by default
	sampleRate := 16000
	duration := 1.0 // 1 second
	numSamples := int(float64(sampleRate) * duration)

	// Create int16 PCM data
	pcmData := make([]int16, numSamples)
	for i := 0; i < numSamples; i++ {
		pcmData[i] = int16(i % 100)
	}

	// Convert to bytes (little-endian)
	audioBytes := make([]byte, numSamples*2)
	for i := 0; i < numSamples; i++ {
		binary.LittleEndian.PutUint16(audioBytes[i*2:], uint16(pcmData[i]))
	}

	// Add audio frames to the analyzer
	for i := 0; i < 10; i++ {
		analyzer.AppendAudio(audioBytes, true)
	}

	// Run analysis
	state, metrics, err := analyzer.AnalyzeEndOfTurn()

	// Verify no error occurred
	if err != nil {
		t.Fatalf("AnalyzeEndOfTurn failed: %v", err)
	}

	// Verify metrics are valid
	if metrics == nil {
		t.Fatalf("Expected metrics, got nil")
	}

	// Verify probability is valid
	if math.IsNaN(metrics.Probability) {
		t.Fatalf("Probability is NaN")
	}
	if math.IsInf(metrics.Probability, 0) {
		t.Fatalf("Probability is Inf")
	}
	if metrics.Probability < 0.0 || metrics.Probability > 1.0 {
		t.Fatalf("Probability out of range [0, 1]: %f", metrics.Probability)
	}

	// Verify state is valid
	if state != TurnComplete && state != TurnIncomplete {
		t.Fatalf("Invalid turn state: %v", state)
	}

	t.Logf("16kHz input test passed: state=%v, probability=%.3f, inference_time=%.1fms",
		state, metrics.Probability, metrics.InferenceTimeMs)
}

// TestLocalSmartTurn_Initialization tests that the analyzer initializes correctly
func TestLocalSmartTurn_Initialization(t *testing.T) {
	config := LocalSmartTurnConfig{
		ModelPath: getModelPath(t),
		Params:    DefaultSmartTurnParams(),
	}

	analyzer, err := NewLocalSmartTurn(config)
	if err != nil {
		t.Fatalf("Failed to create LocalSmartTurn: %v", err)
	}
	defer analyzer.Close()

	// Verify analyzer is initialized
	if analyzer == nil {
		t.Fatalf("Analyzer is nil")
	}

	// Verify we can call AnalyzeEndOfTurn without audio (should return TurnIncomplete)
	state, metrics, err := analyzer.AnalyzeEndOfTurn()

	// Empty audio should return nil metrics and nil error
	if metrics != nil {
		t.Fatalf("Expected nil metrics for empty audio, got %v", metrics)
	}

	if state != TurnIncomplete {
		t.Fatalf("Expected TurnIncomplete for empty audio, got %v", state)
	}

	t.Logf("Initialization test passed")
}
