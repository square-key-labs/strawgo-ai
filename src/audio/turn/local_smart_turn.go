package turn

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/models"
)

// LocalSmartTurnConfig holds configuration for local ONNX Smart Turn
type LocalSmartTurnConfig struct {
	// ModelPath is the path to the ONNX model file
	// If empty, will look for bundled model in standard locations
	ModelPath string
	// CPUCount is the number of CPU threads for inference (default: 1)
	CPUCount int
	// Params are the smart turn parameters
	Params *SmartTurnParams
}

// LocalSmartTurn implements smart turn detection using local ONNX inference
// This uses the Smart Turn v3.1 model which is based on Whisper Tiny
type LocalSmartTurn struct {
	*BaseTurnAnalyzer
	config           *LocalSmartTurnConfig
	params           *SmartTurnParams
	featureExtractor *WhisperFeatureExtractor
	session          *ort.DynamicAdvancedSession
	inputShape       ort.Shape
	outputShape      ort.Shape
	mu               sync.Mutex
	initialized      bool
}

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

// initONNXRuntime initializes the ONNX runtime library once
func initONNXRuntime() error {
	ortInitOnce.Do(func() {
		// Try to find the ONNX runtime shared library
		libPath := findONNXRuntimeLib()
		if libPath == "" {
			ortInitErr = fmt.Errorf("ONNX Runtime shared library not found. Please install onnxruntime")
			return
		}

		ort.SetSharedLibraryPath(libPath)
		ortInitErr = ort.InitializeEnvironment()
		// Ignore "already initialized" error - VAD or another component may have initialized it
		if ortInitErr != nil && strings.Contains(ortInitErr.Error(), "already been initialized") {
			logger.Debug("[LocalSmartTurn] ONNX Runtime already initialized by another component")
			ortInitErr = nil
		} else if ortInitErr != nil {
			ortInitErr = fmt.Errorf("failed to initialize ONNX Runtime: %w", ortInitErr)
		}
	})
	return ortInitErr
}

// findONNXRuntimeLib attempts to find the ONNX runtime shared library
func findONNXRuntimeLib() string {
	var libNames []string
	var searchPaths []string

	switch runtime.GOOS {
	case "darwin":
		libNames = []string{"libonnxruntime.dylib", "libonnxruntime.1.22.0.dylib"}
		searchPaths = []string{
			"/usr/local/lib",
			"/opt/homebrew/lib",
			"/opt/local/lib",
			os.Getenv("HOME") + "/lib",
			".",
		}
	case "linux":
		libNames = []string{"libonnxruntime.so", "libonnxruntime.so.1.22.0"}
		searchPaths = []string{
			"/usr/local/lib",
			"/usr/lib",
			"/usr/lib/x86_64-linux-gnu",
			"/usr/lib/aarch64-linux-gnu",
			os.Getenv("HOME") + "/lib",
			".",
		}
	case "windows":
		libNames = []string{"onnxruntime.dll"}
		searchPaths = []string{
			os.Getenv("ProgramFiles") + "\\onnxruntime\\lib",
			".",
		}
	}

	// Check ORT_LIB_PATH environment variable first
	if envPath := os.Getenv("ORT_LIB_PATH"); envPath != "" {
		if _, err := os.Stat(envPath); err == nil {
			return envPath
		}
		// Also check if it's a directory
		for _, name := range libNames {
			fullPath := filepath.Join(envPath, name)
			if _, err := os.Stat(fullPath); err == nil {
				return fullPath
			}
		}
	}

	// Search standard paths
	for _, searchPath := range searchPaths {
		for _, libName := range libNames {
			fullPath := filepath.Join(searchPath, libName)
			if _, err := os.Stat(fullPath); err == nil {
				return fullPath
			}
		}
	}

	return ""
}

// NewLocalSmartTurn creates a new local ONNX Smart Turn analyzer
func NewLocalSmartTurn(config LocalSmartTurnConfig) (*LocalSmartTurn, error) {
	if config.Params == nil {
		config.Params = DefaultSmartTurnParams()
	}
	if config.CPUCount <= 0 {
		config.CPUCount = 1
	}

	// Initialize ONNX Runtime
	if err := initONNXRuntime(); err != nil {
		return nil, err
	}

	// Find model file
	modelPath := config.ModelPath
	if modelPath == "" {
		modelPath = findSmartTurnModel()
		if modelPath == "" {
			// Auto-download from HuggingFace on first use
			var dlErr error
			modelPath, dlErr = models.EnsureModel(models.SmartTurnURL, models.SmartTurnFile)
			if dlErr != nil {
				return nil, fmt.Errorf("Smart Turn model not found and auto-download failed: %w", dlErr)
			}
		}
	}

	// Verify model file exists
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model file not found: %s", modelPath)
	}

	logger.Info("[LocalSmartTurn] Loading model from %s...", modelPath)

	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Set thread count
	if err := options.SetIntraOpNumThreads(config.CPUCount); err != nil {
		return nil, fmt.Errorf("failed to set thread count: %w", err)
	}

	// Enable all optimizations
	if err := options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return nil, fmt.Errorf("failed to set optimization level: %w", err)
	}

	// Smart Turn v3.1 expects input shape: (batch, n_mels=80, n_frames=800)
	// For 8 seconds of audio at 16kHz with hop_length=160: 8*16000/160 = 800 frames
	inputShape := ort.Shape{1, WhisperNMels, 800}
	outputShape := ort.Shape{1, 1}

	// Create ONNX session
	// Note: Smart Turn v3.1 model uses "logits" as output name (not "output")
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_features"},
		[]string{"logits"},
		options,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	baseParams := &TurnAnalyzerParams{
		StopSecs: config.Params.StopSecs,
	}

	lst := &LocalSmartTurn{
		BaseTurnAnalyzer: NewBaseTurnAnalyzer(WhisperSampleRate, baseParams),
		config:           &config,
		params:           config.Params,
		featureExtractor: NewWhisperFeatureExtractor(),
		session:          session,
		inputShape:       inputShape,
		outputShape:      outputShape,
		initialized:      true,
	}

	logger.Info("[LocalSmartTurn] Model loaded successfully (cpu_threads=%d)", config.CPUCount)
	return lst, nil
}

// findSmartTurnModel looks for the Smart Turn model in standard locations
func findSmartTurnModel() string {
	modelNames := []string{
		"smart-turn-v3.1-cpu.onnx",
		"smart-turn-v3.0.onnx",
	}

	searchPaths := []string{
		"./models",
		"./",
		os.Getenv("HOME") + "/models",
		os.Getenv("HOME") + "/.local/share/strawgo/models",
		"/usr/local/share/strawgo/models",
	}

	// Check SMART_TURN_MODEL_PATH environment variable
	if envPath := os.Getenv("SMART_TURN_MODEL_PATH"); envPath != "" {
		if _, err := os.Stat(envPath); err == nil {
			return envPath
		}
	}

	for _, searchPath := range searchPaths {
		for _, modelName := range modelNames {
			fullPath := filepath.Join(searchPath, modelName)
			if _, err := os.Stat(fullPath); err == nil {
				return fullPath
			}
		}
	}

	return ""
}

// AnalyzeEndOfTurn runs ONNX inference to determine if the turn has ended
func (lst *LocalSmartTurn) AnalyzeEndOfTurn() (EndOfTurnState, *TurnMetrics, error) {
	lst.mu.Lock()
	defer lst.mu.Unlock()

	if !lst.initialized {
		return TurnIncomplete, nil, fmt.Errorf("LocalSmartTurn not initialized")
	}

	startTime := time.Now()

	// Get audio segment for analysis
	segment := lst.GetAudioSegment(lst.params.MaxDurationSecs, lst.params.PreSpeechMs)
	if len(segment) == 0 {
		logger.Debug("[LocalSmartTurn] Empty audio segment, skipping prediction")
		return TurnIncomplete, nil, nil
	}

	// Resample to 16kHz if needed (Whisper features require 16kHz)
	// Convert float32 to int16 for resampling
	var pcm []int16
	for _, sample := range segment {
		pcm = append(pcm, int16(sample*32767))
	}

	// Resample if input is not 16kHz
	if lst.sampleRate != 16000 {
		logger.Debug("[LocalSmartTurn] Resampling from %dHz to 16kHz", lst.sampleRate)
		pcm = resample(pcm, lst.sampleRate, 16000)
	}

	// Convert back to float32 for feature extraction
	resampledSegment := make([]float32, len(pcm))
	for i, sample := range pcm {
		resampledSegment[i] = float32(sample) / 32767.0
	}

	// Extract mel spectrogram features
	// For 8 seconds at 16kHz = 128,000 samples
	maxSamples := int(lst.params.MaxDurationSecs * float64(16000))
	features := lst.featureExtractor.Extract(resampledSegment, maxSamples)

	// Create input tensor
	inputTensor, err := ort.NewTensor(lst.inputShape, features)
	if err != nil {
		return TurnIncomplete, nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Prepare input/output slices for DynamicAdvancedSession
	// Use nil for output to let ONNX runtime allocate it
	inputs := []ort.Value{inputTensor}
	outputs := []ort.Value{nil}

	// Run inference
	inferenceStart := time.Now()
	err = lst.session.Run(inputs, outputs)
	if err != nil {
		return TurnIncomplete, nil, fmt.Errorf("ONNX inference failed: %w", err)
	}
	inferenceTimeMs := float64(time.Since(inferenceStart).Microseconds()) / 1000.0

	// Get output tensor and extract probability
	outputTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return TurnIncomplete, nil, fmt.Errorf("unexpected output tensor type")
	}
	defer outputTensor.Destroy()

	// Get probability from output (sigmoid already applied by model)
	outputData := outputTensor.GetData()
	if len(outputData) == 0 {
		return TurnIncomplete, nil, fmt.Errorf("empty output from model")
	}
	probability := float64(outputData[0])

	// Make prediction (1 for Complete, 0 for Incomplete)
	prediction := 0
	state := TurnIncomplete
	if probability > 0.5 {
		prediction = 1
		state = TurnComplete
		lst.clearState(state)
	}

	totalTimeMs := float64(time.Since(startTime).Microseconds()) / 1000.0

	metrics := &TurnMetrics{
		IsComplete:      prediction == 1,
		Probability:     probability,
		InferenceTimeMs: inferenceTimeMs,
		TotalTimeMs:     totalTimeMs,
	}

	logger.Info("[LocalSmartTurn] Prediction: %s (probability: %.3f, inference: %.1fms, total: %.1fms)",
		state.String(), probability, inferenceTimeMs, totalTimeMs)

	return state, metrics, nil
}

// Close releases resources used by the LocalSmartTurn analyzer
func (lst *LocalSmartTurn) Close() error {
	lst.mu.Lock()
	defer lst.mu.Unlock()

	if lst.session != nil {
		if err := lst.session.Destroy(); err != nil {
			return fmt.Errorf("failed to destroy ONNX session: %w", err)
		}
		lst.session = nil
	}

	lst.initialized = false
	logger.Info("[LocalSmartTurn] Resources released")
	return nil
}
