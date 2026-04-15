package turn

import (
	"encoding/binary"
	"fmt"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// OnnxSmartTurnConfig holds configuration for ONNX-based Smart Turn via Unix socket.
type OnnxSmartTurnConfig struct {
	SockPath string           // path to onnx-worker Unix socket
	Params   *SmartTurnParams // nil = use defaults
}

// OnnxSmartTurn implements TurnAnalyzer using the Rust onnx-worker sidecar.
type OnnxSmartTurn struct {
	*BaseTurnAnalyzer
	client *OnnxTurnClient
	params *SmartTurnParams
	mu     sync.Mutex
}

// NewOnnxSmartTurn creates a new OnnxSmartTurn analyzer.
func NewOnnxSmartTurn(config OnnxSmartTurnConfig) (*OnnxSmartTurn, error) {
	if config.Params == nil {
		config.Params = DefaultSmartTurnParams()
	}
	client, err := NewOnnxTurnClient(config.SockPath)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to onnx-worker at %s: %w", config.SockPath, err)
	}
	baseParams := &TurnAnalyzerParams{StopSecs: config.Params.StopSecs}
	return &OnnxSmartTurn{
		BaseTurnAnalyzer: NewBaseTurnAnalyzer(16000, baseParams),
		client:           client,
		params:           config.Params,
	}, nil
}

// AnalyzeEndOfTurn runs inference via the onnx-worker sidecar to determine if the turn has ended.
func (o *OnnxSmartTurn) AnalyzeEndOfTurn() (EndOfTurnState, *TurnMetrics, error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	startTime := time.Now()

	// Get audio segment from base (float32 [-1,1])
	segment := o.GetAudioSegment(o.params.MaxDurationSecs, o.params.PreSpeechMs)
	if len(segment) == 0 {
		return TurnIncomplete, nil, nil
	}

	// Convert float32 → int16 LE bytes for the wire protocol
	pcmBytes := make([]byte, len(segment)*2)
	for i, s := range segment {
		// Clamp
		if s > 1.0 {
			s = 1.0
		}
		if s < -1.0 {
			s = -1.0
		}
		v := int16(s * 32767.0)
		binary.LittleEndian.PutUint16(pcmBytes[i*2:], uint16(v))
	}

	inferenceStart := time.Now()
	pred, err := o.client.Analyze(pcmBytes, o.sampleRate, 0)
	if err != nil {
		logger.Error("[OnnxSmartTurn] onnx-worker error: %v", err)
		return TurnIncomplete, nil, err
	}
	inferenceMs := float64(time.Since(inferenceStart).Microseconds()) / 1000.0
	totalMs := float64(time.Since(startTime).Microseconds()) / 1000.0

	state := TurnIncomplete
	isComplete := pred.Probability > 0.5
	if isComplete {
		state = TurnComplete
		o.BaseTurnAnalyzer.Clear()
	}

	metrics := &TurnMetrics{
		IsComplete:      isComplete,
		Probability:     float64(pred.Probability),
		InferenceTimeMs: inferenceMs,
		TotalTimeMs:     totalMs,
	}

	logger.Info("[OnnxSmartTurn] Prediction: %s (prob=%.3f, inference=%.1fms, total=%.1fms)",
		state, pred.Probability, inferenceMs, totalMs)

	return state, metrics, nil
}

// Close closes the connection to the onnx-worker.
func (o *OnnxSmartTurn) Close() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.client != nil {
		return o.client.Close()
	}
	return nil
}
