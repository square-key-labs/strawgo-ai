package turn

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// SmartTurnParams holds configuration for smart turn analysis
type SmartTurnParams struct {
	// StopSecs is the duration of silence in seconds before triggering silence-based end of turn
	// Default: 3.0
	StopSecs float64
	// PreSpeechMs is milliseconds of audio to include before speech starts
	// Default: 0.0
	PreSpeechMs float64
	// MaxDurationSecs is the maximum duration in seconds for audio segments
	// Default: 8.0 (model limit)
	MaxDurationSecs float64
}

// DefaultSmartTurnParams returns default parameters
func DefaultSmartTurnParams() *SmartTurnParams {
	return &SmartTurnParams{
		StopSecs:        3.0,
		PreSpeechMs:     0.0,
		MaxDurationSecs: 8.0,
	}
}

// HTTPSmartTurnConfig holds configuration for HTTP-based smart turn
type HTTPSmartTurnConfig struct {
	// URL is the endpoint URL (e.g., "https://fal.run/fal-ai/smart-turn")
	URL string
	// APIKey is the authentication key (e.g., Fal API key)
	APIKey string
	// Params are the smart turn parameters
	Params *SmartTurnParams
	// TimeoutSecs is the HTTP request timeout (default: 5)
	TimeoutSecs int
}

// HTTPSmartTurn implements smart turn detection using an HTTP API
// This can be used with Fal.ai hosted endpoint or any compatible API
type HTTPSmartTurn struct {
	*BaseTurnAnalyzer
	config     *HTTPSmartTurnConfig
	params     *SmartTurnParams
	httpClient *http.Client
}

// NewHTTPSmartTurn creates a new HTTP-based smart turn analyzer
func NewHTTPSmartTurn(config HTTPSmartTurnConfig) *HTTPSmartTurn {
	if config.Params == nil {
		config.Params = DefaultSmartTurnParams()
	}
	if config.TimeoutSecs == 0 {
		config.TimeoutSecs = 5
	}
	if config.URL == "" {
		config.URL = "https://fal.run/fal-ai/smart-turn"
	}

	baseParams := &TurnAnalyzerParams{
		StopSecs: config.Params.StopSecs,
	}

	return &HTTPSmartTurn{
		BaseTurnAnalyzer: NewBaseTurnAnalyzer(16000, baseParams), // Smart Turn expects 16kHz
		config:           &config,
		params:           config.Params,
		httpClient: &http.Client{
			Timeout: time.Duration(config.TimeoutSecs) * time.Second,
		},
	}
}

// AnalyzeEndOfTurn runs ML inference via HTTP to determine if the turn has ended
func (h *HTTPSmartTurn) AnalyzeEndOfTurn() (EndOfTurnState, *TurnMetrics, error) {
	startTime := time.Now()

	// Get audio segment for analysis
	segment := h.GetAudioSegment(h.params.MaxDurationSecs, h.params.PreSpeechMs)
	if len(segment) == 0 {
		logger.Debug("[HTTPSmartTurn] Empty audio segment, skipping prediction")
		return TurnIncomplete, nil, nil
	}

	// Convert float32 to int16 PCM for transmission
	pcmData := float32ToInt16PCM(segment)

	// Encode as base64
	audioBase64 := base64.StdEncoding.EncodeToString(pcmData)

	// Build request
	requestBody := map[string]interface{}{
		"audio_base64":  audioBase64,
		"sample_rate":   h.sampleRate,
		"audio_format":  "pcm_s16le",
		"model_version": "v3.1",
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return TurnIncomplete, nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", h.config.URL, bytes.NewReader(bodyBytes))
	if err != nil {
		return TurnIncomplete, nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if h.config.APIKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Key %s", h.config.APIKey))
	}

	// Send request
	resp, err := h.httpClient.Do(req)
	if err != nil {
		return TurnIncomplete, nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return TurnIncomplete, nil, fmt.Errorf("API error (%d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var response struct {
		Prediction  int     `json:"prediction"`  // 1 = complete, 0 = incomplete
		Probability float64 `json:"probability"` // Confidence score
		Metrics     *struct {
			InferenceTime float64 `json:"inference_time"`
			TotalTime     float64 `json:"total_time"`
		} `json:"metrics,omitempty"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return TurnIncomplete, nil, fmt.Errorf("failed to decode response: %w", err)
	}

	totalTimeMs := time.Since(startTime).Seconds() * 1000

	state := TurnIncomplete
	if response.Prediction == 1 {
		state = TurnComplete
		h.clearState(state)
	}

	metrics := &TurnMetrics{
		IsComplete:  response.Prediction == 1,
		Probability: response.Probability,
		TotalTimeMs: totalTimeMs,
	}

	if response.Metrics != nil {
		metrics.InferenceTimeMs = response.Metrics.InferenceTime * 1000
	}

	logger.Debug("[HTTPSmartTurn] Prediction: %s (probability: %.3f, time: %.1fms)",
		state.String(), response.Probability, totalTimeMs)

	return state, metrics, nil
}

// float32ToInt16PCM converts float32 audio to int16 PCM bytes
func float32ToInt16PCM(audio []float32) []byte {
	buf := new(bytes.Buffer)
	for _, sample := range audio {
		// Clamp to [-1, 1]
		if sample > 1.0 {
			sample = 1.0
		} else if sample < -1.0 {
			sample = -1.0
		}
		// Convert to int16
		int16Sample := int16(sample * 32767)
		binary.Write(buf, binary.LittleEndian, int16Sample)
	}
	return buf.Bytes()
}

// FalSmartTurnConfig is a convenience alias for Fal.ai configuration
type FalSmartTurnConfig struct {
	APIKey string
	Params *SmartTurnParams
}

// NewFalSmartTurn creates a new smart turn analyzer using Fal.ai hosted endpoint
func NewFalSmartTurn(config FalSmartTurnConfig) *HTTPSmartTurn {
	return NewHTTPSmartTurn(HTTPSmartTurnConfig{
		URL:    "https://fal.run/fal-ai/smart-turn",
		APIKey: config.APIKey,
		Params: config.Params,
	})
}
