package turn

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"time"
)

const (
	DefaultFalSmartTurnURL = "https://fal.run/fal-ai/smart-turn/raw"
	maxSmartTurnResponse   = 64 << 10
)

type HTTPSmartTurnFormat int

const (
	SmartTurnFormatNPY HTTPSmartTurnFormat = iota
	SmartTurnFormatJSON
)

type HTTPSmartTurnConfig struct {
	URL         string
	APIKey      string
	Headers     map[string]string
	Format      HTTPSmartTurnFormat
	Params      *SmartTurnParams
	TimeoutSecs int
	Breaker     BreakerConfig
	HTTPClient  *http.Client
}

type HTTPSmartTurn struct {
	*smartTurn
	config     HTTPSmartTurnConfig
	httpClient *http.Client
	body       []byte
}

func NewHTTPSmartTurn(config HTTPSmartTurnConfig) *HTTPSmartTurn {
	if config.URL == "" {
		config.URL = DefaultFalSmartTurnURL
	}
	h := &HTTPSmartTurn{config: config, httpClient: config.HTTPClient}
	if h.httpClient == nil {
		h.httpClient = &http.Client{}
	}
	h.smartTurn = newSmartTurn("HTTPSmartTurn", config.Params, config.Breaker, h.predictHTTP)
	h.requestTimeout = time.Duration(config.TimeoutSecs) * time.Second
	return h
}

func AppendNPYFloat32(dst []byte, samples []int16) []byte {
	dict := "{'descr': '<f4', 'fortran_order': False, 'shape': (" + strconv.Itoa(len(samples)) + ",), }"
	pad := 64 - (10+len(dict)+1)%64
	if pad == 64 {
		pad = 0
	}
	dst = append(dst, 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0)
	dst = binary.LittleEndian.AppendUint16(dst, uint16(len(dict)+pad+1))
	dst = append(dst, dict...)
	for i := 0; i < pad; i++ {
		dst = append(dst, ' ')
	}
	dst = append(dst, '\n')
	for _, s := range samples {
		dst = binary.LittleEndian.AppendUint32(dst, math.Float32bits(float32(s)/32768))
	}
	return dst
}

func (h *HTTPSmartTurn) encode(samples []int16) (string, error) {
	if h.config.Format == SmartTurnFormatNPY {
		h.body = AppendNPYFloat32(h.body[:0], samples)
		return "application/octet-stream", nil
	}
	pcm := int16ToBytes(nil, samples)
	body, err := json.Marshal(map[string]any{
		"audio_base64":  base64.StdEncoding.EncodeToString(pcm),
		"sample_rate":   ModelSampleRate,
		"audio_format":  "pcm_s16le",
		"model_version": "v3.1",
	})
	h.body = body
	return "application/json", err
}

func (h *HTTPSmartTurn) predictHTTP(ctx context.Context, samples []int16) (prediction, error) {
	contentType, err := h.encode(samples)
	if err != nil {
		return prediction{}, fmt.Errorf("smart turn: encode request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.config.URL, bytes.NewReader(h.body))
	if err != nil {
		return prediction{}, fmt.Errorf("smart turn: create request: %w", err)
	}
	req.Header.Set("Content-Type", contentType)
	if h.config.APIKey != "" {
		req.Header.Set("Authorization", "Key "+h.config.APIKey)
	}
	for k, v := range h.config.Headers {
		req.Header.Set(k, v)
	}

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return prediction{}, fmt.Errorf("smart turn: request failed: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxSmartTurnResponse))
	if err != nil {
		return prediction{}, fmt.Errorf("smart turn: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return prediction{}, fmt.Errorf("smart turn: HTTP %d: %s", resp.StatusCode, bytes.TrimSpace(body[:min(len(body), 512)]))
	}
	var out struct {
		Prediction  int     `json:"prediction"`
		Probability float64 `json:"probability"`
		Metrics     *struct {
			InferenceTime float64 `json:"inference_time"`
		} `json:"metrics"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return prediction{}, fmt.Errorf("smart turn: decode response: %w", err)
	}
	pred := prediction{complete: out.Prediction == 1, probability: out.Probability}
	if out.Metrics != nil {
		pred.inferenceMs = out.Metrics.InferenceTime * 1000
	}
	return pred, nil
}

type FalSmartTurnConfig struct {
	URL        string
	APIKey     string
	Params     *SmartTurnParams
	Breaker    BreakerConfig
	HTTPClient *http.Client
}

func NewFalSmartTurn(config FalSmartTurnConfig) *HTTPSmartTurn {
	if config.URL == "" {
		config.URL = DefaultFalSmartTurnURL
	}
	return NewHTTPSmartTurn(HTTPSmartTurnConfig{
		URL:        config.URL,
		APIKey:     config.APIKey,
		Params:     config.Params,
		Breaker:    config.Breaker,
		HTTPClient: config.HTTPClient,
	})
}
