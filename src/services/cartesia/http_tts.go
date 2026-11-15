package cartesia

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const DefaultHTTPBaseURL = "https://api.cartesia.ai"

type HTTPTTSConfig struct {
	APIKey              string
	VoiceID             string
	Model               string
	CartesiaVersion     string
	Language            string
	SampleRate          int
	Encoding            string
	GenerationConfig    *GenerationConfig
	PronunciationDictID string
	BaseURL             string
	HTTPClient          *http.Client
}

type HTTPTTSService struct {
	*tts.Base
	apiKey     string
	version    string
	encoding   string
	generation *GenerationConfig
	dictID     string
	baseURL    string
	client     *http.Client
}

func NewHTTPTTSService(config HTTPTTSConfig) *HTTPTTSService {
	model := config.Model
	if model == "" {
		model = DefaultModel
	}
	version := config.CartesiaVersion
	if version == "" {
		version = DefaultVersion
	}
	language := config.Language
	if language == "" {
		language = "en"
	}
	encoding := config.Encoding
	if encoding == "" {
		encoding = "pcm_s16le"
	}
	baseURL := strings.TrimRight(config.BaseURL, "/")
	if baseURL == "" {
		baseURL = DefaultHTTPBaseURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	codec, bytesPerSample := encodingFormat(encoding)
	s := &HTTPTTSService{
		apiKey:     config.APIKey,
		version:    version,
		encoding:   encoding,
		generation: config.GenerationConfig,
		dictID:     config.PronunciationDictID,
		baseURL:    baseURL,
		client:     client,
	}
	s.Base = tts.NewHTTP("CartesiaHTTPTTS", s, tts.Options{
		Settings:       frames.TTSSettings{Model: model, Voice: config.VoiceID, Language: language},
		SampleRate:     config.SampleRate,
		Codec:          codec,
		BytesPerSample: bytesPerSample,
	})
	return s
}

type httpRequest struct {
	ModelID             string            `json:"model_id"`
	Transcript          string            `json:"transcript"`
	Voice               voice             `json:"voice"`
	OutputFormat        outputFormat      `json:"output_format"`
	Language            string            `json:"language,omitempty"`
	GenerationConfig    *GenerationConfig `json:"generation_config,omitempty"`
	PronunciationDictID string            `json:"pronunciation_dict_id,omitempty"`
}

func (s *HTTPTTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	gen := s.generation
	if req.Settings.Speed != nil {
		g := GenerationConfig{Speed: *req.Settings.Speed}
		if gen != nil {
			g.Volume, g.Emotion = gen.Volume, gen.Emotion
		}
		gen = &g
	}
	payload, err := json.Marshal(httpRequest{
		ModelID:             req.Settings.Model,
		Transcript:          req.Text,
		Voice:               voice{Mode: "id", ID: req.Settings.Voice},
		OutputFormat:        outputFormat{Container: "raw", Encoding: s.encoding, SampleRate: req.SampleRate},
		Language:            req.Settings.Language,
		GenerationConfig:    gen,
		PronunciationDictID: s.dictID,
	})
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+"/tts/bytes", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("X-API-Key", s.apiKey)
	httpReq.Header.Set("Cartesia-Version", s.version)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("cartesia HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return resp.Body, nil
}
