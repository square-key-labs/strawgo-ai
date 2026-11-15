package asyncai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

type HTTPTTSService struct {
	*tts.Base
	apiKey    string
	version   string
	url       string
	encoding  string
	container string
	client    *http.Client
}

func NewHTTPTTSService(config TTSConfig) *HTTPTTSService {
	settings := defaults(&config)
	u := strings.TrimRight(config.URL, "/")
	if u == "" {
		u = DefaultHTTPURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &HTTPTTSService{apiKey: config.APIKey, version: config.Version, url: u, encoding: config.Encoding, container: config.Container, client: client}
	c, bps := codec(config.Encoding)
	s.Base = tts.NewHTTP("AsyncAIHTTPTTS", s, tts.Options{Settings: settings, SampleRate: config.SampleRate, Codec: c, BytesPerSample: bps})
	return s
}

func (s *HTTPTTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	payload, err := json.Marshal(struct {
		ModelID      string       `json:"model_id"`
		Transcript   string       `json:"transcript"`
		Voice        voice        `json:"voice"`
		OutputFormat outputFormat `json:"output_format"`
		Language     *string      `json:"language"`
	}{
		ModelID:      req.Settings.Model,
		Transcript:   req.Text,
		Voice:        voice{Mode: "id", ID: req.Settings.Voice},
		OutputFormat: outputFormat{Container: s.container, Encoding: s.encoding, SampleRate: req.SampleRate},
		Language:     languagePtr(req.Settings.Language),
	})
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url+"/text_to_speech/streaming", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("x-api-key", s.apiKey)
	httpReq.Header.Set("version", s.version)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("asyncai HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return resp.Body, nil
}
