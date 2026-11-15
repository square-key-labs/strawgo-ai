package rime

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
	apiKey string
	url    string
	client *http.Client
}

func NewHTTPTTSService(config TTSConfig) *HTTPTTSService {
	u := config.URL
	if u == "" {
		u = DefaultHTTPURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &HTTPTTSService{apiKey: config.APIKey, url: u, client: client}
	s.Base = tts.NewHTTP("RimeHTTPTTS", s, tts.Options{Settings: defaultSettings(config), SampleRate: config.SampleRate})
	return s
}

func (s *HTTPTTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	body := params(req.Settings, req.SampleRate)
	delete(body, "audioFormat")
	body["text"] = req.Text
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+s.apiKey)
	httpReq.Header.Set("Accept", "audio/pcm")
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("rime HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return resp.Body, nil
}
