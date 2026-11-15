package whisper

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

const (
	WhisperAPIURL     = "https://api.openai.com/v1/audio/transcriptions"
	DefaultModel      = "whisper-1"
	MaxBufferDuration = 30 * time.Second
	DefaultSampleRate = 16000
)

type STTConfig struct {
	APIKey     string
	Model      string
	Language   string
	SampleRate int
	BaseURL    string
	HTTPClient *http.Client
}

type STTService struct {
	*stt.Base
	apiKey     string
	apiURL     string
	httpClient *http.Client
}

func NewSTTService(config STTConfig) *STTService {
	model := config.Model
	if model == "" {
		model = DefaultModel
	}
	apiURL := config.BaseURL
	if apiURL == "" {
		apiURL = WhisperAPIURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	s := &STTService{apiKey: config.APIKey, apiURL: apiURL, httpClient: client}
	s.Base = stt.NewSegmented("WhisperSTT", s, stt.Options{
		Provider:   "whisper",
		Settings:   frames.STTSettings{Model: model, Language: config.Language},
		SampleRate: config.SampleRate,
		TTFSP99:    2 * time.Second,
		MaxSegment: MaxBufferDuration,
	})
	return s
}

func (s *STTService) Transcribe(ctx context.Context, wav []byte) (stt.Transcript, error) {
	settings := s.Settings()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", "audio.wav")
	if err != nil {
		return stt.Transcript{}, err
	}
	if _, err := part.Write(wav); err != nil {
		return stt.Transcript{}, err
	}
	if err := writer.WriteField("model", settings.Model); err != nil {
		return stt.Transcript{}, err
	}
	if settings.Language != "" {
		if err := writer.WriteField("language", settings.Language); err != nil {
			return stt.Transcript{}, err
		}
	}
	if err := writer.Close(); err != nil {
		return stt.Transcript{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.apiURL, &body)
	if err != nil {
		return stt.Transcript{}, err
	}
	req.Header.Set("Authorization", "Bearer "+s.apiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return stt.Transcript{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return stt.Transcript{}, fmt.Errorf("whisper HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var result struct {
		Text     string `json:"text"`
		Language string `json:"language"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return stt.Transcript{}, fmt.Errorf("decode whisper response: %w", err)
	}
	return stt.Transcript{Text: strings.TrimSpace(result.Text), Language: result.Language}, nil
}
