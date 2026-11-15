package groq

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

const (
	DefaultSTTModel     = "whisper-large-v3-turbo"
	DefaultSTTURL       = "https://api.groq.com/openai/v1/audio/transcriptions"
	STTExtraPrompt      = "prompt"
	STTExtraTemperature = "temperature"
)

type STTConfig struct {
	APIKey      string
	Model       string
	Language    string
	Prompt      string
	Temperature *float64
	SampleRate  int
	BaseURL     string
	HTTPClient  *http.Client
}

type STTService struct {
	*stt.Base
	apiKey string
	url    string
	client *http.Client
}

func NewSTTService(config STTConfig) *STTService {
	model := config.Model
	if model == "" {
		model = DefaultSTTModel
	}
	language := config.Language
	if language == "" {
		language = "en"
	}
	url := config.BaseURL
	if url == "" {
		url = DefaultSTTURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	extra := map[string]any{}
	if config.Prompt != "" {
		extra[STTExtraPrompt] = config.Prompt
	}
	if config.Temperature != nil {
		extra[STTExtraTemperature] = *config.Temperature
	}
	s := &STTService{apiKey: config.APIKey, url: url, client: client}
	s.Base = stt.NewSegmented("GroqSTT", s, stt.Options{
		Provider:   "groq",
		Settings:   frames.STTSettings{Model: model, Language: language, Extra: extra},
		SampleRate: config.SampleRate,
		TTFSP99:    1540 * time.Millisecond,
		MaxSegment: 30 * time.Second,
	})
	return s
}

func sttBaseLanguage(lang string) string {
	if i := strings.IndexAny(lang, "-_"); i > 0 {
		lang = lang[:i]
	}
	return strings.ToLower(lang)
}

func (s *STTService) Transcribe(ctx context.Context, wav []byte) (stt.Transcript, error) {
	settings := s.Settings()
	var body bytes.Buffer
	body.Grow(len(wav) + 1024)
	w := multipart.NewWriter(&body)
	part, err := w.CreateFormFile("file", "audio.wav")
	if err != nil {
		return stt.Transcript{}, err
	}
	if _, err := part.Write(wav); err != nil {
		return stt.Transcript{}, err
	}
	lang := sttBaseLanguage(settings.Language)
	fields := [][2]string{{"model", settings.Model}, {"response_format", "json"}}
	if lang != "" {
		fields = append(fields, [2]string{"language", lang})
	}
	if v, ok := settings.Extra[STTExtraPrompt].(string); ok && v != "" {
		fields = append(fields, [2]string{"prompt", v})
	}
	if v, ok := settings.Extra[STTExtraTemperature].(float64); ok {
		fields = append(fields, [2]string{"temperature", strconv.FormatFloat(v, 'f', -1, 64)})
	}
	for _, f := range fields {
		if err := w.WriteField(f[0], f[1]); err != nil {
			return stt.Transcript{}, err
		}
	}
	if err := w.Close(); err != nil {
		return stt.Transcript{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url, &body)
	if err != nil {
		return stt.Transcript{}, err
	}
	req.Header.Set("Authorization", "Bearer "+s.apiKey)
	req.Header.Set("Content-Type", w.FormDataContentType())
	resp, err := s.client.Do(req)
	if err != nil {
		return stt.Transcript{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return stt.Transcript{}, fmt.Errorf("groq stt HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var result struct {
		Text     string `json:"text"`
		Language string `json:"language"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return stt.Transcript{}, fmt.Errorf("decode groq stt response: %w", err)
	}
	if result.Language == "" {
		result.Language = lang
	}
	return stt.Transcript{Text: strings.TrimSpace(result.Text), Language: result.Language}, nil
}
