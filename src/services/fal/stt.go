package fal

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

const (
	DefaultURL        = "https://fal.run/fal-ai/wizper"
	DefaultTask       = "transcribe"
	DefaultChunkLevel = "segment"
	DefaultVersion    = "3"
	dataURIPrefix     = `{"audio_url":"data:audio/x-wav;base64,`
)

type STTConfig struct {
	APIKey     string
	Language   string
	Task       string
	ChunkLevel string
	Version    string
	SampleRate int
	URL        string
	HTTPClient *http.Client
}

type STTService struct {
	*stt.Base
	apiKey     string
	url        string
	task       string
	chunkLevel string
	version    string
	client     *http.Client
}

func NewSTTService(config STTConfig) *STTService {
	language := config.Language
	if language == "" {
		language = "en"
	}
	s := &STTService{
		apiKey:     config.APIKey,
		url:        config.URL,
		task:       config.Task,
		chunkLevel: config.ChunkLevel,
		version:    config.Version,
		client:     config.HTTPClient,
	}
	if s.url == "" {
		s.url = DefaultURL
	}
	if s.task == "" {
		s.task = DefaultTask
	}
	if s.chunkLevel == "" {
		s.chunkLevel = DefaultChunkLevel
	}
	if s.version == "" {
		s.version = DefaultVersion
	}
	if s.client == nil {
		s.client = &http.Client{Timeout: 30 * time.Second}
	}
	s.Base = stt.NewSegmented("FalSTT", s, stt.Options{
		Provider:   "fal",
		Settings:   frames.STTSettings{Language: language},
		SampleRate: config.SampleRate,
		TTFSP99:    2070 * time.Millisecond,
	})
	return s
}

func baseLanguage(lang string) string {
	if i := strings.IndexAny(lang, "-_"); i > 0 {
		lang = lang[:i]
	}
	return strings.ToLower(lang)
}

func (s *STTService) payload(wav []byte) ([]byte, string) {
	lang := baseLanguage(s.Settings().Language)
	rest := map[string]string{"task": s.task, "chunk_level": s.chunkLevel, "version": s.version}
	if lang != "" {
		rest["language"] = lang
	}
	tail, _ := json.Marshal(rest)
	buf := make([]byte, 0, len(dataURIPrefix)+base64.StdEncoding.EncodedLen(len(wav))+len(tail)+2)
	buf = append(buf, dataURIPrefix...)
	buf = base64.StdEncoding.AppendEncode(buf, wav)
	buf = append(buf, '"', ',')
	return append(buf, tail[1:]...), lang
}

func (s *STTService) Transcribe(ctx context.Context, wav []byte) (stt.Transcript, error) {
	body, lang := s.payload(wav)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url, bytes.NewReader(body))
	if err != nil {
		return stt.Transcript{}, err
	}
	req.Header.Set("Authorization", "Key "+s.apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return stt.Transcript{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return stt.Transcript{}, fmt.Errorf("fal stt HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var result struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return stt.Transcript{}, fmt.Errorf("decode fal stt response: %w", err)
	}
	return stt.Transcript{Text: strings.TrimSpace(result.Text), Language: lang}, nil
}
