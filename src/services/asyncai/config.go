package asyncai

import (
	"net/http"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

const (
	DefaultHTTPURL   = "https://api.async.com"
	DefaultVersion   = "v1"
	DefaultModel     = "async_flash_v1.5"
	DefaultEncoding  = "pcm_s16le"
	DefaultContainer = "raw"
)

type TTSConfig struct {
	APIKey     string
	Voice      string
	Model      string
	Language   string
	URL        string
	Version    string
	Encoding   string
	Container  string
	SampleRate int
	HTTPClient *http.Client
}

type voice struct {
	Mode string `json:"mode"`
	ID   string `json:"id"`
}

type outputFormat struct {
	Container  string `json:"container"`
	Encoding   string `json:"encoding"`
	SampleRate int    `json:"sample_rate"`
}

func defaults(config *TTSConfig) frames.TTSSettings {
	if config.Version == "" {
		config.Version = DefaultVersion
	}
	if config.Encoding == "" {
		config.Encoding = DefaultEncoding
	}
	if config.Container == "" {
		config.Container = DefaultContainer
	}
	if config.Model == "" {
		config.Model = DefaultModel
	}
	return frames.TTSSettings{Model: config.Model, Voice: config.Voice, Language: config.Language}
}

func codec(encoding string) (string, int) {
	switch encoding {
	case "pcm_mulaw":
		return "mulaw", 1
	case "pcm_alaw":
		return "alaw", 1
	default:
		return "linear16", 2
	}
}

func languagePtr(language string) *string {
	if language == "" {
		return nil
	}
	return &language
}
