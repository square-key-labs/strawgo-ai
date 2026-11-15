package openaicompat

import (
	"net/http"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

type ServiceConfig struct {
	APIKey         string
	BaseURL        string
	Model          string
	SystemPrompt   string
	Settings       llm.Settings
	Headers        map[string]string
	HTTPClient     *http.Client
	Retry          RetryPolicy
	ServiceOptions []llmservice.Option
}

type Defaults struct {
	Name     string
	BaseURL  string
	Model    string
	Settings llm.Settings
}

func (cfg ServiceConfig) Resolve(d Defaults) (Config, llm.Settings) {
	settings := d.Settings
	settings.Model = d.Model
	settings = settings.Merge(cfg.Settings)
	if cfg.Model != "" {
		settings.Model = cfg.Model
	}
	if cfg.SystemPrompt != "" {
		settings.SystemInstruction = cfg.SystemPrompt
	}
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = d.BaseURL
	}
	return Config{
		Name:       d.Name,
		BaseURL:    baseURL,
		APIKey:     cfg.APIKey,
		Headers:    cfg.Headers,
		HTTPClient: cfg.HTTPClient,
		Retry:      cfg.Retry,
	}, settings
}

func NewService(cfg ServiceConfig, d Defaults, opts ...Option) *llmservice.Service {
	pcfg, settings := cfg.Resolve(d)
	return Wrap(New(pcfg, opts...), settings, cfg.ServiceOptions...)
}

func Wrap(p llm.Provider, settings llm.Settings, opts ...llmservice.Option) *llmservice.Service {
	all := make([]llmservice.Option, 0, len(opts)+1)
	all = append(all, llmservice.WithSettings(settings))
	all = append(all, opts...)
	return llmservice.New(p, all...)
}
