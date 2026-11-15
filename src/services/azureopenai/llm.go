package azureopenai

import (
	"net/url"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

const (
	DefaultModel      = "gpt-4.1"
	DefaultAPIVersion = "2025-04-01-preview"
	v1Path            = "/openai/v1"
)

type LLMConfig struct {
	openaicompat.ServiceConfig
	Endpoint   string
	APIVersion string
}

func IsV1Endpoint(endpoint string) bool {
	return strings.HasSuffix(strings.TrimRight(endpoint, "/"), v1Path)
}

func resolve(cfg LLMConfig) (openaicompat.Config, llm.Settings, []openaicompat.Option) {
	endpoint := strings.TrimRight(cfg.Endpoint, "/")
	sc := cfg.ServiceConfig
	if sc.BaseURL == "" {
		sc.BaseURL = endpoint
	}
	pcfg, settings := sc.Resolve(openaicompat.Defaults{Name: "AzureOpenAI", Model: DefaultModel})
	opts := []openaicompat.Option{openaicompat.WithMaxTokensField("max_completion_tokens")}
	if IsV1Endpoint(endpoint) {
		return pcfg, settings, opts
	}
	version := cfg.APIVersion
	if version == "" {
		version = DefaultAPIVersion
	}
	query := "?api-version=" + url.QueryEscape(version)
	base := strings.TrimRight(pcfg.BaseURL, "/") + "/openai/deployments/"
	return pcfg, settings, append(opts,
		openaicompat.WithAuthHeader("api-key", ""),
		openaicompat.WithURLFunc(func(s llm.Settings) string {
			return base + url.PathEscape(s.Model) + "/chat/completions" + query
		}),
	)
}

func NewProvider(cfg LLMConfig) *openaicompat.Provider {
	pcfg, _, opts := resolve(cfg)
	return openaicompat.New(pcfg, opts...)
}

func NewLLMService(cfg LLMConfig) *llmservice.Service {
	pcfg, settings, opts := resolve(cfg)
	return openaicompat.Wrap(openaicompat.New(pcfg, opts...), settings, cfg.ServiceOptions...)
}
