package together

import (
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

const (
	DefaultBaseURL = "https://api.together.xyz/v1"
	DefaultModel   = "zai-org/GLM-5.2"
)

type LLMConfig = openaicompat.ServiceConfig

var defaults = openaicompat.Defaults{Name: "Together", BaseURL: DefaultBaseURL, Model: DefaultModel}

func Options() []openaicompat.Option {
	return []openaicompat.Option{openaicompat.WithoutDeveloperRole()}
}

func NewProvider(cfg LLMConfig) *openaicompat.Provider {
	pcfg, _ := cfg.Resolve(defaults)
	return openaicompat.New(pcfg, Options()...)
}

func NewLLMService(cfg LLMConfig) *llmservice.Service {
	return openaicompat.NewService(cfg, defaults, Options()...)
}
