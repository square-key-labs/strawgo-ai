package fireworks

import (
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

const (
	DefaultBaseURL = "https://api.fireworks.ai/inference/v1"
	DefaultModel   = "accounts/fireworks/models/nemotron-3-ultra-nvfp4"
)

type LLMConfig = openaicompat.ServiceConfig

var defaults = openaicompat.Defaults{Name: "Fireworks", BaseURL: DefaultBaseURL, Model: DefaultModel}

func Options() []openaicompat.Option {
	return []openaicompat.Option{
		openaicompat.WithoutStreamOptions(),
		openaicompat.WithSeedField(""),
	}
}

func NewProvider(cfg LLMConfig) *openaicompat.Provider {
	pcfg, _ := cfg.Resolve(defaults)
	return openaicompat.New(pcfg, Options()...)
}

func NewLLMService(cfg LLMConfig) *llmservice.Service {
	return openaicompat.NewService(cfg, defaults, Options()...)
}
