package deepseek

import (
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

const (
	DefaultBaseURL = "https://api.deepseek.com/v1"
	DefaultModel   = "deepseek-flash"
)

type LLMConfig = openaicompat.ServiceConfig

var defaults = openaicompat.Defaults{
	Name:     "DeepSeek",
	BaseURL:  DefaultBaseURL,
	Model:    DefaultModel,
	Settings: llm.Settings{Extra: map[string]any{"thinking": map[string]any{"type": "disabled"}}},
}

func Options() []openaicompat.Option {
	return []openaicompat.Option{
		openaicompat.WithoutDeveloperRole(),
		openaicompat.WithSeedField(""),
		openaicompat.WithReasoningContent(),
	}
}

func NewProvider(cfg LLMConfig) *openaicompat.Provider {
	pcfg, _ := cfg.Resolve(defaults)
	return openaicompat.New(pcfg, Options()...)
}

func NewLLMService(cfg LLMConfig) *llmservice.Service {
	return openaicompat.NewService(cfg, defaults, Options()...)
}
