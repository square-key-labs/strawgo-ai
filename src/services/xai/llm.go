package xai

import (
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

const (
	DefaultBaseURL = "https://api.x.ai/v1"
	DefaultModel   = "grok-4.20-non-reasoning"
)

type LLMConfig = openaicompat.ServiceConfig

var defaults = openaicompat.Defaults{Name: "Grok", BaseURL: DefaultBaseURL, Model: DefaultModel}

func Options() []openaicompat.Option {
	return []openaicompat.Option{openaicompat.WithUsageHook(FixUsage)}
}

func FixUsage(u *llm.Usage) {
	if u.ReasoningTokens > 0 && u.TotalTokens == u.PromptTokens+u.CompletionTokens+u.ReasoningTokens {
		u.CompletionTokens += u.ReasoningTokens
	}
}

func NewProvider(cfg LLMConfig) *openaicompat.Provider {
	pcfg, _ := cfg.Resolve(defaults)
	return openaicompat.New(pcfg, Options()...)
}

func NewLLMService(cfg LLMConfig) *llmservice.Service {
	return openaicompat.NewService(cfg, defaults, Options()...)
}
