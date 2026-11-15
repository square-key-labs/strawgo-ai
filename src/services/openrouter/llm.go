package openrouter

import (
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

const (
	DefaultBaseURL = "https://openrouter.ai/api/v1"
	DefaultModel   = "openai/gpt-4.1"
)

type LLMConfig = openaicompat.ServiceConfig

var defaults = openaicompat.Defaults{Name: "OpenRouter", BaseURL: DefaultBaseURL, Model: DefaultModel}

func Options() []openaicompat.Option {
	return []openaicompat.Option{
		openaicompat.WithoutDeveloperRole(),
		openaicompat.WithMessageTransform(FixMessages),
	}
}

func FixMessages(msgs []openaicompat.Message, settings llm.Settings) []openaicompat.Message {
	if !strings.Contains(strings.ToLower(settings.Model), "gemini") {
		return msgs
	}
	seen := false
	for i := range msgs {
		if msgs[i].Role != "system" {
			continue
		}
		if seen {
			msgs[i].Role = "user"
		}
		seen = true
	}
	return msgs
}

func NewProvider(cfg LLMConfig) *openaicompat.Provider {
	pcfg, _ := cfg.Resolve(defaults)
	return openaicompat.New(pcfg, Options()...)
}

func NewLLMService(cfg LLMConfig) *llmservice.Service {
	return openaicompat.NewService(cfg, defaults, Options()...)
}
