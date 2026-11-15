package sarvam

import (
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

const (
	DefaultLLMBaseURL = "https://api.sarvam.ai/v1"
	DefaultLLMModel   = "sarvam-m"
)

type LLMConfig = openaicompat.ServiceConfig

var llmDefaults = openaicompat.Defaults{Name: "Sarvam", BaseURL: DefaultLLMBaseURL, Model: DefaultLLMModel}

func LLMOptions() []openaicompat.Option {
	return []openaicompat.Option{openaicompat.WithoutDeveloperRole()}
}

func NewLLMProvider(cfg LLMConfig) *openaicompat.Provider {
	pcfg, _ := cfg.Resolve(llmDefaults)
	return openaicompat.New(pcfg, LLMOptions()...)
}

func NewLLMService(cfg LLMConfig) *llmservice.Service {
	return openaicompat.NewService(cfg, llmDefaults, LLMOptions()...)
}
