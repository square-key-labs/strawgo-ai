package mistral

import (
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

const (
	DefaultBaseURL = "https://api.mistral.ai/v1"
	DefaultModel   = "mistral-small-latest"
)

type LLMConfig = openaicompat.ServiceConfig

var defaults = openaicompat.Defaults{Name: "Mistral", BaseURL: DefaultBaseURL, Model: DefaultModel}

func Options() []openaicompat.Option {
	return []openaicompat.Option{
		openaicompat.WithoutDeveloperRole(),
		openaicompat.WithoutStreamOptions(),
		openaicompat.WithSeedField("random_seed"),
		openaicompat.WithSkipAnsweredToolCalls(),
		openaicompat.WithMessageTransform(FixMessages),
	}
}

func FixMessages(msgs []openaicompat.Message, _ llm.Settings) []openaicompat.Message {
	if len(msgs) == 0 {
		return msgs
	}
	out := make([]openaicompat.Message, 0, len(msgs)+2)
	leading := true
	for i, m := range msgs {
		if m.Role != "system" {
			leading = false
		} else if !leading {
			m.Role = "user"
		}
		out = append(out, m)
		if m.Role == "tool" && (i == len(msgs)-1 || msgs[i+1].Role != "assistant" && msgs[i+1].Role != "tool") {
			out = append(out, openaicompat.Message{Role: "assistant", Content: " "})
		}
	}
	if last := &out[len(out)-1]; last.Role == "assistant" && len(last.Raw) == 0 {
		last.Prefix = true
	}
	return out
}

func NewProvider(cfg LLMConfig) *openaicompat.Provider {
	pcfg, _ := cfg.Resolve(defaults)
	return openaicompat.New(pcfg, Options()...)
}

func NewLLMService(cfg LLMConfig) *llmservice.Service {
	return openaicompat.NewService(cfg, defaults, Options()...)
}
