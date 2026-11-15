package pipeline

import (
	"context"
	"errors"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

type LLMSwitcher struct {
	*ServiceSwitcher
	llms []*llmservice.Service
}

func NewLLMSwitcher(llms []*llmservice.Service, strategy ServiceSwitcherStrategy) *LLMSwitcher {
	services := make([]processors.FrameProcessor, len(llms))
	for i, l := range llms {
		services[i] = l
	}
	return &LLMSwitcher{
		ServiceSwitcher: NewServiceSwitcher(services, strategy),
		llms:            append([]*llmservice.Service(nil), llms...),
	}
}

func (s *LLMSwitcher) LLMs() []*llmservice.Service {
	return append([]*llmservice.Service(nil), s.llms...)
}

func (s *LLMSwitcher) ActiveLLM() *llmservice.Service {
	return s.llms[s.active.Load()]
}

func (s *LLMSwitcher) RegisterFunction(name string, handler llm.FunctionHandler, opts ...llm.FunctionOptions) {
	for _, l := range s.llms {
		l.RegisterFunction(name, handler, opts...)
	}
}

func (s *LLMSwitcher) UnregisterFunction(name string) {
	for _, l := range s.llms {
		l.UnregisterFunction(name)
	}
}

func (s *LLMSwitcher) RunInference(ctx context.Context, snapshot llm.Snapshot, delta llm.Settings) (string, error) {
	active := s.ActiveLLM()
	if active == nil {
		return "", errors.New("llm switcher: no active llm")
	}
	return active.RunInference(ctx, snapshot, delta)
}
