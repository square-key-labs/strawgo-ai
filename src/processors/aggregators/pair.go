package aggregators

import "github.com/square-key-labs/strawgo-ai/src/llm"

type ContextAggregatorPairParams struct {
	User      UserAggregatorParams
	Assistant AssistantAggregatorParams
}

type ContextAggregatorPair struct {
	user      *LLMUserAggregator
	assistant *LLMAssistantAggregator
}

func NewContextAggregatorPair(c *llm.Context, params ContextAggregatorPairParams) *ContextAggregatorPair {
	if c == nil {
		c = llm.NewContext()
	}
	assistant := params.Assistant
	return &ContextAggregatorPair{
		user:      NewLLMUserAggregatorWithParams(c, params.User),
		assistant: NewLLMAssistantAggregator(c, &assistant),
	}
}

func (p *ContextAggregatorPair) User() *LLMUserAggregator {
	return p.user
}

func (p *ContextAggregatorPair) Assistant() *LLMAssistantAggregator {
	return p.assistant
}

func (p *ContextAggregatorPair) Context() *llm.Context {
	return p.user.context
}
