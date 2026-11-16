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
	assistantParams := params.Assistant
	user := NewLLMUserAggregatorWithParams(c, params.User)
	assistant := NewLLMAssistantAggregator(c, &assistantParams)
	user.assistant = assistant
	return &ContextAggregatorPair{user: user, assistant: assistant}
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
