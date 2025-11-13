package aggregators

import (
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// LLMContextAggregator is the base for all LLM context aggregators
type LLMContextAggregator struct {
	*processors.BaseProcessor

	// Shared state
	context     *services.LLMContext
	role        string // "user" or "assistant"
	aggregation []string
	addSpaces   bool
}

// NewLLMContextAggregator creates a new base context aggregator
func NewLLMContextAggregator(name string, context *services.LLMContext, role string, handler processors.ProcessHandler) *LLMContextAggregator {
	agg := &LLMContextAggregator{
		context:     context,
		role:        role,
		aggregation: make([]string, 0),
		addSpaces:   true,
	}
	agg.BaseProcessor = processors.NewBaseProcessor(name, handler)
	return agg
}

// Reset clears the aggregation state
func (a *LLMContextAggregator) Reset() error {
	a.aggregation = make([]string, 0)
	return nil
}

// AggregationString concatenates all accumulated text
func (a *LLMContextAggregator) AggregationString() string {
	if len(a.aggregation) == 0 {
		return ""
	}

	if a.addSpaces {
		return strings.Join(a.aggregation, " ")
	}
	return strings.Join(a.aggregation, "")
}

// AppendToAggregation adds text to the aggregation buffer
func (a *LLMContextAggregator) AppendToAggregation(text string) {
	a.aggregation = append(a.aggregation, text)
}

// PushContextFrame pushes an LLMContextFrame downstream
func (a *LLMContextAggregator) PushContextFrame(direction frames.FrameDirection) error {
	frame := frames.NewLLMContextFrame(a.context)
	return a.PushFrame(frame, direction)
}

// GetContext returns the LLM context
func (a *LLMContextAggregator) GetContext() *services.LLMContext {
	return a.context
}

// GetRole returns the aggregator's role
func (a *LLMContextAggregator) GetRole() string {
	return a.role
}

// SetAddSpaces sets whether to add spaces between aggregated text
func (a *LLMContextAggregator) SetAddSpaces(addSpaces bool) {
	a.addSpaces = addSpaces
}
