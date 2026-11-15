package aggregators

import (
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type AggregationType string

const (
	AggregationTypeSentence  AggregationType = "sentence"
	AggregationTypeWord      AggregationType = "word"
	AggregationTypeToken     AggregationType = "token"
	AggregationTypeUser      AggregationType = "user"
	AggregationTypeAssistant AggregationType = "assistant"
	AggregationTypeCustom    AggregationType = "custom"
)

type TextAggregationMode int

const (
	TextAggregationModeSentence TextAggregationMode = iota
	TextAggregationModeToken
)

type Aggregation struct {
	Text     string
	Type     AggregationType
	Spoken   bool
	Metadata map[string]interface{}
}

func NewAggregation(text string, aggType AggregationType) *Aggregation {
	return &Aggregation{Text: strings.TrimSpace(text), Type: aggType}
}

func NewSpokenAggregation(text string, aggType AggregationType) *Aggregation {
	return &Aggregation{Text: strings.TrimSpace(text), Type: aggType, Spoken: true}
}

func (a *Aggregation) SetMetadata(key string, value interface{}) {
	if a.Metadata == nil {
		a.Metadata = make(map[string]interface{})
	}
	a.Metadata[key] = value
}

func (a *Aggregation) GetMetadata(key string) (interface{}, bool) {
	v, ok := a.Metadata[key]
	return v, ok
}

func (a *Aggregation) IsEmpty() bool {
	return a.Text == ""
}

func (a *Aggregation) String() string {
	return a.Text
}

type LLMContextAggregator struct {
	*processors.BaseProcessor

	context         *llm.Context
	role            llm.Role
	aggregation     []string
	addSpaces       bool
	aggregationType AggregationType
}

func NewLLMContextAggregator(name string, c *llm.Context, role llm.Role, handler processors.ProcessHandler) *LLMContextAggregator {
	if c == nil {
		c = llm.NewContext()
	}
	aggType := AggregationTypeCustom
	switch role {
	case llm.RoleUser:
		aggType = AggregationTypeUser
	case llm.RoleAssistant:
		aggType = AggregationTypeAssistant
	}
	agg := &LLMContextAggregator{
		context:         c,
		role:            role,
		addSpaces:       true,
		aggregationType: aggType,
	}
	agg.BaseProcessor = processors.NewBaseProcessor(name, handler)
	return agg
}

func (a *LLMContextAggregator) Reset() error {
	a.aggregation = a.aggregation[:0]
	return nil
}

func (a *LLMContextAggregator) AggregationString() string {
	if a.addSpaces {
		return strings.Join(a.aggregation, " ")
	}
	return strings.Join(a.aggregation, "")
}

func (a *LLMContextAggregator) GetAggregation() *Aggregation {
	return NewAggregation(a.AggregationString(), a.aggregationType)
}

func (a *LLMContextAggregator) SetAggregationType(aggType AggregationType) {
	a.aggregationType = aggType
}

func (a *LLMContextAggregator) GetAggregationType() AggregationType {
	return a.aggregationType
}

func (a *LLMContextAggregator) AppendToAggregation(text string) {
	a.aggregation = append(a.aggregation, text)
}

func (a *LLMContextAggregator) PushContextFrame(direction frames.FrameDirection) error {
	return a.PushFrame(frames.NewLLMContextFrame(a.context), direction)
}

func (a *LLMContextAggregator) LLMContext() *llm.Context {
	return a.context
}

func (a *LLMContextAggregator) GetContext() *llm.Context {
	return a.context
}

func (a *LLMContextAggregator) GetRole() llm.Role {
	return a.role
}

func (a *LLMContextAggregator) SetAddSpaces(addSpaces bool) {
	a.addSpaces = addSpaces
}

func (a *LLMContextAggregator) HasAggregation() bool {
	return len(a.aggregation) > 0
}

func (a *LLMContextAggregator) applyMessagesFrame(frame frames.Frame) (handled, runLLM bool) {
	switch f := frame.(type) {
	case *frames.LLMMessagesAppendFrame:
		a.context.Add(f.Messages...)
		return true, f.RunLLM
	case *frames.LLMMessagesUpdateFrame:
		a.context.Set(f.Messages)
		return true, f.RunLLM
	case *frames.LLMMessagesTransformFrame:
		if f.Transform != nil {
			a.context.Transform(f.Transform)
		}
		return true, f.RunLLM != nil && *f.RunLLM
	case *frames.LLMSetToolsFrame:
		a.context.SetTools(f.Tools)
		return true, false
	case *frames.LLMSetToolChoiceFrame:
		a.context.SetToolChoice(f.ToolChoice)
		return true, false
	}
	return false, false
}

type TextAggregator interface {
	Aggregate(text string) (<-chan *Aggregation, string)
	Flush() *Aggregation
	HandleInterruption()
	Reset()
	Type() AggregationType
}

func closedAggregations(aggs []*Aggregation) <-chan *Aggregation {
	ch := make(chan *Aggregation, len(aggs))
	for _, a := range aggs {
		ch <- a
	}
	close(ch)
	return ch
}

type SimpleTextAggregator struct {
	buffer      strings.Builder
	aggType     AggregationType
	sentenceEnd map[rune]bool
}

func NewSimpleTextAggregator() *SimpleTextAggregator {
	return &SimpleTextAggregator{
		aggType:     AggregationTypeSentence,
		sentenceEnd: map[rune]bool{'.': true, '!': true, '?': true},
	}
}

func (s *SimpleTextAggregator) Type() AggregationType {
	return s.aggType
}

func (s *SimpleTextAggregator) Reset() {
	s.buffer.Reset()
}

func (s *SimpleTextAggregator) Aggregate(text string) (<-chan *Aggregation, string) {
	aggs, remainder := s.AggregateSync(text)
	return closedAggregations(aggs), remainder
}

func (s *SimpleTextAggregator) AggregateSync(text string) ([]*Aggregation, string) {
	s.buffer.WriteString(text)
	runes := []rune(s.buffer.String())

	var sentences []*Aggregation
	start := 0
	for i, r := range runes {
		if !s.sentenceEnd[r] {
			continue
		}
		if i == len(runes)-1 || runes[i+1] == ' ' || runes[i+1] == '\n' {
			if sentence := strings.TrimSpace(string(runes[start : i+1])); sentence != "" {
				sentences = append(sentences, NewAggregation(sentence, s.aggType))
			}
			start = i + 1
		}
	}

	remainder := string(runes[start:])
	s.buffer.Reset()
	s.buffer.WriteString(remainder)
	return sentences, remainder
}

func (s *SimpleTextAggregator) Flush() *Aggregation {
	text := strings.TrimSpace(s.buffer.String())
	s.buffer.Reset()
	if text == "" {
		return nil
	}
	return NewAggregation(text, s.aggType)
}

func (s *SimpleTextAggregator) HandleInterruption() {
	s.buffer.Reset()
}

type WordTextAggregator struct {
	buffer  strings.Builder
	aggType AggregationType
}

func NewWordTextAggregator() *WordTextAggregator {
	return &WordTextAggregator{aggType: AggregationTypeWord}
}

func (w *WordTextAggregator) Type() AggregationType {
	return w.aggType
}

func (w *WordTextAggregator) Reset() {
	w.buffer.Reset()
}

func (w *WordTextAggregator) Aggregate(text string) (<-chan *Aggregation, string) {
	aggs, remainder := w.AggregateSync(text)
	return closedAggregations(aggs), remainder
}

func (w *WordTextAggregator) AggregateSync(text string) ([]*Aggregation, string) {
	w.buffer.WriteString(text)
	buffered := w.buffer.String()
	words := strings.Fields(buffered)
	endsWithSpace := len(buffered) > 0 && (buffered[len(buffered)-1] == ' ' || buffered[len(buffered)-1] == '\n')

	complete := words
	remainder := ""
	if !endsWithSpace && len(words) > 0 {
		complete = words[:len(words)-1]
		remainder = words[len(words)-1]
	}

	aggs := make([]*Aggregation, 0, len(complete))
	for _, word := range complete {
		aggs = append(aggs, NewAggregation(word, w.aggType))
	}

	w.buffer.Reset()
	w.buffer.WriteString(remainder)
	return aggs, remainder
}

func (w *WordTextAggregator) Flush() *Aggregation {
	text := strings.TrimSpace(w.buffer.String())
	w.buffer.Reset()
	if text == "" {
		return nil
	}
	return NewAggregation(text, w.aggType)
}

func (w *WordTextAggregator) HandleInterruption() {
	w.buffer.Reset()
}

func CollectAggregations(ch <-chan *Aggregation) []*Aggregation {
	var results []*Aggregation
	for agg := range ch {
		results = append(results, agg)
	}
	return results
}

func MergeAggregationChannels(channels ...<-chan *Aggregation) <-chan *Aggregation {
	var all []*Aggregation
	for _, ch := range channels {
		all = append(all, CollectAggregations(ch)...)
	}
	return closedAggregations(all)
}
