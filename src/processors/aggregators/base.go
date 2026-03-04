package aggregators

import (
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// AggregationType defines the type of aggregation
type AggregationType string

const (
	// AggregationTypeSentence indicates text aggregated by sentence boundaries
	AggregationTypeSentence AggregationType = "sentence"
	// AggregationTypeWord indicates text aggregated word by word
	AggregationTypeWord AggregationType = "word"
	// AggregationTypeToken indicates text aggregated token by token (raw LLM output)
	AggregationTypeToken AggregationType = "token"
	// AggregationTypeUser indicates user input aggregation
	AggregationTypeUser AggregationType = "user"
	// AggregationTypeAssistant indicates assistant response aggregation
	AggregationTypeAssistant AggregationType = "assistant"
	// AggregationTypeCustom indicates a custom aggregation type
	AggregationTypeCustom AggregationType = "custom"
)

// TextAggregationMode defines how text should be aggregated before being sent to TTS
type TextAggregationMode int

const (
	// TextAggregationModeSentence buffers text until sentence boundaries are detected (default)
	TextAggregationModeSentence TextAggregationMode = iota
	// TextAggregationModeToken passes text through immediately without buffering
	TextAggregationModeToken
)

// Aggregation represents aggregated text with metadata
// This allows tracking how text was aggregated and enables downstream processors
// to make decisions based on the aggregation type
type Aggregation struct {
	// Text is the aggregated text content (whitespace trimmed)
	Text string
	// Type describes how the text was aggregated (e.g., "sentence", "word", "user")
	Type AggregationType
	// Spoken indicates whether this text was/will be spoken by TTS
	Spoken bool
	// Metadata holds additional aggregation-specific data
	Metadata map[string]interface{}
}

// NewAggregation creates a new Aggregation with the given text and type
func NewAggregation(text string, aggType AggregationType) *Aggregation {
	return &Aggregation{
		Text:     strings.TrimSpace(text),
		Type:     aggType,
		Spoken:   false,
		Metadata: make(map[string]interface{}),
	}
}

// NewSpokenAggregation creates a new Aggregation that has been/will be spoken
func NewSpokenAggregation(text string, aggType AggregationType) *Aggregation {
	return &Aggregation{
		Text:     strings.TrimSpace(text),
		Type:     aggType,
		Spoken:   true,
		Metadata: make(map[string]interface{}),
	}
}

// SetMetadata sets a metadata value
func (a *Aggregation) SetMetadata(key string, value interface{}) {
	if a.Metadata == nil {
		a.Metadata = make(map[string]interface{})
	}
	a.Metadata[key] = value
}

// GetMetadata gets a metadata value
func (a *Aggregation) GetMetadata(key string) (interface{}, bool) {
	if a.Metadata == nil {
		return nil, false
	}
	v, ok := a.Metadata[key]
	return v, ok
}

// IsEmpty returns true if the aggregation has no text
func (a *Aggregation) IsEmpty() bool {
	return a.Text == ""
}

// String returns the text content (for backward compatibility)
func (a *Aggregation) String() string {
	return a.Text
}

// LLMContextAggregator is the base for all LLM context aggregators
type LLMContextAggregator struct {
	*processors.BaseProcessor

	// Shared state
	context     *services.LLMContext
	role        string // "user" or "assistant"
	aggregation []string
	addSpaces   bool

	// Aggregation type tracking
	aggregationType AggregationType
}

// NewLLMContextAggregator creates a new base context aggregator
func NewLLMContextAggregator(name string, context *services.LLMContext, role string, handler processors.ProcessHandler) *LLMContextAggregator {
	// Determine aggregation type based on role
	aggType := AggregationTypeCustom
	if role == "user" {
		aggType = AggregationTypeUser
	} else if role == "assistant" {
		aggType = AggregationTypeAssistant
	}

	agg := &LLMContextAggregator{
		context:         context,
		role:            role,
		aggregation:     make([]string, 0),
		addSpaces:       true,
		aggregationType: aggType,
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
// Deprecated: Use GetAggregation().Text instead for access to metadata
func (a *LLMContextAggregator) AggregationString() string {
	if len(a.aggregation) == 0 {
		return ""
	}

	if a.addSpaces {
		return strings.Join(a.aggregation, " ")
	}
	return strings.Join(a.aggregation, "")
}

// GetAggregation returns the current aggregation as a typed Aggregation object
// This is the preferred method for accessing aggregated text
func (a *LLMContextAggregator) GetAggregation() *Aggregation {
	text := a.AggregationString()
	return NewAggregation(text, a.aggregationType)
}

// SetAggregationType sets the type of aggregation being performed
func (a *LLMContextAggregator) SetAggregationType(aggType AggregationType) {
	a.aggregationType = aggType
}

// GetAggregationType returns the current aggregation type
func (a *LLMContextAggregator) GetAggregationType() AggregationType {
	return a.aggregationType
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

// HasAggregation returns true if there is any aggregated text
func (a *LLMContextAggregator) HasAggregation() bool {
	return len(a.aggregation) > 0
}

// TextAggregator interface for text aggregation strategies
// Implementations can return multiple aggregations from a single input via channels
type TextAggregator interface {
	// Aggregate processes text and returns aggregations via a channel
	// The channel is closed when all aggregations have been sent
	// Returns (channel of aggregations, remainder text that wasn't aggregated)
	Aggregate(text string) (<-chan *Aggregation, string)

	// Flush returns any pending aggregation that hasn't been emitted yet
	// Called at end of stream (e.g., LLMFullResponseEndFrame)
	Flush() *Aggregation

	// HandleInterruption handles interruptions in the aggregation process
	// May discard pending text or perform internal modifications
	HandleInterruption()

	// Reset clears the aggregator state
	Reset()

	// Type returns the aggregation type this aggregator produces
	Type() AggregationType
}

// SimpleTextAggregator aggregates text by sentence boundaries
// Returns complete sentences as they are detected
type SimpleTextAggregator struct {
	buffer      strings.Builder
	aggType     AggregationType
	sentenceEnd map[rune]bool
}

// NewSimpleTextAggregator creates a new sentence-based text aggregator
func NewSimpleTextAggregator() *SimpleTextAggregator {
	return &SimpleTextAggregator{
		aggType: AggregationTypeSentence,
		sentenceEnd: map[rune]bool{
			'.': true,
			'!': true,
			'?': true,
		},
	}
}

// Type returns the aggregation type
func (s *SimpleTextAggregator) Type() AggregationType {
	return s.aggType
}

// Reset clears the buffer
func (s *SimpleTextAggregator) Reset() {
	s.buffer.Reset()
}

// Aggregate processes text and returns complete sentences via channel
func (s *SimpleTextAggregator) Aggregate(text string) (<-chan *Aggregation, string) {
	ch := make(chan *Aggregation, 10) // Buffered channel for multiple sentences

	// CRITICAL: Compute sentences synchronously to avoid race condition
	// The goroutine only sends pre-computed sentences to the channel
	s.buffer.WriteString(text)
	buffered := s.buffer.String()

	var sentences []*Aggregation
	var current strings.Builder
	runes := []rune(buffered)

	for i := 0; i < len(runes); i++ {
		r := runes[i]
		current.WriteRune(r)

		if s.sentenceEnd[r] {
			// Check if this looks like end of sentence
			// (not abbreviation, followed by space or end)
			isEndOfText := i == len(runes)-1
			isFollowedBySpace := i+1 < len(runes) && (runes[i+1] == ' ' || runes[i+1] == '\n')

			if isEndOfText || isFollowedBySpace {
				sentence := strings.TrimSpace(current.String())
				if sentence != "" {
					sentences = append(sentences, NewAggregation(sentence, s.aggType))
				}
				current.Reset()
			}
		}
	}

	// Update buffer with remainder (synchronously, before returning)
	s.buffer.Reset()
	remainder := current.String()
	if remainder != "" {
		s.buffer.WriteString(remainder)
	}

	// Goroutine only sends pre-computed sentences to channel
	go func() {
		defer close(ch)
		for _, sentence := range sentences {
			ch <- sentence
		}
	}()

	// Return channel and computed remainder
	return ch, s.buffer.String()
}

// AggregateSync is a synchronous version that collects all aggregations
// Useful when you don't need streaming behavior
func (s *SimpleTextAggregator) AggregateSync(text string) ([]*Aggregation, string) {
	ch, remainder := s.Aggregate(text)

	var results []*Aggregation
	for agg := range ch {
		results = append(results, agg)
	}

	return results, remainder
}

// Flush returns any pending text that hasn't been emitted as a complete sentence
func (s *SimpleTextAggregator) Flush() *Aggregation {
	if s.buffer.Len() == 0 {
		return nil
	}
	text := strings.TrimSpace(s.buffer.String())
	s.buffer.Reset()
	if text == "" {
		return nil
	}
	return NewAggregation(text, s.aggType)
}

// HandleInterruption clears any pending text on interruption
func (s *SimpleTextAggregator) HandleInterruption() {
	s.buffer.Reset()
}

// WordTextAggregator aggregates text word by word
type WordTextAggregator struct {
	buffer  strings.Builder
	aggType AggregationType
}

// NewWordTextAggregator creates a new word-based text aggregator
func NewWordTextAggregator() *WordTextAggregator {
	return &WordTextAggregator{
		aggType: AggregationTypeWord,
	}
}

// Type returns the aggregation type
func (w *WordTextAggregator) Type() AggregationType {
	return w.aggType
}

// Reset clears the buffer
func (w *WordTextAggregator) Reset() {
	w.buffer.Reset()
}

// Aggregate processes text and returns complete words via channel
func (w *WordTextAggregator) Aggregate(text string) (<-chan *Aggregation, string) {
	ch := make(chan *Aggregation, 20) // Buffered for multiple words

	// CRITICAL: Compute words synchronously to avoid race condition
	// The goroutine only sends pre-computed words to the channel
	w.buffer.WriteString(text)
	buffered := w.buffer.String()

	// Find word boundaries
	words := strings.Fields(buffered)

	// Check if text ends with space (last word is complete)
	endsWithSpace := len(buffered) > 0 && (buffered[len(buffered)-1] == ' ' || buffered[len(buffered)-1] == '\n')

	var aggregations []*Aggregation
	if len(words) > 0 {
		// Collect all complete words
		completeWords := words
		if !endsWithSpace && len(words) > 0 {
			// Last word might be incomplete
			completeWords = words[:len(words)-1]
		}

		for _, word := range completeWords {
			if word != "" {
				aggregations = append(aggregations, NewAggregation(word, w.aggType))
			}
		}
	}

	// Update buffer with incomplete word (if any) - synchronously
	w.buffer.Reset()
	if !endsWithSpace && len(words) > 0 {
		w.buffer.WriteString(words[len(words)-1])
	}

	// Goroutine only sends pre-computed words to channel
	go func() {
		defer close(ch)
		for _, agg := range aggregations {
			ch <- agg
		}
	}()

	return ch, w.buffer.String()
}

// AggregateSync is a synchronous version that collects all aggregations
func (w *WordTextAggregator) AggregateSync(text string) ([]*Aggregation, string) {
	ch, remainder := w.Aggregate(text)

	var results []*Aggregation
	for agg := range ch {
		results = append(results, agg)
	}

	return results, remainder
}

// Flush returns any pending word that hasn't been emitted
func (w *WordTextAggregator) Flush() *Aggregation {
	if w.buffer.Len() == 0 {
		return nil
	}
	text := strings.TrimSpace(w.buffer.String())
	w.buffer.Reset()
	if text == "" {
		return nil
	}
	return NewAggregation(text, w.aggType)
}

// HandleInterruption clears any pending text on interruption
func (w *WordTextAggregator) HandleInterruption() {
	w.buffer.Reset()
}

// CollectAggregations is a helper function to collect all aggregations from a channel
func CollectAggregations(ch <-chan *Aggregation) []*Aggregation {
	var results []*Aggregation
	for agg := range ch {
		results = append(results, agg)
	}
	return results
}

// MergeAggregationChannels merges multiple aggregation channels into one
// Useful for combining results from multiple aggregators
func MergeAggregationChannels(channels ...<-chan *Aggregation) <-chan *Aggregation {
	out := make(chan *Aggregation, 10)

	go func() {
		defer close(out)
		for _, ch := range channels {
			for agg := range ch {
				out <- agg
			}
		}
	}()

	return out
}
