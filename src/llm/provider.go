package llm

import (
	"context"
	"errors"
	"strings"
)

type Event interface {
	isEvent()
}

type TextDelta struct {
	Text string
}

type ThoughtDelta struct {
	Text      string
	Signature string
}

type ToolCall struct {
	ID        string
	Name      string
	Arguments []byte
	Signature string
}

type Usage struct {
	PromptTokens        int
	CompletionTokens    int
	TotalTokens         int
	CacheReadTokens     int
	CacheCreationTokens int
	ReasoningTokens     int
}

type UsageEvent struct {
	Usage Usage
}

const (
	StopReasonStop          = "stop"
	StopReasonLength        = "length"
	StopReasonToolCalls     = "tool_calls"
	StopReasonContentFilter = "content_filter"
	StopReasonRefusal       = "refusal"
	StopReasonOther         = "other"
)

type Done struct {
	StopReason    string
	RawStopReason string
}

type RefusalCategory string

const (
	RefusalContentFilter RefusalCategory = "content_filter"
	RefusalPromptBlocked RefusalCategory = "prompt_blocked"
	RefusalModel         RefusalCategory = "model_refusal"
	RefusalRecitation    RefusalCategory = "recitation"
	RefusalGuardrail     RefusalCategory = "guardrail"
)

type Refusal struct {
	Category RefusalCategory
	Provider string
	Reason   string
	Detail   string
	Message  string
}

type RefusalError struct {
	Refusal Refusal
	Err     error
}

func (e *RefusalError) Error() string {
	var b strings.Builder
	if e.Refusal.Provider != "" {
		b.WriteString(e.Refusal.Provider)
		b.WriteString(": ")
	}
	b.WriteString("response refused (")
	b.WriteString(string(e.Refusal.Category))
	if e.Refusal.Reason != "" && e.Refusal.Reason != string(e.Refusal.Category) {
		b.WriteString(", ")
		b.WriteString(e.Refusal.Reason)
	}
	b.WriteString(")")
	if e.Refusal.Detail != "" {
		b.WriteString(" [")
		b.WriteString(e.Refusal.Detail)
		b.WriteString("]")
	}
	switch {
	case e.Err != nil:
		b.WriteString(": ")
		b.WriteString(e.Err.Error())
	case e.Refusal.Message != "":
		b.WriteString(": ")
		b.WriteString(e.Refusal.Message)
	}
	return b.String()
}

func (e *RefusalError) Unwrap() error { return e.Err }

func AsRefusal(err error) (Refusal, bool) {
	var re *RefusalError
	if errors.As(err, &re) {
		return re.Refusal, true
	}
	return Refusal{}, false
}

type ResponseInfo struct {
	ResponseID string
	RequestID  string
	Model      string
}

func (r ResponseInfo) Merge(delta ResponseInfo) ResponseInfo {
	if delta.ResponseID != "" {
		r.ResponseID = delta.ResponseID
	}
	if delta.RequestID != "" {
		r.RequestID = delta.RequestID
	}
	if delta.Model != "" {
		r.Model = delta.Model
	}
	return r
}

const (
	ResponseFormatText       = "text"
	ResponseFormatJSONObject = "json_object"
	ResponseFormatJSONSchema = "json_schema"
)

type ResponseFormat struct {
	Type        string
	Name        string
	Description string
	Schema      map[string]any
	Strict      bool
}

func JSONSchemaFormat(name string, schema map[string]any) *ResponseFormat {
	return &ResponseFormat{Type: ResponseFormatJSONSchema, Name: name, Schema: schema, Strict: true}
}

func (f *ResponseFormat) Structured() bool {
	return f != nil && (f.Type == ResponseFormatJSONObject || f.Type == ResponseFormatJSONSchema)
}

func (f *ResponseFormat) SchemaName() string {
	if f.Name != "" {
		return f.Name
	}
	return "response"
}

var ErrResponseFormatUnsupported = errors.New("structured response format is not supported")

func (TextDelta) isEvent()    {}
func (ThoughtDelta) isEvent() {}
func (ToolCall) isEvent()     {}
func (UsageEvent) isEvent()   {}
func (Refusal) isEvent()      {}
func (ResponseInfo) isEvent() {}
func (Done) isEvent()         {}

type Settings struct {
	Model             string
	SystemInstruction string
	Temperature       *float64
	TopP              *float64
	MaxTokens         *int
	Seed              *int
	ResponseFormat    *ResponseFormat
	Extra             map[string]any
}

func (s Settings) Merge(delta Settings) Settings {
	out := s
	if delta.Model != "" {
		out.Model = delta.Model
	}
	if delta.SystemInstruction != "" {
		out.SystemInstruction = delta.SystemInstruction
	}
	if delta.Temperature != nil {
		out.Temperature = delta.Temperature
	}
	if delta.TopP != nil {
		out.TopP = delta.TopP
	}
	if delta.MaxTokens != nil {
		out.MaxTokens = delta.MaxTokens
	}
	if delta.Seed != nil {
		out.Seed = delta.Seed
	}
	if delta.ResponseFormat != nil {
		out.ResponseFormat = delta.ResponseFormat
	}
	if len(delta.Extra) > 0 {
		merged := make(map[string]any, len(s.Extra)+len(delta.Extra))
		for k, v := range s.Extra {
			merged[k] = v
		}
		for k, v := range delta.Extra {
			merged[k] = v
		}
		out.Extra = merged
	}
	return out
}

type Provider interface {
	Name() string
	Stream(ctx context.Context, snapshot Snapshot, settings Settings, emit func(Event)) error
}
