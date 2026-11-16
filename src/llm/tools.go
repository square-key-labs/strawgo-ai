package llm

import (
	"context"
	"encoding/json"
	"time"
)

type FunctionSchema struct {
	Name        string
	Description string
	Properties  map[string]any
	Required    []string
	Strict      bool
}

func (f FunctionSchema) JSONSchema() map[string]any {
	props := f.Properties
	if props == nil {
		props = map[string]any{}
	}
	required := f.Required
	if required == nil {
		required = []string{}
	}
	return map[string]any{
		"type":       "object",
		"properties": props,
		"required":   required,
	}
}

type Tools struct {
	Functions []FunctionSchema
	Custom    map[string][]json.RawMessage
}

func (t Tools) Empty() bool {
	return len(t.Functions) == 0 && len(t.Custom) == 0
}

type ToolChoiceMode string

const (
	ToolChoiceUnset    ToolChoiceMode = ""
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceFunction ToolChoiceMode = "function"
)

type ToolChoice struct {
	Mode ToolChoiceMode
	Name string
}

type FunctionCallParams struct {
	ToolCallID   string
	FunctionName string
	Arguments    map[string]any
	RawArguments json.RawMessage
	Context      *Context
}

type FunctionResult struct {
	Value  any
	RunLLM *bool
}

type FunctionHandler func(ctx context.Context, params FunctionCallParams) (FunctionResult, error)

type FunctionOptions struct {
	CancelOnInterruption bool
	Timeout              time.Duration
}

const ToolResultInProgress = "IN_PROGRESS"

const ToolResultInterrupted = "INTERRUPTED: the call was cancelled before it returned, so its outcome is unknown. It may have completed, partly completed or not run; verify before retrying it or reporting it as done."

func DefaultFunctionOptions() FunctionOptions {
	return FunctionOptions{}
}
