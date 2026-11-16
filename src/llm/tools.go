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
	ToolCallID     string
	FunctionName   string
	Arguments      map[string]any
	RawArguments   json.RawMessage
	Context        *Context
	ReportProgress func(FunctionResult) bool
}

type FunctionResult struct {
	Value    any
	RunLLM   *bool
	Delivery ResultDelivery
}

type ResultDelivery string

const (
	ResultDeliveryDefault   ResultDelivery = ""
	ResultDeliveryInterrupt ResultDelivery = "interrupt"
	ResultDeliveryWhenIdle  ResultDelivery = "when_idle"
	ResultDeliverySilent    ResultDelivery = "silent"
)

type FunctionHandler func(ctx context.Context, params FunctionCallParams) (FunctionResult, error)

type FunctionOptions struct {
	CancelOnInterruption bool
	Timeout              time.Duration
	CancellableByLLM     bool
	Delivery             ResultDelivery
}

const ToolResultInProgress = "IN_PROGRESS"

const ToolResultInterrupted = "INTERRUPTED: the call was cancelled before it returned, so its outcome is unknown. It may have completed, partly completed or not run; verify before retrying it or reporting it as done."

const ToolResultCancelled = "CANCELLED: this tool call was cancelled before it returned a result"

const CancelToolPrefix = "cancel_"

func CancelToolName(functionName string) string {
	return CancelToolPrefix + functionName
}

func CancelToolSchema(functionName string) FunctionSchema {
	return FunctionSchema{
		Name: CancelToolName(functionName),
		Description: "Stop a running " + functionName + " call whose result is no longer needed — the user says to drop it, " +
			"asks for something that replaces it, or says something that makes the pending result stale. Call it with no " +
			"arguments to stop the one running call; only when several " + functionName + " calls are running does it need " +
			"a tool_call_id to say which.",
		Properties: map[string]any{
			"tool_call_id": map[string]any{
				"type": "string",
				"description": "Which " + functionName + " call to stop. Needed only when several are running, and carried " +
					"by the message in the conversation that reported each one running.",
			},
		},
	}
}

func DefaultFunctionOptions() FunctionOptions {
	return FunctionOptions{}
}
