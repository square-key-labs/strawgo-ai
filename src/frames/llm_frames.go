package frames

import (
	"time"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type LLMMarkerFrame struct {
	*DataFrame
	Marker                     string
	AppendToContextImmediately bool
}

func NewLLMMarkerFrame(marker string, appendToContextImmediately bool) *LLMMarkerFrame {
	return &LLMMarkerFrame{
		DataFrame:                  newDataFrame("LLMMarkerFrame"),
		Marker:                     marker,
		AppendToContextImmediately: appendToContextImmediately,
	}
}

type LLMMarkerResponseFrame struct {
	*DataFrame
	Raw     string
	Marker  string
	Kind    string
	Markers []string
}

func NewLLMMarkerResponseFrame(raw, marker, kind string, markers []string) *LLMMarkerResponseFrame {
	return &LLMMarkerResponseFrame{
		DataFrame: newDataFrame("LLMMarkerResponseFrame"),
		Raw:       raw,
		Marker:    marker,
		Kind:      kind,
		Markers:   markers,
	}
}

type LLMContextAssistantTimestampFrame struct {
	*DataFrame
	Timestamp time.Time
}

func NewLLMContextAssistantTimestampFrame(timestamp time.Time) *LLMContextAssistantTimestampFrame {
	return &LLMContextAssistantTimestampFrame{
		DataFrame: newDataFrame("LLMContextAssistantTimestampFrame"),
		Timestamp: timestamp,
	}
}

type LLMContextAssistantTurnFrame struct {
	*DataFrame
	Text      string
	Timestamp time.Time
}

func NewLLMContextAssistantTurnFrame(text string, timestamp time.Time) *LLMContextAssistantTurnFrame {
	return &LLMContextAssistantTurnFrame{
		DataFrame: newDataFrame("LLMContextAssistantTurnFrame"),
		Text:      text,
		Timestamp: timestamp,
	}
}

type LLMContextUserTurnFrame struct {
	*DataFrame
	Text        string
	Timestamp   time.Time
	Perception  []PerceptionSignal
	Speculation bool
}

func NewLLMContextUserTurnFrame(text string, timestamp time.Time) *LLMContextUserTurnFrame {
	return &LLMContextUserTurnFrame{
		DataFrame: newDataFrame("LLMContextUserTurnFrame"),
		Text:      text,
		Timestamp: timestamp,
	}
}

type LLMThoughtStartFrame struct {
	*ControlFrame
	AppendToContext bool
	LLM             string
}

func NewLLMThoughtStartFrame(appendToContext bool, llmName string) *LLMThoughtStartFrame {
	return &LLMThoughtStartFrame{
		ControlFrame:    newControlFrame("LLMThoughtStartFrame"),
		AppendToContext: appendToContext,
		LLM:             llmName,
	}
}

type LLMThoughtTextFrame struct {
	*DataFrame
	Text string
}

func NewLLMThoughtTextFrame(text string) *LLMThoughtTextFrame {
	return &LLMThoughtTextFrame{
		DataFrame: newDataFrame("LLMThoughtTextFrame"),
		Text:      text,
	}
}

type LLMThoughtEndFrame struct {
	*ControlFrame
	Signature string
}

func NewLLMThoughtEndFrame(signature string) *LLMThoughtEndFrame {
	return &LLMThoughtEndFrame{
		ControlFrame: newControlFrame("LLMThoughtEndFrame"),
		Signature:    signature,
	}
}

type LLMRunFrame struct {
	*DataFrame
}

func NewLLMRunFrame() *LLMRunFrame {
	return &LLMRunFrame{DataFrame: newDataFrame("LLMRunFrame")}
}

type LLMMessagesTransformFrame struct {
	*DataFrame
	Transform func([]llm.Message) []llm.Message
	RunLLM    *bool
}

func NewLLMMessagesTransformFrame(transform func([]llm.Message) []llm.Message, runLLM *bool) *LLMMessagesTransformFrame {
	return &LLMMessagesTransformFrame{
		DataFrame: newDataFrame("LLMMessagesTransformFrame"),
		Transform: transform,
		RunLLM:    runLLM,
	}
}

type LLMSetToolsFrame struct {
	*DataFrame
	Tools llm.Tools
}

func NewLLMSetToolsFrame(tools llm.Tools) *LLMSetToolsFrame {
	return &LLMSetToolsFrame{
		DataFrame: newDataFrame("LLMSetToolsFrame"),
		Tools:     tools,
	}
}

type LLMSetToolChoiceFrame struct {
	*DataFrame
	ToolChoice llm.ToolChoice
}

func NewLLMSetToolChoiceFrame(choice llm.ToolChoice) *LLMSetToolChoiceFrame {
	return &LLMSetToolChoiceFrame{
		DataFrame:  newDataFrame("LLMSetToolChoiceFrame"),
		ToolChoice: choice,
	}
}

type LLMEnablePromptCachingFrame struct {
	*DataFrame
	Enable bool
}

func NewLLMEnablePromptCachingFrame(enable bool) *LLMEnablePromptCachingFrame {
	return &LLMEnablePromptCachingFrame{
		DataFrame: newDataFrame("LLMEnablePromptCachingFrame"),
		Enable:    enable,
	}
}

type LLMConfigureOutputFrame struct {
	*DataFrame
	SkipTTS bool
}

func NewLLMConfigureOutputFrame(skipTTS bool) *LLMConfigureOutputFrame {
	return &LLMConfigureOutputFrame{
		DataFrame: newDataFrame("LLMConfigureOutputFrame"),
		SkipTTS:   skipTTS,
	}
}

type LLMAssistantPushAggregationFrame struct {
	*ControlFrame
}

func NewLLMAssistantPushAggregationFrame() *LLMAssistantPushAggregationFrame {
	return &LLMAssistantPushAggregationFrame{ControlFrame: newControlFrame("LLMAssistantPushAggregationFrame")}
}

type LLMContextSummaryRequestFrame struct {
	*ControlFrame
	RequestID            string
	Context              *llm.Context
	MinMessagesToKeep    int
	TargetContextTokens  int
	SummarizationPrompt  string
	SummarizationTimeout time.Duration
}

func NewLLMContextSummaryRequestFrame(requestID string, ctx *llm.Context, minMessagesToKeep, targetContextTokens int, prompt string) *LLMContextSummaryRequestFrame {
	return &LLMContextSummaryRequestFrame{
		ControlFrame:        newControlFrame("LLMContextSummaryRequestFrame"),
		RequestID:           requestID,
		Context:             ctx,
		MinMessagesToKeep:   minMessagesToKeep,
		TargetContextTokens: targetContextTokens,
		SummarizationPrompt: prompt,
	}
}

type LLMContextSummaryResultFrame struct {
	*ControlFrame
	RequestID           string
	Summary             string
	LastSummarizedIndex int
	Error               string
}

func NewLLMContextSummaryResultFrame(requestID, summary string, lastSummarizedIndex int, errMsg string) *LLMContextSummaryResultFrame {
	return &LLMContextSummaryResultFrame{
		ControlFrame:        newUninterruptibleControlFrame("LLMContextSummaryResultFrame"),
		RequestID:           requestID,
		Summary:             summary,
		LastSummarizedIndex: lastSummarizedIndex,
		Error:               errMsg,
	}
}

type VisionFullResponseStartFrame struct {
	*ControlFrame
}

func NewVisionFullResponseStartFrame() *VisionFullResponseStartFrame {
	return &VisionFullResponseStartFrame{ControlFrame: newControlFrame("VisionFullResponseStartFrame")}
}

type VisionFullResponseEndFrame struct {
	*ControlFrame
}

func NewVisionFullResponseEndFrame() *VisionFullResponseEndFrame {
	return &VisionFullResponseEndFrame{ControlFrame: newControlFrame("VisionFullResponseEndFrame")}
}

type LLMUpdateSettingsFrame struct {
	*ControlFrame
	Delta                 llm.Settings
	Service               ProcessorRef
	ReachInactiveServices bool
}

func NewLLMUpdateSettingsFrame(delta llm.Settings) *LLMUpdateSettingsFrame {
	return &LLMUpdateSettingsFrame{
		ControlFrame: newUninterruptibleControlFrame("LLMUpdateSettingsFrame"),
		Delta:        delta,
	}
}

type LLMServiceMetadataFrame struct {
	*ServiceMetadataFrame
	IsRealtimeService bool
}

func NewLLMServiceMetadataFrame(serviceName string, isRealtimeService bool) *LLMServiceMetadataFrame {
	return &LLMServiceMetadataFrame{
		ServiceMetadataFrame: newServiceMetadataFrame("LLMServiceMetadataFrame", serviceName),
		IsRealtimeService:    isRealtimeService,
	}
}

type FunctionCallResultProperties struct {
	RunLLM           *bool
	OnContextUpdated func()
	IsFinal          bool
}

func NewFunctionCallResultProperties() FunctionCallResultProperties {
	return FunctionCallResultProperties{IsFinal: true}
}

type ExternalFunctionCall struct {
	FunctionName string
	ToolCallID   string
}

type ExternalFunctionCallsStartedFrame struct {
	*SystemFrame
	FunctionCalls []ExternalFunctionCall
}

func NewExternalFunctionCallsStartedFrame(calls []ExternalFunctionCall) *ExternalFunctionCallsStartedFrame {
	return &ExternalFunctionCallsStartedFrame{
		SystemFrame:   newSystemFrame("ExternalFunctionCallsStartedFrame"),
		FunctionCalls: calls,
	}
}

type ExternalFunctionCallInProgressFrame struct {
	*SystemFrame
	FunctionName string
	ToolCallID   string
	Arguments    map[string]any
}

func NewExternalFunctionCallInProgressFrame(functionName, toolCallID string, args map[string]any) *ExternalFunctionCallInProgressFrame {
	return &ExternalFunctionCallInProgressFrame{
		SystemFrame:  newSystemFrame("ExternalFunctionCallInProgressFrame"),
		FunctionName: functionName,
		ToolCallID:   toolCallID,
		Arguments:    args,
	}
}

type ExternalFunctionCallResultFrame struct {
	*SystemFrame
	FunctionName string
	ToolCallID   string
	Arguments    map[string]any
	Result       any
	IsFinal      bool
}

func NewExternalFunctionCallResultFrame(functionName, toolCallID string, args map[string]any, result any) *ExternalFunctionCallResultFrame {
	return &ExternalFunctionCallResultFrame{
		SystemFrame:  newSystemFrame("ExternalFunctionCallResultFrame"),
		FunctionName: functionName,
		ToolCallID:   toolCallID,
		Arguments:    args,
		Result:       result,
		IsFinal:      true,
	}
}

type ExternalFunctionCallCancelFrame struct {
	*SystemFrame
	FunctionName string
	ToolCallID   string
}

func NewExternalFunctionCallCancelFrame(functionName, toolCallID string) *ExternalFunctionCallCancelFrame {
	return &ExternalFunctionCallCancelFrame{
		SystemFrame:  newSystemFrame("ExternalFunctionCallCancelFrame"),
		FunctionName: functionName,
		ToolCallID:   toolCallID,
	}
}
