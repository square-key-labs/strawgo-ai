package frames

import (
	"time"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type ControlFrame struct {
	*BaseFrame
}

func (f *ControlFrame) Category() FrameCategory {
	return ControlCategory
}

type LLMFullResponseStartFrame struct {
	*ControlFrame
}

func NewLLMFullResponseStartFrame() *LLMFullResponseStartFrame {
	return &LLMFullResponseStartFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMFullResponseStartFrame"),
		},
	}
}

type LLMFullResponseEndFrame struct {
	*ControlFrame
}

func NewLLMFullResponseEndFrame() *LLMFullResponseEndFrame {
	return &LLMFullResponseEndFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMFullResponseEndFrame"),
		},
	}
}

type TTSStartedFrame struct {
	*ControlFrame
	ContextID string // The context ID for this TTS response (used to filter stale audio)
}

func NewTTSStartedFrame() *TTSStartedFrame {
	return &TTSStartedFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("TTSStartedFrame"),
		},
	}
}

func NewTTSStartedFrameWithContext(contextID string) *TTSStartedFrame {
	return &TTSStartedFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("TTSStartedFrame"),
		},
		ContextID: contextID,
	}
}

type TTSStoppedFrame struct {
	*ControlFrame
	ContextID string // The context ID for this TTS response (used to filter stale audio)
}

func NewTTSStoppedFrame() *TTSStoppedFrame {
	return &TTSStoppedFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("TTSStoppedFrame"),
		},
	}
}

type TTSQueuedTextFrame struct {
	*SystemFrame
	Text            string
	TextType        TextType
	ContextID       string
	PlaybackTracked bool
	Complete        bool
}

func NewTTSQueuedTextFrame(text string, textType TextType, contextID string) *TTSQueuedTextFrame {
	return &TTSQueuedTextFrame{
		SystemFrame: newSystemFrame("TTSQueuedTextFrame"),
		Text:        text,
		TextType:    textType,
		ContextID:   contextID,
	}
}

func NewTTSQueuedTextCompleteFrame(contextID string) *TTSQueuedTextFrame {
	return &TTSQueuedTextFrame{
		SystemFrame: newSystemFrame("TTSQueuedTextFrame"),
		ContextID:   contextID,
		Complete:    true,
	}
}

type PlaybackCompleteFrame struct {
	*ControlFrame
}

const MetadataQueueDrained = "queue_drained"

func NewPlaybackCompleteFrame() *PlaybackCompleteFrame {
	return &PlaybackCompleteFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("PlaybackCompleteFrame"),
		},
	}
}

type HeartbeatFrame struct {
	*ControlFrame
}

func NewHeartbeatFrame() *HeartbeatFrame {
	return &HeartbeatFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("HeartbeatFrame"),
		},
	}
}

type InterruptionTaskFrame struct {
	*ControlFrame
}

func NewInterruptionTaskFrame() *InterruptionTaskFrame {
	return &InterruptionTaskFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("InterruptionTaskFrame"),
		},
	}
}

type LLMContextFrame struct {
	*ControlFrame
	Context     *llm.Context
	Speculation bool
}

func NewLLMContextFrame(context *llm.Context) *LLMContextFrame {
	return &LLMContextFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMContextFrame"),
		},
		Context: context,
	}
}

type LLMSummarizeContextFrame struct {
	*ControlFrame
}

func NewLLMSummarizeContextFrame() *LLMSummarizeContextFrame {
	return &LLMSummarizeContextFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMSummarizeContextFrame"),
		},
	}
}

type LLMMessagesAppendFrame struct {
	*ControlFrame
	Messages []llm.Message
	RunLLM   bool
}

func NewLLMMessagesAppendFrame(messages []llm.Message, runLLM bool) *LLMMessagesAppendFrame {
	return &LLMMessagesAppendFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMMessagesAppendFrame"),
		},
		Messages: messages,
		RunLLM:   runLLM,
	}
}

type LLMMessagesUpdateFrame struct {
	*ControlFrame
	Messages []llm.Message
	RunLLM   bool
}

func NewLLMMessagesUpdateFrame(messages []llm.Message, runLLM bool) *LLMMessagesUpdateFrame {
	return &LLMMessagesUpdateFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMMessagesUpdateFrame"),
		},
		Messages: messages,
		RunLLM:   runLLM,
	}
}

type FunctionCallInfo struct {
	ToolCallID   string
	FunctionName string
}

type FunctionCallsStartedFrame struct {
	*SystemFrame
	FunctionCalls []FunctionCallInfo
}

func NewFunctionCallsStartedFrame(calls []FunctionCallInfo) *FunctionCallsStartedFrame {
	return &FunctionCallsStartedFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("FunctionCallsStartedFrame"),
		},
		FunctionCalls: calls,
	}
}

type FunctionCallInProgressFrame struct {
	*ControlFrame
	ToolCallID           string
	FunctionName         string
	Arguments            map[string]interface{}
	CancelOnInterruption bool
	GroupID              string
	Signature            string
}

func NewFunctionCallInProgressFrame(toolCallID, functionName string, args map[string]interface{}, cancelOnInterruption bool) *FunctionCallInProgressFrame {
	return &FunctionCallInProgressFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewUninterruptibleBaseFrame("FunctionCallInProgressFrame"),
		},
		ToolCallID:           toolCallID,
		FunctionName:         functionName,
		Arguments:            args,
		CancelOnInterruption: cancelOnInterruption,
	}
}

type FunctionCallResultFrame struct {
	*ControlFrame
	ToolCallID   string
	FunctionName string
	Result       interface{}
	RunLLM       *bool // nil means default behavior
	Arguments    map[string]interface{}
	Properties   *FunctionCallResultProperties
	Error        string
}

func NewFunctionCallResultFrame(toolCallID, functionName string, result interface{}, runLLM *bool) *FunctionCallResultFrame {
	return &FunctionCallResultFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewUninterruptibleBaseFrame("FunctionCallResultFrame"),
		},
		ToolCallID:   toolCallID,
		FunctionName: functionName,
		Result:       result,
		RunLLM:       runLLM,
	}
}

type FunctionCallCancelFrame struct {
	*SystemFrame
	ToolCallID   string
	FunctionName string
	RunLLM       bool
}

func NewFunctionCallCancelFrame(toolCallID, functionName string) *FunctionCallCancelFrame {
	return &FunctionCallCancelFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("FunctionCallCancelFrame"),
		},
		ToolCallID:   toolCallID,
		FunctionName: functionName,
	}
}

type UserIdleTimeoutFrame struct {
	*ControlFrame
}

func NewUserIdleTimeoutFrame() *UserIdleTimeoutFrame {
	return &UserIdleTimeoutFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("UserIdleTimeoutFrame"),
		},
	}
}

type UserIdleTimeoutUpdateFrame struct {
	*ControlFrame
	Timeout time.Duration
}

func NewUserIdleTimeoutUpdateFrame(timeout time.Duration) *UserIdleTimeoutUpdateFrame {
	return &UserIdleTimeoutUpdateFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("UserIdleTimeoutUpdateFrame"),
		},
		Timeout: timeout,
	}
}

type EndTaskFrame struct {
	*ControlFrame
	Reason string
}

func NewEndTaskFrame(reason string) *EndTaskFrame {
	return &EndTaskFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewUninterruptibleBaseFrame("EndTaskFrame"),
		},
		Reason: reason,
	}
}

type CancelTaskFrame struct {
	*SystemFrame
	Reason string
}

func NewCancelTaskFrame(reason string) *CancelTaskFrame {
	return &CancelTaskFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("CancelTaskFrame"),
		},
		Reason: reason,
	}
}
