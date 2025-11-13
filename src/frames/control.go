package frames

// ControlFrame is the base for control/configuration frames
type ControlFrame struct {
	*BaseFrame
}

func (f *ControlFrame) Category() FrameCategory {
	return ControlCategory
}

// LLMFullResponseStartFrame marks the beginning of an LLM response
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

// LLMFullResponseEndFrame marks the end of an LLM response
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

// TTSStartedFrame marks the beginning of TTS synthesis
type TTSStartedFrame struct {
	*ControlFrame
}

func NewTTSStartedFrame() *TTSStartedFrame {
	return &TTSStartedFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("TTSStartedFrame"),
		},
	}
}

// TTSStoppedFrame marks the end of TTS synthesis
type TTSStoppedFrame struct {
	*ControlFrame
}

func NewTTSStoppedFrame() *TTSStoppedFrame {
	return &TTSStoppedFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("TTSStoppedFrame"),
		},
	}
}

// HeartbeatFrame is used for pipeline health monitoring
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

// InterruptionTaskFrame signals that the bot should be interrupted
// This frame is pushed upstream to the PipelineTask, which then
// converts it to an InterruptionFrame and sends it downstream
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

// LLMContextFrame carries the conversation context to the LLM
type LLMContextFrame struct {
	*ControlFrame
	Context interface{} // Pointer to services.LLMContext (using interface{} to avoid import cycle)
}

func NewLLMContextFrame(context interface{}) *LLMContextFrame {
	return &LLMContextFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMContextFrame"),
		},
		Context: context,
	}
}

// LLMMessagesAppendFrame appends messages to the context
type LLMMessagesAppendFrame struct {
	*ControlFrame
	Messages interface{} // []services.LLMMessage
	RunLLM   bool
}

func NewLLMMessagesAppendFrame(messages interface{}, runLLM bool) *LLMMessagesAppendFrame {
	return &LLMMessagesAppendFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMMessagesAppendFrame"),
		},
		Messages: messages,
		RunLLM:   runLLM,
	}
}

// LLMMessagesUpdateFrame replaces all messages in the context
type LLMMessagesUpdateFrame struct {
	*ControlFrame
	Messages interface{} // []services.LLMMessage
	RunLLM   bool
}

func NewLLMMessagesUpdateFrame(messages interface{}, runLLM bool) *LLMMessagesUpdateFrame {
	return &LLMMessagesUpdateFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMMessagesUpdateFrame"),
		},
		Messages: messages,
		RunLLM:   runLLM,
	}
}

// FunctionCallInfo describes a function call being initiated
type FunctionCallInfo struct {
	ToolCallID   string
	FunctionName string
}

// FunctionCallsStartedFrame marks the start of function call execution
type FunctionCallsStartedFrame struct {
	*ControlFrame
	FunctionCalls []FunctionCallInfo
}

func NewFunctionCallsStartedFrame(calls []FunctionCallInfo) *FunctionCallsStartedFrame {
	return &FunctionCallsStartedFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("FunctionCallsStartedFrame"),
		},
		FunctionCalls: calls,
	}
}

// FunctionCallInProgressFrame indicates a function is being executed
type FunctionCallInProgressFrame struct {
	*ControlFrame
	ToolCallID           string
	FunctionName         string
	Arguments            map[string]interface{}
	CancelOnInterruption bool
}

func NewFunctionCallInProgressFrame(toolCallID, functionName string, args map[string]interface{}, cancelOnInterruption bool) *FunctionCallInProgressFrame {
	return &FunctionCallInProgressFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("FunctionCallInProgressFrame"),
		},
		ToolCallID:           toolCallID,
		FunctionName:         functionName,
		Arguments:            args,
		CancelOnInterruption: cancelOnInterruption,
	}
}

// FunctionCallResultFrame contains the result of a function execution
type FunctionCallResultFrame struct {
	*ControlFrame
	ToolCallID   string
	FunctionName string
	Result       interface{}
	RunLLM       *bool // nil means default behavior
}

func NewFunctionCallResultFrame(toolCallID, functionName string, result interface{}, runLLM *bool) *FunctionCallResultFrame {
	return &FunctionCallResultFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("FunctionCallResultFrame"),
		},
		ToolCallID:   toolCallID,
		FunctionName: functionName,
		Result:       result,
		RunLLM:       runLLM,
	}
}

// FunctionCallCancelFrame requests cancellation of a function call
type FunctionCallCancelFrame struct {
	*ControlFrame
	ToolCallID   string
	FunctionName string
}

func NewFunctionCallCancelFrame(toolCallID, functionName string) *FunctionCallCancelFrame {
	return &FunctionCallCancelFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("FunctionCallCancelFrame"),
		},
		ToolCallID:   toolCallID,
		FunctionName: functionName,
	}
}
