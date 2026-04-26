package frames

import "time"

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
	ContextID string // The context ID for this TTS response (used to filter stale audio)
}

func NewTTSStartedFrame() *TTSStartedFrame {
	return &TTSStartedFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("TTSStartedFrame"),
		},
	}
}

// NewTTSStartedFrameWithContext creates a TTSStartedFrame with a specific context ID
// This allows downstream processors to filter out stale audio from old contexts
func NewTTSStartedFrameWithContext(contextID string) *TTSStartedFrame {
	return &TTSStartedFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("TTSStartedFrame"),
		},
		ContextID: contextID,
	}
}

// TTSStoppedFrame marks the end of TTS synthesis
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

// PlaybackCompleteFrame signals that the client has finished playing audio.
// Emitted when the transport receives a client-side playback acknowledgement
// (e.g., Twilio "mark" echo or Asterisk "QUEUE_DRAINED"), not on server buffer drain.
type PlaybackCompleteFrame struct {
	*ControlFrame
}

func NewPlaybackCompleteFrame() *PlaybackCompleteFrame {
	return &PlaybackCompleteFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("PlaybackCompleteFrame"),
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
// Deprecated: Use BroadcastInterruption() on BaseProcessor instead. Will be removed in a future version.
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

// STTUpdateSettingsFrame requests a runtime settings update on STT services.
// Settings is a free-form map (e.g. {"language": "es", "model": "nova-3"})
// since each provider exposes a different shape; each STT service picks the
// keys it understands.
//
// Service, when non-empty, scopes the update to a specific service instance
// — useful when the pipeline has more than one STT service of the same
// provider type, or when only one provider in a ServiceSwitcher should
// apply the update. Empty Service means "every STT service in scope".
// Mirrors pipecat #4004.
type STTUpdateSettingsFrame struct {
	*ControlFrame
	Settings map[string]interface{}
	Service  string
}

func NewSTTUpdateSettingsFrame(settings map[string]interface{}) *STTUpdateSettingsFrame {
	return &STTUpdateSettingsFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("STTUpdateSettingsFrame"),
		},
		Settings: settings,
	}
}

func NewSTTUpdateSettingsFrameForService(settings map[string]interface{}, service string) *STTUpdateSettingsFrame {
	return &STTUpdateSettingsFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("STTUpdateSettingsFrame"),
		},
		Settings: settings,
		Service:  service,
	}
}

// LLMMessagesTransformFrame applies a transform function to the conversation
// messages held by the LLM aggregator. Unlike LLMMessagesUpdateFrame (which
// requires the caller to capture a snapshot of messages, transform them, and
// push back — racing with any context edits already in flight), this frame
// runs the transform inside the aggregator's frame-handling loop, so it
// always operates on the latest committed message list.
//
// The Transform field is typed as interface{} to avoid an import cycle on
// services.LLMMessage. Aggregator handlers cast it to:
//
//	func([]services.LLMMessage) []services.LLMMessage
type LLMMessagesTransformFrame struct {
	*ControlFrame
	Transform interface{} // func([]services.LLMMessage) []services.LLMMessage
	RunLLM    bool
}

func NewLLMMessagesTransformFrame(transform interface{}, runLLM bool) *LLMMessagesTransformFrame {
	return &LLMMessagesTransformFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMMessagesTransformFrame"),
		},
		Transform: transform,
		RunLLM:    runLLM,
	}
}

// FunctionCallInfo describes a function call being initiated
type FunctionCallInfo struct {
	ToolCallID   string
	FunctionName string
}

// FunctionCallsStartedFrame marks the start of function call execution.
// All calls in this frame are members of a single LLM response batch
// (group). Pipecat's group_parallel_tools=true semantics: when every call
// in the batch has reported a result, the LLM is triggered exactly once.
//
// GroupID is set by the aggregator on receipt and propagates to each
// downstream FunctionCallInProgressFrame / FunctionCallResultFrame.
// If the producer sets a GroupID, the aggregator preserves it; otherwise
// the aggregator synthesizes a UUID. An empty GroupID means "no group" —
// the historical per-call trigger semantics still apply.
type FunctionCallsStartedFrame struct {
	*ControlFrame
	FunctionCalls []FunctionCallInfo
	GroupID       string
}

func NewFunctionCallsStartedFrame(calls []FunctionCallInfo) *FunctionCallsStartedFrame {
	return &FunctionCallsStartedFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("FunctionCallsStartedFrame"),
		},
		FunctionCalls: calls,
	}
}

// FunctionCallInProgressFrame indicates a function is being executed.
// GroupID identifies the LLM response batch this call belongs to.
// CancelOnInterruption=false marks an async function call: the LLM may
// have continued generating without waiting for this call's result; when
// the result eventually arrives, it is injected back into the context as
// a developer-role message and the LLM is triggered for a new inference.
type FunctionCallInProgressFrame struct {
	*ControlFrame
	ToolCallID           string
	FunctionName         string
	Arguments            map[string]interface{}
	CancelOnInterruption bool
	GroupID              string
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

// FunctionCallResultFrame contains the result of a function execution.
// GroupID matches the originating FunctionCallInProgressFrame's GroupID.
// When the aggregator detects that this is the last result in a group,
// it triggers a single LLM inference for the whole batch (matches pipecat
// group_parallel_tools=true semantics).
//
// RunLLM nil means "let the aggregator decide": for sync calls, run when
// the group is empty; for async calls (CancelOnInterruption=false), run
// immediately and inject the result as a developer-role message.
type FunctionCallResultFrame struct {
	*ControlFrame
	ToolCallID   string
	FunctionName string
	Result       interface{}
	RunLLM       *bool // nil means default behavior
	GroupID      string
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

// UserIdleTimeoutFrame is pushed downstream when the user has been idle
// (not speaking) for longer than the configured timeout after the bot stopped speaking.
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

// UserIdleTimeoutUpdateFrame allows runtime enable/disable/reconfigure of idle detection.
// A Timeout of 0 disables idle detection.
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
