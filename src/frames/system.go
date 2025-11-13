package frames

import "github.com/square-key-labs/strawgo-ai/src/interruptions"

// SystemFrame is the base for all system-level frames
type SystemFrame struct {
	*BaseFrame
}

func (f *SystemFrame) Category() FrameCategory {
	return SystemCategory
}

// StartFrame signals the beginning of pipeline execution
type StartFrame struct {
	*SystemFrame
	AllowInterruptions     bool
	InterruptionStrategies []interruptions.InterruptionStrategy
}

func NewStartFrame() *StartFrame {
	return &StartFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("StartFrame"),
		},
		AllowInterruptions:     false,
		InterruptionStrategies: []interruptions.InterruptionStrategy{},
	}
}

// NewStartFrameWithConfig creates a StartFrame with custom configuration
func NewStartFrameWithConfig(allowInterruptions bool, strategies []interruptions.InterruptionStrategy) *StartFrame {
	return &StartFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("StartFrame"),
		},
		AllowInterruptions:     allowInterruptions,
		InterruptionStrategies: strategies,
	}
}

// EndFrame signals graceful shutdown after flushing all frames
type EndFrame struct {
	*SystemFrame
}

func NewEndFrame() *EndFrame {
	return &EndFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("EndFrame"),
		},
	}
}

// CancelFrame signals immediate shutdown without flushing
type CancelFrame struct {
	*SystemFrame
}

func NewCancelFrame() *CancelFrame {
	return &CancelFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("CancelFrame"),
		},
	}
}

// InterruptionFrame signals user interrupted bot (e.g., started speaking)
type InterruptionFrame struct {
	*SystemFrame
}

func NewInterruptionFrame() *InterruptionFrame {
	return &InterruptionFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("InterruptionFrame"),
		},
	}
}

// ErrorFrame carries error information through the pipeline
type ErrorFrame struct {
	*SystemFrame
	Error error
}

func NewErrorFrame(err error) *ErrorFrame {
	return &ErrorFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("ErrorFrame"),
		},
		Error: err,
	}
}

// UserStartedSpeakingFrame signals VAD detected user speech
type UserStartedSpeakingFrame struct {
	*SystemFrame
}

func NewUserStartedSpeakingFrame() *UserStartedSpeakingFrame {
	return &UserStartedSpeakingFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("UserStartedSpeakingFrame"),
		},
	}
}

// UserStoppedSpeakingFrame signals VAD detected end of user speech
type UserStoppedSpeakingFrame struct {
	*SystemFrame
}

func NewUserStoppedSpeakingFrame() *UserStoppedSpeakingFrame {
	return &UserStoppedSpeakingFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("UserStoppedSpeakingFrame"),
		},
	}
}
