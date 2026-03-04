package frames

import "github.com/square-key-labs/strawgo-ai/src/turns"

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
	AllowInterruptions bool
	TurnStrategies     turns.UserTurnStrategies
}

func NewStartFrame() *StartFrame {
	return &StartFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("StartFrame"),
		},
		AllowInterruptions: false,
		TurnStrategies:     turns.UserTurnStrategies{},
	}
}

// NewStartFrameWithConfig creates a StartFrame with custom configuration
func NewStartFrameWithConfig(allowInterruptions bool, strategies turns.UserTurnStrategies) *StartFrame {
	return &StartFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("StartFrame"),
		},
		AllowInterruptions: allowInterruptions,
		TurnStrategies:     strategies,
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

// UserMuteStartedFrame signals user audio muting has started.
type UserMuteStartedFrame struct {
	*SystemFrame
}

func NewUserMuteStartedFrame() *UserMuteStartedFrame {
	return &UserMuteStartedFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("UserMuteStartedFrame"),
		},
	}
}

// UserMuteStoppedFrame signals user audio muting has stopped.
type UserMuteStoppedFrame struct {
	*SystemFrame
}

func NewUserMuteStoppedFrame() *UserMuteStoppedFrame {
	return &UserMuteStoppedFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("UserMuteStoppedFrame"),
		},
	}
}

// BotStartedSpeakingFrame signals bot started outputting audio
// This is emitted by the output transport when audio starts playing
type BotStartedSpeakingFrame struct {
	*SystemFrame
}

func NewBotStartedSpeakingFrame() *BotStartedSpeakingFrame {
	return &BotStartedSpeakingFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("BotStartedSpeakingFrame"),
		},
	}
}

// BotStoppedSpeakingFrame signals bot stopped outputting audio
// This is emitted by the output transport when audio stops or is interrupted
type BotStoppedSpeakingFrame struct {
	*SystemFrame
}

func NewBotStoppedSpeakingFrame() *BotStoppedSpeakingFrame {
	return &BotStoppedSpeakingFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("BotStoppedSpeakingFrame"),
		},
	}
}

// ClientConnectedFrame signals a client has connected to the transport
type ClientConnectedFrame struct {
	*SystemFrame
}

func NewClientConnectedFrame() *ClientConnectedFrame {
	return &ClientConnectedFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("ClientConnectedFrame"),
		},
	}
}

// BotConnectedFrame signals the bot has connected to the SFU (e.g., Daily WebRTC)
type BotConnectedFrame struct {
	*SystemFrame
}

func NewBotConnectedFrame() *BotConnectedFrame {
	return &BotConnectedFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("BotConnectedFrame"),
		},
	}
}
