package frames

import "github.com/square-key-labs/strawgo-ai/src/turns"

type SystemFrame struct {
	*BaseFrame
}

func (f *SystemFrame) Category() FrameCategory {
	return SystemCategory
}

type StartFrame struct {
	*SystemFrame
	AllowInterruptions    bool
	TurnStrategies        turns.UserTurnStrategies
	AudioInSampleRate     int
	AudioOutSampleRate    int
	EnableMetrics         bool
	EnableUsageMetrics    bool
	ReportOnlyInitialTTFB bool
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

func NewStartFrameWithConfig(allowInterruptions bool, strategies turns.UserTurnStrategies) *StartFrame {
	return &StartFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("StartFrame"),
		},
		AllowInterruptions: allowInterruptions,
		TurnStrategies:     strategies,
	}
}

type EndFrame struct {
	*ControlFrame
	Reason string
}

func NewEndFrame() *EndFrame {
	return &EndFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewUninterruptibleBaseFrame("EndFrame"),
		},
	}
}

type StopFrame struct {
	*ControlFrame
}

func NewStopFrame() *StopFrame {
	return &StopFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewUninterruptibleBaseFrame("StopFrame"),
		},
	}
}

type CancelFrame struct {
	*SystemFrame
	Reason string
}

func NewCancelFrame() *CancelFrame {
	return &CancelFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("CancelFrame"),
		},
	}
}

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

type ErrorFrame struct {
	*SystemFrame
	Error     error
	Fatal     bool
	Processor string
	Category  ErrorCategory
}

func NewErrorFrame(err error) *ErrorFrame {
	return &ErrorFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("ErrorFrame"),
		},
		Error: err,
	}
}

func NewCategorizedErrorFrame(err error, category ErrorCategory) *ErrorFrame {
	f := NewErrorFrame(err)
	f.Category = category
	return f
}

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
