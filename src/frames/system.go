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

// ErrorFrame carries error information through the pipeline.
//
// The "fatal" classification is stored in BaseFrame metadata under
// MetadataKeyFatal, populated by BaseProcessor.PushError. Use IsFatal()
// to read it without callers having to know the metadata key. Errors
// pushed by callers that do not go through PushError (e.g. test
// fixtures or services constructing the frame directly) default to
// non-fatal.
type ErrorFrame struct {
	*SystemFrame
	Error error
}

// MetadataKeyFatal is the BaseFrame metadata key that BaseProcessor.PushError
// populates with a bool indicating whether the error should terminate the
// pipeline (fatal=true) or merely surface upstream for handling (fatal=false).
// Exposed so non-PushError producers (and consumers like ServiceSwitcher)
// can agree on the contract.
const MetadataKeyFatal = "fatal"

func NewErrorFrame(err error) *ErrorFrame {
	return &ErrorFrame{
		SystemFrame: &SystemFrame{
			BaseFrame: NewBaseFrame("ErrorFrame"),
		},
		Error: err,
	}
}

// IsFatal reports whether this ErrorFrame was tagged as fatal via metadata.
// Returns false when the metadata key is absent or non-bool, matching the
// "default to non-fatal" behavior used by ServiceSwitcher's failover path.
func (f *ErrorFrame) IsFatal() bool {
	if f == nil {
		return false
	}
	meta := f.Metadata()
	if meta == nil {
		return false
	}
	v, ok := meta[MetadataKeyFatal]
	if !ok {
		return false
	}
	b, _ := v.(bool)
	return b
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
