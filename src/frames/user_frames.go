package frames

import "time"

type UserSpeakingFrame struct {
	*SystemFrame
}

func NewUserSpeakingFrame() *UserSpeakingFrame {
	return &UserSpeakingFrame{SystemFrame: newSystemFrame("UserSpeakingFrame")}
}

type BotSpeakingFrame struct {
	*SystemFrame
}

func NewBotSpeakingFrame() *BotSpeakingFrame {
	return &BotSpeakingFrame{SystemFrame: newSystemFrame("BotSpeakingFrame")}
}

type VADUserStartedSpeakingFrame struct {
	*SystemFrame
	StartSecs float64
	Timestamp time.Time
}

func NewVADUserStartedSpeakingFrame(startSecs float64) *VADUserStartedSpeakingFrame {
	return &VADUserStartedSpeakingFrame{
		SystemFrame: newSystemFrame("VADUserStartedSpeakingFrame"),
		StartSecs:   startSecs,
		Timestamp:   time.Now(),
	}
}

type VADUserStoppedSpeakingFrame struct {
	*SystemFrame
	StopSecs  float64
	Timestamp time.Time
}

func NewVADUserStoppedSpeakingFrame(stopSecs float64) *VADUserStoppedSpeakingFrame {
	return &VADUserStoppedSpeakingFrame{
		SystemFrame: newSystemFrame("VADUserStoppedSpeakingFrame"),
		StopSecs:    stopSecs,
		Timestamp:   time.Now(),
	}
}

type ProposedUserStartedSpeakingFrame struct {
	*SystemFrame
}

func NewProposedUserStartedSpeakingFrame() *ProposedUserStartedSpeakingFrame {
	return &ProposedUserStartedSpeakingFrame{SystemFrame: newSystemFrame("ProposedUserStartedSpeakingFrame")}
}

type ProposedUserStoppedSpeakingFrame struct {
	*ControlFrame
}

func NewProposedUserStoppedSpeakingFrame() *ProposedUserStoppedSpeakingFrame {
	return &ProposedUserStoppedSpeakingFrame{ControlFrame: newControlFrame("ProposedUserStoppedSpeakingFrame")}
}

type UserTurnResponseCancelFrame struct {
	*SystemFrame
}

func NewUserTurnResponseCancelFrame() *UserTurnResponseCancelFrame {
	return &UserTurnResponseCancelFrame{SystemFrame: newSystemFrame("UserTurnResponseCancelFrame")}
}

type UserTurnInferenceCompletedFrame struct {
	*SystemFrame
}

func NewUserTurnInferenceCompletedFrame() *UserTurnInferenceCompletedFrame {
	return &UserTurnInferenceCompletedFrame{SystemFrame: newSystemFrame("UserTurnInferenceCompletedFrame")}
}

type UserTurnCompletionFrame struct {
	*SystemFrame
	Kind           string
	Marker         string
	ContextFrameID uint64
}

func NewUserTurnCompletionFrame(kind, marker string, contextFrameID uint64) *UserTurnCompletionFrame {
	return &UserTurnCompletionFrame{
		SystemFrame:    newSystemFrame("UserTurnCompletionFrame"),
		Kind:           kind,
		Marker:         marker,
		ContextFrameID: contextFrameID,
	}
}

type PerceptionFrame struct {
	*SystemFrame
	Perception []PerceptionSignal
}

func NewPerceptionFrame(perception []PerceptionSignal) *PerceptionFrame {
	return &PerceptionFrame{
		SystemFrame: newSystemFrame("PerceptionFrame"),
		Perception:  perception,
	}
}
