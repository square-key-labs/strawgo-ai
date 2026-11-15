package user_start

import "github.com/square-key-labs/strawgo-ai/src/turns/internal/frameinfo"

type VADUserTurnStartStrategy struct {
	enableInterruptions bool
}

func NewVADUserTurnStartStrategy(enableInterruptions bool) *VADUserTurnStartStrategy {
	return &VADUserTurnStartStrategy{enableInterruptions: enableInterruptions}
}

func (s *VADUserTurnStartStrategy) ShouldStart(frame any) bool {
	return frameinfo.IsUserStarted(frameinfo.Name(frame))
}

func (s *VADUserTurnStartStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *VADUserTurnStartStrategy) Reset() {}
