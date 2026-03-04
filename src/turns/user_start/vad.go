package user_start

type VADUserTurnStartStrategy struct {
	enableInterruptions bool
}

func NewVADUserTurnStartStrategy(enableInterruptions bool) *VADUserTurnStartStrategy {
	return &VADUserTurnStartStrategy{enableInterruptions: enableInterruptions}
}

func (s *VADUserTurnStartStrategy) ShouldStart(frame any) bool {
	return frameHasName(frame, "UserStartedSpeakingFrame")
}

func (s *VADUserTurnStartStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *VADUserTurnStartStrategy) Reset() {}

type namedFrame interface {
	Name() string
}

func frameHasName(frame any, name string) bool {
	named, ok := frame.(namedFrame)
	if !ok {
		return false
	}

	return named.Name() == name
}
