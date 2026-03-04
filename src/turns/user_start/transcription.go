package user_start

type TranscriptionUserTurnStartStrategy struct {
	enableInterruptions bool
}

func NewTranscriptionUserTurnStartStrategy(enableInterruptions bool) *TranscriptionUserTurnStartStrategy {
	return &TranscriptionUserTurnStartStrategy{enableInterruptions: enableInterruptions}
}

func (s *TranscriptionUserTurnStartStrategy) ShouldStart(frame any) bool {
	return frameHasName(frame, "TranscriptionFrame")
}

func (s *TranscriptionUserTurnStartStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *TranscriptionUserTurnStartStrategy) Reset() {}
