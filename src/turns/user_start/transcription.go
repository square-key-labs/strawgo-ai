package user_start

import "github.com/square-key-labs/strawgo-ai/src/turns/internal/frameinfo"

type TranscriptionUserTurnStartStrategy struct {
	enableInterruptions bool
}

func NewTranscriptionUserTurnStartStrategy(enableInterruptions bool) *TranscriptionUserTurnStartStrategy {
	return &TranscriptionUserTurnStartStrategy{enableInterruptions: enableInterruptions}
}

func (s *TranscriptionUserTurnStartStrategy) ShouldStart(frame any) bool {
	if !frameinfo.IsTranscription(frameinfo.Name(frame)) {
		return false
	}
	text, _ := frameinfo.Text(frame)
	return text != ""
}

func (s *TranscriptionUserTurnStartStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *TranscriptionUserTurnStartStrategy) Reset() {}
