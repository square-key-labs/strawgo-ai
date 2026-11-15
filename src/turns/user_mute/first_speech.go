package user_mute

import (
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/turns/internal/frameinfo"
)

type FirstSpeechUserMuteStrategy struct {
	enableInterruptions bool

	mu              sync.Mutex
	botSpeaking     bool
	firstSpeechDone bool
}

func NewFirstSpeechUserMuteStrategy(enableInterruptions bool) *FirstSpeechUserMuteStrategy {
	return &FirstSpeechUserMuteStrategy{enableInterruptions: enableInterruptions}
}

func (s *FirstSpeechUserMuteStrategy) ShouldMute(frame any) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch frameinfo.Name(frame) {
	case "BotStartedSpeakingFrame":
		s.botSpeaking = true
	case "BotStoppedSpeakingFrame":
		s.botSpeaking = false
		s.firstSpeechDone = true
	}
	return s.botSpeaking && !s.firstSpeechDone
}

func (s *FirstSpeechUserMuteStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *FirstSpeechUserMuteStrategy) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.botSpeaking = false
	s.firstSpeechDone = false
}
