package user_mute

import (
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/turns/internal/frameinfo"
)

type MuteUntilFirstBotCompleteUserMuteStrategy struct {
	mu                 sync.Mutex
	botStartedSpeaking bool
	firstSpeechDone    bool
}

func NewMuteUntilFirstBotCompleteUserMuteStrategy() *MuteUntilFirstBotCompleteUserMuteStrategy {
	return &MuteUntilFirstBotCompleteUserMuteStrategy{}
}

func (s *MuteUntilFirstBotCompleteUserMuteStrategy) ShouldMute(frame any) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch frameinfo.Name(frame) {
	case "BotStartedSpeakingFrame":
		s.botStartedSpeaking = true
	case "BotStoppedSpeakingFrame":
		s.firstSpeechDone = true
	case "ErrorFrame":
		if !s.botStartedSpeaking {
			s.firstSpeechDone = true
		}
	}
	return !s.firstSpeechDone
}

func (s *MuteUntilFirstBotCompleteUserMuteStrategy) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.botStartedSpeaking = false
	s.firstSpeechDone = false
}
