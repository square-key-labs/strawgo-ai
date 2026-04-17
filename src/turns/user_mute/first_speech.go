package user_mute

import (
	"sync"
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

	name := frameName(frame)
	switch name {
	case "TTSStartedFrame", "BotStartedSpeakingFrame":
		s.botSpeaking = true
	case "BotStoppedSpeakingFrame":
		s.botSpeaking = false
		s.firstSpeechDone = true
	}

	return !s.firstSpeechDone
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

type namedFrame interface {
	Name() string
}

func frameName(frame any) string {
	named, ok := frame.(namedFrame)
	if !ok {
		return ""
	}
	return named.Name()
}
