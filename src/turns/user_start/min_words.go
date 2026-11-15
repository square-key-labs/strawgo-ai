package user_start

import (
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/turns/internal/frameinfo"
)

type MinWordsUserTurnStartStrategy struct {
	minWords            int
	enableInterruptions bool

	mu               sync.Mutex
	botSpeaking      bool
	utteranceWords   int
	resetAggregation bool
}

func NewMinWordsUserTurnStartStrategy(minWords int, enableInterruptions bool) *MinWordsUserTurnStartStrategy {
	if minWords < 1 {
		minWords = 1
	}
	return &MinWordsUserTurnStartStrategy{
		minWords:            minWords,
		enableInterruptions: enableInterruptions,
	}
}

func (s *MinWordsUserTurnStartStrategy) ShouldStart(frame any) bool {
	name := frameinfo.Name(frame)
	s.mu.Lock()
	defer s.mu.Unlock()

	switch {
	case name == "BotStartedSpeakingFrame":
		s.botSpeaking = true
		return false
	case name == "BotStoppedSpeakingFrame":
		s.botSpeaking = false
		return false
	case frameinfo.IsUserStarted(name):
		s.utteranceWords = 0
		return false
	case !frameinfo.IsTranscription(name):
		return false
	}

	text, ok := frameinfo.Text(frame)
	if !ok {
		return false
	}
	words := frameinfo.CountWords(text)
	total := s.utteranceWords + words
	final := frameinfo.IsFinal(frame)
	if final {
		s.utteranceWords = total
	}

	threshold := 1
	if s.botSpeaking {
		threshold = s.minWords
	}
	if total >= threshold {
		return true
	}
	if final {
		s.resetAggregation = true
	}
	return false
}

func (s *MinWordsUserTurnStartStrategy) ConsumeAggregationReset() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.resetAggregation
	s.resetAggregation = false
	return r
}

func (s *MinWordsUserTurnStartStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *MinWordsUserTurnStartStrategy) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.botSpeaking = false
	s.utteranceWords = 0
	s.resetAggregation = false
}
