package user_start

import (
	"reflect"
	"strings"
	"sync"
)

type MinWordsUserTurnStartStrategy struct {
	minWords            int
	enableInterruptions bool

	mu        sync.Mutex
	wordCount int
	triggered bool
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
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.triggered {
		return false
	}

	text, ok := transcriptionText(frame)
	if !ok {
		return false
	}

	s.wordCount += len(strings.Fields(text))
	if s.wordCount >= s.minWords {
		s.triggered = true
		return true
	}

	return false
}

func (s *MinWordsUserTurnStartStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *MinWordsUserTurnStartStrategy) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.wordCount = 0
	s.triggered = false
}

func transcriptionText(frame any) (string, bool) {
	if !frameHasName(frame, "TranscriptionFrame") {
		return "", false
	}

	v := reflect.ValueOf(frame)
	if !v.IsValid() {
		return "", false
	}
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return "", false
		}
		v = v.Elem()
	}

	if v.Kind() != reflect.Struct {
		return "", false
	}

	textField := v.FieldByName("Text")
	if !textField.IsValid() || textField.Kind() != reflect.String {
		return "", false
	}

	return textField.String(), true
}
