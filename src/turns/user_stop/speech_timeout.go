package user_stop

import (
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/turns/internal/frameinfo"
)

const DefaultUserSpeechTimeout = 600 * time.Millisecond

type SpeechTimeoutUserTurnStopStrategy struct {
	userSpeechTimeout   time.Duration
	enableInterruptions bool
	now                 func() time.Time

	mu                sync.Mutex
	sttTimeout        time.Duration
	p99Override       bool
	waitForTranscript bool

	hasText        bool
	vadSpeaking    bool
	vadStopped     bool
	finalized      bool
	userSpeechAt   time.Time
	userSpeechDone bool
	sttAt          time.Time
	sttDone        bool
	reported       bool
}

func NewSpeechTimeoutUserTurnStopStrategy(userSpeechTimeout time.Duration, enableInterruptions bool) *SpeechTimeoutUserTurnStopStrategy {
	if userSpeechTimeout < 0 {
		userSpeechTimeout = 0
	}
	return &SpeechTimeoutUserTurnStopStrategy{
		userSpeechTimeout:   userSpeechTimeout,
		enableInterruptions: enableInterruptions,
		waitForTranscript:   true,
		now:                 time.Now,
	}
}

func (s *SpeechTimeoutUserTurnStopStrategy) SetWaitForTranscript(wait bool) {
	s.mu.Lock()
	s.waitForTranscript = wait
	s.mu.Unlock()
}

func (s *SpeechTimeoutUserTurnStopStrategy) SetTTFSP99Latency(d time.Duration) {
	s.mu.Lock()
	s.sttTimeout = d
	s.p99Override = true
	s.mu.Unlock()
}

func (s *SpeechTimeoutUserTurnStopStrategy) ShouldStop(frame any) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.now()
	s.expire(now)

	name := frameinfo.Name(frame)
	switch {
	case name == "STTMetadataFrame":
		if d, ok := frameinfo.P99Latency(frame); ok && !s.p99Override {
			s.sttTimeout = d
		}
	case frameinfo.IsUserStarted(name):
		s.vadSpeaking = true
		s.discardPending()
	case frameinfo.IsUserStopped(name):
		s.vadSpeaking = false
		s.vadStopped = true
		s.startUserSpeechTimer(now)
		s.sttDone = false
		s.sttAt = time.Time{}
		if s.finalized || s.sttTimeout <= 0 {
			s.sttDone = true
		} else {
			s.sttAt = now.Add(s.sttTimeout)
		}
		s.expire(now)
	case name == "InterimTranscriptionFrame":
		s.finalized = false
	case name == "TranscriptionFrame":
		s.handleTranscription(frame, now)
	}

	return s.ready()
}

func (s *SpeechTimeoutUserTurnStopStrategy) handleTranscription(frame any, now time.Time) {
	if !frameinfo.IsFinal(frame) {
		s.finalized = false
		return
	}
	text, _ := frameinfo.Text(frame)
	if text == "" {
		return
	}
	s.hasText = true
	s.finalized = true
	if !s.sttDone {
		s.sttDone = true
		s.sttAt = time.Time{}
	}
	if s.userSpeechDone && s.sttDone {
		return
	}
	if !s.vadSpeaking && !s.vadStopped {
		s.sttDone = true
		s.startUserSpeechTimer(now)
		s.expire(now)
	}
}

func (s *SpeechTimeoutUserTurnStopStrategy) startUserSpeechTimer(now time.Time) {
	s.userSpeechDone = false
	s.userSpeechAt = now.Add(s.userSpeechTimeout)
}

func (s *SpeechTimeoutUserTurnStopStrategy) expire(now time.Time) {
	if !s.userSpeechAt.IsZero() && !now.Before(s.userSpeechAt) {
		s.userSpeechAt = time.Time{}
		s.userSpeechDone = true
	}
	if !s.sttAt.IsZero() && !now.Before(s.sttAt) {
		s.sttAt = time.Time{}
		s.sttDone = true
	}
}

func (s *SpeechTimeoutUserTurnStopStrategy) ready() bool {
	if s.reported || s.vadSpeaking {
		return false
	}
	if s.waitForTranscript && !s.hasText {
		return false
	}
	if !s.userSpeechDone || !s.sttDone {
		return false
	}
	s.reported = true
	return true
}

func (s *SpeechTimeoutUserTurnStopStrategy) discardPending() {
	s.finalized = false
	s.vadStopped = false
	s.userSpeechDone = false
	s.sttDone = false
	s.userSpeechAt = time.Time{}
	s.sttAt = time.Time{}
	s.reported = false
}

func (s *SpeechTimeoutUserTurnStopStrategy) NextDeadline() (time.Time, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var next time.Time
	for _, t := range [2]time.Time{s.userSpeechAt, s.sttAt} {
		if !t.IsZero() && (next.IsZero() || t.Before(next)) {
			next = t
		}
	}
	return next, !next.IsZero()
}

func (s *SpeechTimeoutUserTurnStopStrategy) HandleUserTurnStarted() {
	s.mu.Lock()
	s.hasText = false
	s.discardPending()
	s.mu.Unlock()
}

func (s *SpeechTimeoutUserTurnStopStrategy) HandleUserTurnStopped() {
	s.Reset()
}

func (s *SpeechTimeoutUserTurnStopStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *SpeechTimeoutUserTurnStopStrategy) Reset() {
	s.mu.Lock()
	s.hasText = false
	s.vadSpeaking = false
	s.discardPending()
	s.mu.Unlock()
}
