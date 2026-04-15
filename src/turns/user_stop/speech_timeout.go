package user_stop

import (
	"sync"
	"time"
)

type SpeechTimeoutUserTurnStopStrategy struct {
	timeout             time.Duration
	userSpeechTimeout   time.Duration
	sttP99Latency       time.Duration // P99 STT latency, auto-configured from STTMetadataFrame
	p99Override         bool          // If true, sttP99Latency was explicitly set
	enableInterruptions bool
	now                 func() time.Time

	mu           sync.Mutex
	stopDeadline time.Time
	timerStarted bool
}

func NewSpeechTimeoutUserTurnStopStrategy(timeout time.Duration, enableInterruptions bool) *SpeechTimeoutUserTurnStopStrategy {
	if timeout < 0 {
		timeout = 0
	}

	return &SpeechTimeoutUserTurnStopStrategy{
		timeout:             timeout,
		userSpeechTimeout:   600 * time.Millisecond, // Default 0.6s
		enableInterruptions: enableInterruptions,
		now:                 time.Now,
	}
}

func (s *SpeechTimeoutUserTurnStopStrategy) ShouldStop(frame any) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Handle STTMetadataFrame for auto-configuring timeout
	if provider, ok := frame.(sttLatencyProvider); ok {
		if nf, ok2 := frame.(namedFrame); ok2 && nf.Name() == "STTMetadataFrame" && !s.p99Override {
			s.sttP99Latency = provider.GetTTFSP99Latency()
		}
		return false
	}

	switch frame.(type) {
	case namedFrame:
		name := frame.(namedFrame).Name()
		if name == "UserStoppedSpeakingFrame" {
			s.stopDeadline = s.now().Add(s.timeout + s.sttP99Latency)
			s.timerStarted = true
			return false
		}
		if name == "UserStartedSpeakingFrame" {
			s.timerStarted = false
			return false
		}
		// When a final transcript arrives while the turn timer is running, stop immediately.
		// The transcript proves the utterance is complete — no need to wait out the p99 window.
		if name == "TranscriptionFrame" && s.timerStarted {
			if fp, ok := frame.(finalTranscriptionProvider); ok && fp.IsTranscriptionFinal() {
				s.timerStarted = false
				return true
			}
		}
	}

	if s.timerStarted && !s.now().Before(s.stopDeadline) {
		s.timerStarted = false
		return true
	}

	return false
}

func (s *SpeechTimeoutUserTurnStopStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *SpeechTimeoutUserTurnStopStrategy) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.timerStarted = false
	s.stopDeadline = time.Time{}
}

type namedFrame interface {
	Name() string
}

// sttLatencyProvider is satisfied by frames.STTMetadataFrame via GetTTFSP99Latency()
type sttLatencyProvider interface {
	GetTTFSP99Latency() time.Duration
}

// finalTranscriptionProvider is satisfied by frames.TranscriptionFrame via IsTranscriptionFinal().
// Used to avoid importing the frames package directly.
type finalTranscriptionProvider interface {
	IsTranscriptionFinal() bool
}

// SetTTFSP99Latency explicitly sets the P99 STT latency, preventing auto-configuration from STTMetadataFrame.
func (s *SpeechTimeoutUserTurnStopStrategy) SetTTFSP99Latency(d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sttP99Latency = d
	s.p99Override = true
}
