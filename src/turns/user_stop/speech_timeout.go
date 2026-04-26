package user_stop

import (
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type SpeechTimeoutUserTurnStopStrategy struct {
	timeout             time.Duration
	userSpeechTimeout   time.Duration
	sttP99Latency       time.Duration // P99 STT latency, auto-configured from STTMetadataFrame
	p99Override         bool          // If true, sttP99Latency was explicitly set
	enableInterruptions bool
	now                 func() time.Time

	// vadStopSecs, when > 0, lets the strategy emit a warning if it exceeds
	// the observed STT TTFS-P99 latency. When VAD's stop window is already
	// longer than the time STT typically takes to emit a final transcript,
	// the strategy's "wait for STT to confirm" timeout collapses to 0 and
	// turn detection is delayed. Mirrors pipecat #4115.
	vadStopSecs time.Duration
	// vadWarnFired tracks whether we already emitted the >= p99 warning
	// for the current vadStopSecs/p99 pair so we don't spam the log.
	vadWarnFired bool
	// warnFn is the function used to emit warnings; defaults to logger.Warn.
	// Tests can override via SetWarnFunc to capture invocations.
	warnFn func(format string, args ...interface{})

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
		warnFn:              logger.Warn,
	}
}

// SetWarnFunc overrides the function used to emit warnings. Intended for
// tests; production callers should leave it at the default (logger.Warn).
func (s *SpeechTimeoutUserTurnStopStrategy) SetWarnFunc(fn func(format string, args ...interface{})) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if fn == nil {
		fn = logger.Warn
	}
	s.warnFn = fn
}

func (s *SpeechTimeoutUserTurnStopStrategy) ShouldStop(frame any) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Handle STTMetadataFrame for auto-configuring timeout
	if provider, ok := frame.(sttLatencyProvider); ok {
		if nf, ok2 := frame.(namedFrame); ok2 && nf.Name() == "STTMetadataFrame" && !s.p99Override {
			s.sttP99Latency = provider.GetTTFSP99Latency()
			warn, format, args, fired := s.evalVADWarnLocked()
			if fired {
				// Drop lock before calling warn (defer Unlock above unlocks
				// after this function returns; we briefly release-reacquire
				// to avoid holding the lock across user code).
				s.mu.Unlock()
				warn(format, args...)
				s.mu.Lock()
			}
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
	s.sttP99Latency = d
	s.p99Override = true
	warn, format, args, fired := s.evalVADWarnLocked()
	s.mu.Unlock()
	if fired {
		warn(format, args...)
	}
}

// SetVADStopSecs informs the strategy of the VAD analyzer's stop_secs
// configuration so it can warn when the configured VAD stop window is
// >= STT TTFS-P99 latency. When that condition holds, the strategy's
// "wait for STT to emit a final transcript" timeout collapses to 0,
// causing delayed end-of-turn detection. Mirrors pipecat #4115.
//
// Recommended call site: wherever your pipeline builder wires the VAD
// analyzer to the turn strategies. Example:
//
//	vadParams := vad.DefaultVADParams()
//	vadParams.StopSecs = 0.2
//	stop := user_stop.NewSpeechTimeoutUserTurnStopStrategy(...)
//	stop.SetVADStopSecs(time.Duration(vadParams.StopSecs * float32(time.Second)))
//
// If never called, the strategy silently skips this warning category.
func (s *SpeechTimeoutUserTurnStopStrategy) SetVADStopSecs(d time.Duration) {
	s.mu.Lock()
	if s.vadStopSecs != d {
		s.vadStopSecs = d
		s.vadWarnFired = false
	}
	warn, format, args, fired := s.evalVADWarnLocked()
	s.mu.Unlock()
	if fired {
		warn(format, args...)
	}
}

// evalVADWarnLocked must be called with s.mu held. It returns whether a
// warning is due plus the values to call the warn function with. The
// caller must drop the lock before invoking warn to avoid deadlocking
// against any reentrant call into the strategy from inside warnFn.
func (s *SpeechTimeoutUserTurnStopStrategy) evalVADWarnLocked() (
	warn func(string, ...interface{}),
	format string,
	args []interface{},
	fired bool,
) {
	warn = s.warnFn
	if warn == nil {
		warn = logger.Warn
	}
	if s.vadWarnFired {
		return warn, "", nil, false
	}
	if s.vadStopSecs <= 0 || s.sttP99Latency <= 0 {
		return warn, "", nil, false
	}
	if s.vadStopSecs >= s.sttP99Latency {
		s.vadWarnFired = true
		return warn,
			"[SpeechTimeoutUserTurnStop] VAD stop_secs (%v) is >= STT TTFS-P99 latency (%v). " +
				"This collapses the strategy's STT-wait timeout to 0 and may delay turn detection. " +
				"Re-run https://github.com/pipecat-ai/stt-benchmark with your VAD settings.",
			[]interface{}{s.vadStopSecs, s.sttP99Latency},
			true
	}
	return warn, "", nil, false
}
