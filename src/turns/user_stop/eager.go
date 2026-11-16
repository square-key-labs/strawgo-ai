package user_stop

import (
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/square-key-labs/strawgo-ai/src/turns/internal/frameinfo"
)

const DefaultSpeculationTimeout = 5 * time.Second

type UserTurnSpeculation struct {
	Text string
}

type Speculator interface {
	TakeSpeculation() (start *UserTurnSpeculation, cancel bool)
	ConfirmsSpeculation() bool
	SpeculationDeadline() (time.Time, bool)
}

type EagerMatchPolicy interface {
	Matches(eager, final string) bool
}

type ExactMatch struct{}

func (ExactMatch) Matches(eager, final string) bool {
	return eager == final
}

type NormalizedMatch struct{}

func (NormalizedMatch) Matches(eager, final string) bool {
	return normalizeTranscript(eager) == normalizeTranscript(final)
}

func normalizeTranscript(text string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(text) {
		switch {
		case r == '\'' || r == '’' || r == 'ʼ':
		case unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsMark(r) || r == '_':
			b.WriteRune(r)
		default:
			b.WriteRune(' ')
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

type EagerUserTurnStopStrategy struct {
	policy  EagerMatchPolicy
	timeout time.Duration
	now     func() time.Time

	mu          sync.Mutex
	speculation *UserTurnSpeculation
	deadline    time.Time
	confirmed   *UserTurnSpeculation
	committed   []string
	started     *UserTurnSpeculation
	cancelled   bool
	inTurn      bool
}

func NewEagerUserTurnStopStrategy(policy EagerMatchPolicy, speculationTimeout time.Duration) *EagerUserTurnStopStrategy {
	if policy == nil {
		policy = NormalizedMatch{}
	}
	if speculationTimeout <= 0 {
		speculationTimeout = DefaultSpeculationTimeout
	}
	return &EagerUserTurnStopStrategy{policy: policy, timeout: speculationTimeout, now: time.Now}
}

func (s *EagerUserTurnStopStrategy) MatchPolicy() EagerMatchPolicy {
	return s.policy
}

func (s *EagerUserTurnStopStrategy) ShouldStop(frame any) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.now()
	if s.speculation != nil && !now.Before(s.deadline) {
		s.speculation = nil
		s.cancelled = true
	}

	name := frameinfo.Name(frame)
	switch {
	case name == "EagerTranscriptionFrame":
		text, _ := frameinfo.Text(frame)
		if strings.TrimSpace(text) == "" {
			return false
		}
		spec := &UserTurnSpeculation{Text: text}
		s.speculation, s.confirmed, s.committed = spec, nil, nil
		s.deadline = now.Add(s.timeout)
		s.started = nil
		if s.inTurn {
			s.started = spec
		}
	case name == "UserTurnResponseCancelFrame":
		s.speculation, s.confirmed, s.started = nil, nil, nil
	case frameinfo.IsUserStarted(name) || name == "ProposedUserStartedSpeakingFrame":
		s.confirmed = nil
	case name == "TranscriptionFrame" && frameinfo.IsFinal(frame):
		if text, _ := frameinfo.Text(frame); strings.TrimSpace(text) != "" {
			s.committed = append(s.committed, strings.TrimSpace(text))
		}
	case name == "ProposedUserStoppedSpeakingFrame":
		s.decide()
		return true
	}
	return false
}

func (s *EagerUserTurnStopStrategy) decide() {
	spec := s.speculation
	s.speculation = nil
	if spec == nil {
		spec = s.confirmed
	}
	if spec == nil {
		return
	}
	if s.policy.Matches(spec.Text, strings.Join(s.committed, " ")) {
		s.confirmed = spec
		return
	}
	s.confirmed = nil
	s.cancelled = true
}

func (s *EagerUserTurnStopStrategy) TakeSpeculation() (*UserTurnSpeculation, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	start, cancel := s.started, s.cancelled
	s.started, s.cancelled = nil, false
	return start, cancel
}

func (s *EagerUserTurnStopStrategy) ConfirmsSpeculation() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.confirmed != nil
}

func (s *EagerUserTurnStopStrategy) SpeculationDeadline() (time.Time, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.speculation == nil {
		return time.Time{}, false
	}
	return s.deadline, true
}

func (s *EagerUserTurnStopStrategy) HandleUserTurnStarted() {
	s.mu.Lock()
	early := s.speculation
	s.mu.Unlock()
	s.reset()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inTurn = true
	if early != nil {
		s.speculation, s.started, s.cancelled = early, early, false
	}
}

func (s *EagerUserTurnStopStrategy) HandleUserTurnStopped() {
	s.reset()
	s.mu.Lock()
	s.inTurn = false
	s.mu.Unlock()
}

func (s *EagerUserTurnStopStrategy) Reset() {
	s.reset()
}

func (s *EagerUserTurnStopStrategy) reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.speculation != nil {
		s.cancelled = true
	}
	s.speculation, s.confirmed, s.committed, s.started = nil, nil, nil, nil
}
