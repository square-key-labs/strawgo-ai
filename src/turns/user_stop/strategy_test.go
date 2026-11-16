package user_stop_test

import (
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

type mockStopStrategy struct {
	stop       bool
	resetCount int
}

func (m *mockStopStrategy) ShouldStop(frame any) bool {
	return m.stop && frame != nil
}

func (m *mockStopStrategy) Reset() {
	m.resetCount++
}

var _ user_stop.UserTurnStopStrategy = (*mockStopStrategy)(nil)

func TestUserTurnStopStrategyInterface(t *testing.T) {
	strategy := &mockStopStrategy{stop: true}
	frame := frames.NewUserStoppedSpeakingFrame()

	if !strategy.ShouldStop(frame) {
		t.Fatalf("expected ShouldStop to return true")
	}

	strategy.Reset()
	if strategy.resetCount != 1 {
		t.Fatalf("expected resetCount to be 1, got %d", strategy.resetCount)
	}
}

func eagerTurn(s *user_stop.EagerUserTurnStopStrategy, eager, final string) bool {
	s.HandleUserTurnStarted()
	s.ShouldStop(frames.NewProposedUserStartedSpeakingFrame())
	s.ShouldStop(frames.NewEagerTranscriptionFrame(eager, ""))
	s.ShouldStop(frames.NewTranscriptionFrame(final, true))
	return s.ShouldStop(frames.NewProposedUserStoppedSpeakingFrame())
}

func TestEagerMatchPolicies(t *testing.T) {
	cases := []struct {
		policy       user_stop.EagerMatchPolicy
		eager, final string
		want         bool
	}{
		{user_stop.NormalizedMatch{}, "its a table for two", "It's a table for two.", true},
		{user_stop.NormalizedMatch{}, "book a table", "Book a table, please.", false},
		{user_stop.NormalizedMatch{}, "hello   world", "Hello, world!", true},
		{user_stop.ExactMatch{}, "book a table", "Book a table.", false},
		{user_stop.ExactMatch{}, "book a table", "book a table", true},
	}
	for _, c := range cases {
		if got := c.policy.Matches(c.eager, c.final); got != c.want {
			t.Errorf("%T.Matches(%q, %q) = %v, want %v", c.policy, c.eager, c.final, got, c.want)
		}
	}
}

func TestEagerStopStrategyStartsAndConfirmsSpeculation(t *testing.T) {
	s := user_stop.NewEagerUserTurnStopStrategy(nil, 0)
	s.HandleUserTurnStarted()
	if s.ShouldStop(frames.NewEagerTranscriptionFrame("book a table", "")) {
		t.Fatal("an eager end of turn must not end the turn")
	}
	start, cancel := s.TakeSpeculation()
	if start == nil || start.Text != "book a table" || cancel {
		t.Fatalf("start=%v cancel=%v", start, cancel)
	}
	if _, pending := s.SpeculationDeadline(); !pending {
		t.Fatal("expected a deadline while the speculation is unresolved")
	}
	s.ShouldStop(frames.NewTranscriptionFrame("Book a table.", true))
	if !s.ShouldStop(frames.NewProposedUserStoppedSpeakingFrame()) {
		t.Fatal("the committed end of turn must end the turn")
	}
	if !s.ConfirmsSpeculation() {
		t.Fatal("matching transcript should confirm the speculation")
	}
	if start, cancel := s.TakeSpeculation(); start != nil || cancel {
		t.Fatalf("unexpected events start=%v cancel=%v", start, cancel)
	}
	s.HandleUserTurnStopped()
	if s.ConfirmsSpeculation() {
		t.Fatal("confirmation must not outlive the turn")
	}
}

func TestEagerStopStrategyCancelsOnChangedText(t *testing.T) {
	s := user_stop.NewEagerUserTurnStopStrategy(nil, 0)
	if !eagerTurn(s, "book a table", "Book a table for six.") {
		t.Fatal("expected the turn to end")
	}
	if s.ConfirmsSpeculation() {
		t.Fatal("a different transcript must not confirm")
	}
	if _, cancel := s.TakeSpeculation(); !cancel {
		t.Fatal("expected the speculation to be cancelled")
	}
}

func TestEagerStopStrategyExactMatch(t *testing.T) {
	s := user_stop.NewEagerUserTurnStopStrategy(user_stop.ExactMatch{}, 0)
	eagerTurn(s, "book a table", "Book a table.")
	if s.ConfirmsSpeculation() {
		t.Fatal("ExactMatch must reject a reformatted transcript")
	}
}

func TestEagerStopStrategyResumedSpeechVoidsConfirmation(t *testing.T) {
	s := user_stop.NewEagerUserTurnStopStrategy(nil, 0)
	eagerTurn(s, "book a table", "book a table")
	s.ShouldStop(frames.NewUserStartedSpeakingFrame())
	if s.ConfirmsSpeculation() {
		t.Fatal("speech after the decision must void the confirmation")
	}
}

func TestEagerStopStrategyForgetsWithdrawnSpeculation(t *testing.T) {
	s := user_stop.NewEagerUserTurnStopStrategy(nil, 0)
	s.ShouldStop(frames.NewEagerTranscriptionFrame("book a table", ""))
	s.TakeSpeculation()
	s.ShouldStop(frames.NewUserTurnResponseCancelFrame())
	if _, pending := s.SpeculationDeadline(); pending {
		t.Fatal("a withdrawn speculation must stop its clock")
	}
	s.ShouldStop(frames.NewTranscriptionFrame("book a table", true))
	s.ShouldStop(frames.NewProposedUserStoppedSpeakingFrame())
	if s.ConfirmsSpeculation() {
		t.Fatal("a withdrawn speculation must not be confirmed")
	}
	if _, cancel := s.TakeSpeculation(); cancel {
		t.Fatal("a withdrawn speculation must not be cancelled twice")
	}
}

func TestEagerStopStrategyTimesOutUnresolvedSpeculation(t *testing.T) {
	s := user_stop.NewEagerUserTurnStopStrategy(nil, 20*time.Millisecond)
	s.ShouldStop(frames.NewEagerTranscriptionFrame("book a table", ""))
	s.TakeSpeculation()
	time.Sleep(30 * time.Millisecond)
	s.ShouldStop(nil)
	if _, cancel := s.TakeSpeculation(); !cancel {
		t.Fatal("expected the unresolved speculation to be withdrawn")
	}
}

func TestEagerStopStrategyTurnEndWithdrawsUnresolvedSpeculation(t *testing.T) {
	s := user_stop.NewEagerUserTurnStopStrategy(nil, 0)
	s.ShouldStop(frames.NewEagerTranscriptionFrame("book a table", ""))
	s.TakeSpeculation()
	s.HandleUserTurnStopped()
	if _, cancel := s.TakeSpeculation(); !cancel {
		t.Fatal("expected a turn end to withdraw the unresolved speculation")
	}
	if _, pending := s.SpeculationDeadline(); pending {
		t.Fatal("expected no clock after the turn ended")
	}
}
