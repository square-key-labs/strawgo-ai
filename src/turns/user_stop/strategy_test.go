package user_stop_test

import (
	"testing"

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
