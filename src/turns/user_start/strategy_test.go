package user_start_test

import (
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
)

type mockStartStrategy struct {
	start               bool
	enableInterruptions bool
	resetCount          int
}

func (m *mockStartStrategy) ShouldStart(frame any) bool {
	return m.start && frame != nil
}

func (m *mockStartStrategy) EnableInterruptions() bool {
	return m.enableInterruptions
}

func (m *mockStartStrategy) Reset() {
	m.resetCount++
}

var _ user_start.UserTurnStartStrategy = (*mockStartStrategy)(nil)

func TestUserTurnStartStrategyInterface(t *testing.T) {
	strategy := &mockStartStrategy{start: true, enableInterruptions: true}
	frame := frames.NewUserStartedSpeakingFrame()

	if !strategy.ShouldStart(frame) {
		t.Fatalf("expected ShouldStart to return true")
	}

	if !strategy.EnableInterruptions() {
		t.Fatalf("expected EnableInterruptions to return true")
	}

	strategy.Reset()
	if strategy.resetCount != 1 {
		t.Fatalf("expected resetCount to be 1, got %d", strategy.resetCount)
	}
}
