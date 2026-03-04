package user_mute_test

import (
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
)

type mockMuteStrategy struct {
	mute       bool
	resetCount int
}

func (m *mockMuteStrategy) ShouldMute(frame any) bool {
	return m.mute && frame != nil
}

func (m *mockMuteStrategy) Reset() {
	m.resetCount++
}

var _ user_mute.UserMuteStrategy = (*mockMuteStrategy)(nil)

func TestUserMuteStrategyInterface(t *testing.T) {
	strategy := &mockMuteStrategy{mute: true}
	frame := frames.NewUserMuteStartedFrame()

	if !strategy.ShouldMute(frame) {
		t.Fatalf("expected ShouldMute to return true")
	}

	strategy.Reset()
	if strategy.resetCount != 1 {
		t.Fatalf("expected resetCount to be 1, got %d", strategy.resetCount)
	}
}
