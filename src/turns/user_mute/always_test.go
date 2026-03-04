package user_mute_test

import (
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
)

func TestAlwaysMuteStrategy(t *testing.T) {
	strategy := user_mute.NewAlwaysUserMuteStrategy(true)

	if !strategy.ShouldMute(frames.NewTextFrame("test")) {
		t.Fatalf("expected always mute for text frame")
	}

	if !strategy.ShouldMute(frames.NewTTSStartedFrame()) {
		t.Fatalf("expected always mute for control frame")
	}

	strategy.Reset()

	if !strategy.ShouldMute(frames.NewFunctionCallsStartedFrame(nil)) {
		t.Fatalf("expected always mute after reset")
	}

	if !strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be true")
	}
}
