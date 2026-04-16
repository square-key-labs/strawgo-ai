package user_mute_test

import (
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
)

func TestFirstSpeechMuteStrategy(t *testing.T) {
	strategy := user_mute.NewFirstSpeechUserMuteStrategy(true)

	if !strategy.ShouldMute(frames.NewTextFrame("before first speech")) {
		t.Fatalf("expected mute before first bot speech completes")
	}

	if !strategy.ShouldMute(frames.NewTTSStartedFrame()) {
		t.Fatalf("expected mute while first speech is active")
	}

	if strategy.ShouldMute(frames.NewBotStoppedSpeakingFrame()) {
		t.Fatalf("expected unmute after first speech completes")
	}

	if strategy.ShouldMute(frames.NewTextFrame("after first speech")) {
		t.Fatalf("expected to stay unmuted after first speech")
	}

	strategy.Reset()

	if !strategy.ShouldMute(frames.NewTextFrame("after reset")) {
		t.Fatalf("expected reset to restore initial mute state")
	}

	if !strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be true")
	}
}
