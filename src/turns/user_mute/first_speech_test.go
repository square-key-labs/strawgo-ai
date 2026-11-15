package user_mute_test

import (
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
)

func TestFirstSpeechMuteStrategy(t *testing.T) {
	strategy := user_mute.NewFirstSpeechUserMuteStrategy()

	if strategy.ShouldMute(frames.NewTextFrame("before first speech")) {
		t.Fatalf("expected no mute before the bot starts speaking")
	}
	if !strategy.ShouldMute(frames.NewBotStartedSpeakingFrame()) {
		t.Fatalf("expected mute while first speech is active")
	}
	if !strategy.ShouldMute(frames.NewInterruptionFrame()) {
		t.Fatalf("expected mute to persist through other frames")
	}
	if strategy.ShouldMute(frames.NewBotStoppedSpeakingFrame()) {
		t.Fatalf("expected unmute after first speech completes")
	}
	if strategy.ShouldMute(frames.NewBotStartedSpeakingFrame()) {
		t.Fatalf("expected to stay unmuted for later speech")
	}

	strategy.Reset()
	if !strategy.ShouldMute(frames.NewBotStartedSpeakingFrame()) {
		t.Fatalf("expected reset to restore initial state")
	}
}

func TestMuteUntilFirstBotCompleteStrategy(t *testing.T) {
	strategy := user_mute.NewMuteUntilFirstBotCompleteUserMuteStrategy()

	if !strategy.ShouldMute(frames.NewTextFrame("before")) {
		t.Fatalf("expected mute from the start")
	}
	strategy.ShouldMute(frames.NewBotStartedSpeakingFrame())
	if strategy.ShouldMute(frames.NewBotStoppedSpeakingFrame()) {
		t.Fatalf("expected unmute after first bot speech")
	}

	failed := user_mute.NewMuteUntilFirstBotCompleteUserMuteStrategy()
	if failed.ShouldMute(frames.NewErrorFrame(nil)) {
		t.Fatalf("expected an error before any speech to release the mute")
	}
}
