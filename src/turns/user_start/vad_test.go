package user_start_test

import (
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
)

func TestVADStartStrategy(t *testing.T) {
	strategy := user_start.NewVADUserTurnStartStrategy(true)

	if !strategy.ShouldStart(frames.NewUserStartedSpeakingFrame()) {
		t.Fatalf("expected user started speaking frame to trigger start")
	}

	if strategy.ShouldStart(frames.NewTextFrame("hello")) {
		t.Fatalf("expected non-VAD frame to not trigger start")
	}

	if !strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be true")
	}

	strategy.Reset()
	if strategy.ShouldStart(frames.NewTextFrame("after reset")) {
		t.Fatalf("expected reset to preserve non-trigger behavior for other frames")
	}
}
