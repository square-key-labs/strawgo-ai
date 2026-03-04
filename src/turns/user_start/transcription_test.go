package user_start_test

import (
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
)

func TestTranscriptionStartStrategy(t *testing.T) {
	strategy := user_start.NewTranscriptionUserTurnStartStrategy(false)

	if !strategy.ShouldStart(frames.NewTranscriptionFrame("hello", false)) {
		t.Fatalf("expected transcription frame to trigger start")
	}

	if strategy.ShouldStart(frames.NewUserStartedSpeakingFrame()) {
		t.Fatalf("expected non-transcription frame to not trigger start")
	}

	if strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be false")
	}

	strategy.Reset()
}
