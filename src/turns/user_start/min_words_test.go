package user_start_test

import (
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
)

func TestMinWordsStartStrategy(t *testing.T) {
	strategy := user_start.NewMinWordsUserTurnStartStrategy(3, true)

	if strategy.ShouldStart(frames.NewTextFrame("ignore")) {
		t.Fatalf("expected non-transcription frame to not trigger")
	}

	if strategy.ShouldStart(frames.NewTranscriptionFrame("hello", false)) {
		t.Fatalf("expected one word to not trigger")
	}

	if !strategy.ShouldStart(frames.NewTranscriptionFrame("there now", true)) {
		t.Fatalf("expected threshold to trigger")
	}

	if strategy.ShouldStart(frames.NewTranscriptionFrame("extra words", true)) {
		t.Fatalf("expected strategy to trigger once before reset")
	}

	strategy.Reset()

	if strategy.ShouldStart(frames.NewTranscriptionFrame("two words", true)) {
		t.Fatalf("expected reset strategy to wait for threshold again")
	}

	if !strategy.ShouldStart(frames.NewTranscriptionFrame("third", true)) {
		t.Fatalf("expected trigger after reset and reaching threshold")
	}

	if !strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be true")
	}
}

func TestMinWordsStartStrategyMinValueBoundary(t *testing.T) {
	strategy := user_start.NewMinWordsUserTurnStartStrategy(0, false)

	if !strategy.ShouldStart(frames.NewTranscriptionFrame("one", false)) {
		t.Fatalf("expected min words to clamp to one")
	}

	if strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be false")
	}
}
