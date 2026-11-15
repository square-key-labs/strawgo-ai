package user_start_test

import (
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
)

func TestMinWordsStartStrategyBotSpeaking(t *testing.T) {
	strategy := user_start.NewMinWordsUserTurnStartStrategy(3, true)
	strategy.ShouldStart(frames.NewBotStartedSpeakingFrame())

	if strategy.ShouldStart(frames.NewTextFrame("ignore these many words")) {
		t.Fatalf("expected non-transcription frame to not trigger")
	}
	if strategy.ShouldStart(frames.NewTranscriptionFrame("hello", true)) {
		t.Fatalf("expected one word to not trigger while bot speaks")
	}
	if !strategy.ConsumeAggregationReset() {
		t.Fatalf("expected rejected final transcript to request an aggregation reset")
	}
	if !strategy.ShouldStart(frames.NewTranscriptionFrame("there now", true)) {
		t.Fatalf("expected words of the same utterance to reach the threshold")
	}
	if !strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be true")
	}
}

func TestMinWordsStartStrategyCountsCurrentUtteranceOnly(t *testing.T) {
	strategy := user_start.NewMinWordsUserTurnStartStrategy(3, true)
	strategy.ShouldStart(frames.NewBotStartedSpeakingFrame())

	strategy.ShouldStart(frames.NewUserStartedSpeakingFrame())
	if strategy.ShouldStart(frames.NewTranscriptionFrame("yeah", true)) {
		t.Fatalf("expected one word to not trigger")
	}
	strategy.ShouldStart(frames.NewUserStartedSpeakingFrame())
	if strategy.ShouldStart(frames.NewTranscriptionFrame("uh huh", true)) {
		t.Fatalf("expected words of a previous utterance not to count")
	}
	strategy.ShouldStart(frames.NewUserStartedSpeakingFrame())
	for i := 0; i < 3; i++ {
		if strategy.ShouldStart(frames.NewTranscriptionFrame("ok", false)) {
			t.Fatalf("expected repeated interim results not to accumulate")
		}
	}
	if !strategy.ShouldStart(frames.NewTranscriptionFrame("wait a second", false)) {
		t.Fatalf("expected interim with enough words to trigger")
	}
}

func TestMinWordsStartStrategyBotSilent(t *testing.T) {
	strategy := user_start.NewMinWordsUserTurnStartStrategy(0, false)

	if !strategy.ShouldStart(frames.NewTranscriptionFrame("one", false)) {
		t.Fatalf("expected a single word to trigger when the bot is silent")
	}
	if strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be false")
	}
	strategy.ShouldStart(frames.NewBotStartedSpeakingFrame())
	strategy.Reset()
	if !strategy.ShouldStart(frames.NewTranscriptionFrame("one", true)) {
		t.Fatalf("expected reset to clear bot speaking state")
	}
}
