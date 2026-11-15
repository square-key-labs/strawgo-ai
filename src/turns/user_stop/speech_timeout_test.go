package user_stop_test

import (
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

func waitDeadline(t *testing.T, s *user_stop.SpeechTimeoutUserTurnStopStrategy) {
	t.Helper()
	deadline, ok := s.NextDeadline()
	if !ok {
		t.Fatalf("expected a scheduled deadline")
	}
	time.Sleep(time.Until(deadline) + 5*time.Millisecond)
}

func TestSpeechTimeoutStopStrategy(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30*time.Millisecond, true)

	if strategy.ShouldStop(frames.NewTextFrame("idle")) {
		t.Fatalf("expected idle frame to not stop")
	}
	if _, ok := strategy.NextDeadline(); ok {
		t.Fatalf("expected no deadline before the user stops")
	}
	strategy.ShouldStop(frames.NewUserStartedSpeakingFrame())
	strategy.ShouldStop(frames.NewTranscriptionFrame("hello there", true))
	if strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame()) {
		t.Fatalf("expected stop timer start to not immediately stop")
	}
	if strategy.ShouldStop(nil) {
		t.Fatalf("expected not to stop before timeout")
	}
	waitDeadline(t, strategy)
	if !strategy.ShouldStop(nil) {
		t.Fatalf("expected stop when timeout elapses")
	}
	if strategy.ShouldStop(nil) {
		t.Fatalf("expected one-shot stop")
	}
	if !strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be true")
	}
}

func TestSpeechTimeoutWaitsForTranscript(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(20*time.Millisecond, true)
	strategy.ShouldStop(frames.NewUserStartedSpeakingFrame())
	strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame())
	waitDeadline(t, strategy)
	if strategy.ShouldStop(nil) {
		t.Fatalf("expected no stop without a transcript")
	}
	if !strategy.ShouldStop(frames.NewTranscriptionFrame("late words", true)) {
		t.Fatalf("expected late transcript to end the turn once timers are done")
	}
}

func TestSpeechTimeoutStopStrategyResetsOnNewSpeechAndReset(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30*time.Millisecond, false)
	strategy.ShouldStop(frames.NewTranscriptionFrame("hi", true))
	strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame())
	time.Sleep(10 * time.Millisecond)
	strategy.ShouldStop(frames.NewUserStartedSpeakingFrame())
	time.Sleep(35 * time.Millisecond)
	if strategy.ShouldStop(nil) {
		t.Fatalf("expected started speaking to cancel pending timeout")
	}

	strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame())
	strategy.Reset()
	time.Sleep(35 * time.Millisecond)
	if strategy.ShouldStop(nil) {
		t.Fatalf("expected reset to clear timer")
	}
	if strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be false")
	}
}

func TestSpeechTimeoutTranscriptWithoutVAD(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(20*time.Millisecond, true)
	if strategy.ShouldStop(frames.NewTranscriptionFrame("typed", true)) {
		t.Fatalf("expected fallback timer before stopping")
	}
	waitDeadline(t, strategy)
	if !strategy.ShouldStop(nil) {
		t.Fatalf("expected stop after inactivity since the transcript")
	}
}

func TestSTTMetadata_AutoConfigures(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30*time.Millisecond, true)
	if strategy.ShouldStop(frames.NewSTTMetadataFrame("azure", 200*time.Millisecond)) {
		t.Fatalf("STTMetadataFrame should never trigger stop")
	}
	strategy.ShouldStop(frames.NewUserStartedSpeakingFrame())
	strategy.ShouldStop(frames.NewTranscriptionFrame("hello", false))
	strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame())

	time.Sleep(60 * time.Millisecond)
	if strategy.ShouldStop(nil) {
		t.Fatalf("expected the p99 wait to hold the turn open")
	}
	strategy.ShouldStop(frames.NewTranscriptionFrame("hello", true))
	if strategy.ShouldStop(nil) {
		t.Fatalf("expected one-shot stop already reported by the final transcript")
	}
}

func TestSTTMetadata_FinalTranscriptShortCircuits(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30*time.Millisecond, true)
	strategy.ShouldStop(frames.NewSTTMetadataFrame("azure", 500*time.Millisecond))
	strategy.ShouldStop(frames.NewUserStartedSpeakingFrame())
	strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame())
	time.Sleep(40 * time.Millisecond)
	if !strategy.ShouldStop(frames.NewTranscriptionFrame("done", true)) {
		t.Fatalf("expected final transcript to skip the remaining p99 wait")
	}
}

func TestSTTMetadata_OverridePreventsAutoConfig(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30*time.Millisecond, true)
	strategy.SetTTFSP99Latency(10 * time.Millisecond)
	strategy.ShouldStop(frames.NewSTTMetadataFrame("azure", 500*time.Millisecond))
	strategy.ShouldStop(frames.NewTranscriptionFrame("hi", false))
	strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame())
	strategy.SetWaitForTranscript(false)
	time.Sleep(50 * time.Millisecond)
	if !strategy.ShouldStop(nil) {
		t.Fatalf("expected to stop with the override p99, not the auto-configured one")
	}
}
