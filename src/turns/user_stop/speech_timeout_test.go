package user_stop_test

import (
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
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
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30 * time.Millisecond)

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
}

func TestSpeechTimeoutWaitsForTranscript(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(20 * time.Millisecond)
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
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30 * time.Millisecond)
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
}

func TestSpeechTimeoutTranscriptWithoutVAD(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(20 * time.Millisecond)
	if strategy.ShouldStop(frames.NewTranscriptionFrame("typed", true)) {
		t.Fatalf("expected fallback timer before stopping")
	}
	waitDeadline(t, strategy)
	if !strategy.ShouldStop(nil) {
		t.Fatalf("expected stop after inactivity since the transcript")
	}
}

func TestSTTMetadata_AutoConfigures(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30 * time.Millisecond)
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
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30 * time.Millisecond)
	strategy.ShouldStop(frames.NewSTTMetadataFrame("azure", 500*time.Millisecond))
	strategy.ShouldStop(frames.NewUserStartedSpeakingFrame())
	strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame())
	time.Sleep(40 * time.Millisecond)
	if !strategy.ShouldStop(frames.NewTranscriptionFrame("done", true)) {
		t.Fatalf("expected final transcript to skip the remaining p99 wait")
	}
}

func TestSTTMetadata_OverridePreventsAutoConfig(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30 * time.Millisecond)
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

func analyzerPause(s *user_stop.SpeechTimeoutUserTurnStopStrategy, turnParams any) {
	s.ShouldStop(frames.NewSpeechControlParamsFrame(&frames.VADParams{StopSecs: 0.2}, turnParams))
	s.ShouldStop(frames.NewUserStartedSpeakingFrame())
	s.ShouldStop(frames.NewVADUserStartedSpeakingFrame(0.2))
	s.ShouldStop(frames.NewTranscriptionFrame("my number is", true))
	s.ShouldStop(frames.NewVADUserStoppedSpeakingFrame(0.2))
}

func TestSpeechTimeoutWaitsForTurnAnalyzerAtPause(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(20 * time.Millisecond)
	analyzerPause(strategy, turn.TurnAnalyzerParams{StopSecs: 3})
	waitDeadline(t, strategy)
	if strategy.ShouldStop(nil) {
		t.Fatalf("expected the turn to stay open while the turn analyzer has not said complete")
	}
	if _, ok := strategy.NextDeadline(); ok {
		t.Fatalf("expected no pending deadline while waiting for the turn analyzer")
	}

	strategy.ShouldStop(frames.NewUserStartedSpeakingFrame())
	strategy.ShouldStop(frames.NewVADUserStartedSpeakingFrame(0.2))
	strategy.ShouldStop(frames.NewTranscriptionFrame("five five five", true))
	strategy.ShouldStop(frames.NewVADUserStoppedSpeakingFrame(0.2))
	if strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame()) {
		t.Fatalf("expected the speech timeout to still apply after the analyzer says complete")
	}
	waitDeadline(t, strategy)
	if !strategy.ShouldStop(nil) {
		t.Fatalf("expected stop once the analyzer said complete and the timeout elapsed")
	}
}

func TestSpeechTimeoutStopsWhenTurnAnalyzerCompletesAfterTimeout(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(20 * time.Millisecond)
	analyzerPause(strategy, turn.TurnAnalyzerParams{StopSecs: 3})
	waitDeadline(t, strategy)
	if strategy.ShouldStop(nil) {
		t.Fatalf("expected no stop before the turn analyzer says complete")
	}
	if !strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame()) {
		t.Fatalf("expected stop as soon as the turn analyzer says complete")
	}
}

func TestSpeechTimeoutWithoutTurnAnalyzerStopsAtPause(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(20 * time.Millisecond)
	analyzerPause(strategy, nil)
	waitDeadline(t, strategy)
	if !strategy.ShouldStop(nil) {
		t.Fatalf("expected stop after the timeout when no turn analyzer is attached")
	}
}

func TestSpeechTimeoutWithTurnAnalyzerStopsOnTranscriptWithoutVAD(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(20 * time.Millisecond)
	strategy.ShouldStop(frames.NewSpeechControlParamsFrame(&frames.VADParams{StopSecs: 0.2}, turn.TurnAnalyzerParams{StopSecs: 3}))
	strategy.ShouldStop(frames.NewTranscriptionFrame("yes", true))
	waitDeadline(t, strategy)
	if !strategy.ShouldStop(nil) {
		t.Fatalf("expected stop for a transcript the VAD never saw")
	}
}
