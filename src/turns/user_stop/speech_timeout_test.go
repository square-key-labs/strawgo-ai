package user_stop_test

import (
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

func TestSpeechTimeoutStopStrategy(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30*time.Millisecond, true)

	if strategy.ShouldStop(frames.NewTextFrame("idle")) {
		t.Fatalf("expected idle frame to not stop")
	}

	if strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame()) {
		t.Fatalf("expected stop timer start to not immediately stop")
	}

	time.Sleep(10 * time.Millisecond)
	if strategy.ShouldStop(frames.NewTextFrame("still waiting")) {
		t.Fatalf("expected not to stop before timeout")
	}

	time.Sleep(35 * time.Millisecond)
	if !strategy.ShouldStop(frames.NewTextFrame("timeout reached")) {
		t.Fatalf("expected stop when timeout elapses")
	}

	if strategy.ShouldStop(frames.NewTextFrame("already fired")) {
		t.Fatalf("expected one-shot stop until new user stop event")
	}

	if !strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be true")
	}
}

func TestSpeechTimeoutStopStrategyResetsOnNewSpeechAndReset(t *testing.T) {
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30*time.Millisecond, false)

	strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame())
	time.Sleep(10 * time.Millisecond)
	strategy.ShouldStop(frames.NewUserStartedSpeakingFrame())
	time.Sleep(35 * time.Millisecond)

	if strategy.ShouldStop(frames.NewTextFrame("after restart")) {
		t.Fatalf("expected started speaking to cancel pending timeout")
	}

	strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame())
	strategy.Reset()
	time.Sleep(35 * time.Millisecond)

	if strategy.ShouldStop(frames.NewTextFrame("after reset")) {
		t.Fatalf("expected reset to clear timer")
	}

	if strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be false")
	}
}

func TestSpeechTimeout_UserSpeechTimeout(t *testing.T) {
	// Verify that UserSpeechTimeout defaults to 0.6s (600ms)
	// Create strategy with 600ms timeout to match the default
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(600*time.Millisecond, true)

	strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame())

	// Sleep for 500ms (less than 600ms timeout)
	time.Sleep(500 * time.Millisecond)
	if strategy.ShouldStop(frames.NewTextFrame("before timeout")) {
		t.Fatalf("expected not to stop before 600ms timeout")
	}

	// Sleep for 150ms more (total 650ms, exceeds 600ms timeout)
	time.Sleep(150 * time.Millisecond)
	if !strategy.ShouldStop(frames.NewTextFrame("after timeout")) {
		t.Fatalf("expected to stop after 600ms timeout")
	}
}

func TestSTTMetadata_AutoConfigures(t *testing.T) {
	// Base timeout of 30ms, no P99 override
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30*time.Millisecond, true)

	// Send STTMetadataFrame with 500ms P99 latency
	metaFrame := frames.NewSTTMetadataFrame("azure", 500*time.Millisecond)
	if strategy.ShouldStop(metaFrame) {
		t.Fatalf("STTMetadataFrame should never trigger stop")
	}

	// Now trigger UserStoppedSpeaking — effective timeout should be 30ms + 500ms = 530ms
	if strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame()) {
		t.Fatalf("expected stop timer start to not immediately stop")
	}

	// At 400ms — should NOT have stopped yet (530ms total needed)
	time.Sleep(400 * time.Millisecond)
	if strategy.ShouldStop(frames.NewTextFrame("before combined timeout")) {
		t.Fatalf("expected not to stop before combined timeout (30ms + 500ms P99)")
	}

	// At 600ms total — should have stopped (> 530ms)
	time.Sleep(200 * time.Millisecond)
	if !strategy.ShouldStop(frames.NewTextFrame("after combined timeout")) {
		t.Fatalf("expected to stop after combined timeout (30ms + 500ms P99)")
	}
}

func TestSTTMetadata_OverridePreventsAutoConfig(t *testing.T) {
	// Base timeout of 30ms with explicit P99 override of 10ms
	strategy := user_stop.NewSpeechTimeoutUserTurnStopStrategy(30*time.Millisecond, true)
	strategy.SetTTFSP99Latency(10 * time.Millisecond)

	// Send STTMetadataFrame with 500ms P99 — should be ignored due to override
	metaFrame := frames.NewSTTMetadataFrame("azure", 500*time.Millisecond)
	strategy.ShouldStop(metaFrame)

	// Trigger stop timer — effective timeout should be 30ms + 10ms = 40ms (NOT 530ms)
	strategy.ShouldStop(frames.NewUserStoppedSpeakingFrame())

	// At 50ms — should have stopped (40ms timeout)
	time.Sleep(50 * time.Millisecond)
	if !strategy.ShouldStop(frames.NewTextFrame("after override timeout")) {
		t.Fatalf("expected to stop with override P99 (30ms + 10ms), not auto-configured 500ms")
	}
}
