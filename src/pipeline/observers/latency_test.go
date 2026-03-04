package observers

import (
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

func TestUserBotLatencyObserverReportsBreakdownAndFirstSpeech(t *testing.T) {
	observer := NewUserBotLatencyObserver()

	breakdownCh := make(chan LatencyBreakdown, 1)
	firstSpeechCh := make(chan time.Duration, 1)

	observer.OnLatencyBreakdown = func(breakdown LatencyBreakdown) {
		breakdownCh <- breakdown
	}
	observer.OnFirstBotSpeechLatency = func(latency time.Duration) {
		firstSpeechCh <- latency
	}

	base := time.Unix(1, 0)
	connected := base
	userStopped := base.Add(100 * time.Millisecond)
	transcription := base.Add(250 * time.Millisecond)
	llm := base.Add(400 * time.Millisecond)
	firstAudio := base.Add(550 * time.Millisecond)

	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewClientConnectedFrame(), Timestamp: connected})
	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewUserStoppedSpeakingFrame(), Timestamp: userStopped})
	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewTranscriptionFrame("hello", true), Timestamp: transcription})
	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewLLMTextFrame("world"), Timestamp: llm})
	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewTTSAudioFrame([]byte{1}, 16000, 1), Timestamp: firstAudio})

	breakdown := waitForBreakdown(t, breakdownCh)
	if breakdown.TotalLatency != 450*time.Millisecond {
		t.Fatalf("unexpected total latency: got %v want %v", breakdown.TotalLatency, 450*time.Millisecond)
	}
	if breakdown.STTLatency != 150*time.Millisecond {
		t.Fatalf("unexpected STT latency: got %v want %v", breakdown.STTLatency, 150*time.Millisecond)
	}
	if breakdown.LLMLatency != 150*time.Millisecond {
		t.Fatalf("unexpected LLM latency: got %v want %v", breakdown.LLMLatency, 150*time.Millisecond)
	}
	if breakdown.TTSLatency != 150*time.Millisecond {
		t.Fatalf("unexpected TTS latency: got %v want %v", breakdown.TTSLatency, 150*time.Millisecond)
	}

	firstSpeechLatency := waitForDuration(t, firstSpeechCh)
	if firstSpeechLatency != 550*time.Millisecond {
		t.Fatalf("unexpected first bot speech latency: got %v want %v", firstSpeechLatency, 550*time.Millisecond)
	}
}

func TestUserBotLatencyObserverOnlyEmitsOncePerTurn(t *testing.T) {
	observer := NewUserBotLatencyObserver()

	breakdownCh := make(chan LatencyBreakdown, 2)
	observer.OnLatencyBreakdown = func(breakdown LatencyBreakdown) {
		breakdownCh <- breakdown
	}

	base := time.Unix(2, 0)
	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewUserStoppedSpeakingFrame(), Timestamp: base})
	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewTranscriptionFrame("a", true), Timestamp: base.Add(10 * time.Millisecond)})
	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewLLMTextFrame("b"), Timestamp: base.Add(20 * time.Millisecond)})
	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewTTSAudioFrame([]byte{1}, 16000, 1), Timestamp: base.Add(30 * time.Millisecond)})
	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewTTSAudioFrame([]byte{1}, 16000, 1), Timestamp: base.Add(40 * time.Millisecond)})

	_ = waitForBreakdown(t, breakdownCh)

	select {
	case <-breakdownCh:
		t.Fatal("expected single latency breakdown callback per turn")
	case <-time.After(100 * time.Millisecond):
	}
}

func waitForBreakdown(t *testing.T, ch <-chan LatencyBreakdown) LatencyBreakdown {
	t.Helper()

	select {
	case breakdown := <-ch:
		return breakdown
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for latency breakdown")
		return LatencyBreakdown{}
	}
}

func waitForDuration(t *testing.T, ch <-chan time.Duration) time.Duration {
	t.Helper()

	select {
	case duration := <-ch:
		return duration
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for duration callback")
		return 0
	}
}
