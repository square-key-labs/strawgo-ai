package observers

import (
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

func push(o pipeline.Observer, f frames.Frame, ts time.Time, processor string) {
	o.OnPushFrame(pipeline.PushFrameEvent{Frame: f, Timestamp: ts, ProcessorName: processor, Direction: frames.Downstream})
}

func TestUserBotLatencyObserverReportsBreakdownAndFirstSpeech(t *testing.T) {
	observer := NewUserBotLatencyObserver()

	breakdownCh := make(chan LatencyBreakdown, 2)
	measuredCh := make(chan time.Duration, 2)
	firstSpeechCh := make(chan time.Duration, 1)

	observer.OnLatencyBreakdown(func(b LatencyBreakdown) { breakdownCh <- b })
	observer.OnLatencyMeasured(func(d time.Duration) { measuredCh <- d })
	observer.OnFirstBotSpeechLatency(func(d time.Duration) { firstSpeechCh <- d })

	base := time.Unix(1, 0)
	push(observer, frames.NewClientConnectedFrame(), base, "transport")
	push(observer, frames.NewBotStartedSpeakingFrame(), base.Add(50*time.Millisecond), "transport")

	if got := waitForDuration(t, firstSpeechCh); got != 50*time.Millisecond {
		t.Fatalf("unexpected first bot speech latency: got %v", got)
	}
	_ = waitForBreakdown(t, breakdownCh)

	userStopped := base.Add(100 * time.Millisecond)
	push(observer, frames.NewUserStartedSpeakingFrame(), base.Add(60*time.Millisecond), "vad")
	push(observer, frames.NewUserStoppedSpeakingFrame(), userStopped, "vad")
	push(observer, frames.NewTranscriptionFrame("hello", true), base.Add(250*time.Millisecond), "stt")
	llmStart := frames.NewLLMFullResponseStartFrame()
	push(observer, llmStart, base.Add(300*time.Millisecond), "llm")
	push(observer, llmStart, base.Add(310*time.Millisecond), "tts")
	push(observer, frames.NewMetricsFrame(frames.TTFBMetricsData{MetricsBase: frames.MetricsBase{Processor: "llm", Model: "m"}, Value: 100 * time.Millisecond}), base.Add(400*time.Millisecond), "llm")
	push(observer, frames.NewLLMTextFrame("world"), base.Add(400*time.Millisecond), "llm")
	push(observer, frames.NewTTSAudioFrame([]byte{1}, 16000, 1), base.Add(550*time.Millisecond), "tts")
	push(observer, frames.NewBotStartedSpeakingFrame(), base.Add(600*time.Millisecond), "transport")

	if got := waitForDuration(t, measuredCh); got != 500*time.Millisecond {
		t.Fatalf("unexpected measured latency: got %v want 500ms", got)
	}
	b := waitForBreakdown(t, breakdownCh)
	if b.MeasuredFrom != MeasuredFromUserSilence {
		t.Fatalf("unexpected measured from: %q", b.MeasuredFrom)
	}
	if b.Total != 500*time.Millisecond {
		t.Fatalf("unexpected total: got %v want 500ms", b.Total)
	}
	if got := b.Contribution("transcription"); got != 150*time.Millisecond {
		t.Fatalf("unexpected transcription contribution: %v", got)
	}
	if got := b.Contribution("turn_detection"); got != 50*time.Millisecond {
		t.Fatalf("unexpected turn detection contribution: %v", got)
	}
	if got := b.Contribution("llm_inference"); got != 100*time.Millisecond {
		t.Fatalf("unexpected LLM contribution: %v", got)
	}
	if got := b.Contribution("speech_synthesis"); got != 150*time.Millisecond {
		t.Fatalf("unexpected TTS contribution: %v", got)
	}
	if got := b.Contribution("output_transport"); got != 50*time.Millisecond {
		t.Fatalf("unexpected output transport contribution: %v", got)
	}
	if len(b.TTFB) != 1 || b.TTFB[0].Processor != "llm" || b.TTFB[0].Duration != 100*time.Millisecond {
		t.Fatalf("unexpected TTFB breakdown: %+v", b.TTFB)
	}
}

func TestUserBotLatencyObserverOnlyEmitsOncePerTurn(t *testing.T) {
	observer := NewUserBotLatencyObserver()

	breakdownCh := make(chan LatencyBreakdown, 2)
	observer.OnLatencyBreakdown(func(b LatencyBreakdown) { breakdownCh <- b })

	base := time.Unix(2, 0)
	push(observer, frames.NewUserStoppedSpeakingFrame(), base, "vad")
	push(observer, frames.NewTranscriptionFrame("a", true), base.Add(10*time.Millisecond), "stt")
	push(observer, frames.NewLLMTextFrame("b"), base.Add(20*time.Millisecond), "llm")
	push(observer, frames.NewBotStartedSpeakingFrame(), base.Add(30*time.Millisecond), "transport")
	push(observer, frames.NewBotStartedSpeakingFrame(), base.Add(40*time.Millisecond), "transport")

	_ = waitForBreakdown(t, breakdownCh)

	select {
	case <-breakdownCh:
		t.Fatal("expected single latency breakdown callback per turn")
	default:
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
