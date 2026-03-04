package observers

import (
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

func TestTurnMetricsDataStartStopAndFrameConversion(t *testing.T) {
	metrics := NewTurnMetricsData("turn-analyzer")
	metrics.Start()
	time.Sleep(5 * time.Millisecond)
	metrics.Stop()

	if metrics.StartTime.IsZero() {
		t.Fatal("expected StartTime to be set")
	}
	if metrics.EndTime.IsZero() {
		t.Fatal("expected EndTime to be set")
	}
	if metrics.Duration <= 0 {
		t.Fatalf("expected positive duration, got %v", metrics.Duration)
	}

	frame := metrics.ToFrame()
	if frame.ProcessorName != "turn-analyzer" {
		t.Fatalf("unexpected processor name: got %q want %q", frame.ProcessorName, "turn-analyzer")
	}
	if frame.StartTime.IsZero() || frame.EndTime.IsZero() {
		t.Fatal("expected frame times to be populated")
	}
	if frame.Duration <= 0 {
		t.Fatalf("expected positive frame duration, got %v", frame.Duration)
	}
}

func TestTurnMetricsObserverCollectsTurnMetricsFrame(t *testing.T) {
	observer := NewTurnMetricsObserver()
	callbackCh := make(chan TurnMetricsData, 1)
	observer.OnTurnMetrics = func(metrics TurnMetricsData) {
		callbackCh <- metrics
	}

	start := time.Unix(20, 0)
	end := start.Add(250 * time.Millisecond)
	frame := frames.NewTurnMetricsFrame("smart-turn", start, end, 250*time.Millisecond)

	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frame, Timestamp: end})

	select {
	case observed := <-callbackCh:
		if observed.ProcessorName != "smart-turn" {
			t.Fatalf("unexpected callback processor name: got %q want %q", observed.ProcessorName, "smart-turn")
		}
		if observed.Duration != 250*time.Millisecond {
			t.Fatalf("unexpected callback duration: got %v want %v", observed.Duration, 250*time.Millisecond)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for turn metrics callback")
	}

	allMetrics := observer.Metrics()
	if len(allMetrics) != 1 {
		t.Fatalf("expected 1 metrics record, got %d", len(allMetrics))
	}
	if allMetrics[0].ProcessorName != "smart-turn" {
		t.Fatalf("unexpected stored processor name: got %q want %q", allMetrics[0].ProcessorName, "smart-turn")
	}
}

func TestTurnMetricsObserverIgnoresNonTurnMetricsFrame(t *testing.T) {
	observer := NewTurnMetricsObserver()
	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewTextFrame("ignored"), Timestamp: time.Now()})

	allMetrics := observer.Metrics()
	if len(allMetrics) != 0 {
		t.Fatalf("expected no metrics records, got %d", len(allMetrics))
	}
}
