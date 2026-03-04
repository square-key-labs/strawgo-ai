package observers

import (
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

func TestStartupTimingObserverReportsTransportTiming(t *testing.T) {
	observer := NewStartupTimingObserver()
	reportCh := make(chan TransportTimingReport, 1)

	observer.OnTransportTimingReport = func(report TransportTimingReport) {
		reportCh <- report
	}

	base := time.Unix(10, 0)
	start := base
	connected := base.Add(325 * time.Millisecond)

	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewStartFrame(), Timestamp: start})
	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewClientConnectedFrame(), Timestamp: connected, ProcessorName: "transport-output"})

	select {
	case report := <-reportCh:
		if report.StartupDuration != 325*time.Millisecond {
			t.Fatalf("unexpected startup duration: got %v want %v", report.StartupDuration, 325*time.Millisecond)
		}
		if report.ProcessorName != "transport-output" {
			t.Fatalf("unexpected processor name: got %q want %q", report.ProcessorName, "transport-output")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for startup timing report")
	}
}

func TestStartupTimingObserverIgnoresClientConnectedWithoutStart(t *testing.T) {
	observer := NewStartupTimingObserver()
	reportCh := make(chan TransportTimingReport, 1)

	observer.OnTransportTimingReport = func(report TransportTimingReport) {
		reportCh <- report
	}

	observer.OnPushFrame(pipeline.PushFrameEvent{Frame: frames.NewClientConnectedFrame(), Timestamp: time.Now(), ProcessorName: "transport-output"})

	select {
	case <-reportCh:
		t.Fatal("did not expect startup timing report without StartFrame")
	case <-time.After(100 * time.Millisecond):
	}
}
