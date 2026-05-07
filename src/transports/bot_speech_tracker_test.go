package transports

import (
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// frameCollector records every frame pushed upstream by the tracker.
type frameCollector struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (c *frameCollector) push(f frames.Frame, _ frames.FrameDirection) error {
	c.mu.Lock()
	c.frames = append(c.frames, f)
	c.mu.Unlock()
	return nil
}

func (c *frameCollector) waitForName(t *testing.T, name string, timeout time.Duration) bool {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		c.mu.Lock()
		for _, f := range c.frames {
			if f.Name() == name {
				c.mu.Unlock()
				return true
			}
		}
		c.mu.Unlock()
		time.Sleep(5 * time.Millisecond)
	}
	return false
}

func (c *frameCollector) countByName(name string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for _, f := range c.frames {
		if f.Name() == name {
			n++
		}
	}
	return n
}

// driveAckTrackerLocal builds a real PlaybackAckTracker in stratLocal
// mode so RequestComplete fires Done() synchronously — matches the
// "audio path went idle, immediately stop speaking" path in BotSpeechTracker.
func driveAckTrackerLocal(t *testing.T) *PlaybackAckTracker {
	t.Helper()
	r := &strategyResolver{}
	r.set(stratLocal)
	tr := NewPlaybackAckTracker(playbackTransportFns{
		send:             func(interface{}) error { return nil },
		serializer:       &mockSerializer{},
		resolveStrategy:  r.get,
		drainPad:         func() time.Duration { return 0 },
		fallbackDuration: time.Second,
	})
	t.Cleanup(func() { tr.Stop() })
	return tr
}

func newBotSpeechTrackerForTest(t *testing.T, ack *PlaybackAckTracker) (*BotSpeechTracker, *frameCollector, chan PacerEvent) {
	t.Helper()
	collector := &frameCollector{}
	events := make(chan PacerEvent, 8)
	tr := NewBotSpeechTracker(events, ack, collector.push, nil)
	t.Cleanup(func() { tr.Stop() })
	return tr, collector, events
}

func TestBotSpeechTracker_EmitsStartedOnPacerStarted(t *testing.T) {
	ack := driveAckTrackerLocal(t)
	tr, collector, events := newBotSpeechTrackerForTest(t, ack)
	_ = tr

	events <- PacerStarted

	if !collector.waitForName(t, "BotStartedSpeakingFrame", time.Second) {
		t.Fatal("expected BotStartedSpeakingFrame")
	}
	if got := collector.countByName("BotStartedSpeakingFrame"); got != 1 {
		t.Fatalf("expected exactly 1 BotStartedSpeakingFrame, got %d", got)
	}
}

func TestBotSpeechTracker_PacerIdleWaitsForLLMEnd(t *testing.T) {
	ack := driveAckTrackerLocal(t)
	tr, collector, events := newBotSpeechTrackerForTest(t, ack)

	events <- PacerStarted
	collector.waitForName(t, "BotStartedSpeakingFrame", time.Second)

	events <- PacerIdle

	// Without LLMResponseEnded, no stop frame should be emitted.
	if collector.waitForName(t, "BotStoppedSpeakingFrame", 100*time.Millisecond) {
		t.Fatal("BotStoppedSpeakingFrame emitted before LLM end")
	}

	tr.LLMResponseEnded()
	events <- PacerIdle

	if !collector.waitForName(t, "BotStoppedSpeakingFrame", time.Second) {
		t.Fatal("BotStoppedSpeakingFrame not emitted after LLM end + PacerIdle")
	}
}

func TestBotSpeechTracker_PacerIdleIgnoredWhenNotSpeaking(t *testing.T) {
	ack := driveAckTrackerLocal(t)
	tr, collector, events := newBotSpeechTrackerForTest(t, ack)

	tr.LLMResponseEnded()
	events <- PacerIdle

	if collector.waitForName(t, "BotStoppedSpeakingFrame", 100*time.Millisecond) {
		t.Fatal("PacerIdle should be ignored when not speaking")
	}
}

func TestBotSpeechTracker_InterruptEmitsStopAndCancelsAck(t *testing.T) {
	ack := driveAckTrackerLocal(t)
	tr, collector, events := newBotSpeechTrackerForTest(t, ack)

	events <- PacerStarted
	if !collector.waitForName(t, "BotStartedSpeakingFrame", time.Second) {
		t.Fatal("missed BotStartedSpeakingFrame")
	}

	tr.Interrupt()

	if !collector.waitForName(t, "BotStoppedSpeakingFrame", time.Second) {
		t.Fatal("Interrupt did not emit BotStoppedSpeakingFrame")
	}
}

func TestBotSpeechTracker_InterruptWhileNotSpeakingIsNoop(t *testing.T) {
	ack := driveAckTrackerLocal(t)
	tr, collector, _ := newBotSpeechTrackerForTest(t, ack)

	tr.Interrupt()

	if collector.waitForName(t, "BotStoppedSpeakingFrame", 50*time.Millisecond) {
		t.Fatal("Interrupt should not emit Stop when not speaking")
	}
}

func TestBotSpeechTracker_ResetLLMStateClearsGate(t *testing.T) {
	ack := driveAckTrackerLocal(t)
	tr, collector, events := newBotSpeechTrackerForTest(t, ack)

	tr.LLMResponseEnded()
	tr.ResetLLMState()

	events <- PacerStarted
	collector.waitForName(t, "BotStartedSpeakingFrame", time.Second)
	events <- PacerIdle

	// After Reset, LLM is no longer "ended" — Stop should not fire.
	if collector.waitForName(t, "BotStoppedSpeakingFrame", 100*time.Millisecond) {
		t.Fatal("ResetLLMState did not clear the LLM-end gate")
	}
}

func TestBotSpeechTracker_StopIsIdempotent(t *testing.T) {
	ack := driveAckTrackerLocal(t)
	tr, _, _ := newBotSpeechTrackerForTest(t, ack)

	tr.Stop()
	tr.Stop() // must not panic
}
