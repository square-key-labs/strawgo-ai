package transports

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type queuedFrameCapture struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (c *queuedFrameCapture) record(frame frames.Frame) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.frames = append(c.frames, frame)
}

func (c *queuedFrameCapture) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	c.record(frame)
	return nil
}

func (c *queuedFrameCapture) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	c.record(frame)
	return nil
}

func (c *queuedFrameCapture) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	c.record(frame)
	return nil
}

func (c *queuedFrameCapture) Link(next processors.FrameProcessor) {}

func (c *queuedFrameCapture) SetPrev(prev processors.FrameProcessor) {}

func (c *queuedFrameCapture) Start(ctx context.Context) error { return nil }

func (c *queuedFrameCapture) Stop() error { return nil }

func (c *queuedFrameCapture) Name() string { return "queued-capture" }

func (c *queuedFrameCapture) count(name string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	count := 0
	for _, frame := range c.frames {
		if frame.Name() == name {
			count++
		}
	}
	return count
}

func (c *queuedFrameCapture) find(name string) frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, frame := range c.frames {
		if frame.Name() == name {
			return frame
		}
	}
	return nil
}

func markID(h *harness) string {
	for _, w := range h.conn.snapshot() {
		if s := string(w.msg.Data); strings.HasPrefix(s, "mark:") {
			return s[5:]
		}
	}
	return ""
}

func TestPlaybackAckDelaysBotStoppedSpeaking(t *testing.T) {
	h := newOutputHarness(t, TransportParams{PlaybackAck: true, PlaybackAckTimeout: 5 * time.Second})
	out := h.sess.Output()
	queue(t, out, ttsAudio(60, 8000))
	queue(t, out, frames.NewTTSStoppedFrame())

	waitUntil(t, time.Second, "mark request", func() bool { return markID(h) != "" })
	time.Sleep(50 * time.Millisecond)
	if h.up.count("BotStoppedSpeakingFrame") != 0 {
		t.Fatal("BotStoppedSpeakingFrame emitted before the playback ack")
	}

	out.ack("some-other-mark")
	time.Sleep(30 * time.Millisecond)
	if h.up.count("BotStoppedSpeakingFrame") != 0 {
		t.Fatal("stale playback ack accepted")
	}

	out.ack(markID(h))
	waitUntil(t, time.Second, "bot stopped after ack", func() bool { return h.up.count("BotStoppedSpeakingFrame") == 1 })
}

func TestPlaybackAckTimeoutFallback(t *testing.T) {
	h := newOutputHarness(t, TransportParams{PlaybackAck: true, PlaybackAckTimeout: 80 * time.Millisecond})
	out := h.sess.Output()
	queue(t, out, ttsAudio(40, 8000))
	queue(t, out, frames.NewTTSStoppedFrame())

	waitUntil(t, time.Second, "mark request", func() bool { return markID(h) != "" })
	sent := time.Now()
	waitUntil(t, time.Second, "fallback bot stopped", func() bool { return h.up.count("BotStoppedSpeakingFrame") == 1 })
	if time.Since(sent) < 60*time.Millisecond {
		t.Fatal("fallback fired before the configured timeout")
	}
}

func TestInterruptionResetsPendingPlaybackAck(t *testing.T) {
	h := newOutputHarness(t, TransportParams{PlaybackAck: true, PlaybackAckTimeout: 5 * time.Second})
	out := h.sess.Output()
	queue(t, out, ttsAudio(40, 8000))
	queue(t, out, frames.NewTTSStoppedFrame())
	waitUntil(t, time.Second, "mark request", func() bool { return markID(h) != "" })

	queue(t, out, frames.NewInterruptionFrame())
	waitUntil(t, time.Second, "bot stopped on interruption", func() bool { return h.up.count("BotStoppedSpeakingFrame") == 1 })

	out.ack(markID(h))
	time.Sleep(30 * time.Millisecond)
	if h.up.count("BotStoppedSpeakingFrame") != 1 {
		t.Fatal("late playback ack produced a second BotStoppedSpeakingFrame")
	}
}
