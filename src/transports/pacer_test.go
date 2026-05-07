package transports

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// captureSender records every payload pushed to the pacer's wire and
// counts sends atomically so tests can poll without races.
type captureSender struct {
	mu       sync.Mutex
	payloads [][]byte
	count    atomic.Int32
	failNext atomic.Bool
}

func (s *captureSender) send(data interface{}) error {
	if s.failNext.CompareAndSwap(true, false) {
		return errors.New("simulated send error")
	}
	if b, ok := data.([]byte); ok {
		s.mu.Lock()
		s.payloads = append(s.payloads, b)
		s.mu.Unlock()
	}
	s.count.Add(1)
	return nil
}

// passthroughSerialize wraps the chunk's bytes back into the queue
// payload so captureSender can record them.
func passthroughSerialize(f frames.Frame) (interface{}, error) {
	if af, ok := f.(*frames.TTSAudioFrame); ok {
		// Copy so the slice we record isn't aliased with the chunk's
		// underlying array (which Submit re-slices).
		out := make([]byte, len(af.Data))
		copy(out, af.Data)
		return out, nil
	}
	return nil, nil
}

func mulawAudioFrame(size int) *frames.TTSAudioFrame {
	f := frames.NewTTSAudioFrame(make([]byte, size), 8000, 1)
	f.SetMetadata("codec", "mulaw")
	return f
}

func waitForCount(t *testing.T, sender *captureSender, want int32, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if sender.count.Load() >= want {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timeout waiting for %d sends, got %d", want, sender.count.Load())
}

func TestPacer_SubmitChunksMulawFrame(t *testing.T) {
	sender := &captureSender{}
	p := NewPacer(sender.send, nil)
	defer p.Cleanup()

	// 480 bytes at mulaw 8kHz / chunkSize 160 → 3 chunks, 20ms apart
	frame := mulawAudioFrame(480)
	if err := p.Submit(frame, passthroughSerialize); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	waitForCount(t, sender, 3, time.Second)

	sender.mu.Lock()
	defer sender.mu.Unlock()
	for i, payload := range sender.payloads {
		if len(payload) != 160 {
			t.Errorf("chunk %d: len=%d, want 160", i, len(payload))
		}
	}
}

func TestPacer_StartedEventOnFirstChunk(t *testing.T) {
	sender := &captureSender{}
	p := NewPacer(sender.send, nil)
	defer p.Cleanup()

	if err := p.Submit(mulawAudioFrame(480), passthroughSerialize); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	select {
	case ev := <-p.Events():
		if ev != PacerStarted {
			t.Errorf("expected PacerStarted, got %v", ev)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for PacerStarted")
	}
}

func TestPacer_IdleEventAfterInactivity(t *testing.T) {
	sender := &captureSender{}
	p := NewPacer(sender.send, nil)
	defer p.Cleanup()

	if err := p.Submit(mulawAudioFrame(160), passthroughSerialize); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	waitForCount(t, sender, 1, time.Second)

	// First event is Started
	select {
	case ev := <-p.Events():
		if ev != PacerStarted {
			t.Fatalf("expected PacerStarted first, got %v", ev)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for PacerStarted")
	}

	// Idle should arrive ~pacerIdleAfter (350ms) later
	select {
	case ev := <-p.Events():
		if ev != PacerIdle {
			t.Errorf("expected PacerIdle, got %v", ev)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for PacerIdle")
	}
}

func TestPacer_DrainSuppressesPendingSends(t *testing.T) {
	sender := &captureSender{}
	p := NewPacer(sender.send, nil)
	defer p.Cleanup()

	// 1600 bytes mulaw → 10 chunks, 20ms each = 200ms total
	if err := p.Submit(mulawAudioFrame(1600), passthroughSerialize); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	// Drain immediately. At least the first chunk likely landed before
	// the call; assert the rest are dropped.
	time.Sleep(10 * time.Millisecond)
	p.Drain()

	// Wait long enough that any non-drained chunks would have shipped.
	time.Sleep(300 * time.Millisecond)

	got := sender.count.Load()
	if got >= 8 {
		t.Errorf("expected most chunks dropped after Drain, but %d/10 reached the wire", got)
	}

	// Submit during drain → no-op
	before := sender.count.Load()
	if err := p.Submit(mulawAudioFrame(160), passthroughSerialize); err != nil {
		t.Fatalf("Submit during drain: %v", err)
	}
	time.Sleep(50 * time.Millisecond)
	if after := sender.count.Load(); after != before {
		t.Errorf("Submit during drain should no-op; count went %d → %d", before, after)
	}
}

func TestPacer_ResumeAfterDrainEmitsStartedAgain(t *testing.T) {
	sender := &captureSender{}
	p := NewPacer(sender.send, nil)
	defer p.Cleanup()

	// First burst
	if err := p.Submit(mulawAudioFrame(160), passthroughSerialize); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	waitForCount(t, sender, 1, time.Second)
	if ev := <-p.Events(); ev != PacerStarted {
		t.Fatalf("expected PacerStarted, got %v", ev)
	}

	// Drain mid-flight, then Resume
	p.Drain()
	p.Resume()

	// Second burst should re-emit PacerStarted
	if err := p.Submit(mulawAudioFrame(160), passthroughSerialize); err != nil {
		t.Fatalf("Submit after resume: %v", err)
	}
	waitForCount(t, sender, 2, time.Second)

	// Drain any idle event from the first burst that may have queued
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		select {
		case ev := <-p.Events():
			if ev == PacerStarted {
				return // success
			}
			// PacerIdle from prior burst; keep waiting
		case <-time.After(20 * time.Millisecond):
		}
	}
	t.Fatal("expected second PacerStarted after Resume")
}

func TestPacer_CleanupIsIdempotent(t *testing.T) {
	sender := &captureSender{}
	p := NewPacer(sender.send, nil)

	if err := p.Cleanup(); err != nil {
		t.Errorf("first Cleanup: %v", err)
	}
	if err := p.Cleanup(); err != nil {
		t.Errorf("second Cleanup: %v", err)
	}
}

func TestCalculateSendInterval(t *testing.T) {
	cases := []struct {
		name       string
		chunkSize  int
		sampleRate int
		codec      string
		want       time.Duration
	}{
		{"mulaw 8kHz 160-byte", 160, 8000, "mulaw", 20 * time.Millisecond},
		{"alaw 8kHz 160-byte", 160, 8000, "alaw", 20 * time.Millisecond},
		{"linear16 16kHz 320-byte", 320, 16000, "linear16", 10 * time.Millisecond},
		{"float32 48kHz 1920-byte", 1920, 48000, "float32", 10 * time.Millisecond},
		{"empty codec defaults to linear16", 320, 16000, "", 10 * time.Millisecond},
		{"sample rate zero falls back to 8kHz", 160, 0, "mulaw", 20 * time.Millisecond},
		{"interval clamped to ≥ 1ms", 1, 8000, "mulaw", time.Millisecond},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := calculateSendInterval(tc.chunkSize, tc.sampleRate, tc.codec)
			if got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}
