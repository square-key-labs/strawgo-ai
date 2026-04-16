package aggregators

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// ---- test helpers --------------------------------------------------------

// captureProc records every frame queued to it.
type captureProc struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (c *captureProc) ProcessFrame(_ context.Context, _ frames.Frame, _ frames.FrameDirection) error {
	return nil
}
func (c *captureProc) QueueFrame(f frames.Frame, _ frames.FrameDirection) error {
	c.mu.Lock()
	c.frames = append(c.frames, f)
	c.mu.Unlock()
	return nil
}
func (c *captureProc) PushFrame(_ frames.Frame, _ frames.FrameDirection) error { return nil }
func (c *captureProc) Link(_ processors.FrameProcessor)                        {}
func (c *captureProc) SetPrev(_ processors.FrameProcessor)                     {}
func (c *captureProc) Start(_ context.Context) error                           { return nil }
func (c *captureProc) Stop() error                                             { return nil }
func (c *captureProc) Name() string                                            { return "capture" }

func (c *captureProc) get() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]frames.Frame, len(c.frames))
	copy(out, c.frames)
	return out
}

func (c *captureProc) first() frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.frames) == 0 {
		return nil
	}
	return c.frames[0]
}

func (c *captureProc) waitFor(t *testing.T, name string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		c.mu.Lock()
		for _, f := range c.frames {
			if f.Name() == name {
				c.mu.Unlock()
				return
			}
		}
		c.mu.Unlock()
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timeout waiting for frame %q", name)
}

// prevCapture records frames pushed upstream (to prev).
type prevCapture struct {
	captureProc
}

// setupProcessor builds a started UserTurnCompletionProcessor linked to
// a downstream captureProc and a prev captureProc for upstream assertions.
func setupProcessor(t *testing.T, cfg UserTurnCompletionConfig) (*UserTurnCompletionProcessor, *captureProc, *captureProc) {
	t.Helper()

	ctx := services.NewLLMContext("test system prompt")
	p := NewUserTurnCompletionProcessor(ctx, cfg)

	down := &captureProc{}
	up := &captureProc{}

	// Wire: up ← p → down
	p.Link(down)
	p.SetPrev(up)

	if err := p.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = p.Stop() })

	return p, down, up
}

// send queues a downstream frame to the processor and waits briefly for it to be handled.
func send(t *testing.T, p *UserTurnCompletionProcessor, f frames.Frame) {
	t.Helper()
	if err := p.QueueFrame(f, frames.Downstream); err != nil {
		t.Fatalf("QueueFrame: %v", err)
	}
	time.Sleep(20 * time.Millisecond) // allow single goroutine to process
}

// sendTokens sends multiple LLMTextFrame tokens one at a time.
func sendTokens(t *testing.T, p *UserTurnCompletionProcessor, tokens ...string) {
	t.Helper()
	for _, tok := range tokens {
		send(t, p, frames.NewLLMTextFrame(tok))
	}
}

// ---- tests ---------------------------------------------------------------

// TestWrapSystemPromptWithTurnCompletion verifies the prompt wrapping utility.
func TestWrapSystemPromptWithTurnCompletion(t *testing.T) {
	base := "You are a helpful assistant."
	result := WrapSystemPromptWithTurnCompletion(base, DefaultUserTurnCompletionConfig())
	if len(result) <= len(base) {
		t.Error("expected result to be longer than base prompt")
	}
	if result[:len(base)] != base {
		t.Error("expected result to start with base prompt")
	}
	for _, marker := range []string{TurnCompleteMarker, TurnIncompleteShortMarker, TurnIncompleteLongMarker} {
		if !containsStr(result, marker) {
			t.Errorf("instructions missing marker %q", marker)
		}
	}
}

// TestTurnComplete verifies that ✓ strips the marker and forwards remainder to TTS.
func TestTurnComplete(t *testing.T) {
	p, down, _ := setupProcessor(t, DefaultUserTurnCompletionConfig())

	send(t, p, frames.NewLLMFullResponseStartFrame())
	sendTokens(t, p, "✓ Hello there, how can I help?")
	send(t, p, frames.NewLLMFullResponseEndFrame())

	captured := down.get()

	// Find the SkipTTS frame (marker prefix).
	var skipTTSFound bool
	var normalText string
	for _, f := range captured {
		if lf, ok := f.(*frames.LLMTextFrame); ok {
			if lf.SkipTTS {
				skipTTSFound = true
			} else if lf.Text != "" {
				normalText += lf.Text
			}
		}
	}

	if !skipTTSFound {
		t.Error("expected a SkipTTS=true frame for the marker portion")
	}
	if normalText == "" {
		t.Error("expected normal (non-SkipTTS) text after the marker")
	}
}

// TestTurnIncompleteShort verifies that ○ suppresses text with SkipTTS and starts timer.
func TestTurnIncompleteShort(t *testing.T) {
	cfg := DefaultUserTurnCompletionConfig()
	cfg.IncompleteShortTimeout = 50 * time.Millisecond // fast for test
	p, down, up := setupProcessor(t, cfg)

	send(t, p, frames.NewLLMFullResponseStartFrame())
	sendTokens(t, p, "○") // marker only
	send(t, p, frames.NewLLMFullResponseEndFrame())

	// Verify TTS-suppressed frame reached downstream.
	skipFound := false
	for _, f := range down.get() {
		if lf, ok := f.(*frames.LLMTextFrame); ok && lf.SkipTTS {
			skipFound = true
		}
	}
	if !skipFound {
		t.Error("expected SkipTTS=true frame to reach downstream (for context storage)")
	}

	// Wait for timer to fire and push LLMMessagesAppendFrame upstream.
	up.waitFor(t, "LLMMessagesAppendFrame", 500*time.Millisecond)
}

// TestTurnIncompleteLong verifies ◐ uses the long timeout.
func TestTurnIncompleteLong(t *testing.T) {
	cfg := DefaultUserTurnCompletionConfig()
	cfg.IncompleteLongTimeout = 60 * time.Millisecond
	p, _, up := setupProcessor(t, cfg)

	send(t, p, frames.NewLLMFullResponseStartFrame())
	sendTokens(t, p, "◐")
	send(t, p, frames.NewLLMFullResponseEndFrame())

	up.waitFor(t, "LLMMessagesAppendFrame", 500*time.Millisecond)
}

// TestInterruptionCancelsTimer verifies that InterruptionFrame cancels the timer.
func TestInterruptionCancelsTimer(t *testing.T) {
	cfg := DefaultUserTurnCompletionConfig()
	cfg.IncompleteShortTimeout = 100 * time.Millisecond
	p, _, up := setupProcessor(t, cfg)

	send(t, p, frames.NewLLMFullResponseStartFrame())
	sendTokens(t, p, "○")
	send(t, p, frames.NewLLMFullResponseEndFrame())

	// Immediately interrupt — should cancel the timer.
	if err := p.QueueFrame(frames.NewInterruptionFrame(), frames.Downstream); err != nil {
		t.Fatalf("QueueFrame interruption: %v", err)
	}
	time.Sleep(20 * time.Millisecond)

	// Wait past the would-be timeout; no LLMMessagesAppendFrame should arrive.
	time.Sleep(300 * time.Millisecond)

	for _, f := range up.get() {
		if f.Name() == "LLMMessagesAppendFrame" {
			t.Error("LLMMessagesAppendFrame should not have been pushed after interruption")
		}
	}
}

// TestGracefulFallback verifies that text is flushed to TTS when no marker arrives.
func TestGracefulFallback(t *testing.T) {
	p, down, _ := setupProcessor(t, DefaultUserTurnCompletionConfig())

	send(t, p, frames.NewLLMFullResponseStartFrame())
	sendTokens(t, p, "Hello, no marker here.")
	send(t, p, frames.NewLLMFullResponseEndFrame())

	var normalText string
	for _, f := range down.get() {
		if lf, ok := f.(*frames.LLMTextFrame); ok && !lf.SkipTTS {
			normalText += lf.Text
		}
	}
	if normalText == "" {
		t.Error("expected buffered text to be flushed as normal frame on fallback")
	}
}

// TestMarkerSplitAcrossTokens verifies buffer accumulation handles split Unicode.
func TestMarkerSplitAcrossTokens(t *testing.T) {
	cfg := DefaultUserTurnCompletionConfig()
	cfg.IncompleteShortTimeout = 50 * time.Millisecond
	p, _, up := setupProcessor(t, cfg)

	// ○ is a 3-byte UTF-8 sequence: 0xE2 0x97 0x8B
	// Simulate it arriving split across two streaming tokens.
	raw := []byte(TurnIncompleteShortMarker)
	part1 := string(raw[:2])
	part2 := string(raw[2:])

	send(t, p, frames.NewLLMFullResponseStartFrame())
	sendTokens(t, p, part1, part2)
	send(t, p, frames.NewLLMFullResponseEndFrame())

	up.waitFor(t, "LLMMessagesAppendFrame", 500*time.Millisecond)
}

// TestMultipleResponseCycles verifies state resets properly between LLM turns.
func TestMultipleResponseCycles(t *testing.T) {
	p, down, _ := setupProcessor(t, DefaultUserTurnCompletionConfig())

	// First response: complete turn.
	send(t, p, frames.NewLLMFullResponseStartFrame())
	sendTokens(t, p, "✓ First response.")
	send(t, p, frames.NewLLMFullResponseEndFrame())

	// Second response: also complete — processor must have reset cleanly.
	send(t, p, frames.NewLLMFullResponseStartFrame())
	sendTokens(t, p, "✓ Second response.")
	send(t, p, frames.NewLLMFullResponseEndFrame())

	var normalCount int
	for _, f := range down.get() {
		if lf, ok := f.(*frames.LLMTextFrame); ok && !lf.SkipTTS && lf.Text != "" {
			normalCount++
		}
	}
	if normalCount < 2 {
		t.Errorf("expected at least 2 normal text frames (one per turn), got %d", normalCount)
	}
}

// TestSkipTTSInFrameTypes verifies SkipTTS field exists and is respected.
func TestSkipTTSInFrameTypes(t *testing.T) {
	lf := frames.NewLLMTextFrame("test")
	if lf.SkipTTS {
		t.Error("new LLMTextFrame should have SkipTTS=false by default")
	}
	lf.SkipTTS = true
	if !lf.SkipTTS {
		t.Error("SkipTTS should be settable to true")
	}

	tf := frames.NewTextFrame("test")
	if tf.SkipTTS {
		t.Error("new TextFrame should have SkipTTS=false by default")
	}
	tf.SkipTTS = true
	if !tf.SkipTTS {
		t.Error("TextFrame SkipTTS should be settable to true")
	}
}

// ---- helpers --------------------------------------------------------

func containsStr(s, sub string) bool {
	return strings.Contains(s, sub)
}
