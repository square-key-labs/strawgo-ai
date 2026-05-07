package processors

import (
	"context"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// audioFrame is a small helper that builds a TTSAudioFrame with a generation
// stamped on both the typed field and the metadata map.
func audioFrame(genID string) *frames.TTSAudioFrame {
	f := frames.NewTTSAudioFrame([]byte("audio"), 8000, 1)
	f.ContextID = genID
	f.SetMetadata("context_id", genID)
	return f
}

func TestAudioGuard_AdmitFrameWithoutGenerationTag(t *testing.T) {
	g := NewAudioGuard()

	f := frames.NewTTSAudioFrame([]byte("audio"), 8000, 1)
	if !g.admit(f) {
		t.Fatal("untagged frame should be admitted")
	}
}

func TestAudioGuard_BeginGenerationLocksOnFirstMatchingAudio(t *testing.T) {
	g := NewAudioGuard()

	g.beginGeneration("gen-1")

	g.mu.Lock()
	if g.expectedGeneration != "gen-1" {
		t.Errorf("expectedGeneration = %q, want %q", g.expectedGeneration, "gen-1")
	}
	if g.activeGeneration != "" {
		t.Errorf("activeGeneration should be empty before first audio, got %q", g.activeGeneration)
	}
	g.mu.Unlock()

	if !g.admit(audioFrame("gen-1")) {
		t.Fatal("first matching audio should be admitted")
	}

	g.mu.Lock()
	if g.activeGeneration != "gen-1" {
		t.Errorf("activeGeneration = %q, want %q", g.activeGeneration, "gen-1")
	}
	g.mu.Unlock()
}

func TestAudioGuard_DropsAudioFromMismatchedGeneration(t *testing.T) {
	g := NewAudioGuard()

	g.beginGeneration("gen-1")

	if g.admit(audioFrame("gen-2")) {
		t.Fatal("audio from non-expected generation should not admit")
	}
}

func TestAudioGuard_AdmitsSubsequentAudioFromActiveGeneration(t *testing.T) {
	g := NewAudioGuard()

	g.beginGeneration("gen-1")
	g.admit(audioFrame("gen-1"))

	if !g.admit(audioFrame("gen-1")) {
		t.Fatal("subsequent audio from active generation should be admitted")
	}
}

func TestAudioGuard_InvalidateRequiresNewGenerationToResume(t *testing.T) {
	g := NewAudioGuard()

	g.beginGeneration("gen-1")
	g.admit(audioFrame("gen-1"))
	g.invalidate()

	// Old generation no longer admitted after invalidate
	if g.admit(audioFrame("gen-1")) {
		t.Fatal("audio from invalidated generation should not admit")
	}

	// New generation lifecycle starts cleanly
	g.beginGeneration("gen-2")
	if !g.admit(audioFrame("gen-2")) {
		t.Fatal("audio from new generation should admit after invalidate")
	}

	g.mu.Lock()
	if g.invalidated {
		t.Error("invalidated flag should clear on new generation lock-in")
	}
	g.mu.Unlock()
}

func TestAudioGuard_StaleAudioInFlightAfterInvalidateIsRejected(t *testing.T) {
	g := NewAudioGuard()

	g.beginGeneration("gen-1")
	g.admit(audioFrame("gen-1"))
	g.invalidate()
	g.beginGeneration("gen-2")

	// Audio from gen-1 still in flight after the new TTSStartedFrame arrived
	if g.admit(audioFrame("gen-1")) {
		t.Fatal("stale in-flight audio from gen-1 should not admit after gen-2 starts")
	}
}

func TestAudioGuard_ShadowModeAdmitsEverythingThroughHandleFrame(t *testing.T) {
	g := NewAudioGuardShadow()
	capture := &frameCaptureProcessor{}
	g.Link(capture)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := g.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer g.Stop()

	// Sequence that WOULD drop in active mode: gen-2 audio without gen-2 start
	if err := g.QueueFrame(frames.NewTTSStartedFrameWithContext("gen-1"), frames.Downstream); err != nil {
		t.Fatalf("queue start: %v", err)
	}
	if err := g.QueueFrame(audioFrame("gen-2"), frames.Downstream); err != nil {
		t.Fatalf("queue audio: %v", err)
	}

	capture.waitForFrame(t, "TTSAudioFrame", 500*time.Millisecond)

	if !capture.hasFrameOfType("TTSAudioFrame") {
		t.Error("shadow mode should still admit mismatched audio downstream")
	}
}

func TestAudioGuard_ActiveModeDropsMismatchedAudioFromHandleFrame(t *testing.T) {
	g := NewAudioGuard()
	g.Active.Store(true)
	capture := &frameCaptureProcessor{}
	g.Link(capture)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := g.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer g.Stop()

	if err := g.QueueFrame(frames.NewTTSStartedFrameWithContext("gen-1"), frames.Downstream); err != nil {
		t.Fatalf("queue start: %v", err)
	}
	if err := g.QueueFrame(audioFrame("gen-2"), frames.Downstream); err != nil {
		t.Fatalf("queue stale audio: %v", err)
	}
	if err := g.QueueFrame(audioFrame("gen-1"), frames.Downstream); err != nil {
		t.Fatalf("queue good audio: %v", err)
	}

	capture.waitForFrame(t, "TTSAudioFrame", 500*time.Millisecond)

	// Count TTSAudioFrame instances — only gen-1 should reach downstream
	audioCount := 0
	for _, f := range capture.capturedFrames() {
		if af, ok := f.(*frames.TTSAudioFrame); ok {
			audioCount++
			if af.ContextID != "gen-1" {
				t.Errorf("active mode admitted mismatched audio: got generation %q", af.ContextID)
			}
		}
	}
	if audioCount != 1 {
		t.Errorf("expected 1 admitted audio frame in active mode, got %d", audioCount)
	}
}

func TestAudioGuard_InterruptionFrameInvalidatesViaHandleFrame(t *testing.T) {
	g := NewAudioGuard()

	g.beginGeneration("gen-1")
	g.admit(audioFrame("gen-1"))

	// Drive InterruptionFrame through HandleFrame
	if err := g.HandleFrame(context.Background(), frames.NewInterruptionFrame(), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame interruption: %v", err)
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.invalidated {
		t.Error("InterruptionFrame should set invalidated=true via HandleFrame")
	}
	if g.activeGeneration != "" {
		t.Errorf("activeGeneration should clear on invalidation, got %q", g.activeGeneration)
	}
}
