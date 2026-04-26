package transports

import (
	"context"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/turns"
)

func enableInterruptions(t *testing.T, p *WebSocketOutputProcessor) {
	t.Helper()
	if err := p.HandleFrame(context.Background(), frames.NewStartFrameWithConfig(true, turns.UserTurnStrategies{}), frames.Downstream); err != nil {
		t.Fatalf("StartFrame: %v", err)
	}
}

func newConcurrentTestTransport() *WebSocketTransport {
	return NewWebSocketTransport(WebSocketConfig{
		Port:                  9090,
		Path:                  "/ws",
		Serializer:            &mockSerializer{},
		ConcurrentTTSContexts: true,
	})
}

func TestConcurrentTTSContextsConfigPropagated(t *testing.T) {
	tt := newConcurrentTestTransport()
	if !tt.outputProc.concurrentContexts {
		t.Fatal("expected concurrentContexts=true on output processor")
	}
	// Default transport should remain single-context.
	def := NewWebSocketTransport(WebSocketConfig{Port: 9091, Serializer: &mockSerializer{}})
	if def.outputProc.concurrentContexts {
		t.Fatal("default transport must not enable concurrent contexts")
	}
}

func TestConcurrentMultipleExpectedContextsAccepted(t *testing.T) {
	tt := newConcurrentTestTransport()
	p := tt.outputProc
	ctx := context.Background()

	idA := services.GenerateContextID()
	idB := services.GenerateContextID()

	// Two TTSStartedFrames in flight before any audio arrives.
	if err := p.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(idA), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame A: %v", err)
	}
	if err := p.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(idB), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame B: %v", err)
	}

	p.interruptionMu.Lock()
	if _, ok := p.expectedContexts[idA]; !ok {
		t.Errorf("expected idA in expectedContexts")
	}
	if _, ok := p.expectedContexts[idB]; !ok {
		t.Errorf("expected idB in expectedContexts")
	}
	p.interruptionMu.Unlock()

	// Audio for B arrives first — must be accepted, B promotes to current.
	audioB := frames.NewTTSAudioFrame([]byte("Bbb"), 24000, 1)
	audioB.SetMetadata("context_id", idB)
	if err := p.HandleFrame(ctx, audioB, frames.Downstream); err != nil {
		t.Fatalf("audio B: %v", err)
	}

	// Audio for A arrives — also accepted, A promotes alongside B.
	audioA := frames.NewTTSAudioFrame([]byte("Aaa"), 24000, 1)
	audioA.SetMetadata("context_id", idA)
	if err := p.HandleFrame(ctx, audioA, frames.Downstream); err != nil {
		t.Fatalf("audio A: %v", err)
	}

	p.interruptionMu.Lock()
	defer p.interruptionMu.Unlock()
	if _, ok := p.currentContexts[idA]; !ok {
		t.Error("idA must have been promoted to currentContexts")
	}
	if _, ok := p.currentContexts[idB]; !ok {
		t.Error("idB must have been promoted to currentContexts")
	}
	if _, lingering := p.expectedContexts[idA]; lingering {
		t.Error("idA must have been removed from expectedContexts after promotion")
	}
}

func TestConcurrentStaleAudioBlocked(t *testing.T) {
	tt := newConcurrentTestTransport()
	p := tt.outputProc
	ctx := context.Background()

	known := services.GenerateContextID()
	unknown := services.GenerateContextID()

	if err := p.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(known), frames.Downstream); err != nil {
		t.Fatalf("known started: %v", err)
	}

	// Audio for an unknown context (e.g. one we never saw a TTSStartedFrame for)
	// must be blocked and must NOT register itself in either set.
	stale := frames.NewTTSAudioFrame([]byte("stale"), 24000, 1)
	stale.SetMetadata("context_id", unknown)
	if err := p.HandleFrame(ctx, stale, frames.Downstream); err != nil {
		t.Fatalf("stale: %v", err)
	}

	p.interruptionMu.Lock()
	defer p.interruptionMu.Unlock()
	if _, ok := p.currentContexts[unknown]; ok {
		t.Error("unknown context must not slip into currentContexts")
	}
	if _, ok := p.expectedContexts[unknown]; ok {
		t.Error("unknown context must not slip into expectedContexts")
	}
	if p.staleAudioBlockedCount == 0 {
		t.Error("expected stale audio counter to advance")
	}
}

func TestConcurrentInterruptionWipesAllSets(t *testing.T) {
	tt := newConcurrentTestTransport()
	p := tt.outputProc
	ctx := context.Background()
	enableInterruptions(t, p)

	idA := services.GenerateContextID()
	idB := services.GenerateContextID()

	if err := p.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(idA), frames.Downstream); err != nil {
		t.Fatalf("started A: %v", err)
	}
	if err := p.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(idB), frames.Downstream); err != nil {
		t.Fatalf("started B: %v", err)
	}

	// Promote one of them.
	audioA := frames.NewTTSAudioFrame([]byte("a"), 24000, 1)
	audioA.SetMetadata("context_id", idA)
	if err := p.HandleFrame(ctx, audioA, frames.Downstream); err != nil {
		t.Fatalf("audioA: %v", err)
	}

	if err := p.HandleFrame(ctx, frames.NewInterruptionFrame(), frames.Downstream); err != nil {
		t.Fatalf("interruption: %v", err)
	}

	p.interruptionMu.Lock()
	if !p.interrupted {
		t.Error("interrupted must be true after InterruptionFrame")
	}
	if len(p.expectedContexts) != 0 {
		t.Errorf("expectedContexts must be empty after interruption, got %v", p.expectedContexts)
	}
	if len(p.currentContexts) != 0 {
		t.Errorf("currentContexts must be empty after interruption, got %v", p.currentContexts)
	}
	p.interruptionMu.Unlock()

	// Old in-flight audio after interrupt must remain blocked.
	leftover := frames.NewTTSAudioFrame([]byte("leftover"), 24000, 1)
	leftover.SetMetadata("context_id", idB)
	if err := p.HandleFrame(ctx, leftover, frames.Downstream); err != nil {
		t.Fatalf("leftover: %v", err)
	}

	p.interruptionMu.Lock()
	if _, ok := p.currentContexts[idB]; ok {
		t.Error("post-interrupt audio for old context must not register")
	}
	p.interruptionMu.Unlock()

	// New TTSStartedFrame after interrupt repopulates expectedContexts.
	idC := services.GenerateContextID()
	if err := p.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(idC), frames.Downstream); err != nil {
		t.Fatalf("started C: %v", err)
	}
	audioC := frames.NewTTSAudioFrame([]byte("c"), 24000, 1)
	audioC.SetMetadata("context_id", idC)
	if err := p.HandleFrame(ctx, audioC, frames.Downstream); err != nil {
		t.Fatalf("audioC: %v", err)
	}

	p.interruptionMu.Lock()
	defer p.interruptionMu.Unlock()
	if p.interrupted {
		t.Error("interrupted must clear when new context audio is accepted")
	}
	if _, ok := p.currentContexts[idC]; !ok {
		t.Error("idC must be promoted to currentContexts")
	}
}

func TestConcurrentTTSStoppedFramePrunesContext(t *testing.T) {
	tt := newConcurrentTestTransport()
	p := tt.outputProc
	ctx := context.Background()

	idA := services.GenerateContextID()
	idB := services.GenerateContextID()

	for _, id := range []string{idA, idB} {
		if err := p.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(id), frames.Downstream); err != nil {
			t.Fatalf("started %s: %v", id, err)
		}
		af := frames.NewTTSAudioFrame([]byte("x"), 24000, 1)
		af.SetMetadata("context_id", id)
		if err := p.HandleFrame(ctx, af, frames.Downstream); err != nil {
			t.Fatalf("audio %s: %v", id, err)
		}
	}

	// Pre-stage idA in expectedContexts too (simulate the case where a stop
	// frame arrives before any audio promotes it).
	idC := services.GenerateContextID()
	if err := p.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(idC), frames.Downstream); err != nil {
		t.Fatalf("started C: %v", err)
	}

	// Per-context stop frame for idA — should drop idA from currentContexts
	// without affecting idB. Stop frame for idC must drop it from
	// expectedContexts.
	if err := p.HandleFrame(ctx, frames.NewTTSStoppedFrameWithContext(idA), frames.Downstream); err != nil {
		t.Fatalf("stopped A: %v", err)
	}
	if err := p.HandleFrame(ctx, frames.NewTTSStoppedFrameWithContext(idC), frames.Downstream); err != nil {
		t.Fatalf("stopped C: %v", err)
	}

	p.interruptionMu.Lock()
	if _, ok := p.currentContexts[idA]; ok {
		t.Error("idA should be pruned from currentContexts after TTSStoppedFrameWithContext(idA)")
	}
	if _, ok := p.expectedContexts[idA]; ok {
		t.Error("idA should also be pruned from expectedContexts on stop")
	}
	if _, ok := p.currentContexts[idB]; !ok {
		t.Error("idB should remain in currentContexts")
	}
	if _, ok := p.expectedContexts[idC]; ok {
		t.Error("idC should be pruned from expectedContexts after stop (never promoted)")
	}
	if _, ok := p.currentContexts[idC]; ok {
		t.Error("idC should not be in currentContexts (was never promoted)")
	}
	p.interruptionMu.Unlock()

	// Stale audio for idA should be blocked since idA is gone.
	stale := frames.NewTTSAudioFrame([]byte("late"), 24000, 1)
	stale.SetMetadata("context_id", idA)
	if err := p.HandleFrame(ctx, stale, frames.Downstream); err != nil {
		t.Fatalf("stale: %v", err)
	}
	p.interruptionMu.Lock()
	if _, ok := p.currentContexts[idA]; ok {
		t.Error("late audio for pruned context must not re-register idA")
	}
	if p.staleAudioBlockedCount == 0 {
		t.Error("late audio after stop should advance stale counter")
	}
	p.interruptionMu.Unlock()
}

func TestConcurrentLegacyStoppedFrameDoesNotTouchSets(t *testing.T) {
	// Empty-ContextID TTSStoppedFrame must NOT enter the prune branch,
	// even in concurrent mode. Pre-populate the sets and verify they
	// survive the legacy frame.
	tt := newConcurrentTestTransport()
	p := tt.outputProc
	ctx := context.Background()

	idA := services.GenerateContextID()
	if err := p.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(idA), frames.Downstream); err != nil {
		t.Fatalf("started A: %v", err)
	}

	if err := p.HandleFrame(ctx, frames.NewTTSStoppedFrame(), frames.Downstream); err != nil {
		t.Fatalf("legacy stop: %v", err)
	}

	p.interruptionMu.Lock()
	defer p.interruptionMu.Unlock()
	if _, ok := p.expectedContexts[idA]; !ok {
		t.Error("legacy empty-ContextID stop must not prune sets")
	}
}

func TestConcurrentInterleavedAudioAcrossContexts(t *testing.T) {
	tt := newConcurrentTestTransport()
	p := tt.outputProc
	ctx := context.Background()

	idA := services.GenerateContextID()
	idB := services.GenerateContextID()

	for _, id := range []string{idA, idB} {
		if err := p.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(id), frames.Downstream); err != nil {
			t.Fatalf("started %s: %v", id, err)
		}
	}

	// Interleave four chunks across the two contexts; all should pass.
	mkAudio := func(id string, payload string) *frames.TTSAudioFrame {
		f := frames.NewTTSAudioFrame([]byte(payload), 24000, 1)
		f.SetMetadata("context_id", id)
		return f
	}
	for _, f := range []*frames.TTSAudioFrame{
		mkAudio(idA, "a1"),
		mkAudio(idB, "b1"),
		mkAudio(idB, "b2"),
		mkAudio(idA, "a2"),
	} {
		if err := p.HandleFrame(ctx, f, frames.Downstream); err != nil {
			t.Fatalf("audio: %v", err)
		}
	}

	p.interruptionMu.Lock()
	defer p.interruptionMu.Unlock()
	if _, ok := p.currentContexts[idA]; !ok {
		t.Error("idA missing from currentContexts after interleaved audio")
	}
	if _, ok := p.currentContexts[idB]; !ok {
		t.Error("idB missing from currentContexts after interleaved audio")
	}
}
