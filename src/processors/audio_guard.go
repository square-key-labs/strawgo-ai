package processors

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// AudioGuard enforces the invariant that TTSAudioFrames reaching downstream
// belong to the active Generation. Sits in the pipeline between the TTS
// service and the output transport.
//
// Lifecycle:
//   - TTSStartedFrame sets the expected Generation ID.
//   - The first TTSAudioFrame whose context_id matches the expectation locks
//     in as the active Generation. Subsequent frames must match.
//   - InterruptionFrame invalidates the active Generation. The next
//     TTSStartedFrame begins a new one.
//
// In Phase 1 of the rollout it operates in SHADOW mode (Active=false): the
// state machine runs and logs every frame it WOULD drop, but every frame is
// still admitted downstream. This lets us validate the gate before deleting
// the duplicate gate that lives in the output transport. Set Active=true to
// flip the gate from shadow to authoritative.
//
// See docs/adr/0001-audio-guard-as-dedicated-processor.md.
type AudioGuard struct {
	*BaseProcessor

	// Active controls whether the gate enforces (true) or shadows (false).
	// Default false.
	Active atomic.Bool

	mu                 sync.Mutex
	expectedGeneration string
	activeGeneration   string
	invalidated        bool

	wouldDropCount     uint64
	lastWouldDropGenID string

	log *logger.Logger
}

// NewAudioGuard creates a new AudioGuard in shadow mode (admits all frames).
func NewAudioGuard() *AudioGuard {
	g := &AudioGuard{
		log: logger.WithPrefix("AudioGuard"),
	}
	g.BaseProcessor = NewBaseProcessor("AudioGuard", g)
	return g
}

// HandleFrame routes frames through the gate state machine.
func (g *AudioGuard) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.TTSStartedFrame:
		g.beginGeneration(f.ContextID)
	case *frames.TTSAudioFrame:
		if !g.admit(f) && g.Active.Load() {
			return nil
		}
	case *frames.InterruptionFrame:
		g.invalidate()
	}
	return g.PushFrame(frame, direction)
}

// beginGeneration records the expected Generation from a TTSStartedFrame.
// Resets the active Generation; the next matching TTSAudioFrame locks it in.
func (g *AudioGuard) beginGeneration(genID string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.wouldDropCount > 0 {
		g.log.Debug("Generation reset; %d frames %s from %s",
			g.wouldDropCount, g.dropVerb(), g.lastWouldDropGenID)
		g.wouldDropCount = 0
		g.lastWouldDropGenID = ""
	}

	g.expectedGeneration = genID
	g.activeGeneration = ""
	g.invalidated = false
	g.log.Debug("Begin Generation: %s", genID)
}

// invalidate marks the active Generation as cancelled. After invalidate, no
// audio is admitted until the next TTSStartedFrame (which clears the flag).
func (g *AudioGuard) invalidate() {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.invalidated = true
	g.activeGeneration = ""
	g.expectedGeneration = ""
	g.log.Debug("Invalidated; awaiting next Generation")
}

// admit decides whether a TTSAudioFrame should pass downstream. Returns true
// if the frame belongs to the active or expected Generation. The state
// machine runs regardless of Active mode; the caller chooses whether to
// honor a false return.
func (g *AudioGuard) admit(f *frames.TTSAudioFrame) bool {
	frameGen := frameGenerationID(f)

	g.mu.Lock()
	defer g.mu.Unlock()

	if frameGen == "" {
		return true
	}

	// After invalidation we admit nothing until the next TTSStartedFrame
	// resets state (which clears the flag and sets a new expected generation).
	if g.invalidated && g.expectedGeneration == "" {
		g.recordDropLocked(frameGen, "invalidated")
		return false
	}

	if g.expectedGeneration != "" && frameGen != g.expectedGeneration {
		g.recordDropLocked(frameGen, "expected="+g.expectedGeneration)
		return false
	}

	if g.activeGeneration == "" {
		g.activeGeneration = frameGen
		g.invalidated = false
		return true
	}

	if frameGen != g.activeGeneration {
		g.recordDropLocked(frameGen, "active="+g.activeGeneration)
		return false
	}

	return true
}

// recordDropLocked logs/counts a drop. Caller must hold g.mu.
func (g *AudioGuard) recordDropLocked(frameGen, reason string) {
	if g.lastWouldDropGenID != frameGen {
		if g.wouldDropCount > 0 {
			g.log.Debug("%s %d frames from %s (%s)",
				g.dropVerb(), g.wouldDropCount, g.lastWouldDropGenID, reason)
		}
		g.lastWouldDropGenID = frameGen
		g.wouldDropCount = 1
		g.log.Debug("%s audio from %s (%s)", g.dropVerbSingle(), frameGen, reason)
		return
	}
	g.wouldDropCount++
}

func (g *AudioGuard) dropVerb() string {
	if g.Active.Load() {
		return "dropped"
	}
	return "would-drop"
}

func (g *AudioGuard) dropVerbSingle() string {
	if g.Active.Load() {
		return "drop"
	}
	return "would-drop"
}

// frameGenerationID returns the generation ID stamped on a TTSAudioFrame.
// Prefers the typed ContextID field; falls back to Metadata for services
// that only stamp via metadata.
func frameGenerationID(f *frames.TTSAudioFrame) string {
	if f.ContextID != "" {
		return f.ContextID
	}
	if v, ok := f.Metadata()["context_id"].(string); ok {
		return v
	}
	return ""
}
