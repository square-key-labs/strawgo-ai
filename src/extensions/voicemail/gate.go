package voicemail

import (
	"context"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type gateState int

const (
	gateHolding gateState = iota
	gateOpen
	gateDiscarding
)

type heldFrame struct {
	frame     frames.Frame
	direction frames.FrameDirection
}

type TTSGate struct {
	*processors.BaseProcessor

	mu      sync.Mutex
	state   gateState
	held    []heldFrame
	max     int
	dropped int
}

func newTTSGate(max int) *TTSGate {
	g := &TTSGate{max: max}
	g.BaseProcessor = processors.NewBaseProcessor("VoicemailTTSGate", g)
	return g
}

func isTTSOutput(frame frames.Frame) bool {
	switch frame.(type) {
	case *frames.TTSStartedFrame, *frames.TTSStoppedFrame, *frames.TTSTextFrame, *frames.TTSAudioFrame:
		return true
	}
	return false
}

func (g *TTSGate) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch frame.(type) {
	case *frames.InterruptionFrame, *frames.EndFrame, *frames.CancelFrame:
		g.mu.Lock()
		g.clearLocked()
		g.mu.Unlock()
		return g.PushFrame(frame, direction)
	}
	if isTTSOutput(frame) {
		g.mu.Lock()
		if g.state == gateHolding {
			if g.max <= 0 || len(g.held) < g.max {
				g.held = append(g.held, heldFrame{frame: frame, direction: direction})
			} else {
				g.dropped++
			}
			g.mu.Unlock()
			return nil
		}
		g.mu.Unlock()
	}
	return g.PushFrame(frame, direction)
}

func (g *TTSGate) release() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.state != gateHolding {
		return
	}
	g.state = gateOpen
	if g.dropped > 0 {
		logger.Warn("[%s] Dropped %d TTS frames over the %d frame hold limit", g.Name(), g.dropped, g.max)
	}
	for _, h := range g.held {
		_ = g.PushFrame(h.frame, h.direction)
	}
	g.clearLocked()
}

func (g *TTSGate) discard() {
	g.mu.Lock()
	if g.state == gateHolding {
		g.state = gateDiscarding
	}
	g.clearLocked()
	g.mu.Unlock()
}

func (g *TTSGate) clearLocked() {
	g.held = nil
	g.dropped = 0
}
