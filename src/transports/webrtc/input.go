package webrtc

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type InputProcessor struct {
	*processors.BaseProcessor
	session *Session

	startOnce  sync.Once
	ending     atomic.Bool
	targetRate int
	emit       func(frames.Frame)
}

func newInputProcessor(s *Session) *InputProcessor {
	p := &InputProcessor{session: s}
	p.BaseProcessor = processors.NewBaseProcessor("WebRTCInput", p)
	p.emit = func(f frames.Frame) { _ = p.PushFrame(f, frames.Downstream) }
	return p
}

func (p *InputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		err := p.PushFrame(frame, direction)
		p.startOnce.Do(func() {
			p.targetRate = p.session.params.AudioInSampleRate
			if p.targetRate == 0 {
				p.targetRate = f.AudioInSampleRate
			}
			close(p.session.startedCh)
		})
		return err
	case *frames.EndFrame, *frames.CancelFrame:
		p.ending.Store(true)
	}
	return p.PushFrame(frame, direction)
}

func (p *InputProcessor) deliverAudio(f *frames.AudioFrame) {
	if a := p.session.params.AudioInAnalyzer; a != nil {
		a.AnalyzeAudio(f, p.emit)
	}
	if !p.session.params.DisableAudioInPassthrough {
		_ = p.PushFrame(f, frames.Downstream)
	}
}
