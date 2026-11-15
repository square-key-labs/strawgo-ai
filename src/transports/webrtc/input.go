package webrtc

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/audio/filter"
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

	audioMu    sync.Mutex
	filter     filter.Filter
	filterRate int
	filterBuf  []byte
}

func newInputProcessor(s *Session) *InputProcessor {
	p := &InputProcessor{session: s, filter: s.params.InputFilter}
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
		p.stopFilter()
	case *frames.FilterEnableFrame, *frames.FilterUpdateSettingsFrame:
		p.filterControl(frame)
	}
	return p.PushFrame(frame, direction)
}

func (p *InputProcessor) filterControl(frame frames.Frame) {
	p.audioMu.Lock()
	defer p.audioMu.Unlock()
	if p.filter == nil {
		return
	}
	if err := p.filter.ProcessFrame(frame); err != nil {
		p.session.log.Warn("input filter: %v", err)
	}
}

func (p *InputProcessor) stopFilter() {
	p.audioMu.Lock()
	defer p.audioMu.Unlock()
	if p.filter != nil && p.filterRate != 0 {
		p.filterRate = 0
		_ = p.filter.Stop()
	}
}

func (p *InputProcessor) deliverAudio(f *frames.AudioFrame) {
	p.audioMu.Lock()
	if p.filter != nil {
		p.applyFilter(f)
	}
	if a := p.session.params.AudioInAnalyzer; a != nil {
		a.AnalyzeAudio(f, p.emit)
	}
	p.audioMu.Unlock()
	if !p.session.params.DisableAudioInPassthrough {
		_ = p.PushFrame(f, frames.Downstream)
	}
}

func (p *InputProcessor) applyFilter(f *frames.AudioFrame) {
	if f.SampleRate <= 0 {
		return
	}
	if p.filterRate != f.SampleRate {
		if p.filterRate != 0 {
			_ = p.filter.Stop()
		}
		if err := p.filter.Start(f.SampleRate); err != nil {
			p.session.log.Warn("input filter start: %v", err)
			p.filter = nil
			return
		}
		p.filterRate = f.SampleRate
	}
	p.filterBuf = p.filter.Filter(p.filterBuf[:0], f.Data)
	if len(p.filterBuf) == len(f.Data) {
		copy(f.Data, p.filterBuf)
	} else {
		f.Data = append([]byte(nil), p.filterBuf...)
	}
}
