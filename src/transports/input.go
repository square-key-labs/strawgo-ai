package transports

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/audio/filter"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type InputProcessor struct {
	*processors.BaseProcessor
	session *Session

	startedCh   chan struct{}
	startedOnce sync.Once
	ending      atomic.Bool

	targetRate int
	emit       func(frames.Frame)

	audioMu    sync.Mutex
	rs         audio.Resampler
	filter     filter.Filter
	filterRate int
	filterBuf  []byte
	pcmBuf     []byte
	pcmFrame   *frames.AudioFrame
}

func newInputProcessor(s *Session) *InputProcessor {
	p := &InputProcessor{session: s, startedCh: make(chan struct{}), filter: s.params.InputFilter}
	p.BaseProcessor = processors.NewBaseProcessor("WebSocketInput", p)
	p.emit = func(f frames.Frame) { _ = p.PushFrame(f, frames.Downstream) }
	return p
}

func (p *InputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		err := p.PushFrame(frame, direction)
		p.onStart(f)
		return err
	case *frames.EndFrame, *frames.CancelFrame:
		p.ending.Store(true)
		p.session.stopTimer()
		p.stopFilter()
		return p.PushFrame(frame, direction)
	case *frames.FilterEnableFrame, *frames.FilterUpdateSettingsFrame:
		p.filterControl(frame)
	case *frames.AudioFrame:
		if direction == frames.Downstream {
			p.deliverAudio(f)
			return nil
		}
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

func (p *InputProcessor) onStart(f *frames.StartFrame) {
	if p.started() {
		return
	}
	p.targetRate = p.session.params.AudioInSampleRate
	if p.targetRate == 0 {
		p.targetRate = f.AudioInSampleRate
	}
	_ = p.PushFrame(frames.NewClientConnectedFrame(), frames.Downstream)
	if cb := p.session.params.OnClientConnected; cb != nil {
		cb(p.session)
	}
	p.startedOnce.Do(func() { close(p.startedCh) })
	p.session.startTimer()
	p.session.maybeHandleDisconnect()
}

func (p *InputProcessor) started() bool {
	select {
	case <-p.startedCh:
		return true
	default:
		return false
	}
}

func (p *InputProcessor) waitStarted(ctx context.Context) bool {
	select {
	case <-p.startedCh:
		return true
	case <-ctx.Done():
		return false
	}
}

func (p *InputProcessor) deliver(frame frames.Frame) {
	switch f := frame.(type) {
	case *frames.AudioFrame:
		p.deliverAudio(f)
	case *frames.InterruptionFrame:
		_ = p.BroadcastInterruption(p.session.ctx)
	case *frames.InputTransportMessageFrame:
		msg := f.Message
		_ = p.BroadcastFrame(p.session.ctx, func() frames.Frame { return frames.NewInputTransportMessageFrame(msg) })
	default:
		_ = p.PushFrame(frame, frames.Downstream)
	}
}

func (p *InputProcessor) deliverAudio(f *frames.AudioFrame) {
	p.audioMu.Lock()
	codec := frameCodec(f)
	native := codec != audio.CodecLinear16
	if !native && p.targetRate > 0 && f.SampleRate > 0 && f.SampleRate != p.targetRate {
		f.Data = p.rs.Resample(make([]byte, 0, p.rs.MaxOutputBytes(len(f.Data), f.SampleRate, p.targetRate)), f.Data, f.SampleRate, p.targetRate)
		f.SampleRate = p.targetRate
	}
	analyzer := p.session.params.AudioInAnalyzer
	var pcm []byte
	if native && (p.filter != nil || analyzer != nil) {
		var err error
		if p.pcmBuf, err = audio.DecodePCM16(p.pcmBuf[:0], f.Data, codec); err == nil {
			pcm = p.pcmBuf
		}
	}
	if p.filter != nil {
		pcm = p.applyFilter(f, codec, pcm)
	}
	if analyzer != nil {
		if native {
			if pcm != nil {
				if p.pcmFrame == nil {
					p.pcmFrame = frames.NewAudioFrame(nil, 0, 1)
				}
				p.pcmFrame.Data, p.pcmFrame.SampleRate = pcm, f.SampleRate
				analyzer.AnalyzeAudio(p.pcmFrame, p.emit)
			}
		} else {
			analyzer.AnalyzeAudio(f, p.emit)
		}
	}
	p.audioMu.Unlock()
	if !p.session.params.DisableAudioInPassthrough {
		_ = p.PushFrame(f, frames.Downstream)
	}
}

func (p *InputProcessor) applyFilter(f *frames.AudioFrame, codec string, decoded []byte) []byte {
	if f.SampleRate <= 0 {
		return decoded
	}
	if p.filterRate != f.SampleRate {
		if p.filterRate != 0 {
			_ = p.filter.Stop()
		}
		if err := p.filter.Start(f.SampleRate); err != nil {
			p.session.log.Warn("input filter start: %v", err)
			p.filter = nil
			return decoded
		}
		p.filterRate = f.SampleRate
	}
	if codec == audio.CodecLinear16 {
		p.filterBuf = p.filter.Filter(p.filterBuf[:0], f.Data)
		if len(p.filterBuf) == len(f.Data) {
			copy(f.Data, p.filterBuf)
		} else {
			f.Data = append([]byte(nil), p.filterBuf...)
		}
		return nil
	}
	if decoded == nil {
		return nil
	}
	p.filterBuf = p.filter.Filter(p.filterBuf[:0], decoded)
	if out, err := audio.EncodePCM16(f.Data[:0], p.filterBuf, codec); err == nil {
		f.Data = out
	}
	return p.filterBuf
}
