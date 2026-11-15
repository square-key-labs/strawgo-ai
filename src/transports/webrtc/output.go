package webrtc

import (
	"bytes"
	"context"
	"encoding/binary"
	"sync"
	"time"

	"github.com/pion/webrtc/v4/pkg/media"
	"github.com/square-key-labs/strawgo-ai/src/audio"
	pcmpkg "github.com/square-key-labs/strawgo-ai/src/audio/pcm"
	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

type outKind uint8

const (
	itemAudio outKind = iota
	itemFrame
	itemEnd
)

type outItem struct {
	kind  outKind
	pcm   []byte
	ctx   context.Context
	frame frames.Frame
}

func alive(ctx context.Context) bool {
	return ctx == nil || ctx.Err() == nil
}

type OutputProcessor struct {
	*processors.BaseProcessor
	session *Session

	queue     chan outItem
	urgent    chan frames.Frame
	interrupt chan struct{}

	senderCancel context.CancelFunc
	senderDone   chan struct{}

	wireRate int
	rs       *resample.Resampler
	rsIn     int
	rsDirty  bool
	rsTail   []int16
	lastCtx  context.Context
	scratch  []byte

	finishOnce sync.Once
}

func newOutputProcessor(s *Session) *OutputProcessor {
	p := &OutputProcessor{
		session:   s,
		queue:     make(chan outItem, outputQueueSize),
		urgent:    make(chan frames.Frame, 16),
		interrupt: make(chan struct{}, 1),
	}
	p.BaseProcessor = processors.NewBaseProcessor("WebRTCOutput", p, processors.WithConcurrentSystemFrames())
	return p
}

func (p *OutputProcessor) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	err := p.BaseProcessor.QueueFrame(frame, direction)
	if _, ok := frame.(*frames.InterruptionFrame); ok && direction == frames.Downstream {
		select {
		case p.interrupt <- struct{}{}:
		default:
		}
	}
	return err
}

func (p *OutputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if direction == frames.Upstream {
		return p.PushFrame(frame, direction)
	}
	switch f := frame.(type) {
	case *frames.StartFrame:
		err := p.PushFrame(frame, direction)
		p.onStart()
		return err
	case *frames.CancelFrame:
		p.cancelSender()
		p.finish(frame)
		return p.PushFrame(frame, direction)
	case *frames.EndFrame:
		p.finish(frame)
		return p.PushFrame(frame, direction)
	case *frames.OutputTransportMessageUrgentFrame:
		p.sendUrgent(f)
		return nil
	}
	if frames.IsSystem(frame) {
		return p.PushFrame(frame, direction)
	}
	switch f := frame.(type) {
	case *frames.TTSAudioFrame:
		return p.handleAudio(ctx, f)
	case *frames.AudioFrame:
		return p.PushFrame(frame, direction)
	case *frames.TTSStoppedFrame:
		p.flushResampler(ctx)
	}
	if p.senderDone == nil {
		return p.PushFrame(frame, direction)
	}
	if frame.SurvivesInterruption() {
		p.enqueue(p.Context(), outItem{kind: itemFrame, frame: frame})
		return nil
	}
	p.enqueue(ctx, outItem{kind: itemFrame, frame: frame, ctx: ctx})
	return nil
}

func (p *OutputProcessor) onStart() {
	if p.senderDone != nil {
		return
	}
	pc, err := newPacer(p)
	if err != nil {
		_ = p.PushError("webrtc output encoder", err, true)
		return
	}
	ctx, cancel := context.WithCancel(p.Context())
	p.senderCancel = cancel
	p.senderDone = pc.done
	go pc.run(ctx)
}

func (p *OutputProcessor) cancelSender() {
	if p.senderCancel != nil {
		p.senderCancel()
	}
}

func (p *OutputProcessor) enqueue(ctx context.Context, it outItem) {
	select {
	case p.queue <- it:
	case <-ctx.Done():
	case <-p.senderDone:
	}
}

func (p *OutputProcessor) sendUrgent(frame frames.Frame) {
	if p.senderDone == nil {
		return
	}
	select {
	case p.urgent <- frame:
	case <-p.senderDone:
	case <-p.Context().Done():
	}
}

func (p *OutputProcessor) resetResampler() {
	p.rsDirty = false
	if p.rs != nil {
		p.rs.Reset()
	}
}

func (p *OutputProcessor) trackEpoch(ctx context.Context) {
	if p.lastCtx != ctx {
		if p.lastCtx != nil && p.lastCtx.Err() != nil {
			p.resetResampler()
		}
		p.lastCtx = ctx
	}
}

func (p *OutputProcessor) flushResampler(ctx context.Context) {
	p.trackEpoch(ctx)
	if ctx.Err() != nil {
		p.resetResampler()
		return
	}
	if !p.rsDirty || p.rs == nil {
		return
	}
	p.rsDirty = false
	p.rsTail = p.rs.Flush(p.rsTail[:0])
	if len(p.rsTail) == 0 || p.senderDone == nil || p.wireRate == 0 {
		return
	}
	p.enqueue(ctx, outItem{kind: itemAudio, pcm: pcmpkg.FromInt16(nil, p.rsTail), ctx: ctx})
}

func (p *OutputProcessor) handleAudio(ctx context.Context, f *frames.TTSAudioFrame) error {
	p.trackEpoch(ctx)
	if ctx.Err() != nil || p.senderDone == nil || p.wireRate == 0 || len(f.Data) == 0 {
		return nil
	}
	pcm, err := p.toPCM(f)
	if err != nil {
		return p.PushError("output audio conversion failed", err, false)
	}
	p.enqueue(ctx, outItem{kind: itemAudio, pcm: pcm, ctx: ctx, frame: f})
	return nil
}

func (p *OutputProcessor) toPCM(f *frames.TTSAudioFrame) ([]byte, error) {
	codec, _ := f.Metadata()["codec"].(string)
	codec = audio.NormalizeCodec(codec)
	rate := f.SampleRate
	if rate <= 0 {
		rate = p.wireRate
	}
	src := f.Data
	if codec != audio.CodecLinear16 || f.Channels > 1 {
		pcm, err := audio.DecodePCM16(p.scratch[:0], f.Data, codec)
		if err != nil {
			return nil, err
		}
		if f.Channels > 1 {
			pcm = downmix(pcm, f.Channels)
		}
		p.scratch = pcm
		src = pcm
	}
	if rate == p.wireRate {
		return bytes.Clone(src[:len(src)&^1]), nil
	}
	if p.rs == nil || p.rsIn != rate {
		rs, err := resample.New(rate, p.wireRate)
		if err != nil {
			return nil, err
		}
		p.rs, p.rsIn = rs, rate
		p.rsDirty = false
	}
	p.rsDirty = p.rsDirty || len(src) > 1
	return p.rs.ProcessBytes(make([]byte, 0, 2*p.rs.MaxOutput(len(src)/2)), src), nil
}

func downmix(pcm []byte, channels int) []byte {
	frameBytes := 2 * channels
	n := len(pcm) / frameBytes
	out := pcm[:0]
	for i := 0; i < n; i++ {
		sum := 0
		for c := 0; c < channels; c++ {
			sum += int(int16(binary.LittleEndian.Uint16(pcm[i*frameBytes+2*c:])))
		}
		out = binary.LittleEndian.AppendUint16(out, uint16(int16(sum/channels)))
	}
	return out
}

func (p *OutputProcessor) finish(frame frames.Frame) {
	p.finishOnce.Do(func() {
		if p.senderDone != nil {
			if _, end := frame.(*frames.EndFrame); end {
				select {
				case p.queue <- outItem{kind: itemEnd}:
				case <-p.senderDone:
				case <-p.Context().Done():
				}
			}
			select {
			case <-p.senderDone:
			case <-p.Context().Done():
				p.cancelSender()
				<-p.senderDone
			}
		}
		_ = p.session.Close()
	})
}

type pacer struct {
	out    *OutputProcessor
	s      *Session
	enc    Encoder
	params Params
	done   chan struct{}

	chunkBytes int
	buf        []byte
	bufCtx     context.Context
	held       outItem
	hasHeld    bool
	silence    []byte
	encBuf     []byte
	mixOut     []byte

	fadeBytes int
	fading    bool

	mixerStarted bool
	nextSend     time.Time
	timer        *time.Timer
	armed        bool

	speaking       bool
	lastSpeakingAt time.Time
	quiet          time.Duration
}

func newPacer(out *OutputProcessor) (*pacer, error) {
	s := out.session
	p := &pacer{
		out:    out,
		s:      s,
		params: s.params,
		done:   make(chan struct{}),
		timer:  newStoppedTimer(),
	}
	if s.codec == nil {
		return p, nil
	}
	enc, err := s.codec.NewEncoder()
	if err != nil {
		return nil, err
	}
	rate := s.codec.SampleRate()
	p.enc = enc
	out.wireRate = rate
	p.chunkBytes = 2 * int(int64(rate)*int64(audioChunk)/int64(time.Second))
	p.silence = make([]byte, p.chunkBytes)
	p.fadeBytes = 2 * transports.FadeSamples(rate, p.params.InterruptionFade)
	p.buf = make([]byte, 0, max(4*p.chunkBytes, p.fadeBytes+p.chunkBytes))
	if m := p.params.AudioOutMixer; m != nil {
		if err := m.Start(rate); err != nil {
			s.log.Warn("mixer start: %v", err)
		} else {
			p.mixerStarted = true
		}
	}
	return p, nil
}

func (p *pacer) run(ctx context.Context) {
	defer close(p.done)
	defer p.shutdown()
	for {
		if p.progress(ctx) {
			return
		}
		p.arm()
		var queueC <-chan outItem
		if !p.hasHeld && !p.fading && (p.chunkBytes == 0 || len(p.buf) < p.chunkBytes) {
			queueC = p.out.queue
		}
		var tickC <-chan time.Time
		if p.armed {
			tickC = p.timer.C
		}
		select {
		case <-ctx.Done():
			return
		case <-p.out.interrupt:
			p.onInterrupt()
		case f := <-p.out.urgent:
			if u, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
				p.sendMessage(ctx, u.Message)
			}
		case it := <-queueC:
			p.accept(it)
		case <-tickC:
			p.armed = false
			p.tick()
		}
	}
}

func (p *pacer) shutdown() {
	p.timer.Stop()
	if p.mixerStarted {
		_ = p.params.AudioOutMixer.Stop()
	}
	p.botStopped()
}

func (p *pacer) progress(ctx context.Context) bool {
	for p.hasHeld {
		if !alive(p.held.ctx) {
			p.held, p.hasHeld = outItem{}, false
			continue
		}
		if p.fading {
			return false
		}
		if p.held.kind == itemAudio {
			it := p.held
			p.held, p.hasHeld = outItem{}, false
			p.accept(it)
			continue
		}
		if !alive(p.bufCtx) {
			p.buf = p.buf[:0]
		}
		if len(p.buf) > 0 {
			if r := len(p.buf) % p.chunkBytes; r != 0 {
				p.buf = append(p.buf, p.silence[:p.chunkBytes-r]...)
			}
			return false
		}
		it := p.held
		p.held, p.hasHeld = outItem{}, false
		if it.kind == itemEnd {
			p.waitTick(ctx)
			return true
		}
		p.handleFrame(ctx, it)
	}
	return false
}

func (p *pacer) waitTick(ctx context.Context) {
	if !p.armed {
		return
	}
	select {
	case <-p.timer.C:
	case <-ctx.Done():
	}
	p.armed = false
}

func (p *pacer) arm() {
	if p.armed || p.chunkBytes == 0 {
		return
	}
	if len(p.buf) < p.chunkBytes && !p.speaking && p.params.DisableAudioOutSilence {
		p.nextSend = time.Time{}
		return
	}
	now := time.Now()
	if p.nextSend.IsZero() || now.Sub(p.nextSend) > 2*audioChunk {
		p.nextSend = now
	}
	p.timer.Reset(p.nextSend.Sub(now))
	p.armed = true
}

func (p *pacer) accept(it outItem) {
	if !alive(it.ctx) {
		return
	}
	if it.kind != itemAudio {
		p.held, p.hasHeld = it, true
		return
	}
	if !alive(p.bufCtx) {
		p.buf = p.buf[:0]
	}
	p.bufCtx = it.ctx
	p.buf = append(p.buf, it.pcm...)
	if it.frame != nil {
		_ = p.out.PushFrame(it.frame, frames.Downstream)
	}
}

func (p *pacer) tick() {
	now := time.Now()
	p.nextSend = p.nextSend.Add(audioChunk)
	if p.fading && len(p.buf) < p.chunkBytes {
		p.fading = false
		p.buf = p.buf[:0]
		p.botStopped()
	}
	if !alive(p.bufCtx) && !p.startFade() {
		p.buf = p.buf[:0]
	}
	if len(p.buf) >= p.chunkBytes {
		p.write(p.buf[:p.chunkBytes])
		p.buf = p.buf[:copy(p.buf, p.buf[p.chunkBytes:])]
		p.onSpeech(now)
		return
	}
	if !p.params.DisableAudioOutSilence {
		p.write(p.silence)
	}
	if p.speaking {
		p.quiet += audioChunk
		if p.quiet >= p.params.BotStopTimeout {
			p.botStopped()
		}
	}
}

func (p *pacer) write(pcm []byte) {
	if p.mixerStarted {
		p.mixOut = p.params.AudioOutMixer.Mix(p.mixOut[:0], pcm)
		pcm = p.mixOut
	}
	var err error
	p.encBuf, err = p.enc.Encode(p.encBuf[:0], pcm)
	if err != nil {
		p.s.log.Debug("encode: %v", err)
		return
	}
	if err := p.s.track.WriteSample(media.Sample{Data: p.encBuf, Duration: audioChunk}); err != nil && !p.s.closing.Load() {
		p.s.log.Debug("write sample: %v", err)
	}
}

func (p *pacer) handleFrame(ctx context.Context, it outItem) {
	switch f := it.frame.(type) {
	case *frames.TTSStoppedFrame:
		p.botStopped()
	case *frames.OutputTransportMessageFrame:
		p.sendMessage(ctx, f.Message)
	case *frames.MixerUpdateSettingsFrame, *frames.MixerEnableFrame:
		if p.mixerStarted {
			if err := p.params.AudioOutMixer.ProcessFrame(f); err != nil {
				p.s.log.Warn("mixer: %v", err)
			}
		}
	}
	if alive(it.ctx) {
		_ = p.out.PushFrame(it.frame, frames.Downstream)
	}
}

func (p *pacer) sendMessage(ctx context.Context, msg any) {
	m, err := encodeMessage(msg)
	if err != nil {
		p.s.log.Warn("encode transport message: %v", err)
		return
	}
	_ = p.s.enqueue(ctx, m)
}

func (p *pacer) onSpeech(now time.Time) {
	p.quiet = 0
	if !p.speaking {
		p.speaking = true
		p.lastSpeakingAt = time.Time{}
		p.broadcast(func() frames.Frame { return frames.NewBotStartedSpeakingFrame() })
	}
	if now.Sub(p.lastSpeakingAt) >= botSpeakingPeriod {
		p.lastSpeakingAt = now
		p.broadcast(func() frames.Frame { return frames.NewBotSpeakingFrame() })
	}
}

func (p *pacer) botStopped() {
	if !p.speaking {
		return
	}
	p.speaking = false
	p.quiet = 0
	p.broadcast(func() frames.Frame { return frames.NewBotStoppedSpeakingFrame() })
}

func (p *pacer) broadcast(ctor func() frames.Frame) {
	_ = p.out.BroadcastFrame(context.Background(), ctor)
}

func (p *pacer) startFade() bool {
	if p.fading {
		return true
	}
	if p.fadeBytes == 0 || p.chunkBytes == 0 {
		return false
	}
	if alive(p.bufCtx) {
		if len(p.buf) > 0 {
			return false
		}
	} else if len(p.buf) > p.fadeBytes {
		p.buf = p.buf[:p.fadeBytes]
	}
	for len(p.buf) < p.fadeBytes && !p.hasHeld {
		select {
		case it := <-p.out.queue:
			if it.kind == itemEnd || alive(it.ctx) {
				p.held, p.hasHeld = it, true
				continue
			}
			if it.kind == itemAudio {
				p.buf = append(p.buf, it.pcm[:min(len(it.pcm), p.fadeBytes-len(p.buf))]...)
			}
			continue
		default:
		}
		break
	}
	p.bufCtx = nil
	p.buf = p.buf[:len(p.buf)&^1]
	if len(p.buf) == 0 {
		return false
	}
	pcmpkg.FadeOut(p.buf)
	if r := len(p.buf) % p.chunkBytes; r != 0 {
		p.buf = append(p.buf, p.silence[:p.chunkBytes-r]...)
	}
	p.fading = true
	return true
}

func (p *pacer) onInterrupt() {
	if p.startFade() {
		return
	}
	if !alive(p.bufCtx) {
		p.buf = p.buf[:0]
	}
	if p.hasHeld && !alive(p.held.ctx) {
		p.held, p.hasHeld = outItem{}, false
	}
	p.botStopped()
}
