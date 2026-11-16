package transports

import (
	"context"
	"strconv"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/audio/dtmf"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

type outKind uint8

const (
	itemAudio outKind = iota
	itemFrame
	itemEnd
)

type outItem struct {
	kind   outKind
	pcm    []byte
	rate   int
	codec  string
	ctx    context.Context
	frame  frames.Frame
	speech bool
}

func alive(ctx context.Context) bool {
	return ctx == nil || ctx.Err() == nil
}

type pacer struct {
	out    *OutputProcessor
	sess   *Session
	ser    serializers.FrameSerializer
	params TransportParams
	mixer  AudioMixer
	native serializers.NativeAudioSerializer
	done   chan struct{}

	rate        int
	samples     int
	chunkBytes  int
	chunkDur    time.Duration
	carry       []byte
	carryCtx    context.Context
	carrySpeech bool
	carryCodec  string
	silence     []byte
	mixOut      []byte
	nextSend    time.Time
	msgs        []serializers.Message
	deferred    []frames.Frame

	speaking       bool
	ttsAudio       bool
	playedAny      bool
	lastSpeakingAt time.Time

	timer        *time.Timer
	idle         *time.Timer
	idleC        <-chan time.Time
	ackTimer     *time.Timer
	ackC         <-chan time.Time
	awaiting     string
	ackSeq       uint64
	mixerStarted bool

	fadeSamples  int
	fadeBuf      []byte
	fading       bool
	clearPending bool
	cur          []byte
	curCtx       context.Context
	curCodec     string
	pending      outItem
	hasPending   bool
}

func newPacer(out *OutputProcessor) *pacer {
	p := &pacer{
		out:      out,
		sess:     out.session,
		ser:      out.session.serializer,
		params:   out.session.params,
		mixer:    out.session.params.outputMixer(),
		done:     make(chan struct{}),
		timer:    newStoppedTimer(),
		idle:     newStoppedTimer(),
		ackTimer: newStoppedTimer(),
	}
	p.native, _ = p.ser.(serializers.NativeAudioSerializer)
	p.setRate(out.outRate)
	return p
}

func newStoppedTimer() *time.Timer {
	t := time.NewTimer(time.Hour)
	t.Stop()
	return t
}

var readyChan = func() chan time.Time {
	c := make(chan time.Time)
	close(c)
	return c
}()

func (p *pacer) run(ctx context.Context) {
	defer close(p.done)
	defer p.shutdown()
	for {
		for len(p.deferred) > 0 {
			f := p.deferred[0]
			p.deferred = p.deferred[1:]
			p.handleUrgent(ctx, f, false)
		}
		if p.hasPending {
			it := p.pending
			p.pending, p.hasPending = outItem{}, false
			if it.kind == itemEnd {
				p.drainEnd(ctx)
				return
			}
			p.process(ctx, it)
			continue
		}
		var mixerC <-chan time.Time
		if p.mixerStarted {
			mixerC = readyChan
		}
		select {
		case <-ctx.Done():
			return
		case <-p.out.interrupt:
			p.onInterrupt(ctx)
		case f := <-p.out.urgent:
			p.handleUrgent(ctx, f, false)
		case a := <-p.out.acks:
			p.onAck(a)
		case <-p.ackC:
			p.ackTimeout()
		case <-p.idleC:
			p.idleC = nil
			if p.awaiting == "" {
				p.botStopped()
			}
		case <-mixerC:
			if len(p.out.queue) == 0 && !p.hasPending {
				p.writeChunk(ctx, nil, p.silence, "", false)
			}
		case it := <-p.out.queue:
			if it.kind == itemEnd {
				p.drainEnd(ctx)
				return
			}
			p.process(ctx, it)
		}
	}
}

func (p *pacer) shutdown() {
	p.timer.Stop()
	p.idle.Stop()
	p.ackTimer.Stop()
	if p.mixerStarted {
		_ = p.mixer.Stop()
	}
	if p.speaking {
		p.botStopped()
	}
}

func (p *pacer) setRate(rate int) {
	if rate <= 0 || rate == p.rate {
		return
	}
	samples := int(int64(rate) * int64(p.params.AudioOutChunk) / int64(time.Second))
	if samples < 1 {
		samples = 1
	}
	p.rate = rate
	p.samples = samples
	p.chunkBytes = samples * 2
	p.chunkDur = time.Duration(samples) * time.Second / time.Duration(rate)
	p.silence = make([]byte, p.chunkBytes)
	p.carry = make([]byte, 0, p.chunkBytes)
	p.carryCtx = nil
	p.carryCodec = ""
	p.cur = nil
	p.fadeSamples = FadeSamples(rate, p.params.InterruptionFade)
	p.fadeBuf = nil
	if p.fadeSamples > 0 {
		chunks := (p.fadeSamples + samples - 1) / samples
		p.fadeBuf = make([]byte, 0, chunks*p.chunkBytes)
	}
	if p.mixer != nil && !p.mixerStarted {
		if err := p.mixer.Start(rate); err != nil {
			p.sess.log.Warn("mixer start: %v", err)
		} else {
			p.mixerStarted = true
		}
	}
}

func (p *pacer) process(ctx context.Context, it outItem) {
	if !alive(it.ctx) {
		if it.kind == itemAudio && p.fadeSamples > 0 && it.rate == p.rate {
			p.cur, p.curCtx, p.curCodec = it.pcm, it.ctx, it.codec
			if it.codec == audio.CodecLinear16 {
				p.curCodec = ""
			}
			p.fadeDead(ctx)
		}
		return
	}
	switch it.kind {
	case itemAudio:
		p.cancelAck()
		p.setRate(it.rate)
		if p.play(ctx, it.pcm, it.codec, it.ctx, it.speech) && it.frame != nil && alive(it.ctx) {
			_ = p.out.PushFrame(it.frame, frames.Downstream)
		}
	case itemFrame:
		p.handleFrame(ctx, it)
	}
}

func (p *pacer) handleFrame(ctx context.Context, it outItem) {
	switch f := it.frame.(type) {
	case *frames.TTSStoppedFrame:
		p.flushCarry(ctx)
		if p.waitSlot(ctx, it.ctx) && p.ttsAudio {
			p.finishSpeech()
		}
	case *frames.OutputTransportMessageFrame:
		p.writeFrame(f)
	case *frames.OutputDTMFFrame:
		p.writeDTMF(ctx, f.Buttons, it.ctx)
	case *frames.TransferCallFrame:
		p.flushCarry(ctx)
		p.waitSlot(ctx, nil)
		go p.out.transfer(f)
	case *frames.MixerUpdateSettingsFrame, *frames.MixerEnableFrame:
		if p.mixer != nil {
			if err := p.mixer.ProcessFrame(f); err != nil {
				p.sess.log.Warn("mixer: %v", err)
			}
		}
	}
	if alive(it.ctx) {
		_ = p.out.PushFrame(it.frame, frames.Downstream)
	}
}

func (p *pacer) handleUrgent(ctx context.Context, f frames.Frame, inWait bool) {
	switch u := f.(type) {
	case *frames.OutputTransportMessageUrgentFrame:
		p.writeFrame(u)
	case *frames.OutputDTMFUrgentFrame:
		if _, native := p.ser.(serializers.DTMFSerializer); !native && inWait {
			p.deferred = append(p.deferred, f)
			return
		}
		p.writeDTMF(ctx, u.Buttons, nil)
	}
}

func (p *pacer) chunkSize(codec string) int {
	if codec == "" {
		return p.chunkBytes
	}
	return p.samples * audio.BytesPerSample(codec)
}

func (p *pacer) play(ctx context.Context, data []byte, codec string, ictx context.Context, speech bool) bool {
	if p.chunkBytes == 0 {
		return false
	}
	if codec == audio.CodecLinear16 {
		codec = ""
	}
	if !alive(p.carryCtx) {
		p.carry = p.carry[:0]
	}
	if len(p.carry) > 0 && p.carryCodec != codec {
		p.flushCarry(ctx)
	}
	p.carryCtx = ictx
	p.carrySpeech = speech
	p.carryCodec = codec
	cb := p.chunkSize(codec)
	if len(p.carry) > 0 {
		n := min(cb-len(p.carry), len(data))
		p.carry = append(p.carry, data[:n]...)
		data = data[n:]
		if len(p.carry) < cb {
			return true
		}
		p.cur, p.curCtx, p.curCodec = data, ictx, codec
		if !p.writeChunk(ctx, ictx, p.carry, codec, speech) {
			return false
		}
		p.carry = p.carry[:0]
	}
	for len(data) >= cb {
		p.cur, p.curCtx, p.curCodec = data, ictx, codec
		if !p.writeChunk(ctx, ictx, data[:cb], codec, speech) {
			return false
		}
		data = data[cb:]
	}
	p.cur = nil
	p.carry = append(p.carry, data...)
	return true
}

func (p *pacer) flushCarry(ctx context.Context) {
	if len(p.carry) == 0 {
		return
	}
	if !alive(p.carryCtx) {
		p.carry = p.carry[:0]
		return
	}
	n := len(p.carry)
	p.carry = p.carry[:p.chunkSize(p.carryCodec)]
	fill := byte(0)
	if p.carryCodec != "" {
		fill = audio.SilenceByte(p.carryCodec)
	}
	for i := n; i < len(p.carry); i++ {
		p.carry[i] = fill
	}
	p.writeChunk(ctx, p.carryCtx, p.carry, p.carryCodec, p.carrySpeech)
	p.carry = p.carry[:0]
}

func (p *pacer) writeChunk(ctx context.Context, ictx context.Context, chunk []byte, codec string, speech bool) bool {
	if !p.waitSlot(ctx, ictx) || !p.waitFlow(ctx, ictx) {
		return false
	}
	var err error
	if codec != "" && p.native != nil {
		p.msgs, err = p.native.SerializeNativeAudio(p.msgs[:0], chunk)
	} else {
		data := chunk
		if p.mixerStarted {
			p.mixOut = p.mixer.Mix(p.mixOut[:0], chunk)
			data = p.mixOut
		}
		p.msgs, err = p.ser.SerializeAudio(p.msgs[:0], data, p.rate)
	}
	if err != nil {
		p.sess.log.Warn("serialize audio: %v", err)
		return true
	}
	p.writeMsgs()
	now := time.Now()
	p.nextSend = p.nextSend.Add(p.chunkDur)
	if speech {
		p.playedAny = true
		p.onSpeech(now)
	}
	return true
}

func (p *pacer) waitFlow(ctx context.Context, ictx context.Context) bool {
	if !p.out.xoff.Load() {
		return true
	}
	var ictxDone <-chan struct{}
	if ictx != nil {
		ictxDone = ictx.Done()
	}
	for p.out.xoff.Load() {
		select {
		case <-p.out.flowWake:
		case <-ictxDone:
			return false
		case <-p.out.interrupt:
			p.onInterrupt(ctx)
			if !alive(ictx) {
				return false
			}
		case f := <-p.out.urgent:
			p.handleUrgent(ctx, f, true)
		case a := <-p.out.acks:
			p.onAck(a)
		case <-ctx.Done():
			return false
		}
	}
	p.nextSend = time.Now()
	return alive(ictx) && ctx.Err() == nil
}

func (p *pacer) waitSlot(ctx context.Context, ictx context.Context) bool {
	now := time.Now()
	if p.nextSend.IsZero() || now.Sub(p.nextSend) > 2*p.chunkDur {
		p.nextSend = now
		return alive(ictx) && ctx.Err() == nil
	}
	deadline := p.nextSend
	d := deadline.Sub(now)
	if d <= 0 {
		return alive(ictx) && ctx.Err() == nil
	}
	var ictxDone <-chan struct{}
	if ictx != nil {
		ictxDone = ictx.Done()
	}
	p.timer.Reset(d)
	for {
		select {
		case <-p.timer.C:
			return alive(ictx) && ctx.Err() == nil
		case <-ictxDone:
			p.timer.Stop()
			p.fadeDead(ctx)
			return false
		case <-p.out.interrupt:
			p.timer.Stop()
			p.onInterrupt(ctx)
			if !alive(ictx) {
				return false
			}
			if d := time.Until(deadline); d > 0 {
				p.timer.Reset(d)
			} else {
				return ctx.Err() == nil
			}
		case f := <-p.out.urgent:
			p.handleUrgent(ctx, f, true)
		case a := <-p.out.acks:
			p.onAck(a)
		case <-ctx.Done():
			p.timer.Stop()
			return false
		}
	}
}

func (p *pacer) writeMsgs() {
	for _, m := range p.msgs {
		if p.sess.write(m) != nil {
			break
		}
	}
	p.msgs = p.msgs[:0]
}

func (p *pacer) writeFrame(f frames.Frame) {
	var err error
	p.msgs, err = p.ser.Serialize(p.msgs[:0], f)
	if err != nil {
		p.sess.log.Warn("serialize %s: %v", f.Name(), err)
		return
	}
	p.writeMsgs()
}

func (p *pacer) writeDTMF(ctx context.Context, buttons []frames.KeypadEntry, ictx context.Context) {
	if len(buttons) == 0 {
		return
	}
	if ds, ok := p.ser.(serializers.DTMFSerializer); ok {
		var err error
		if p.msgs, err = ds.SerializeDTMF(p.msgs[:0], buttons); err != nil {
			p.sess.log.Warn("serialize dtmf: %v", err)
			return
		}
		p.writeMsgs()
		return
	}
	if p.out.dtmfReq != nil {
		select {
		case p.out.dtmfReq <- append([]frames.KeypadEntry(nil), buttons...):
			return
		default:
			p.sess.log.Warn("dtmf sender busy, falling back to in-band tones")
		}
	}
	if p.rate == 0 {
		p.setRate(8000)
	}
	p.flushCarry(ctx)
	for _, b := range buttons {
		if len(b) == 0 {
			continue
		}
		tone, err := dtmf.Tone(b[0], p.rate)
		if err != nil {
			p.sess.log.Warn("dtmf tone %q: %v", b, err)
			continue
		}
		if !p.play(ctx, tone, "", ictx, false) {
			return
		}
		p.flushCarry(ctx)
	}
}

func (p *pacer) onSpeech(now time.Time) {
	p.ttsAudio = true
	if !p.speaking {
		p.speaking = true
		p.lastSpeakingAt = time.Time{}
		p.broadcast(func() frames.Frame { return frames.NewBotStartedSpeakingFrame() })
	}
	if now.Sub(p.lastSpeakingAt) >= botSpeakingPeriod {
		p.lastSpeakingAt = now
		p.broadcast(func() frames.Frame { return frames.NewBotSpeakingFrame() })
	}
	p.idle.Reset(p.nextSend.Sub(now) + p.params.BotStopTimeout)
	p.idleC = p.idle.C
}

func (p *pacer) broadcast(ctor func() frames.Frame) {
	_ = p.out.BroadcastFrame(context.Background(), ctor)
}

func (p *pacer) finishSpeech() {
	if !p.speaking {
		return
	}
	if p.params.PlaybackAck {
		if p.requestAck() {
			return
		}
	}
	p.botStopped()
}

func (p *pacer) requestAck() bool {
	ackSer, ok := p.ser.(serializers.PlaybackAckSerializer)
	if !ok {
		return false
	}
	p.ackSeq++
	id := "strawgo-" + p.sess.id + "-" + strconv.FormatUint(p.ackSeq, 10)
	msg, err := ackSer.SerializePlaybackAck(id)
	if err != nil || p.sess.write(msg) != nil {
		return false
	}
	p.out.markSeq.Store(p.ackSeq)
	p.awaiting = id
	p.ackTimer.Reset(p.params.PlaybackAckTimeout)
	p.ackC = p.ackTimer.C
	return true
}

func (p *pacer) onAck(a playbackAck) {
	if p.awaiting == "" {
		return
	}
	if a.drained {
		if a.markSeq != p.ackSeq {
			return
		}
	} else if a.id != p.awaiting {
		return
	}
	p.cancelAck()
	p.botStopped()
}

func (p *pacer) ackTimeout() {
	p.ackC = nil
	if p.awaiting == "" {
		return
	}
	p.awaiting = ""
	p.botStopped()
}

func (p *pacer) cancelAck() {
	p.awaiting = ""
	p.ackTimer.Stop()
	p.ackC = nil
}

func (p *pacer) botStopped() {
	if !p.speaking {
		return
	}
	p.speaking = false
	p.ttsAudio = false
	p.idle.Stop()
	p.idleC = nil
	p.broadcast(func() frames.Frame { return frames.NewBotStoppedSpeakingFrame() })
}

func (p *pacer) onInterrupt(ctx context.Context) {
	if p.fading {
		p.clearPending = true
		return
	}
	p.fadeOut(ctx)
	p.clearOutput()
}

func (p *pacer) fadeDead(ctx context.Context) {
	p.fadeOut(ctx)
	if p.clearPending {
		p.clearOutput()
	}
}

func (p *pacer) clearOutput() {
	p.clearPending = false
	p.carry = p.carry[:0]
	p.carryCtx = nil
	p.carryCodec = ""
	p.cur = nil
	p.cancelAck()
	p.nextSend = time.Time{}
	p.writeFrame(frames.NewInterruptionFrame())
	p.botStopped()
}

func (p *pacer) fadeOut(ctx context.Context) {
	if p.fading || p.fadeSamples == 0 || ctx.Err() != nil || p.out.xoff.Load() {
		return
	}
	if p.nextSend.IsZero() || time.Since(p.nextSend) > 2*p.chunkDur {
		return
	}
	codec := p.collectTail()
	if len(p.fadeBuf) == 0 {
		return
	}
	FadeOut(p.fadeBuf, codec)
	cb := p.chunkSize(codec)
	if r := len(p.fadeBuf) % cb; r != 0 {
		n := len(p.fadeBuf)
		p.fadeBuf = p.fadeBuf[:n+cb-r]
		fill := byte(0)
		if codec != "" {
			fill = audio.SilenceByte(codec)
		}
		for i := n; i < len(p.fadeBuf); i++ {
			p.fadeBuf[i] = fill
		}
	}
	speech := p.speaking
	p.fading = true
	for off := 0; off < len(p.fadeBuf); off += cb {
		if !p.writeChunk(ctx, nil, p.fadeBuf[off:off+cb], codec, speech) {
			break
		}
	}
	p.waitSlot(ctx, nil)
	p.fading = false
	p.nextSend = time.Time{}
}

func (p *pacer) collectTail() string {
	p.fadeBuf = p.fadeBuf[:0]
	codec := ""
	started := false
	add := func(data []byte, c string) {
		if !started {
			codec, started = c, true
		}
		if c != codec {
			return
		}
		room := p.fadeSamples*audio.BytesPerSample(codec) - len(p.fadeBuf)
		p.fadeBuf = append(p.fadeBuf, data[:min(room, len(data))]...)
	}
	full := func() bool {
		return started && len(p.fadeBuf) >= p.fadeSamples*audio.BytesPerSample(codec)
	}
	if len(p.carry) > 0 && !alive(p.carryCtx) {
		add(p.carry, p.carryCodec)
		p.carry = p.carry[:0]
	}
	if len(p.cur) > 0 && !alive(p.curCtx) {
		add(p.cur, p.curCodec)
	}
	p.cur = nil
	for !full() && !p.hasPending {
		select {
		case it := <-p.out.queue:
			if it.kind == itemEnd || alive(it.ctx) {
				p.pending, p.hasPending = it, true
				continue
			}
			if it.kind != itemAudio || it.rate != p.rate {
				continue
			}
			c := it.codec
			if c == audio.CodecLinear16 {
				c = ""
			}
			add(it.pcm, c)
		default:
			return codec
		}
	}
	return codec
}

func (p *pacer) drainEnd(ctx context.Context) {
	p.flushCarry(ctx)
	p.waitSlot(ctx, nil)
	if p.playedAny && p.requestAck() {
		p.waitAck(ctx)
	} else if p.params.AudioOutEndSilence > 0 && p.chunkBytes > 0 {
		for remaining := p.params.AudioOutEndSilence; remaining > 0; remaining -= p.chunkDur {
			if !p.writeChunk(ctx, nil, p.silence, "", false) {
				break
			}
		}
		p.waitSlot(ctx, nil)
	}
	p.botStopped()
}

func (p *pacer) waitAck(ctx context.Context) {
	for p.awaiting != "" {
		select {
		case a := <-p.out.acks:
			p.onAck(a)
		case <-p.ackC:
			p.ackTimeout()
		case <-ctx.Done():
			return
		}
	}
}
