package transports

import (
	"context"
	"encoding/binary"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

type OutputProcessor struct {
	*processors.BaseProcessor
	session *Session

	queue     chan outItem
	urgent    chan frames.Frame
	interrupt chan struct{}
	acks      chan playbackAck
	markSeq   atomic.Uint64
	flowWake  chan struct{}
	xoff      atomic.Bool
	dtmfReq   chan []frames.KeypadEntry

	senderCancel context.CancelFunc
	senderDone   chan struct{}

	outRate int
	rs      audio.Resampler
	decBuf  []byte
	lastCtx context.Context

	finishOnce sync.Once
}

func newOutputProcessor(s *Session) *OutputProcessor {
	p := &OutputProcessor{
		session:   s,
		queue:     make(chan outItem, outputQueueSize),
		urgent:    make(chan frames.Frame, 16),
		interrupt: make(chan struct{}, 1),
		acks:      make(chan playbackAck, 4),
		flowWake:  make(chan struct{}, 1),
	}
	p.BaseProcessor = processors.NewBaseProcessor("WebSocketOutput", p, processors.WithConcurrentSystemFrames())
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
		p.onStart(f)
		return err
	case *frames.CancelFrame:
		p.cancelSender()
		p.finish(frame)
		return p.PushFrame(frame, direction)
	case *frames.EndFrame:
		p.finish(frame)
		return p.PushFrame(frame, direction)
	case *frames.OutputTransportMessageUrgentFrame, *frames.OutputDTMFUrgentFrame:
		p.sendUrgent(frame)
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
		if f, ok := frame.(*frames.TransferCallFrame); ok {
			go p.transfer(f)
		}
		return p.PushFrame(frame, direction)
	}
	if frame.SurvivesInterruption() {
		p.enqueue(p.Context(), outItem{kind: itemFrame, frame: frame})
		return nil
	}
	p.enqueue(ctx, outItem{kind: itemFrame, frame: frame, ctx: ctx})
	return nil
}

func (p *OutputProcessor) dropItem(it outItem) {
	if f, ok := it.frame.(*frames.TransferCallFrame); ok {
		p.dropTransfer(f)
	}
}

func (p *OutputProcessor) dropTransfer(f *frames.TransferCallFrame) {
	if f.Claim() {
		f.Complete(serializers.ErrCallEnded)
		return
	}
	f.Complete(serializers.ErrTransferCancelled)
}

func (p *OutputProcessor) drainQueue() {
	for {
		select {
		case it := <-p.queue:
			p.dropItem(it)
		default:
			return
		}
	}
}

func (p *OutputProcessor) onStart(f *frames.StartFrame) {
	if p.senderDone != nil {
		return
	}
	rate := 0
	if wf, ok := p.session.serializer.(serializers.WireFormat); ok {
		_, rate = wf.WireSampleRates()
	}
	if rate == 0 {
		rate = p.session.params.AudioOutSampleRate
	}
	if rate == 0 {
		rate = f.AudioOutSampleRate
	}
	p.outRate = rate
	ctx, cancel := context.WithCancel(p.Context())
	pc := newPacer(p)
	p.senderCancel = cancel
	p.senderDone = pc.done
	if ds, ok := p.session.serializer.(serializers.DTMFSender); ok && ds.CanSendDTMF() {
		p.dtmfReq = make(chan []frames.KeypadEntry, 8)
		go p.dtmfWorker(ctx, ds)
	}
	go pc.run(ctx)
}

func (p *OutputProcessor) dtmfWorker(ctx context.Context, ds serializers.DTMFSender) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-p.senderDone:
			return
		case buttons := <-p.dtmfReq:
			sctx, cancel := context.WithTimeout(ctx, p.session.params.HangupTimeout)
			if err := ds.SendDTMF(sctx, buttons); err != nil {
				p.session.log.Warn("send dtmf: %v", err)
			}
			cancel()
		}
	}
}

func (p *OutputProcessor) transfer(f *frames.TransferCallFrame) {
	if !f.Claim() {
		f.Complete(serializers.ErrTransferCancelled)
		return
	}
	err := serializers.ErrTransferUnsupported
	if t, ok := p.session.serializer.(serializers.Transferer); ok && t.CanTransfer() {
		ctx, cancel := context.WithTimeout(context.WithoutCancel(p.Context()), p.session.params.HangupTimeout)
		err = t.Transfer(ctx, serializers.TransferRequest{Destination: f.Destination, CallerID: f.CallerID, Announcement: f.Announcement})
		cancel()
	}
	err = logger.RedactError(err)
	if err != nil {
		p.session.log.Warn("transfer: %v", err)
	} else {
		p.session.log.Info("call transfer accepted by the carrier")
	}
	f.Complete(err)
	_ = p.PushFrame(frames.NewTransferCallResultFrame(f.Destination, err), frames.Upstream)
	if err == nil {
		_ = p.PushFrame(frames.NewEndTaskFrame("call transferred"), frames.Upstream)
	}
}

func (p *OutputProcessor) setFlow(paused bool) {
	p.xoff.Store(paused)
	select {
	case p.flowWake <- struct{}{}:
	default:
	}
}

func (p *OutputProcessor) cancelSender() {
	if p.senderCancel != nil {
		p.senderCancel()
	}
}

func (p *OutputProcessor) enqueue(ctx context.Context, it outItem) {
	select {
	case p.queue <- it:
		select {
		case <-p.senderDone:
			p.drainQueue()
		default:
		}
	case <-ctx.Done():
		p.dropItem(it)
	case <-p.senderDone:
		p.dropItem(it)
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

type playbackAck struct {
	id      string
	drained bool
	markSeq uint64
}

func (p *OutputProcessor) ack(correlationID string, drained bool) {
	select {
	case p.acks <- playbackAck{id: correlationID, drained: drained, markSeq: p.markSeq.Load()}:
	default:
	}
}

func (p *OutputProcessor) trackEpoch(ctx context.Context) {
	if p.lastCtx != ctx {
		if p.lastCtx != nil && p.lastCtx.Err() != nil {
			p.rs.Reset()
		}
		p.lastCtx = ctx
	}
}

func (p *OutputProcessor) flushResampler(ctx context.Context) {
	p.trackEpoch(ctx)
	if ctx.Err() != nil {
		p.rs.Reset()
		return
	}
	tail := p.rs.Flush(nil)
	if len(tail) == 0 || p.senderDone == nil {
		return
	}
	p.enqueue(ctx, outItem{kind: itemAudio, pcm: tail, rate: p.outRate, ctx: ctx, speech: true})
}

func (p *OutputProcessor) handleAudio(ctx context.Context, f *frames.TTSAudioFrame) error {
	p.trackEpoch(ctx)
	if ctx.Err() != nil || p.senderDone == nil || len(f.Data) == 0 {
		return nil
	}
	if p.outRate == 0 {
		p.outRate = f.SampleRate
	}
	if codec, ok := p.nativeCodec(f); ok {
		p.enqueue(ctx, outItem{kind: itemAudio, pcm: f.Data, rate: p.outRate, codec: codec, ctx: ctx, frame: f, speech: true})
		return nil
	}
	pcm, err := p.toPCM(f)
	if err != nil {
		return p.PushError("output audio conversion failed", err, false)
	}
	p.enqueue(ctx, outItem{kind: itemAudio, pcm: pcm, rate: p.outRate, ctx: ctx, frame: f, speech: true})
	return nil
}

func (p *OutputProcessor) nativeCodec(f *frames.TTSAudioFrame) (string, bool) {
	if !p.session.params.AudioOutNative || p.session.params.outputMixer() != nil || f.Channels > 1 {
		return "", false
	}
	ns, ok := p.session.serializer.(serializers.NativeAudioSerializer)
	if !ok {
		return "", false
	}
	codec := frameCodec(f)
	wireCodec, wireRate := ns.NativeAudioFormat()
	if codec == audio.CodecLinear16 || codec != audio.NormalizeCodec(wireCodec) || wireRate != p.outRate || (f.SampleRate > 0 && f.SampleRate != wireRate) {
		return "", false
	}
	return codec, true
}

func (p *OutputProcessor) toPCM(f *frames.TTSAudioFrame) ([]byte, error) {
	codec := frameCodec(f)
	rate := f.SampleRate
	if rate <= 0 {
		rate = p.outRate
	}
	if codec == audio.CodecLinear16 && f.Channels <= 1 && rate == p.outRate {
		return f.Data, nil
	}
	var pcm []byte
	var err error
	if rate != p.outRate {
		p.decBuf, err = audio.DecodePCM16(p.decBuf[:0], f.Data, codec)
		pcm = p.decBuf
	} else {
		pcm, err = audio.DecodePCM16(nil, f.Data, codec)
	}
	if err != nil {
		return nil, err
	}
	if f.Channels > 1 {
		pcm = downmix(pcm, f.Channels)
	}
	if rate != p.outRate {
		pcm = p.rs.Resample(make([]byte, 0, p.rs.MaxOutputBytes(len(pcm), rate, p.outRate)), pcm, rate, p.outRate)
	}
	return pcm, nil
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
		s := p.session
		if msgs, err := s.serializer.Serialize(nil, frame); err == nil {
			for _, m := range msgs {
				if s.write(m) != nil {
					break
				}
			}
		}
		if h, ok := s.serializer.(serializers.Hangupper); ok {
			ctx, cancel := context.WithTimeout(context.WithoutCancel(p.Context()), s.params.HangupTimeout)
			if err := h.Hangup(ctx); err != nil {
				s.log.Warn("hangup: %v", err)
			}
			cancel()
		}
		_ = s.Close()
	})
}
