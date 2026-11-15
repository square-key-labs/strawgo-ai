package realtime

import (
	"encoding/binary"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type Pusher interface {
	Name() string
	PushFrame(frame frames.Frame, direction frames.FrameDirection) error
	StartConfig() processors.StartConfig
}

type Output struct {
	p        Pusher
	channels int

	mu           sync.Mutex
	model        string
	responding   bool
	speaking     bool
	contextID    string
	requestAt    time.Time
	ttfbPending  bool
	ttfbReported bool
}

func NewOutput(p Pusher, model string, channels int) *Output {
	if channels <= 0 {
		channels = 1
	}
	return &Output{p: p, model: model, channels: channels}
}

func (o *Output) SetModel(model string) {
	o.mu.Lock()
	o.model = model
	o.mu.Unlock()
}

func (o *Output) base() frames.MetricsBase {
	return frames.MetricsBase{Processor: o.p.Name(), Model: o.model}
}

func (o *Output) MarkRequest() {
	o.mu.Lock()
	o.requestAt = time.Now()
	o.ttfbPending = true
	o.mu.Unlock()
}

func (o *Output) BeginResponse() {
	o.mu.Lock()
	o.beginLocked()
	o.mu.Unlock()
}

func (o *Output) beginLocked() {
	if o.responding {
		return
	}
	o.responding = true
	if o.requestAt.IsZero() {
		o.requestAt = time.Now()
	}
	_ = o.p.PushFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)
}

func (o *Output) firstOutputLocked() {
	if !o.ttfbPending {
		return
	}
	o.ttfbPending = false
	cfg := o.p.StartConfig()
	if !cfg.EnableMetrics || (cfg.ReportOnlyInitialTTFB && o.ttfbReported) {
		return
	}
	o.ttfbReported = true
	_ = o.p.PushFrame(frames.NewMetricsFrame(frames.TTFBMetricsData{MetricsBase: o.base(), Value: time.Since(o.requestAt)}), frames.Downstream)
}

func (o *Output) Audio(data []byte, sampleRate int, codec string) {
	if len(data) == 0 {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.beginLocked()
	o.firstOutputLocked()
	if !o.speaking {
		o.speaking = true
		o.contextID = uuid.NewString()
		_ = o.p.PushFrame(frames.NewTTSStartedFrameWithContext(o.contextID), frames.Downstream)
	}
	f := frames.NewTTSAudioFrame(data, sampleRate, o.channels)
	f.ContextID = o.contextID
	if codec != "" {
		f.SetMetadata("codec", codec)
	}
	_ = o.p.PushFrame(f, frames.Downstream)
}

func (o *Output) Text(text string, spoken bool) {
	if text == "" {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.beginLocked()
	o.firstOutputLocked()
	_ = o.p.PushFrame(frames.NewLLMTextFrame(text), frames.Downstream)
	if spoken {
		t := frames.NewTTSTextFrame(text, frames.TextTypeWord, o.contextID)
		_ = o.p.PushFrame(t, frames.Downstream)
	}
}

func (o *Output) Speaking() bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.speaking
}

func (o *Output) Responding() bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.responding
}

func (o *Output) End(usage *llm.Usage) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.endLocked()
	cfg := o.p.StartConfig()
	var metrics []frames.MetricsData
	if cfg.EnableMetrics && !o.requestAt.IsZero() {
		metrics = append(metrics, frames.ProcessingMetricsData{MetricsBase: o.base(), Value: time.Since(o.requestAt)})
	}
	if cfg.EnableUsageMetrics && usage != nil {
		metrics = append(metrics, frames.LLMUsageMetricsData{MetricsBase: o.base(), Value: *usage})
	}
	o.requestAt = time.Time{}
	if len(metrics) > 0 {
		_ = o.p.PushFrame(frames.NewMetricsFrame(metrics...), frames.Downstream)
	}
}

func (o *Output) Interrupt() {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.endLocked()
	o.ttfbPending = false
	o.requestAt = time.Time{}
}

func (o *Output) endLocked() {
	if o.speaking {
		o.speaking = false
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = o.contextID
		_ = o.p.PushFrame(stopped, frames.Downstream)
	}
	if o.responding {
		o.responding = false
		_ = o.p.PushFrame(frames.NewLLMFullResponseEndFrame(), frames.Downstream)
	}
}

func AppendResamplePCM16(dst, src []byte, inRate, outRate int) []byte {
	n := len(src) / 2
	if n == 0 || inRate <= 0 || outRate <= 0 || inRate == outRate {
		return append(dst, src[:n*2]...)
	}
	outN := int(int64(n) * int64(outRate) / int64(inRate))
	for i := 0; i < outN; i++ {
		pos := int64(i) * int64(inRate)
		idx := int(pos / int64(outRate))
		frac := pos % int64(outRate)
		a := int64(int16(binary.LittleEndian.Uint16(src[idx*2:])))
		b := a
		if idx+1 < n {
			b = int64(int16(binary.LittleEndian.Uint16(src[(idx+1)*2:])))
		}
		v := a + (b-a)*frac/int64(outRate)
		dst = binary.LittleEndian.AppendUint16(dst, uint16(int16(v)))
	}
	return dst
}
