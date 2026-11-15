package classify

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	DefaultWindow  = time.Second
	pendingWindows = 2
)

type Config struct {
	Name     string
	Window   time.Duration
	Hop      time.Duration
	OnResult func(result Result, offset, duration time.Duration)
	NoFrames bool
}

type job struct {
	audio    []byte
	rate     int
	channels int
	offset   time.Duration
}

type Processor struct {
	*processors.BaseProcessor
	classifier Classifier
	cfg        Config

	rate        int
	channels    int
	windowBytes int
	hopBytes    int
	buf         []byte
	consumed    int64

	jobs       chan job
	free       chan []byte
	workerOnce sync.Once
	skipped    atomic.Uint64
}

func NewProcessor(classifier Classifier, cfg Config) *Processor {
	if cfg.Name == "" {
		cfg.Name = "AudioClassifier"
	}
	if cfg.Window <= 0 {
		cfg.Window = DefaultWindow
	}
	if cfg.Hop <= 0 || cfg.Hop > cfg.Window {
		cfg.Hop = cfg.Window
	}
	p := &Processor{
		classifier: classifier,
		cfg:        cfg,
		jobs:       make(chan job, 1),
		free:       make(chan []byte, pendingWindows),
	}
	for i := 0; i < pendingWindows; i++ {
		p.free <- nil
	}
	p.BaseProcessor = processors.NewBaseProcessor(cfg.Name, p)
	return p
}

func (p *Processor) Skipped() uint64 {
	return p.skipped.Load()
}

func (p *Processor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		p.workerOnce.Do(func() { go p.worker(p.Context()) })
	case *frames.AudioFrame:
		if direction == frames.Downstream {
			p.handleAudio(f)
		}
	}
	return p.PushFrame(frame, direction)
}

func (p *Processor) configure(rate, channels int) {
	p.rate = rate
	p.channels = channels
	bytesPerSecond := int64(rate) * int64(channels) * 2
	align := 2 * channels
	p.windowBytes = max(int(bytesPerSecond*int64(p.cfg.Window)/int64(time.Second))/align*align, align)
	p.hopBytes = max(int(bytesPerSecond*int64(p.cfg.Hop)/int64(time.Second))/align*align, align)
	p.buf = p.buf[:0]
	p.consumed = 0
}

func (p *Processor) handleAudio(f *frames.AudioFrame) {
	channels := max(f.Channels, 1)
	if f.SampleRate <= 0 {
		return
	}
	if f.SampleRate != p.rate || channels != p.channels {
		p.configure(f.SampleRate, channels)
	}
	p.buf = append(p.buf, f.Data...)
	off := 0
	for len(p.buf)-off >= p.windowBytes {
		p.submit(p.buf[off:off+p.windowBytes], p.consumed)
		off += p.hopBytes
		p.consumed += int64(p.hopBytes)
	}
	if off > 0 {
		off = min(off, len(p.buf))
		p.buf = p.buf[:copy(p.buf, p.buf[off:])]
	}
}

func (p *Processor) submit(window []byte, consumed int64) {
	var audio []byte
	select {
	case audio = <-p.free:
	default:
		p.skipped.Add(1)
		return
	}
	audio = append(audio[:0], window...)
	offset := time.Duration(consumed/int64(2*p.channels)) * time.Second / time.Duration(p.rate)
	select {
	case p.jobs <- job{audio: audio, rate: p.rate, channels: p.channels, offset: offset}:
	default:
		p.free <- audio
		p.skipped.Add(1)
	}
}

func (p *Processor) worker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case j := <-p.jobs:
			window := Window{Audio: j.audio, SampleRate: j.rate, Channels: j.channels, Offset: j.offset}
			duration := window.Duration()
			result, err := p.classify(ctx, window)
			p.free <- j.audio
			if ctx.Err() != nil {
				return
			}
			if err != nil {
				logger.Error("[%s] classification failed: %v", p.Name(), err)
				continue
			}
			if p.cfg.OnResult != nil {
				p.cfg.OnResult(result, j.offset, duration)
			}
			if !p.cfg.NoFrames {
				if err := p.PushFrame(NewResultFrame(p.Name(), result, j.offset, duration), frames.Downstream); err != nil {
					logger.Error("[%s] push classification: %v", p.Name(), err)
				}
			}
		}
	}
}

func (p *Processor) classify(ctx context.Context, window Window) (result Result, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = &PanicError{Value: r}
		}
	}()
	return p.classifier.Classify(ctx, window)
}
