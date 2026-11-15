package classify

import (
	"context"
	"encoding/binary"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

func tone(samples int, amplitude float64) []byte {
	out := make([]byte, samples*2)
	for i := 0; i < samples; i++ {
		v := int16(amplitude * 32767 * math.Sin(2*math.Pi*440*float64(i)/16000))
		binary.LittleEndian.PutUint16(out[2*i:], uint16(v))
	}
	return out
}

func TestEnergyClassifierLabelsSilenceAndSound(t *testing.T) {
	c := NewEnergyClassifier()
	silence, err := c.Classify(context.Background(), Window{Audio: make([]byte, 3200), SampleRate: 16000, Channels: 1})
	if err != nil || silence.Label != LabelSilence || silence.Confidence != 1 {
		t.Fatalf("silence = %+v, %v", silence, err)
	}
	loud, err := c.Classify(context.Background(), Window{Audio: tone(1600, 0.5), SampleRate: 16000, Channels: 1})
	if err != nil || loud.Label != LabelSound || loud.Confidence < 0.9 {
		t.Fatalf("tone = %+v, %v", loud, err)
	}
	faint, _ := c.Classify(context.Background(), Window{Audio: tone(1600, 0.006), SampleRate: 16000, Channels: 1})
	if faint.Confidence >= loud.Confidence {
		t.Fatalf("near-threshold confidence %.2f should be below loud confidence %.2f", faint.Confidence, loud.Confidence)
	}
}

type collector struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (c *collector) ProcessFrame(context.Context, frames.Frame, frames.FrameDirection) error {
	return nil
}
func (c *collector) QueueFrame(frame frames.Frame, _ frames.FrameDirection) error {
	c.mu.Lock()
	c.frames = append(c.frames, frame)
	c.mu.Unlock()
	return nil
}
func (c *collector) PushFrame(frames.Frame, frames.FrameDirection) error { return nil }
func (c *collector) Link(processors.FrameProcessor)                      {}
func (c *collector) SetPrev(processors.FrameProcessor)                   {}
func (c *collector) Start(context.Context) error                         { return nil }
func (c *collector) Stop() error                                         { return nil }
func (c *collector) Name() string                                        { return "collector" }

func (c *collector) results() []*ResultFrame {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []*ResultFrame
	for _, f := range c.frames {
		if r, ok := f.(*ResultFrame); ok {
			out = append(out, r)
		}
	}
	return out
}

func (c *collector) audioFrames() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for _, f := range c.frames {
		if _, ok := f.(*frames.AudioFrame); ok {
			n++
		}
	}
	return n
}

func TestProcessorClassifiesWindowsAndPassesAudioThrough(t *testing.T) {
	var mu sync.Mutex
	var callbacks []Result
	p := NewProcessor(NewEnergyClassifier(), Config{
		Window: 500 * time.Millisecond,
		OnResult: func(result Result, _, _ time.Duration) {
			mu.Lock()
			callbacks = append(callbacks, result)
			mu.Unlock()
		},
	})
	sink := &collector{}
	p.Link(sink)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := p.Start(ctx); err != nil {
		t.Fatal(err)
	}
	defer p.Stop()
	if err := p.QueueFrame(frames.NewStartFrame(), frames.Downstream); err != nil {
		t.Fatal(err)
	}

	feed := func(data []byte, wantResults int) {
		for off := 0; off < len(data); off += 640 {
			if err := p.QueueFrame(frames.NewAudioFrame(data[off:off+640], 16000, 1), frames.Downstream); err != nil {
				t.Fatal(err)
			}
		}
		deadline := time.Now().Add(5 * time.Second)
		for len(sink.results()) < wantResults {
			if time.Now().After(deadline) {
				t.Fatalf("got %d results, want %d", len(sink.results()), wantResults)
			}
			time.Sleep(time.Millisecond)
		}
	}
	feed(make([]byte, 16000), 1)
	feed(tone(8000, 0.5), 2)

	results := sink.results()
	if results[0].Result.Label != LabelSilence || results[1].Result.Label != LabelSound {
		t.Fatalf("labels = %q, %q", results[0].Result.Label, results[1].Result.Label)
	}
	if results[1].Offset != 500*time.Millisecond || results[1].Duration != 500*time.Millisecond {
		t.Fatalf("second window offset=%v duration=%v", results[1].Offset, results[1].Duration)
	}
	if results[1].Classifier != "AudioClassifier" {
		t.Fatalf("classifier name = %q", results[1].Classifier)
	}
	if n := sink.audioFrames(); n != 50 {
		t.Fatalf("passed through %d audio frames, want 50", n)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(callbacks) != 2 {
		t.Fatalf("callback ran %d times, want 2", len(callbacks))
	}
}

func TestProcessorSkipsWindowsInsteadOfBlockingAudio(t *testing.T) {
	release := make(chan struct{})
	slow := ClassifierFunc(func(ctx context.Context, _ Window) (Result, error) {
		select {
		case <-release:
		case <-ctx.Done():
		}
		return Result{Label: "slow", Confidence: 1}, nil
	})
	p := NewProcessor(slow, Config{Window: 20 * time.Millisecond})
	sink := &collector{}
	p.Link(sink)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := p.Start(ctx); err != nil {
		t.Fatal(err)
	}
	defer p.Stop()
	_ = p.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	for i := 0; i < 50; i++ {
		_ = p.QueueFrame(frames.NewAudioFrame(make([]byte, 640), 16000, 1), frames.Downstream)
	}
	deadline := time.Now().Add(5 * time.Second)
	for sink.audioFrames() < 50 {
		if time.Now().After(deadline) {
			t.Fatalf("audio blocked behind a slow classifier: %d of 50 frames passed", sink.audioFrames())
		}
		time.Sleep(time.Millisecond)
	}
	if p.Skipped() < 47 {
		t.Fatalf("skipped %d windows, want at least 47", p.Skipped())
	}
	close(release)
}
