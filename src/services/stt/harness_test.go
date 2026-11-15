package stt

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const guard = 3 * time.Second

type collector struct {
	name   string
	mu     sync.Mutex
	frames []frames.Frame
	notify chan struct{}
}

func newCollector(name string) *collector {
	return &collector{name: name, notify: make(chan struct{}, 1)}
}

func (c *collector) QueueFrame(f frames.Frame, _ frames.FrameDirection) error {
	c.mu.Lock()
	c.frames = append(c.frames, f)
	c.mu.Unlock()
	select {
	case c.notify <- struct{}{}:
	default:
	}
	return nil
}

func (c *collector) ProcessFrame(context.Context, frames.Frame, frames.FrameDirection) error {
	return nil
}
func (c *collector) PushFrame(frames.Frame, frames.FrameDirection) error { return nil }
func (c *collector) Link(processors.FrameProcessor)                      {}
func (c *collector) SetPrev(processors.FrameProcessor)                   {}
func (c *collector) Start(context.Context) error                         { return nil }
func (c *collector) Stop() error                                         { return nil }
func (c *collector) Name() string                                        { return c.name }

func (c *collector) snapshot() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]frames.Frame(nil), c.frames...)
}

func (c *collector) waitFor(t *testing.T, what string, pred func([]frames.Frame) bool) []frames.Frame {
	t.Helper()
	timer := time.NewTimer(guard)
	defer timer.Stop()
	for {
		fs := c.snapshot()
		if pred(fs) {
			return fs
		}
		select {
		case <-c.notify:
		case <-timer.C:
			t.Fatalf("%s: timed out waiting for %s; got %s", c.name, what, names(fs))
		}
	}
}

func names(fs []frames.Frame) []string {
	out := make([]string, len(fs))
	for i, f := range fs {
		out[i] = f.Name()
	}
	return out
}

func ofType[T frames.Frame](fs []frames.Frame) []T {
	var out []T
	for _, f := range fs {
		if v, ok := f.(T); ok {
			out = append(out, v)
		}
	}
	return out
}

func atLeast[T frames.Frame](n int) func([]frames.Frame) bool {
	return func(fs []frames.Frame) bool { return len(ofType[T](fs)) >= n }
}

func metrics[T frames.MetricsData](fs []frames.Frame) []T {
	var out []T
	for _, m := range ofType[*frames.MetricsFrame](fs) {
		for _, d := range m.Data {
			if v, ok := d.(T); ok {
				out = append(out, v)
			}
		}
	}
	return out
}

type rig struct {
	b    *Base
	up   *collector
	down *collector
}

func newRig(t *testing.T, b *Base, start *frames.StartFrame) *rig {
	t.Helper()
	r := &rig{b: b, up: newCollector("up"), down: newCollector("down")}
	b.SetPrev(r.up)
	b.Link(r.down)
	if err := b.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = b.Stop() })
	if start == nil {
		start = frames.NewStartFrame()
	}
	r.send(start)
	r.down.waitFor(t, "StartFrame", atLeast[*frames.StartFrame](1))
	return r
}

func (r *rig) send(f frames.Frame) {
	_ = r.b.QueueFrame(f, frames.Downstream)
}

func (r *rig) systemBarrier(t *testing.T) {
	t.Helper()
	m := frames.NewMetricsFrame()
	r.send(m)
	r.down.waitFor(t, "system barrier", func(fs []frames.Frame) bool {
		for _, f := range fs {
			if f == frames.Frame(m) {
				return true
			}
		}
		return false
	})
}

func (r *rig) dataBarrier(t *testing.T) {
	t.Helper()
	m := frames.NewTextFrame("barrier")
	r.send(m)
	r.down.waitFor(t, "data barrier", func(fs []frames.Frame) bool {
		for _, f := range fs {
			if f == frames.Frame(m) {
				return true
			}
		}
		return false
	})
}

func pcm(n int, v byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = v
	}
	return b
}
