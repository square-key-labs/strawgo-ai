package strawtest

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const DefaultTimeout = 5 * time.Second

type SleepFrame struct {
	*frames.SystemFrame
	Duration time.Duration
}

func Sleep(d time.Duration) *SleepFrame {
	return &SleepFrame{SystemFrame: &frames.SystemFrame{BaseFrame: frames.NewBaseFrame("SleepFrame")}, Duration: d}
}

type barrierFrame struct {
	*frames.DataFrame
}

func newBarrier() *barrierFrame {
	return &barrierFrame{DataFrame: &frames.DataFrame{BaseFrame: frames.NewUninterruptibleBaseFrame("StrawtestBarrierFrame")}}
}

type Option func(*runOptions)

type runOptions struct {
	direction frames.FrameDirection
	config    *pipeline.PipelineTaskConfig
	timeout   time.Duration
	keepStart bool
	noEnd     bool
}

func WithDirection(d frames.FrameDirection) Option {
	return func(o *runOptions) { o.direction = d }
}

func WithTaskConfig(cfg *pipeline.PipelineTaskConfig) Option {
	return func(o *runOptions) { o.config = cfg }
}

func WithTimeout(d time.Duration) Option {
	return func(o *runOptions) { o.timeout = d }
}

func WithStartFrame() Option {
	return func(o *runOptions) { o.keepStart = true }
}

func WithoutEndFrame() Option {
	return func(o *runOptions) { o.noEnd = true }
}

type frameLog struct {
	mu      sync.Mutex
	down    []frames.Frame
	up      []frames.Frame
	changed chan struct{}
}

func newFrameLog() *frameLog {
	return &frameLog{changed: make(chan struct{})}
}

func (l *frameLog) add(f frames.Frame, d frames.FrameDirection) {
	l.mu.Lock()
	if d == frames.Upstream {
		l.up = append(l.up, f)
	} else {
		l.down = append(l.down, f)
	}
	close(l.changed)
	l.changed = make(chan struct{})
	l.mu.Unlock()
}

func (l *frameLog) snapshot() (down, up []frames.Frame) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]frames.Frame(nil), l.down...), append([]frames.Frame(nil), l.up...)
}

func (l *frameLog) waitFor(f frames.Frame, d frames.FrameDirection, deadline <-chan time.Time) bool {
	for {
		l.mu.Lock()
		list := l.down
		if d == frames.Upstream {
			list = l.up
		}
		found := false
		for _, g := range list {
			if g == f {
				found = true
				break
			}
		}
		ch := l.changed
		l.mu.Unlock()
		if found {
			return true
		}
		select {
		case <-ch:
		case <-deadline:
			return false
		}
	}
}

func RunProcessors(tb testing.TB, procs []processors.FrameProcessor, send []frames.Frame, opts ...Option) (down, up []frames.Frame) {
	tb.Helper()
	o := runOptions{timeout: DefaultTimeout}
	for _, opt := range opts {
		opt(&o)
	}
	cfg := o.config
	if cfg == nil {
		cfg = pipeline.DefaultPipelineTaskConfig()
		cfg.CancelOnIdleTimeout = false
	}

	task := pipeline.NewPipelineTaskWithConfig(pipeline.NewPipeline(procs), cfg)
	log := newFrameLog()
	task.OnFrameReachedDownstream(func(f frames.Frame) { log.add(f, frames.Downstream) })
	task.OnFrameReachedUpstream(func(f frames.Frame) { log.add(f, frames.Upstream) })
	started := make(chan struct{})
	var startOnce sync.Once
	task.OnFrameReachedDownstream(func(frames.Frame) { startOnce.Do(func() { close(started) }) }, (*frames.StartFrame)(nil))

	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	timer := time.NewTimer(o.timeout)
	defer timer.Stop()
	var end frames.Frame

	stop := func() {
		task.Cancel()
		select {
		case <-done:
		case <-time.After(cfg.CancelTimeout + time.Second):
		}
	}
	fail := func(what string) {
		d, u := log.snapshot()
		stop()
		tb.Fatalf("strawtest: %s within %v\nreceived down: %s\nreceived up:   %s", what, o.timeout, FrameTypes(filterFrames(d, end, o.keepStart)), FrameTypes(filterFrames(u, end, o.keepStart)))
	}

	select {
	case <-started:
	case <-timer.C:
		fail("the StartFrame did not reach the end of the pipeline")
		return nil, nil
	}

	for _, f := range send {
		if s, ok := f.(*SleepFrame); ok {
			time.Sleep(s.Duration)
			continue
		}
		if err := task.QueueFrame(f, o.direction); err != nil {
			stop()
			tb.Fatalf("strawtest: queueing %s: %v", f.Name(), err)
			return nil, nil
		}
	}

	if len(send) > 0 {
		b := newBarrier()
		if err := task.QueueFrame(b, o.direction); err == nil && !log.waitFor(b, o.direction, timer.C) {
			fail("the sent frames did not pass through every processor (a processor must forward frames it does not handle)")
			return nil, nil
		}
	}

	if !o.noEnd {
		end = frames.NewEndFrame()
		if err := task.QueueFrame(end); err != nil {
			end = nil
		}
	}

	select {
	case err := <-done:
		if err != nil {
			tb.Errorf("strawtest: pipeline task returned %v", err)
		}
	case <-timer.C:
		fail("the pipeline did not finish")
		return nil, nil
	}

	d, u := log.snapshot()
	return filterFrames(d, end, o.keepStart), filterFrames(u, end, o.keepStart)
}

func filterFrames(list []frames.Frame, end frames.Frame, keepStart bool) []frames.Frame {
	out := make([]frames.Frame, 0, len(list))
	for _, f := range list {
		if _, ok := f.(*barrierFrame); ok {
			continue
		}
		if _, ok := f.(*frames.StartFrame); ok && !keepStart {
			continue
		}
		if end != nil && f == end {
			continue
		}
		out = append(out, f)
	}
	return out
}

func frameType(f frames.Frame) reflect.Type {
	t := reflect.TypeOf(f)
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return t
}

func typeName(f frames.Frame) string {
	if t := frameType(f); t != nil {
		return t.Name()
	}
	return "nil"
}

func FrameTypes(list []frames.Frame) string {
	names := make([]string, len(list))
	for i, f := range list {
		names[i] = typeName(f)
	}
	return "[" + strings.Join(names, ", ") + "]"
}

func AssertFrameTypes(tb testing.TB, got []frames.Frame, want ...frames.Frame) bool {
	tb.Helper()
	return assertTypes(tb, got, nil, want)
}

func AssertFrameTypesIgnoring(tb testing.TB, got []frames.Frame, ignore []frames.Frame, want ...frames.Frame) bool {
	tb.Helper()
	return assertTypes(tb, got, ignore, want)
}

func assertTypes(tb testing.TB, got, ignore, want []frames.Frame) bool {
	tb.Helper()
	skip := map[reflect.Type]bool{}
	for _, f := range ignore {
		skip[frameType(f)] = true
	}
	kept := make([]frames.Frame, 0, len(got))
	for _, f := range got {
		if !skip[frameType(f)] {
			kept = append(kept, f)
		}
	}
	at := -1
	for i := 0; i < len(kept) || i < len(want); i++ {
		if i >= len(kept) || i >= len(want) || frameType(kept[i]) != frameType(want[i]) {
			at = i
			break
		}
	}
	if at < 0 {
		return true
	}
	expected, received := "nothing", "nothing"
	if at < len(want) {
		expected = typeName(want[at])
	}
	if at < len(kept) {
		received = typeName(kept[at])
	}
	msg := fmt.Sprintf("frame types do not match at position %d: expected %s, received %s\nexpected: %s\nreceived: %s", at, expected, received, FrameTypes(want), FrameTypes(kept))
	if len(ignore) > 0 {
		msg += "\nignoring: " + FrameTypes(ignore)
	}
	tb.Errorf("%s", msg)
	return false
}
