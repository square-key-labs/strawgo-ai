package processors

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

const engineWait = 2 * time.Second

type sinkEvent struct {
	frame     frames.Frame
	direction frames.FrameDirection
}

type chanSink struct {
	name string
	ch   chan sinkEvent
}

func newChanSink(name string) *chanSink {
	return &chanSink{name: name, ch: make(chan sinkEvent, 4096)}
}

func (s *chanSink) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}

func (s *chanSink) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	s.ch <- sinkEvent{frame: frame, direction: direction}
	return nil
}

func (s *chanSink) PushFrame(frame frames.Frame, direction frames.FrameDirection) error { return nil }
func (s *chanSink) Link(next FrameProcessor)                                            {}
func (s *chanSink) SetPrev(prev FrameProcessor)                                         {}
func (s *chanSink) Start(ctx context.Context) error                                     { return nil }
func (s *chanSink) Stop() error                                                         { return nil }
func (s *chanSink) Name() string                                                        { return s.name }

func (s *chanSink) next(t *testing.T) frames.Frame {
	t.Helper()
	select {
	case ev := <-s.ch:
		return ev.frame
	case <-time.After(engineWait):
		t.Fatalf("%s: timed out waiting for frame", s.name)
		return nil
	}
}

func (s *chanSink) collect(t *testing.T, n int) []frames.Frame {
	t.Helper()
	out := make([]frames.Frame, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, s.next(t))
	}
	return out
}

func (s *chanSink) waitFor(t *testing.T, match func(frames.Frame) bool) frames.Frame {
	t.Helper()
	deadline := time.After(engineWait)
	for {
		select {
		case ev := <-s.ch:
			if match(ev.frame) {
				return ev.frame
			}
		case <-deadline:
			t.Fatalf("%s: timed out waiting for matching frame", s.name)
			return nil
		}
	}
}

func (s *chanSink) assertEmpty(t *testing.T, window time.Duration) {
	t.Helper()
	select {
	case ev := <-s.ch:
		t.Fatalf("%s: unexpected frame %s", s.name, ev.frame.Name())
	case <-time.After(window):
	}
}

type handlerFunc func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error

func (h handlerFunc) HandleFrame(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
	return h(ctx, f, d)
}

type harness struct {
	p    *BaseProcessor
	down *chanSink
	up   *chanSink
}

func newHarness(t *testing.T, h func(p *BaseProcessor) handlerFunc, opts ...Option) *harness {
	t.Helper()
	hs := &harness{down: newChanSink("down"), up: newChanSink("up")}
	var proc *BaseProcessor
	var handler ProcessHandler
	if h != nil {
		handler = handlerFunc(func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
			return h(proc)(ctx, f, d)
		})
	}
	proc = NewBaseProcessor(t.Name(), handler, opts...)
	proc.Link(hs.down)
	proc.SetPrev(hs.up)
	hs.p = proc
	if err := proc.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() { _ = proc.Stop() })
	return hs
}

func forward(p *BaseProcessor) handlerFunc {
	return func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
		return p.PushFrame(f, d)
	}
}

func textOf(f frames.Frame) string {
	if tf, ok := f.(*frames.TextFrame); ok {
		return tf.Text
	}
	return f.Name()
}

func isText(text string) func(frames.Frame) bool {
	return func(f frames.Frame) bool {
		tf, ok := f.(*frames.TextFrame)
		return ok && tf.Text == text
	}
}

func TestEngineOrderingPerDirection(t *testing.T) {
	hs := newHarness(t, nil)
	const n = 200
	for i := 0; i < n; i++ {
		if err := hs.p.QueueFrame(frames.NewTextFrame(fmt.Sprintf("d%d", i)), frames.Downstream); err != nil {
			t.Fatal(err)
		}
		if err := hs.p.QueueFrame(frames.NewTextFrame(fmt.Sprintf("u%d", i)), frames.Upstream); err != nil {
			t.Fatal(err)
		}
	}
	for i, f := range hs.down.collect(t, n) {
		if got, want := textOf(f), fmt.Sprintf("d%d", i); got != want {
			t.Fatalf("downstream[%d] = %s, want %s", i, got, want)
		}
	}
	for i, f := range hs.up.collect(t, n) {
		if got, want := textOf(f), fmt.Sprintf("u%d", i); got != want {
			t.Fatalf("upstream[%d] = %s, want %s", i, got, want)
		}
	}
}

func TestEngineSystemFramesKeepOrderAmongThemselves(t *testing.T) {
	hs := newHarness(t, nil)
	var ids []uint64
	for i := 0; i < 50; i++ {
		f := frames.NewUserStartedSpeakingFrame()
		ids = append(ids, f.ID())
		if err := hs.p.QueueFrame(f, frames.Downstream); err != nil {
			t.Fatal(err)
		}
	}
	for i, f := range hs.down.collect(t, len(ids)) {
		if f.ID() != ids[i] {
			t.Fatalf("system frame %d out of order", i)
		}
	}
}

func TestEngineSystemFrameOvertakesQueuedData(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{})
	var once sync.Once
	hs := newHarness(t, func(p *BaseProcessor) handlerFunc {
		return func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
			if tf, ok := f.(*frames.TextFrame); ok && tf.Text == "block" {
				once.Do(func() { close(entered) })
				<-release
			}
			return p.PushFrame(f, d)
		}
	})
	if err := hs.p.QueueFrame(frames.NewTextFrame("block"), frames.Downstream); err != nil {
		t.Fatal(err)
	}
	<-entered
	const n = 300
	for i := 0; i < n; i++ {
		_ = hs.p.QueueFrame(frames.NewTextFrame(fmt.Sprintf("t%d", i)), frames.Downstream)
	}
	sys := frames.NewUserStartedSpeakingFrame()
	_ = hs.p.QueueFrame(sys, frames.Downstream)
	waitCond(t, func() bool { s, _ := hs.p.QueueLen(); return s == 0 })
	close(release)

	got := hs.down.collect(t, n+2)
	pos := -1
	for i, f := range got {
		if f.ID() == sys.ID() {
			pos = i
		}
	}
	if pos < 0 {
		t.Fatal("system frame not delivered")
	}
	if pos >= n {
		t.Fatalf("system frame delivered at %d, after nearly all %d queued data frames", pos, n)
	}
	var texts []string
	for _, f := range got {
		if tf, ok := f.(*frames.TextFrame); ok {
			texts = append(texts, tf.Text)
		}
	}
	if texts[0] != "block" {
		t.Fatalf("first data frame = %s", texts[0])
	}
	for i := 0; i < n; i++ {
		if texts[i+1] != fmt.Sprintf("t%d", i) {
			t.Fatalf("data order broken at %d: %s", i, texts[i+1])
		}
	}
}

func waitCond(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(engineWait)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("condition not met")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestEngineInterruptionDropsInterruptibleKeepsSurvivors(t *testing.T) {
	entered := make(chan struct{})
	blockedErr := make(chan error, 1)
	freshErr := make(chan error, 1)
	var blockedCtx atomic.Value
	hs := newHarness(t, func(p *BaseProcessor) handlerFunc {
		return func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
			if tf, ok := f.(*frames.TextFrame); ok {
				switch tf.Text {
				case "block":
					blockedCtx.Store(ctx)
					close(entered)
					<-ctx.Done()
					blockedErr <- ctx.Err()
					return ctx.Err()
				case "after":
					if old, _ := blockedCtx.Load().(context.Context); old == ctx {
						freshErr <- errors.New("same ctx reused")
					} else {
						freshErr <- ctx.Err()
					}
				}
			}
			return p.PushFrame(f, d)
		}
	})

	_ = hs.p.QueueFrame(frames.NewTextFrame("block"), frames.Downstream)
	<-entered

	audio := frames.NewTTSAudioFrame([]byte{1, 2}, 16000, 1)
	audio.SetSurvivesInterruption(true)
	dropAudio := frames.NewTTSAudioFrame([]byte{3}, 16000, 1)
	result := frames.NewFunctionCallResultFrame("call-1", "fn", "ok", nil)
	end := frames.NewEndFrame()
	queued := []frames.Frame{
		frames.NewTextFrame("drop1"),
		audio,
		dropAudio,
		frames.NewTextFrame("drop2"),
		result,
		frames.NewTextFrame("drop3"),
		end,
	}
	for _, f := range queued {
		_ = hs.p.QueueFrame(f, frames.Downstream)
	}
	intr := frames.NewInterruptionFrame()
	_ = hs.p.QueueFrame(intr, frames.Downstream)

	select {
	case err := <-blockedErr:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("blocked handler ctx err = %v", err)
		}
	case <-time.After(engineWait):
		t.Fatal("handler ctx not cancelled by interruption")
	}

	_ = hs.p.QueueFrame(frames.NewTextFrame("after"), frames.Downstream)
	select {
	case err := <-freshErr:
		if err != nil {
			t.Fatalf("post-interruption ctx: %v", err)
		}
	case <-time.After(engineWait):
		t.Fatal("frame after interruption not handled")
	}

	got := hs.down.collect(t, 5)
	var data []frames.Frame
	sawIntr := false
	for _, f := range got {
		if f.ID() == intr.ID() {
			sawIntr = true
			continue
		}
		data = append(data, f)
	}
	if !sawIntr {
		t.Fatal("interruption frame not forwarded")
	}
	want := []uint64{audio.ID(), result.ID(), end.ID()}
	if len(data) != 4 {
		t.Fatalf("got %d data frames, want 4", len(data))
	}
	for i, id := range want {
		if data[i].ID() != id {
			t.Fatalf("survivor %d = %s", i, data[i].Name())
		}
	}
	if textOf(data[3]) != "after" {
		t.Fatalf("last frame = %s", textOf(data[3]))
	}
	hs.down.assertEmpty(t, 30*time.Millisecond)
}

func TestEngineStartFrameAppliedOnceAndGatesData(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{}, 2)
	hs := newHarness(t, func(p *BaseProcessor) handlerFunc {
		return func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
			if _, ok := f.(*frames.StartFrame); ok {
				entered <- struct{}{}
				<-release
			}
			return p.PushFrame(f, d)
		}
	}, WithConcurrentSystemFrames())

	first := frames.NewStartFrame()
	first.AllowInterruptions = true
	first.AudioInSampleRate = 16000
	first.AudioOutSampleRate = 24000
	first.EnableMetrics = true
	first.ReportOnlyInitialTTFB = true
	_ = hs.p.QueueFrame(first, frames.Downstream)
	_ = hs.p.QueueFrame(frames.NewTextFrame("gated"), frames.Downstream)
	<-entered

	hs.down.assertEmpty(t, 30*time.Millisecond)
	if _, dataLen := hs.p.QueueLen(); dataLen != 1 {
		t.Fatalf("data queue length while StartFrame pending = %d, want 1", dataLen)
	}
	close(release)

	if f := hs.down.next(t); f.ID() != first.ID() {
		t.Fatalf("first delivered = %s, want StartFrame", f.Name())
	}
	if f := hs.down.next(t); textOf(f) != "gated" {
		t.Fatalf("second delivered = %s, want gated text", textOf(f))
	}

	cfg := hs.p.StartConfig()
	if !cfg.AllowInterruptions || cfg.AudioInSampleRate != 16000 || cfg.AudioOutSampleRate != 24000 || !cfg.EnableMetrics || !cfg.ReportOnlyInitialTTFB || cfg.EnableUsageMetrics {
		t.Fatalf("start config not applied: %+v", cfg)
	}

	second := frames.NewStartFrame()
	second.AudioInSampleRate = 8000
	_ = hs.p.QueueFrame(second, frames.Downstream)
	<-entered
	hs.down.waitFor(t, func(f frames.Frame) bool { return f.ID() == second.ID() })
	if cfg2 := hs.p.StartConfig(); cfg2.AudioInSampleRate != 16000 || !cfg2.AllowInterruptions {
		t.Fatalf("second StartFrame changed config: %+v", cfg2)
	}
	if !hs.p.InterruptionsAllowed() {
		t.Fatal("InterruptionsAllowed false")
	}
}

func TestEngineHandlerPanicBecomesErrorFrame(t *testing.T) {
	cases := []struct {
		name  string
		frame func() frames.Frame
	}{
		{"data", func() frames.Frame { return frames.NewTextFrame("boom") }},
		{"system", func() frames.Frame { return frames.NewUserStartedSpeakingFrame() }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hs := newHarness(t, func(p *BaseProcessor) handlerFunc {
				return func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
					if tf, ok := f.(*frames.TextFrame); ok && tf.Text == "boom" {
						panic("kaboom")
					}
					if _, ok := f.(*frames.UserStartedSpeakingFrame); ok {
						panic("kaboom")
					}
					return p.PushFrame(f, d)
				}
			})
			_ = hs.p.QueueFrame(tc.frame(), frames.Downstream)
			ef, ok := hs.up.next(t).(*frames.ErrorFrame)
			if !ok {
				t.Fatal("expected ErrorFrame upstream")
			}
			if ef.Fatal {
				t.Fatal("panic error should not be fatal")
			}
			if ef.Processor != hs.p.Name() || ef.Error == nil {
				t.Fatalf("bad error frame: %+v", ef)
			}
			_ = hs.p.QueueFrame(frames.NewTextFrame("ok"), frames.Downstream)
			_ = hs.p.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
			hs.down.waitFor(t, isText("ok"))
		})
	}
}

func TestEngineDirectMode(t *testing.T) {
	var gid atomic.Int64
	hs := newHarness(t, func(p *BaseProcessor) handlerFunc {
		return func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
			gid.Add(1)
			return p.PushFrame(f, d)
		}
	}, WithDirectMode())
	for i := 0; i < 10; i++ {
		f := frames.NewTextFrame(fmt.Sprintf("x%d", i))
		if err := hs.p.QueueFrame(f, frames.Downstream); err != nil {
			t.Fatal(err)
		}
		select {
		case ev := <-hs.down.ch:
			if ev.frame.ID() != f.ID() {
				t.Fatal("direct mode delivered wrong frame")
			}
		default:
			t.Fatal("direct mode did not deliver synchronously")
		}
	}
	sys := frames.NewUserStartedSpeakingFrame()
	_ = hs.p.QueueFrame(sys, frames.Upstream)
	select {
	case ev := <-hs.up.ch:
		if ev.frame.ID() != sys.ID() {
			t.Fatal("wrong upstream frame")
		}
	default:
		t.Fatal("direct mode system frame not synchronous")
	}
	if gid.Load() != 11 {
		t.Fatalf("handler calls = %d", gid.Load())
	}
	if s, d := hs.p.QueueLen(); s != 0 || d != 0 {
		t.Fatalf("direct mode queued frames: %d %d", s, d)
	}
}

func TestEngineConcurrentSystemFrames(t *testing.T) {
	sysDone := make(chan struct{})
	entered := make(chan struct{})
	hs := newHarness(t, func(p *BaseProcessor) handlerFunc {
		return func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
			switch f.(type) {
			case *frames.TextFrame:
				close(entered)
				select {
				case <-sysDone:
				case <-time.After(engineWait):
					return errors.New("system frame did not run concurrently")
				}
			case *frames.UserStartedSpeakingFrame:
				defer close(sysDone)
			}
			return p.PushFrame(f, d)
		}
	}, WithConcurrentSystemFrames())
	_ = hs.p.QueueFrame(frames.NewTextFrame("slow"), frames.Downstream)
	<-entered
	sys := frames.NewUserStartedSpeakingFrame()
	_ = hs.p.QueueFrame(sys, frames.Downstream)
	if f := hs.down.next(t); f.ID() != sys.ID() {
		t.Fatalf("first delivered = %s, want system frame", f.Name())
	}
	if f := hs.down.next(t); textOf(f) != "slow" {
		t.Fatalf("second delivered = %s", textOf(f))
	}
}

func TestEngineSerializedHandlersByDefault(t *testing.T) {
	var active, maxActive atomic.Int32
	hs := newHarness(t, func(p *BaseProcessor) handlerFunc {
		return func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
			n := active.Add(1)
			for {
				m := maxActive.Load()
				if n <= m || maxActive.CompareAndSwap(m, n) {
					break
				}
			}
			runtime.Gosched()
			active.Add(-1)
			return p.PushFrame(f, d)
		}
	})
	for i := 0; i < 200; i++ {
		_ = hs.p.QueueFrame(frames.NewTextFrame("d"), frames.Downstream)
		_ = hs.p.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	}
	hs.down.collect(t, 400)
	if maxActive.Load() != 1 {
		t.Fatalf("handlers overlapped: max concurrency %d", maxActive.Load())
	}
}

func TestEngineStopWhileHandlerBlocksOnCtx(t *testing.T) {
	entered := make(chan struct{})
	hs := newHarness(t, func(p *BaseProcessor) handlerFunc {
		return func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
			close(entered)
			<-ctx.Done()
			return ctx.Err()
		}
	})
	_ = hs.p.QueueFrame(frames.NewTextFrame("block"), frames.Downstream)
	<-entered
	done := make(chan error, 1)
	go func() { done <- hs.p.Stop() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("stop: %v", err)
		}
	case <-time.After(engineWait):
		t.Fatal("Stop did not return while handler blocked on ctx")
	}
}

func TestEngineQueueAfterStopAndDoubleStop(t *testing.T) {
	var calls atomic.Int32
	hs := newHarness(t, func(p *BaseProcessor) handlerFunc {
		return func(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
			calls.Add(1)
			return p.PushFrame(f, d)
		}
	})
	if err := hs.p.Stop(); err != nil {
		t.Fatal(err)
	}
	if err := hs.p.Stop(); err != nil {
		t.Fatalf("second stop: %v", err)
	}
	for _, f := range []frames.Frame{frames.NewTextFrame("late"), frames.NewUserStartedSpeakingFrame(), frames.NewEndFrame()} {
		if err := hs.p.QueueFrame(f, frames.Downstream); err == nil {
			t.Fatalf("QueueFrame(%s) after Stop returned nil", f.Name())
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("handler called %d times after Stop", calls.Load())
	}
	hs.down.assertEmpty(t, 20*time.Millisecond)
}

func TestEngineStopBeforeStartAndStartTwice(t *testing.T) {
	p := NewBaseProcessor("idle", nil)
	if err := p.Stop(); err != nil {
		t.Fatalf("stop before start: %v", err)
	}
	q := NewBaseProcessor("twice", nil)
	if err := q.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer q.Stop()
	if err := q.Start(context.Background()); err == nil {
		t.Fatal("second Start should fail")
	}
}

func TestEngineQueueNilFrame(t *testing.T) {
	hs := newHarness(t, nil)
	if err := hs.p.QueueFrame(nil, frames.Downstream); err == nil {
		t.Fatal("nil frame accepted")
	}
}

func TestEngineEndAndCancelForwardedByPassthrough(t *testing.T) {
	hs := newHarness(t, nil)
	end := frames.NewEndFrame()
	cancel := frames.NewCancelFrame()
	_ = hs.p.QueueFrame(end, frames.Downstream)
	_ = hs.p.QueueFrame(cancel, frames.Downstream)
	seen := map[uint64]bool{}
	for _, f := range hs.down.collect(t, 2) {
		seen[f.ID()] = true
	}
	if !seen[end.ID()] || !seen[cancel.ID()] {
		t.Fatal("EndFrame or CancelFrame not forwarded")
	}
}
