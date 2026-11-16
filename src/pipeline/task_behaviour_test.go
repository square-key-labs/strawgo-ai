package pipeline

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const waitLimit = 3 * time.Second

type seen struct {
	frame frames.Frame
	dir   frames.FrameDirection
}

type recorder struct {
	mu      sync.Mutex
	log     []seen
	changed chan struct{}
}

func newRecorder() *recorder {
	return &recorder{changed: make(chan struct{})}
}

func (r *recorder) add(f frames.Frame, d frames.FrameDirection) {
	r.mu.Lock()
	r.log = append(r.log, seen{frame: f, dir: d})
	close(r.changed)
	r.changed = make(chan struct{})
	r.mu.Unlock()
}

func (r *recorder) snapshot() []seen {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]seen(nil), r.log...)
}

func (r *recorder) waitFor(t *testing.T, what string, pred func([]seen) bool) []seen {
	t.Helper()
	timer := time.NewTimer(waitLimit)
	defer timer.Stop()
	for {
		r.mu.Lock()
		log := append([]seen(nil), r.log...)
		ch := r.changed
		r.mu.Unlock()
		if pred(log) {
			return log
		}
		select {
		case <-ch:
		case <-timer.C:
			t.Fatalf("timed out waiting for %s; saw %s", what, describe(log))
			return nil
		}
	}
}

func describe(log []seen) string {
	out := ""
	for i, s := range log {
		if i > 0 {
			out += ", "
		}
		out += s.frame.Name()
		if tf, ok := s.frame.(*frames.TextFrame); ok {
			out += "(" + tf.Text + ")"
		}
		if s.dir == frames.Upstream {
			out += "^"
		}
	}
	return "[" + out + "]"
}

type hookFunc func(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool

type probe struct {
	*processors.BaseProcessor
	*recorder
	hook hookFunc
}

func newProbe(name string, hook hookFunc, opts ...processors.Option) *probe {
	p := &probe{recorder: newRecorder(), hook: hook}
	p.BaseProcessor = processors.NewBaseProcessor(name, p, opts...)
	return p
}

func (p *probe) HandleFrame(ctx context.Context, f frames.Frame, d frames.FrameDirection) error {
	forward := true
	if p.hook != nil {
		forward = p.hook(ctx, p, f, d)
	}
	if forward {
		if err := p.PushFrame(f, d); err != nil {
			return err
		}
	}
	p.add(f, d)
	return nil
}

func sinkRecorder(task *PipelineTask) *recorder {
	r := newRecorder()
	task.OnFrameReachedDownstream(func(f frames.Frame) { r.add(f, frames.Downstream) })
	return r
}

func texts(log []seen, dir frames.FrameDirection) []string {
	var out []string
	for _, s := range log {
		if tf, ok := s.frame.(*frames.TextFrame); ok && s.dir == dir {
			out = append(out, tf.Text)
		}
	}
	return out
}

func countOf[T frames.Frame](log []seen, dir frames.FrameDirection) int {
	n := 0
	for _, s := range log {
		if _, ok := s.frame.(T); ok && s.dir == dir {
			n++
		}
	}
	return n
}

func has[T frames.Frame](dir frames.FrameDirection) func([]seen) bool {
	return func(log []seen) bool { return countOf[T](log, dir) > 0 }
}

func hasTexts(dir frames.FrameDirection, n int) func([]seen) bool {
	return func(log []seen) bool { return len(texts(log, dir)) >= n }
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func runTask(task *PipelineTask, ctx context.Context) <-chan error {
	done := make(chan error, 1)
	go func() { done <- task.Run(ctx) }()
	return done
}

func awaitRun(t *testing.T, done <-chan error, limit time.Duration) time.Duration {
	t.Helper()
	start := time.Now()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run returned %v", err)
		}
	case <-time.After(limit):
		t.Fatalf("Run did not return within %v", limit)
	}
	return time.Since(start)
}

func mustQueue(t *testing.T, task *PipelineTask, f frames.Frame, dir ...frames.FrameDirection) {
	t.Helper()
	if err := task.QueueFrame(f, dir...); err != nil {
		t.Fatalf("QueueFrame(%s): %v", f.Name(), err)
	}
}

func onText(text string, fn func(p *probe)) hookFunc {
	return func(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool {
		if tf, ok := f.(*frames.TextFrame); ok && tf.Text == text {
			fn(p)
			return false
		}
		return true
	}
}

func TestTaskEndFrameReachesSinkAndRunReturnsNil(t *testing.T) {
	p := newProbe("p", nil)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{p}))
	sink := sinkRecorder(task)
	done := runTask(task, context.Background())

	mustQueue(t, task, frames.NewTextFrame("a"))
	mustQueue(t, task, frames.NewTextFrame("b"))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	log := sink.snapshot()
	if countOf[*frames.EndFrame](log, frames.Downstream) != 1 {
		t.Fatalf("sink should see one EndFrame, saw %s", describe(log))
	}
	if countOf[*frames.StartFrame](log, frames.Downstream) != 1 {
		t.Fatalf("sink should see one StartFrame, saw %s", describe(log))
	}
	if got := texts(log, frames.Downstream); !equalStrings(got, []string{"a", "b"}) {
		t.Fatalf("sink texts = %v", got)
	}
	if _, ok := log[len(log)-1].frame.(*frames.EndFrame); !ok {
		t.Fatalf("EndFrame should be the last frame, saw %s", describe(log))
	}
	if !task.HasFinished() {
		t.Fatal("HasFinished should be true after Run returns")
	}
	if err := task.QueueFrame(frames.NewTextFrame("late")); err == nil {
		t.Fatal("QueueFrame after finish should fail")
	}
}

func TestTaskFramesQueuedBeforeRunAreDelivered(t *testing.T) {
	p := newProbe("p", nil)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{p}))
	for _, s := range []string{"1", "2", "3"} {
		mustQueue(t, task, frames.NewTextFrame(s))
	}
	mustQueue(t, task, frames.NewEndFrame())

	awaitRun(t, runTask(task, context.Background()), waitLimit)

	log := p.snapshot()
	if _, ok := log[0].frame.(*frames.StartFrame); !ok {
		t.Fatalf("StartFrame should be handled first, saw %s", describe(log))
	}
	if got := texts(log, frames.Downstream); !equalStrings(got, []string{"1", "2", "3"}) {
		t.Fatalf("texts = %v", got)
	}
	if _, ok := log[len(log)-1].frame.(*frames.EndFrame); !ok {
		t.Fatalf("EndFrame should be last, saw %s", describe(log))
	}
}

func swallowCancel(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool {
	_, isCancel := f.(*frames.CancelFrame)
	return !isCancel
}

func TestTaskCancelReturnsWithinCancelTimeoutWhenCancelFrameIsSwallowed(t *testing.T) {
	cases := []struct {
		name   string
		cancel func(task *PipelineTask, stop context.CancelFunc)
	}{
		{"Cancel method", func(task *PipelineTask, _ context.CancelFunc) { task.Cancel() }},
		{"caller ctx", func(_ *PipelineTask, stop context.CancelFunc) { stop() }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultPipelineTaskConfig()
			cfg.CancelTimeout = 150 * time.Millisecond
			p := newProbe("swallower", swallowCancel)
			task := NewPipelineTaskWithConfig(NewPipeline([]processors.FrameProcessor{p}), cfg)
			ctx, stop := context.WithCancel(context.Background())
			defer stop()
			done := runTask(task, ctx)
			p.waitFor(t, "StartFrame", has[*frames.StartFrame](frames.Downstream))

			tc.cancel(task, stop)
			p.waitFor(t, "CancelFrame at swallower", has[*frames.CancelFrame](frames.Downstream))
			elapsed := awaitRun(t, done, cfg.CancelTimeout+2*time.Second)
			if elapsed > cfg.CancelTimeout+time.Second {
				t.Fatalf("Run took %v after cancel", elapsed)
			}
			if !task.HasFinished() {
				t.Fatal("task should be finished")
			}
		})
	}
}

func TestTaskRunWithAlreadyCancelledContext(t *testing.T) {
	p := newProbe("p", nil)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{p}))
	sink := sinkRecorder(task)
	ctx, stop := context.WithCancel(context.Background())
	stop()

	awaitRun(t, runTask(task, ctx), waitLimit)
	if countOf[*frames.CancelFrame](sink.snapshot(), frames.Downstream) != 1 {
		t.Fatalf("CancelFrame should reach the sink, saw %s", describe(sink.snapshot()))
	}
}

func TestTaskCancelBeforeRun(t *testing.T) {
	p := newProbe("p", nil)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{p}))
	sink := sinkRecorder(task)
	task.Cancel()

	awaitRun(t, runTask(task, context.Background()), waitLimit)
	if countOf[*frames.CancelFrame](sink.snapshot(), frames.Downstream) != 1 {
		t.Fatalf("CancelFrame should reach the sink, saw %s", describe(sink.snapshot()))
	}
}

func TestTaskIdleTimeout(t *testing.T) {
	t.Run("cancels", func(t *testing.T) {
		cfg := DefaultPipelineTaskConfig()
		cfg.IdleTimeout = 80 * time.Millisecond
		cfg.HeartbeatInterval = 10 * time.Millisecond
		task := NewPipelineTaskWithConfig(NewPipeline([]processors.FrameProcessor{newProbe("p", nil)}), cfg)
		sink := sinkRecorder(task)
		idle := make(chan struct{}, 8)
		task.OnIdleTimeout(func() { idle <- struct{}{} })

		awaitRun(t, runTask(task, context.Background()), waitLimit)
		select {
		case <-idle:
		default:
			t.Fatal("OnIdleTimeout was not called")
		}
		if countOf[*frames.CancelFrame](sink.snapshot(), frames.Downstream) == 0 {
			t.Fatal("idle timeout should cancel the pipeline")
		}
	})

	t.Run("keeps running", func(t *testing.T) {
		cfg := DefaultPipelineTaskConfig()
		cfg.IdleTimeout = 50 * time.Millisecond
		cfg.HeartbeatInterval = 10 * time.Millisecond
		cfg.CancelOnIdleTimeout = false
		task := NewPipelineTaskWithConfig(NewPipeline([]processors.FrameProcessor{newProbe("p", nil)}), cfg)
		sink := sinkRecorder(task)
		idle := make(chan struct{}, 64)
		task.OnIdleTimeout(func() {
			select {
			case idle <- struct{}{}:
			default:
			}
		})
		done := runTask(task, context.Background())
		select {
		case <-idle:
		case <-time.After(waitLimit):
			t.Fatal("OnIdleTimeout was not called")
		}
		if task.HasFinished() {
			t.Fatal("task should keep running when CancelOnIdleTimeout is false")
		}
		mustQueue(t, task, frames.NewEndFrame())
		awaitRun(t, done, waitLimit)
		if countOf[*frames.CancelFrame](sink.snapshot(), frames.Downstream) != 0 {
			t.Fatal("no CancelFrame expected")
		}
	})
}

func TestTaskFatalErrorFrame(t *testing.T) {
	for _, keep := range []bool{false, true} {
		name := "cancels"
		if keep {
			name = "KeepRunningOnFatal"
		}
		t.Run(name, func(t *testing.T) {
			cfg := DefaultPipelineTaskConfig()
			cfg.KeepRunningOnFatal = keep
			bad := errors.New("bad")
			p := newProbe("p", onText("boom", func(p *probe) {
				_ = p.PushError("exploded", bad, true)
			}))
			task := NewPipelineTaskWithConfig(NewPipeline([]processors.FrameProcessor{p}), cfg)
			sink := sinkRecorder(task)
			errs := make(chan error, 4)
			task.OnError(func(err error) { errs <- err })
			done := runTask(task, context.Background())

			mustQueue(t, task, frames.NewTextFrame("boom"))
			select {
			case err := <-errs:
				if err == nil {
					t.Fatal("OnError got nil error")
				}
			case <-time.After(waitLimit):
				t.Fatal("OnError not called for fatal ErrorFrame")
			}

			if keep {
				mustQueue(t, task, frames.NewTextFrame("after"))
				sink.waitFor(t, "text after error", hasTexts(frames.Downstream, 1))
				mustQueue(t, task, frames.NewEndFrame())
				awaitRun(t, done, waitLimit)
			} else {
				select {
				case err := <-done:
					if !errors.Is(err, bad) {
						t.Fatalf("Run returned %v, want an error wrapping %v", err, bad)
					}
				case <-time.After(waitLimit):
					t.Fatalf("Run did not return within %v", waitLimit)
				}
			}

			cancels := countOf[*frames.CancelFrame](sink.snapshot(), frames.Downstream)
			if keep && cancels != 0 {
				t.Fatalf("KeepRunningOnFatal should not cancel, saw %s", describe(sink.snapshot()))
			}
			if !keep && cancels != 1 {
				t.Fatalf("fatal error should cancel, saw %s", describe(sink.snapshot()))
			}
		})
	}
}

func TestTaskNonFatalErrorDoesNotCancel(t *testing.T) {
	p := newProbe("p", onText("oops", func(p *probe) {
		_ = p.PushError("minor", errors.New("x"), false)
	}))
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{p}))
	sink := sinkRecorder(task)
	errs := make(chan error, 4)
	task.OnError(func(err error) { errs <- err })
	done := runTask(task, context.Background())

	mustQueue(t, task, frames.NewTextFrame("oops"))
	select {
	case <-errs:
	case <-time.After(waitLimit):
		t.Fatal("OnError not called")
	}
	mustQueue(t, task, frames.NewTextFrame("still"))
	sink.waitFor(t, "text after error", hasTexts(frames.Downstream, 1))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)
	if countOf[*frames.CancelFrame](sink.snapshot(), frames.Downstream) != 0 {
		t.Fatal("non-fatal error must not cancel")
	}
}

func TestTaskProcessorUnusablePolicy(t *testing.T) {
	cases := []struct {
		name   string
		policy ProcessorUnusablePolicy
	}{
		{"continue", ProcessorUnusableContinue},
		{"end", ProcessorUnusableEnd},
		{"cancel", ProcessorUnusableCancel},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultPipelineTaskConfig()
			cfg.ProcessorUnusablePolicy = tc.policy
			rejected := errors.New("bad key")
			p := newProbe("p", func(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool {
				tf, ok := f.(*frames.TextFrame)
				if !ok || d != frames.Downstream {
					return true
				}
				switch tf.Text {
				case "auth":
					_ = p.PushErrorFrame(frames.NewCategorizedErrorFrame(rejected, frames.ErrorCategoryAuthentication))
					return false
				case "busy":
					_ = p.PushErrorFrame(frames.NewCategorizedErrorFrame(errors.New("slow down"), frames.ErrorCategoryRateLimit))
					return false
				case "malformed":
					_ = p.PushErrorFrame(frames.NewCategorizedErrorFrame(errors.New("bad tool call"), frames.ErrorCategoryInvalidRequest))
					return false
				}
				return true
			})
			task := NewPipelineTaskWithConfig(NewPipeline([]processors.FrameProcessor{p}), cfg)
			sink := sinkRecorder(task)
			errs := make(chan error, 8)
			task.OnError(func(err error) { errs <- err })
			done := runTask(task, context.Background())

			mustQueue(t, task, frames.NewTextFrame("busy"))
			select {
			case <-errs:
			case <-time.After(waitLimit):
				t.Fatal("OnError not called for rate limit")
			}
			mustQueue(t, task, frames.NewTextFrame("malformed"))
			select {
			case <-errs:
			case <-time.After(waitLimit):
				t.Fatal("OnError not called for invalid request")
			}
			mustQueue(t, task, frames.NewTextFrame("after transient errors"))
			sink.waitFor(t, "text after transient errors", hasTexts(frames.Downstream, 1))

			mustQueue(t, task, frames.NewTextFrame("auth"))
			select {
			case <-errs:
			case <-time.After(waitLimit):
				t.Fatal("OnError not called for authentication error")
			}

			switch tc.policy {
			case ProcessorUnusableContinue:
				mustQueue(t, task, frames.NewTextFrame("auth"))
				mustQueue(t, task, frames.NewTextFrame("still running"))
				sink.waitFor(t, "text after unusable", hasTexts(frames.Downstream, 2))
				mustQueue(t, task, frames.NewEndFrame())
				if err := <-done; err != nil {
					t.Fatalf("Run returned %v", err)
				}
			case ProcessorUnusableEnd:
				select {
				case err := <-done:
					if err != nil {
						t.Fatalf("Run returned %v", err)
					}
				case <-time.After(waitLimit):
					t.Fatal("pipeline did not end")
				}
				if countOf[*frames.EndFrame](sink.snapshot(), frames.Downstream) != 1 {
					t.Fatalf("expected a graceful end, saw %s", describe(sink.snapshot()))
				}
			case ProcessorUnusableCancel:
				select {
				case err := <-done:
					if !errors.Is(err, rejected) {
						t.Fatalf("Run returned %v, want an error wrapping %v", err, rejected)
					}
				case <-time.After(waitLimit):
					t.Fatal("pipeline was not cancelled")
				}
			}
			cancels := countOf[*frames.CancelFrame](sink.snapshot(), frames.Downstream)
			if (tc.policy == ProcessorUnusableCancel) != (cancels == 1) {
				t.Fatalf("policy %s: cancel frames = %d", tc.name, cancels)
			}
		})
	}
}

func TestTaskEndAndCancelTaskFramesFromProcessor(t *testing.T) {
	t.Run("EndTaskFrame", func(t *testing.T) {
		p := newProbe("p", onText("bye", func(p *probe) {
			_ = p.PushFrame(frames.NewEndTaskFrame("done talking"), frames.Upstream)
		}))
		task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{newProbe("head", nil), p}))
		sink := sinkRecorder(task)
		done := runTask(task, context.Background())
		mustQueue(t, task, frames.NewTextFrame("bye"))
		awaitRun(t, done, waitLimit)

		var end *frames.EndFrame
		for _, s := range sink.snapshot() {
			if f, ok := s.frame.(*frames.EndFrame); ok {
				end = f
			}
		}
		if end == nil {
			t.Fatalf("EndFrame should reach the sink, saw %s", describe(sink.snapshot()))
		}
		if end.Reason != "done talking" {
			t.Fatalf("EndFrame reason = %q", end.Reason)
		}
	})

	t.Run("CancelTaskFrame", func(t *testing.T) {
		p := newProbe("p", onText("abort", func(p *probe) {
			_ = p.PushFrame(frames.NewCancelTaskFrame("stop now"), frames.Upstream)
		}))
		task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{p}))
		sink := sinkRecorder(task)
		done := runTask(task, context.Background())
		mustQueue(t, task, frames.NewTextFrame("abort"))
		awaitRun(t, done, waitLimit)
		if countOf[*frames.CancelFrame](sink.snapshot(), frames.Downstream) != 1 {
			t.Fatalf("CancelFrame should reach the sink, saw %s", describe(sink.snapshot()))
		}
	})
}

func TestNestedPipelineBothDirections(t *testing.T) {
	head := newProbe("head", nil)
	inner1 := newProbe("inner1", nil)
	inner2 := newProbe("inner2", nil)
	tail := newProbe("tail", onText("ping", func(p *probe) {
		_ = p.PushFrame(frames.NewTextFrame("pong"), frames.Upstream)
	}))
	inner := NewPipeline([]processors.FrameProcessor{inner1, inner2})
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{head, inner, tail}))
	sink := sinkRecorder(task)
	done := runTask(task, context.Background())

	for _, s := range []string{"x", "y", "ping"} {
		mustQueue(t, task, frames.NewTextFrame(s))
	}
	head.waitFor(t, "pong upstream at head", hasTexts(frames.Upstream, 1))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	for _, p := range []*probe{head, inner1, inner2} {
		if got := texts(p.snapshot(), frames.Downstream); !equalStrings(got, []string{"x", "y", "ping"}) {
			t.Fatalf("%s downstream texts = %v", p.Name(), got)
		}
	}
	for _, p := range []*probe{inner1, inner2, head} {
		if got := texts(p.snapshot(), frames.Upstream); !equalStrings(got, []string{"pong"}) {
			t.Fatalf("%s upstream texts = %v", p.Name(), got)
		}
	}
	if got := texts(sink.snapshot(), frames.Downstream); !equalStrings(got, []string{"x", "y"}) {
		t.Fatalf("sink texts = %v", got)
	}
	log := sink.snapshot()
	if countOf[*frames.StartFrame](log, frames.Downstream) != 1 || countOf[*frames.EndFrame](log, frames.Downstream) != 1 {
		t.Fatalf("sink lifecycle frames: %s", describe(log))
	}
}

type lifecycleObserver struct {
	mu      sync.Mutex
	started int
	stopped int
	process int
}

func (o *lifecycleObserver) OnProcessFrame(ProcessFrameEvent) {
	o.mu.Lock()
	o.process++
	o.mu.Unlock()
}

func (o *lifecycleObserver) OnPushFrame(PushFrameEvent) {}

func (o *lifecycleObserver) OnPipelineStarted() {
	o.mu.Lock()
	o.started++
	o.mu.Unlock()
}

func (o *lifecycleObserver) OnPipelineStopped() {
	o.mu.Lock()
	o.stopped++
	o.mu.Unlock()
}

func (o *lifecycleObserver) counts() (int, int, int) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.started, o.stopped, o.process
}

func TestTaskObserverStartedStoppedAndClosed(t *testing.T) {
	obs := &lifecycleObserver{}
	to := NewTaskObserver()
	to.AddObserver(obs)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{newProbe("p", nil)}))
	task.SetObserver(to)
	done := runTask(task, context.Background())
	mustQueue(t, task, frames.NewTextFrame("x"))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	started, stopped, process := obs.counts()
	if started != 1 || stopped != 1 {
		t.Fatalf("started=%d stopped=%d, want 1 and 1", started, stopped)
	}
	if process == 0 {
		t.Fatal("observer should receive process events")
	}

	late := &lifecycleObserver{}
	to.AddObserver(late)
	to.OnPipelineStarted()
	if s, _, _ := late.counts(); s != 0 {
		t.Fatal("closed TaskObserver must not accept new observers")
	}
}

func TestTaskRunsDoNotLeakGoroutines(t *testing.T) {
	runOnce := func() {
		to := NewTaskObserver()
		to.AddObserver(&lifecycleObserver{})
		task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{newProbe("a", nil), newProbe("b", nil)}))
		task.SetObserver(to)
		done := runTask(task, context.Background())
		mustQueue(t, task, frames.NewTextFrame("x"))
		mustQueue(t, task, frames.NewEndFrame())
		awaitRun(t, done, waitLimit)
	}
	runOnce()
	before := runtime.NumGoroutine()
	for i := 0; i < 50; i++ {
		runOnce()
	}
	deadline := time.Now().Add(2 * time.Second)
	after := runtime.NumGoroutine()
	for after > before+2 && time.Now().Before(deadline) {
		runtime.Gosched()
		time.Sleep(10 * time.Millisecond)
		after = runtime.NumGoroutine()
	}
	if after > before+2 {
		t.Fatalf("goroutines grew from %d to %d across 50 runs", before, after)
	}
}
