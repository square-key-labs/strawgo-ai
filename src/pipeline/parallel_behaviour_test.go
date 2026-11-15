package pipeline

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

func prefixer(prefix string) hookFunc {
	return func(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool {
		if tf, ok := f.(*frames.TextFrame); ok && d == frames.Downstream {
			_ = p.PushFrame(frames.NewTextFrame(prefix+tf.Text), d)
			return false
		}
		return true
	}
}

func TestParallelPipelineLifecycleOnceAndBranchOutputs(t *testing.T) {
	a := newProbe("a", prefixer("a-"))
	b := newProbe("b", prefixer("b-"))
	post := newProbe("post", nil)
	pp := NewParallelPipeline([]processors.FrameProcessor{a}, []processors.FrameProcessor{b})
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{pp, post}))
	done := runTask(task, context.Background())

	mustQueue(t, task, frames.NewTextFrame("x"))
	mustQueue(t, task, frames.NewTextFrame("y"))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	log := post.snapshot()
	if n := countOf[*frames.StartFrame](log, frames.Downstream); n != 1 {
		t.Fatalf("StartFrame emitted %d times: %s", n, describe(log))
	}
	if n := countOf[*frames.EndFrame](log, frames.Downstream); n != 1 {
		t.Fatalf("EndFrame emitted %d times: %s", n, describe(log))
	}
	if _, ok := log[0].frame.(*frames.StartFrame); !ok {
		t.Fatalf("StartFrame should be first: %s", describe(log))
	}
	if _, ok := log[len(log)-1].frame.(*frames.EndFrame); !ok {
		t.Fatalf("EndFrame should be last: %s", describe(log))
	}
	got := map[string]bool{}
	for _, s := range texts(log, frames.Downstream) {
		got[s] = true
	}
	for _, want := range []string{"a-x", "a-y", "b-x", "b-y"} {
		if !got[want] {
			t.Fatalf("missing %q in %s", want, describe(log))
		}
	}
	var aOrder, bOrder []string
	for _, s := range texts(log, frames.Downstream) {
		switch s[0] {
		case 'a':
			aOrder = append(aOrder, s)
		case 'b':
			bOrder = append(bOrder, s)
		}
	}
	if !equalStrings(aOrder, []string{"a-x", "a-y"}) || !equalStrings(bOrder, []string{"b-x", "b-y"}) {
		t.Fatalf("per-branch order lost: %v %v", aOrder, bOrder)
	}
}

func TestParallelPipelineDeduplicatesPassthroughFrames(t *testing.T) {
	a := newProbe("a", nil)
	b := newProbe("b", nil)
	post := newProbe("post", nil)
	pp := NewParallelPipeline([]processors.FrameProcessor{a}, []processors.FrameProcessor{b})
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{pp, post}))
	done := runTask(task, context.Background())
	mustQueue(t, task, frames.NewTextFrame("x"))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)
	if got := texts(post.snapshot(), frames.Downstream); !equalStrings(got, []string{"x"}) {
		t.Fatalf("texts = %v, want one x", got)
	}
}

func TestParallelPipelineInterruptionWaitsForAllBranches(t *testing.T) {
	release := make(chan struct{})
	blocked := make(chan struct{})
	a := newProbe("a", nil)
	b := newProbe("b", func(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool {
		if _, ok := f.(*frames.InterruptionFrame); ok {
			close(blocked)
			<-release
		}
		return true
	})
	post := newProbe("post", nil)
	pp := NewParallelPipeline([]processors.FrameProcessor{a}, []processors.FrameProcessor{b})
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{pp, post}))
	done := runTask(task, context.Background())
	post.waitFor(t, "StartFrame", has[*frames.StartFrame](frames.Downstream))

	mustQueue(t, task, frames.NewInterruptionFrame())
	a.waitFor(t, "interruption in branch a", has[*frames.InterruptionFrame](frames.Downstream))
	select {
	case <-blocked:
	case <-time.After(waitLimit):
		t.Fatal("branch b never saw the interruption")
	}
	time.Sleep(50 * time.Millisecond)
	if n := countOf[*frames.InterruptionFrame](post.snapshot(), frames.Downstream); n != 0 {
		t.Fatalf("interruption forwarded before all branches handled it")
	}

	close(release)
	post.waitFor(t, "interruption after release", has[*frames.InterruptionFrame](frames.Downstream))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)
	if n := countOf[*frames.InterruptionFrame](post.snapshot(), frames.Downstream); n != 1 {
		t.Fatalf("interruption forwarded %d times", n)
	}
}

func TestParallelPipelineUpstreamFromBranches(t *testing.T) {
	a := newProbe("a", onText("ask", func(p *probe) {
		_ = p.PushFrame(frames.NewTextFrame("from-a"), frames.Upstream)
	}))
	b := newProbe("b", onText("ask", func(p *probe) {
		_ = p.PushFrame(frames.NewTextFrame("from-b"), frames.Upstream)
	}))
	head := newProbe("head", nil)
	pp := NewParallelPipeline([]processors.FrameProcessor{a}, []processors.FrameProcessor{b})
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{head, pp}))
	done := runTask(task, context.Background())
	mustQueue(t, task, frames.NewTextFrame("ask"))
	log := head.waitFor(t, "both upstream outputs", hasTexts(frames.Upstream, 2))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)
	got := map[string]bool{}
	for _, s := range texts(log, frames.Upstream) {
		got[s] = true
	}
	if !got["from-a"] || !got["from-b"] {
		t.Fatalf("upstream texts = %v", texts(log, frames.Upstream))
	}
}

func TestSyncParallelPipelineRoundOrdering(t *testing.T) {
	a := newProbe("a", prefixer("a-"))
	b := newProbe("b", prefixer("b-"))
	post := newProbe("post", nil)
	sp := NewSyncParallelPipeline([]processors.FrameProcessor{a}, []processors.FrameProcessor{b})
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sp, post}))
	done := runTask(task, context.Background())

	for _, s := range []string{"1", "2", "3"} {
		mustQueue(t, task, frames.NewTextFrame(s))
	}
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	log := post.snapshot()
	want := []string{"a-1", "b-1", "a-2", "b-2", "a-3", "b-3"}
	if got := texts(log, frames.Downstream); !equalStrings(got, want) {
		t.Fatalf("texts = %v, want %v", got, want)
	}
	if countOf[*frames.EndFrame](log, frames.Downstream) != 1 || countOf[*frames.StartFrame](log, frames.Downstream) != 1 {
		t.Fatalf("lifecycle frames: %s", describe(log))
	}
	if _, ok := log[len(log)-1].frame.(*frames.EndFrame); !ok {
		t.Fatalf("EndFrame should be last: %s", describe(log))
	}
}

func tagger(name string) hookFunc {
	return func(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool {
		tf, ok := f.(*frames.TextFrame)
		if !ok || d != frames.Downstream {
			return true
		}
		if tf.Text == "boom" {
			_ = p.PushError("service failed", errors.New("unavailable"), false)
			return false
		}
		_ = p.PushFrame(frames.NewTextFrame(name+":"+tf.Text), d)
		return false
	}
}

func TestServiceSwitcherManual(t *testing.T) {
	s1 := newProbe("svc1", tagger("svc1"))
	s2 := newProbe("svc2", tagger("svc2"))
	sw := NewServiceSwitcher([]processors.FrameProcessor{s1, s2}, nil)
	switched := make(chan processors.FrameProcessor, 4)
	sw.OnServiceSwitched(func(p processors.FrameProcessor) { switched <- p })
	post := newProbe("post", nil)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sw, post}))
	done := runTask(task, context.Background())

	mustQueue(t, task, frames.NewTextFrame("x"))
	post.waitFor(t, "svc1 output", hasTexts(frames.Downstream, 1))
	mustQueue(t, task, frames.NewManuallySwitchServiceFrame(s2))
	mustQueue(t, task, frames.NewTextFrame("y"))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	log := post.snapshot()
	if got := texts(log, frames.Downstream); !equalStrings(got, []string{"svc1:x", "svc2:y"}) {
		t.Fatalf("texts = %v", got)
	}
	if countOf[*frames.StartFrame](log, frames.Downstream) != 1 || countOf[*frames.EndFrame](log, frames.Downstream) != 1 {
		t.Fatalf("lifecycle frames: %s", describe(log))
	}
	if countOf[*frames.ManuallySwitchServiceFrame](log, frames.Downstream) != 0 {
		t.Fatal("switch frame should be consumed by the switcher")
	}
	select {
	case p := <-switched:
		if p != s2 {
			t.Fatalf("switched to %s", p.Name())
		}
	default:
		t.Fatal("OnServiceSwitched not called")
	}
	if sw.ActiveService() != s2 {
		t.Fatalf("active = %s", sw.ActiveService().Name())
	}
}

func TestServiceSwitcherFailover(t *testing.T) {
	s1 := newProbe("primary", tagger("primary"))
	s2 := newProbe("backup", tagger("backup"))
	sw := NewServiceSwitcher([]processors.FrameProcessor{s1, s2}, FailoverSwitchStrategy{})
	switched := make(chan processors.FrameProcessor, 4)
	sw.OnServiceSwitched(func(p processors.FrameProcessor) { switched <- p })
	post := newProbe("post", nil)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sw, post}))
	sink := sinkRecorder(task)
	errs := make(chan error, 4)
	task.OnError(func(err error) { errs <- err })
	done := runTask(task, context.Background())

	mustQueue(t, task, frames.NewTextFrame("x"))
	post.waitFor(t, "primary output", hasTexts(frames.Downstream, 1))
	mustQueue(t, task, frames.NewTextFrame("boom"))
	select {
	case p := <-switched:
		if p != s2 {
			t.Fatalf("failover switched to %s", p.Name())
		}
	case <-time.After(waitLimit):
		t.Fatal("failover did not switch")
	}
	mustQueue(t, task, frames.NewTextFrame("y"))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	if got := texts(post.snapshot(), frames.Downstream); !equalStrings(got, []string{"primary:x", "backup:y"}) {
		t.Fatalf("texts = %v", got)
	}
	select {
	case err := <-errs:
		t.Fatalf("handled service error should not reach the task: %v", err)
	default:
	}
	if countOf[*frames.CancelFrame](sink.snapshot(), frames.Downstream) != 0 {
		t.Fatal("failover must not cancel the pipeline")
	}
}
