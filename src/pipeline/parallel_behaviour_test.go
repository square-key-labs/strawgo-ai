package pipeline

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
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

func categorizedTagger(name string) hookFunc {
	return func(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool {
		tf, ok := f.(*frames.TextFrame)
		if !ok || d != frames.Downstream {
			return true
		}
		if category, failed := strings.CutPrefix(tf.Text, "fail:"); failed {
			_ = p.PushErrorFrame(frames.NewCategorizedErrorFrame(errors.New(name+" failed"), frames.ErrorCategory(category)))
			return false
		}
		_ = p.PushFrame(frames.NewTextFrame(name+":"+tf.Text), d)
		return false
	}
}

func expectSwitch(t *testing.T, switched <-chan processors.FrameProcessor, want processors.FrameProcessor) {
	t.Helper()
	select {
	case p := <-switched:
		if p != want {
			t.Fatalf("switched to %s, want %s", p.Name(), want.Name())
		}
	case <-time.After(waitLimit):
		t.Fatalf("did not switch to %s", want.Name())
	}
}

func expectTaskError(t *testing.T, errs <-chan error) {
	t.Helper()
	select {
	case <-errs:
	case <-time.After(waitLimit):
		t.Fatal("error did not reach the task")
	}
}

func TestServiceSwitcherFailoverSkipsUnusableService(t *testing.T) {
	s1 := newProbe("primary", categorizedTagger("primary"))
	s2 := newProbe("backup", categorizedTagger("backup"))
	sw := NewServiceSwitcher([]processors.FrameProcessor{s1, s2}, FailoverSwitchStrategy{})
	switched := make(chan processors.FrameProcessor, 4)
	sw.OnServiceSwitched(func(p processors.FrameProcessor) { switched <- p })
	post := newProbe("post", nil)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sw, post}))
	errs := make(chan error, 4)
	task.OnError(func(err error) { errs <- err })
	done := runTask(task, context.Background())

	mustQueue(t, task, frames.NewTextFrame("fail:authentication"))
	expectSwitch(t, switched, s2)
	if sw.ServiceUsable(s1) || !sw.ServiceUsable(s2) {
		t.Fatalf("usable: primary=%v backup=%v", sw.ServiceUsable(s1), sw.ServiceUsable(s2))
	}

	mustQueue(t, task, frames.NewTextFrame("fail:server"))
	expectTaskError(t, errs)
	mustQueue(t, task, frames.NewTextFrame("x"))
	post.waitFor(t, "backup output", hasTexts(frames.Downstream, 1))
	if sw.ActiveService() != s2 {
		t.Fatalf("switched back to an unusable service: active = %s", sw.ActiveService().Name())
	}

	if !sw.SetServiceUsable(s1, true) {
		t.Fatal("SetServiceUsable did not find the service")
	}
	mustQueue(t, task, frames.NewTextFrame("fail:rate_limit"))
	expectSwitch(t, switched, s1)
	mustQueue(t, task, frames.NewTextFrame("y"))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	if got := texts(post.snapshot(), frames.Downstream); !equalStrings(got, []string{"backup:x", "primary:y"}) {
		t.Fatalf("texts = %v", got)
	}
}

func TestServiceSwitcherFailoverOnlyWhenUnusable(t *testing.T) {
	s1 := newProbe("primary", categorizedTagger("primary"))
	s2 := newProbe("backup", categorizedTagger("backup"))
	sw := NewServiceSwitcher([]processors.FrameProcessor{s1, s2}, FailoverSwitchStrategy{SwitchOnlyWhenUnusable: true})
	switched := make(chan processors.FrameProcessor, 4)
	sw.OnServiceSwitched(func(p processors.FrameProcessor) { switched <- p })
	post := newProbe("post", nil)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sw, post}))
	errs := make(chan error, 8)
	task.OnError(func(err error) { errs <- err })
	done := runTask(task, context.Background())

	for _, category := range []frames.ErrorCategory{
		frames.ErrorCategoryRateLimit,
		frames.ErrorCategoryConnectivity,
		frames.ErrorCategoryServer,
		frames.ErrorCategoryInvalidRequest,
		frames.ErrorCategoryUnknown,
	} {
		mustQueue(t, task, frames.NewTextFrame("fail:"+string(category)))
		expectTaskError(t, errs)
		if sw.ActiveService() != s1 {
			t.Fatalf("%s switched service", category)
		}
		if !sw.ServiceUsable(s1) {
			t.Fatalf("%s marked the service unusable", category)
		}
	}
	mustQueue(t, task, frames.NewTextFrame("fail:quota"))
	expectSwitch(t, switched, s2)
	mustQueue(t, task, frames.NewTextFrame("x"))
	mustQueue(t, task, frames.NewTextFrame("fail:configuration"))
	expectTaskError(t, errs)
	if sw.ServiceUsable(s2) || sw.ActiveService() != s2 {
		t.Fatalf("configuration error: backup usable=%v active=%s", sw.ServiceUsable(s2), sw.ActiveService().Name())
	}
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	if got := texts(post.snapshot(), frames.Downstream); !equalStrings(got, []string{"backup:x"}) {
		t.Fatalf("texts = %v", got)
	}
	select {
	case err := <-errs:
		t.Fatalf("unexpected error reached the task: %v", err)
	default:
	}
}

type customFailover struct {
	FailoverSwitchStrategy
}

func TestServiceSwitcherEmbeddedFailoverSkipsUnusableService(t *testing.T) {
	s1 := newProbe("a", categorizedTagger("a"))
	s2 := newProbe("b", categorizedTagger("b"))
	s3 := newProbe("c", categorizedTagger("c"))
	sw := NewServiceSwitcher([]processors.FrameProcessor{s1, s2, s3}, customFailover{})
	switched := make(chan processors.FrameProcessor, 4)
	sw.OnServiceSwitched(func(p processors.FrameProcessor) { switched <- p })
	post := newProbe("post", nil)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sw, post}))
	done := runTask(task, context.Background())

	mustQueue(t, task, frames.NewTextFrame("fail:authentication"))
	expectSwitch(t, switched, s2)
	mustQueue(t, task, frames.NewTextFrame("fail:server"))
	expectSwitch(t, switched, s3)
	mustQueue(t, task, frames.NewTextFrame("fail:server"))
	expectSwitch(t, switched, s2)
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)
}

func TestSyncParallelPipelineDropsBranchOutputFromBeforeInterruption(t *testing.T) {
	release := make(chan struct{})
	blocked := make(chan struct{})
	a := newProbe("a", func(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool {
		if _, ok := f.(*frames.AudioBufferStartRecordingFrame); ok {
			_ = p.PushFrame(frames.NewTextFrame("stale"), d)
		}
		return true
	})
	b := newProbe("b", func(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool {
		if _, ok := f.(*frames.AudioBufferStartRecordingFrame); ok {
			close(blocked)
			<-release
		}
		return true
	})
	post := newProbe("post", nil)
	sp := NewSyncParallelPipeline([]processors.FrameProcessor{a}, []processors.FrameProcessor{b})
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sp, post}))
	done := runTask(task, context.Background())
	post.waitFor(t, "StartFrame", has[*frames.StartFrame](frames.Downstream))

	mustQueue(t, task, frames.NewAudioBufferStartRecordingFrame())
	a.waitFor(t, "round frame in branch a", has[*frames.AudioBufferStartRecordingFrame](frames.Downstream))
	select {
	case <-blocked:
	case <-time.After(waitLimit):
		t.Fatal("branch b never saw the round frame")
	}
	time.Sleep(20 * time.Millisecond)
	mustQueue(t, task, frames.NewInterruptionFrame())
	post.waitFor(t, "interruption", has[*frames.InterruptionFrame](frames.Downstream))
	close(release)
	post.waitFor(t, "round frame", has[*frames.AudioBufferStartRecordingFrame](frames.Downstream))
	mustQueue(t, task, frames.NewTextFrame("fresh"))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	log := post.snapshot()
	interrupted := false
	for _, s := range log {
		switch f := s.frame.(type) {
		case *frames.InterruptionFrame:
			interrupted = true
		case *frames.TextFrame:
			if interrupted && f.Text == "stale" {
				t.Fatalf("output produced before the interruption was emitted after it: %s", describe(log))
			}
		}
	}
	if got := texts(log, frames.Downstream); len(got) == 0 || got[len(got)-1] != "fresh" {
		t.Fatalf("texts after interruption = %v, want fresh last", got)
	}
}

func TestSyncParallelPipelineRecoversWhenBranchSwallowsInterruption(t *testing.T) {
	a := newProbe("a", func(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool {
		_, interruption := f.(*frames.InterruptionFrame)
		return !interruption
	})
	b := newProbe("b", nil)
	post := newProbe("post", nil)
	sp := NewSyncParallelPipeline([]processors.FrameProcessor{a, newProbe("a2", prefixer("a-"))}, []processors.FrameProcessor{b})
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sp, post}))
	done := runTask(task, context.Background())
	post.waitFor(t, "StartFrame", has[*frames.StartFrame](frames.Downstream))

	mustQueue(t, task, frames.NewInterruptionFrame())
	post.waitFor(t, "interruption", has[*frames.InterruptionFrame](frames.Downstream))
	mustQueue(t, task, frames.NewTextFrame("one"))
	mustQueue(t, task, frames.NewTextFrame("two"))
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	got := map[string]bool{}
	for _, s := range texts(post.snapshot(), frames.Downstream) {
		got[s] = true
	}
	for _, want := range []string{"a-one", "a-two"} {
		if !got[want] {
			t.Fatalf("missing %q after a swallowed interruption: %s", want, describe(post.snapshot()))
		}
	}
}

type countingProvider struct {
	name  string
	reply string
	calls atomic.Int32
}

func (p *countingProvider) Name() string { return p.name }

func (p *countingProvider) Stream(_ context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
	p.calls.Add(1)
	if p.reply == "" {
		return errors.New("unavailable")
	}
	emit(llm.TextDelta{Text: p.reply})
	return nil
}

func TestLLMSwitcherFailoverReplay(t *testing.T) {
	cases := []struct {
		name         string
		replay       bool
		embedded     bool
		backupReply  string
		wantBackup   int32
		wantAnswered bool
	}{
		{"off", false, false, "from backup", 0, false},
		{"replays to backup", true, false, "from backup", 1, true},
		{"embedded strategy replays to backup", true, true, "from backup", 1, true},
		{"every service fails", true, false, "", 1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			primary := &countingProvider{name: "primary"}
			backup := &countingProvider{name: "backup", reply: tc.backupReply}
			var strategy ServiceSwitcherStrategy = FailoverSwitchStrategy{ReplayFailedTurn: tc.replay}
			if tc.embedded {
				strategy = customFailover{FailoverSwitchStrategy{ReplayFailedTurn: tc.replay}}
			}
			sw := NewLLMSwitcher([]*llmservice.Service{llmservice.New(primary), llmservice.New(backup)}, strategy)
			switched := make(chan processors.FrameProcessor, 8)
			sw.OnServiceSwitched(func(p processors.FrameProcessor) { switched <- p })
			post := newProbe("post", nil)
			task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sw, post}))
			done := runTask(task, context.Background())
			post.waitFor(t, "StartFrame", has[*frames.StartFrame](frames.Downstream))

			mustQueue(t, task, frames.NewLLMContextFrame(llm.NewContext(llm.User("hi"))))
			select {
			case <-switched:
			case <-time.After(waitLimit):
				t.Fatal("failover did not switch")
			}
			if tc.wantAnswered {
				post.waitFor(t, "backup answer", has[*frames.LLMTextFrame](frames.Downstream))
			}
			time.Sleep(100 * time.Millisecond)
			mustQueue(t, task, frames.NewEndFrame())
			awaitRun(t, done, waitLimit)

			if got := primary.calls.Load(); got != 1 {
				t.Fatalf("primary called %d times, want 1", got)
			}
			if got := backup.calls.Load(); got != tc.wantBackup {
				t.Fatalf("backup called %d times, want %d", got, tc.wantBackup)
			}
			answered := countOf[*frames.LLMTextFrame](post.snapshot(), frames.Downstream) > 0
			if answered != tc.wantAnswered {
				t.Fatalf("answered = %v, want %v: %s", answered, tc.wantAnswered, describe(post.snapshot()))
			}
		})
	}
}

func TestLLMSwitcherDoesNotReplayAnsweredTurn(t *testing.T) {
	primary := &countingProvider{name: "primary", reply: "from primary"}
	backup := &countingProvider{name: "backup", reply: "from backup"}
	sw := NewLLMSwitcher([]*llmservice.Service{llmservice.New(primary), llmservice.New(backup)}, FailoverSwitchStrategy{ReplayFailedTurn: true})
	failed := make(chan struct{})
	post := newProbe("post", func(ctx context.Context, p *probe, f frames.Frame, d frames.FrameDirection) bool {
		if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
			_ = p.PushError("downstream failure", errors.New("tts unavailable"), false)
			close(failed)
		}
		return true
	})
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sw, post}))
	done := runTask(task, context.Background())
	post.waitFor(t, "StartFrame", has[*frames.StartFrame](frames.Downstream))

	mustQueue(t, task, frames.NewLLMContextFrame(llm.NewContext(llm.User("hi"))))
	select {
	case <-failed:
	case <-time.After(waitLimit):
		t.Fatal("primary never finished its response")
	}
	time.Sleep(150 * time.Millisecond)
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)

	if got := backup.calls.Load(); got != 0 {
		t.Fatalf("backup called %d times for a turn the primary answered", got)
	}
	if got := countOf[*frames.LLMTextFrame](post.snapshot(), frames.Downstream); got != 1 {
		t.Fatalf("got %d answers, want 1: %s", got, describe(post.snapshot()))
	}
}

type scriptedProvider struct {
	name    string
	replies []string
	started chan int
	gate    chan struct{}
	calls   atomic.Int32
	mu      sync.Mutex
	sizes   []int
}

func (p *scriptedProvider) Name() string { return p.name }

func (p *scriptedProvider) Stream(ctx context.Context, snap llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
	n := int(p.calls.Add(1))
	p.mu.Lock()
	p.sizes = append(p.sizes, len(snap.Messages))
	p.mu.Unlock()
	if p.started != nil {
		p.started <- n
	}
	if n == 1 && p.gate != nil {
		select {
		case <-p.gate:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	reply := ""
	if n <= len(p.replies) {
		reply = p.replies[n-1]
	}
	if reply == "" {
		return errors.New("unavailable")
	}
	emit(llm.TextDelta{Text: reply})
	return nil
}

func llmTexts(log []seen) []string {
	var out []string
	for _, s := range log {
		if f, ok := s.frame.(*frames.LLMTextFrame); ok && s.dir == frames.Downstream {
			out = append(out, f.Text)
		}
	}
	return out
}

func TestLLMSwitcherFailoverReplayWithOverlappingRequests(t *testing.T) {
	shared := llm.NewContext(llm.User("hi"))
	cases := []struct {
		name         string
		second       *llm.Context
		primary      []string
		wantPrimary  int32
		wantBackup   int32
		wantSizes    []int
		wantAnswered []string
	}{
		{"queued request for the same conversation is answered by the failed service", shared, []string{"", "primary"}, 2, 0, nil, []string{"primary"}},
		{"queued request for the same conversation fails too and is replayed once", shared, []string{"", ""}, 2, 1, []int{1}, []string{"backup"}},
		{"failed request for another conversation is replayed alone", llm.NewContext(llm.User("other"), llm.User("again")), []string{"", "primary"}, 2, 1, []int{1}, []string{"backup", "primary"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			primary := &scriptedProvider{name: "primary", replies: tc.primary, started: make(chan int, 4), gate: make(chan struct{})}
			backup := &scriptedProvider{name: "backup", replies: []string{"backup"}}
			sw := NewLLMSwitcher([]*llmservice.Service{llmservice.New(primary), llmservice.New(backup)}, FailoverSwitchStrategy{ReplayFailedTurn: true})
			pre := newProbe("pre", nil)
			post := newProbe("post", nil)
			task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{pre, sw, post}))
			done := runTask(task, context.Background())
			post.waitFor(t, "StartFrame", has[*frames.StartFrame](frames.Downstream))

			mustQueue(t, task, frames.NewLLMContextFrame(shared))
			select {
			case <-primary.started:
			case <-time.After(waitLimit):
				t.Fatal("primary never started the first request")
			}
			second := frames.NewLLMContextFrame(tc.second)
			mustQueue(t, task, second)
			pre.waitFor(t, "second request", func(log []seen) bool {
				for _, s := range log {
					if s.frame == frames.Frame(second) {
						return true
					}
				}
				return false
			})
			time.Sleep(20 * time.Millisecond)
			close(primary.gate)

			post.waitFor(t, "answers", func(log []seen) bool { return len(llmTexts(log)) >= len(tc.wantAnswered) })
			time.Sleep(150 * time.Millisecond)
			mustQueue(t, task, frames.NewEndFrame())
			awaitRun(t, done, waitLimit)

			if got := primary.calls.Load(); got != tc.wantPrimary {
				t.Fatalf("primary called %d times, want %d", got, tc.wantPrimary)
			}
			if got := backup.calls.Load(); got != tc.wantBackup {
				t.Fatalf("backup called %d times, want %d", got, tc.wantBackup)
			}
			backup.mu.Lock()
			sizes := append([]int(nil), backup.sizes...)
			backup.mu.Unlock()
			if len(sizes) != len(tc.wantSizes) || (len(sizes) > 0 && sizes[0] != tc.wantSizes[0]) {
				t.Fatalf("backup saw contexts of %v messages, want %v", sizes, tc.wantSizes)
			}
			got := llmTexts(post.snapshot())
			want := map[string]int{}
			for _, w := range tc.wantAnswered {
				want[w]++
			}
			for _, g := range got {
				want[g]--
			}
			for k, v := range want {
				if v != 0 || len(got) != len(tc.wantAnswered) {
					t.Fatalf("answers %v, want %v (mismatch on %q): %s", got, tc.wantAnswered, k, describe(post.snapshot()))
				}
			}
		})
	}
}

func TestLLMSwitcherDoesNotReplayTurnSupersededAfterManualSwitch(t *testing.T) {
	shared := llm.NewContext(llm.User("hi"))
	primary := &scriptedProvider{name: "primary", replies: []string{""}, started: make(chan int, 4), gate: make(chan struct{})}
	backup := &scriptedProvider{name: "backup", replies: []string{"backup-1", "backup-2"}}
	p, b := llmservice.New(primary), llmservice.New(backup)
	sw := NewLLMSwitcher([]*llmservice.Service{p, b}, FailoverSwitchStrategy{ReplayFailedTurn: true})
	post := newProbe("post", nil)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sw, post}))
	done := runTask(task, context.Background())
	post.waitFor(t, "StartFrame", has[*frames.StartFrame](frames.Downstream))

	mustQueue(t, task, frames.NewLLMContextFrame(shared))
	<-primary.started
	time.Sleep(50 * time.Millisecond)
	sw.SwitchTo(b)
	mustQueue(t, task, frames.NewLLMContextFrame(shared))
	post.waitFor(t, "first answer", func(log []seen) bool { return len(llmTexts(log)) >= 1 })
	close(primary.gate)
	time.Sleep(300 * time.Millisecond)
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)
	got := llmTexts(post.snapshot())
	if len(got) != 1 {
		t.Fatalf("answers = %v, want one", got)
	}
}

func TestLLMSwitcherReplaysAfterForeignResponseStartPassesThrough(t *testing.T) {
	primary := &scriptedProvider{name: "primary", replies: []string{"ok", ""}, started: make(chan int, 4), gate: make(chan struct{})}
	backup := &scriptedProvider{name: "backup", replies: []string{"backup"}}
	sw := NewLLMSwitcher([]*llmservice.Service{llmservice.New(primary), llmservice.New(backup)}, FailoverSwitchStrategy{ReplayFailedTurn: true})
	post := newProbe("post", nil)
	task := NewPipelineTask(NewPipeline([]processors.FrameProcessor{sw, post}))
	done := runTask(task, context.Background())
	post.waitFor(t, "StartFrame", has[*frames.StartFrame](frames.Downstream))

	mustQueue(t, task, frames.NewLLMContextFrame(llm.NewContext(llm.User("one"))))
	<-primary.started
	mustQueue(t, task, frames.NewLLMFullResponseStartFrame())
	mustQueue(t, task, frames.NewLLMFullResponseEndFrame())
	mustQueue(t, task, frames.NewLLMContextFrame(llm.NewContext(llm.User("two"))))
	time.Sleep(100 * time.Millisecond)
	close(primary.gate)
	time.Sleep(400 * time.Millisecond)
	mustQueue(t, task, frames.NewEndFrame())
	awaitRun(t, done, waitLimit)
	if backup.calls.Load() != 1 {
		t.Fatalf("backup calls = %d, want 1", backup.calls.Load())
	}
}
