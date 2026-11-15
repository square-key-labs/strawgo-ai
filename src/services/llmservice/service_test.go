package llmservice

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const waitTimeout = 3 * time.Second

type streamFunc func(ctx context.Context, snap llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error

type fakeProvider struct {
	name string
	fn   streamFunc
}

func (p *fakeProvider) Name() string { return p.name }

func (p *fakeProvider) Stream(ctx context.Context, snap llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error {
	return p.fn(ctx, snap, settings, emit)
}

func scripted(events ...llm.Event) streamFunc {
	return func(ctx context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
		for _, e := range events {
			emit(e)
		}
		return nil
	}
}

type capture struct {
	name   string
	mu     sync.Mutex
	frames []frames.Frame
	notify chan struct{}
}

func newCapture(name string) *capture {
	return &capture{name: name, notify: make(chan struct{}, 1)}
}

func (c *capture) ProcessFrame(context.Context, frames.Frame, frames.FrameDirection) error {
	return nil
}

func (c *capture) QueueFrame(f frames.Frame, _ frames.FrameDirection) error {
	c.mu.Lock()
	c.frames = append(c.frames, f)
	c.mu.Unlock()
	select {
	case c.notify <- struct{}{}:
	default:
	}
	return nil
}

func (c *capture) PushFrame(frames.Frame, frames.FrameDirection) error { return nil }
func (c *capture) Link(processors.FrameProcessor)                      {}
func (c *capture) SetPrev(processors.FrameProcessor)                   {}
func (c *capture) Start(context.Context) error                         { return nil }
func (c *capture) Stop() error                                         { return nil }
func (c *capture) Name() string                                        { return c.name }

func (c *capture) all() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]frames.Frame(nil), c.frames...)
}

func (c *capture) waitFor(t *testing.T, desc string, pred func([]frames.Frame) bool) []frames.Frame {
	t.Helper()
	deadline := time.NewTimer(waitTimeout)
	defer deadline.Stop()
	for {
		fs := c.all()
		if pred(fs) {
			return fs
		}
		select {
		case <-c.notify:
		case <-deadline.C:
			t.Fatalf("%s: timed out waiting for %s; got %v", c.name, desc, names(c.all()))
		}
	}
}

func (c *capture) waitForType(t *testing.T, match func(frames.Frame) bool, desc string) frames.Frame {
	t.Helper()
	var found frames.Frame
	c.waitFor(t, desc, func(fs []frames.Frame) bool {
		for _, f := range fs {
			if match(f) {
				found = f
				return true
			}
		}
		return false
	})
	return found
}

func names(fs []frames.Frame) []string {
	out := make([]string, len(fs))
	for i, f := range fs {
		switch v := f.(type) {
		case *frames.LLMTextFrame:
			out[i] = "LLMTextFrame(" + v.Text + ")"
		default:
			out[i] = f.Name()
		}
	}
	return out
}

func is[T frames.Frame](f frames.Frame) bool {
	_, ok := f.(T)
	return ok
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

func indexOf(fs []frames.Frame, match func(frames.Frame) bool) int {
	for i, f := range fs {
		if match(f) {
			return i
		}
	}
	return -1
}

type harness struct {
	svc  *Service
	up   *capture
	down *capture
}

func newHarness(t *testing.T, fn streamFunc, start *frames.StartFrame, opts ...Option) *harness {
	t.Helper()
	h := &harness{
		svc:  New(&fakeProvider{name: "fake", fn: fn}, opts...),
		up:   newCapture("up"),
		down: newCapture("down"),
	}
	h.svc.SetPrev(h.up)
	h.svc.Link(h.down)
	if err := h.svc.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = h.svc.Stop() })
	if start == nil {
		start = frames.NewStartFrame()
	}
	h.queue(t, start)
	h.down.waitForType(t, is[*frames.StartFrame], "StartFrame")
	return h
}

func (h *harness) queue(t *testing.T, f frames.Frame) {
	t.Helper()
	if err := h.svc.QueueFrame(f, frames.Downstream); err != nil {
		t.Fatalf("queue %s: %v", f.Name(), err)
	}
}

func (h *harness) generate(t *testing.T, c *llm.Context) {
	t.Helper()
	h.queue(t, frames.NewLLMContextFrame(c))
}

func (h *harness) drain(t *testing.T) []frames.Frame {
	t.Helper()
	end := frames.NewEndFrame()
	h.queue(t, end)
	return h.down.waitFor(t, "EndFrame", func(fs []frames.Frame) bool {
		for _, f := range fs {
			if f == frames.Frame(end) {
				return true
			}
		}
		return false
	})
}

func TestTextResponseFrameSequence(t *testing.T) {
	h := newHarness(t, scripted(
		llm.TextDelta{Text: "Hel"},
		llm.TextDelta{Text: ""},
		llm.TextDelta{Text: "lo"},
		llm.UsageEvent{Usage: llm.Usage{PromptTokens: 3}},
		llm.Done{StopReason: "stop"},
	), nil)
	h.generate(t, llm.NewContext(llm.User("hi")))
	got := names(h.drain(t))
	want := []string{"StartFrame", "LLMFullResponseStartFrame", "LLMTextFrame(Hel)", "LLMTextFrame(lo)", "LLMFullResponseEndFrame", "EndFrame"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("got %v want %v", got, want)
	}
	if up := h.up.all(); len(up) != 0 {
		t.Fatalf("unexpected upstream frames %v", names(up))
	}
}

func TestProviderReceivesSnapshotAndSettings(t *testing.T) {
	var gotSnap llm.Snapshot
	var gotSettings llm.Settings
	h := newHarness(t, func(_ context.Context, snap llm.Snapshot, s llm.Settings, _ func(llm.Event)) error {
		gotSnap, gotSettings = snap, s
		return nil
	}, nil, WithSettings(llm.Settings{Model: "m1"}), WithName("named"))
	if h.svc.Name() != "named" {
		t.Fatalf("name = %q", h.svc.Name())
	}
	c := llm.NewContext(llm.System("s"), llm.User("u"))
	h.generate(t, c)
	h.drain(t)
	if len(gotSnap.Messages) != 2 || gotSnap.Messages[1].Text() != "u" || gotSettings.Model != "m1" {
		t.Fatalf("snap=%+v settings=%+v", gotSnap, gotSettings)
	}
}

func TestUpstreamContextFrameIsForwarded(t *testing.T) {
	called := false
	h := newHarness(t, func(context.Context, llm.Snapshot, llm.Settings, func(llm.Event)) error {
		called = true
		return nil
	}, nil)
	if err := h.svc.QueueFrame(frames.NewLLMContextFrame(llm.NewContext()), frames.Upstream); err != nil {
		t.Fatal(err)
	}
	h.up.waitForType(t, is[*frames.LLMContextFrame], "upstream LLMContextFrame")
	h.drain(t)
	if called {
		t.Fatal("upstream context frame triggered generation")
	}
}

func metricsData(fs []frames.Frame) (ttfb, processing, usage int, firstTTFB, firstText int) {
	firstTTFB, firstText = -1, -1
	for i, f := range fs {
		if is[*frames.LLMTextFrame](f) && firstText < 0 {
			firstText = i
		}
		m, ok := f.(*frames.MetricsFrame)
		if !ok {
			continue
		}
		for _, d := range m.Data {
			switch d.(type) {
			case frames.TTFBMetricsData:
				ttfb++
				if firstTTFB < 0 {
					firstTTFB = i
				}
			case frames.ProcessingMetricsData:
				processing++
			case frames.LLMUsageMetricsData:
				usage++
			}
		}
	}
	return
}

func TestMetrics(t *testing.T) {
	cases := []struct {
		name           string
		metrics, usage bool
		onlyInitial    bool
		wantTTFB       int
		wantProcessing int
		wantUsage      int
	}{
		{"disabled", false, false, false, 0, 0, 0},
		{"metrics every ttfb", true, false, false, 2, 2, 0},
		{"metrics only initial ttfb", true, false, true, 1, 2, 0},
		{"usage only", false, true, false, 0, 0, 2},
		{"all", true, true, false, 2, 2, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			start := frames.NewStartFrame()
			start.EnableMetrics = tc.metrics
			start.EnableUsageMetrics = tc.usage
			start.ReportOnlyInitialTTFB = tc.onlyInitial
			h := newHarness(t, scripted(
				llm.TextDelta{Text: "a"},
				llm.UsageEvent{Usage: llm.Usage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7}},
			), start, WithSettings(llm.Settings{Model: "mod"}))
			c := llm.NewContext(llm.User("hi"))
			h.generate(t, c)
			h.generate(t, c)
			fs := h.drain(t)
			ttfb, processing, usage, firstTTFB, firstText := metricsData(fs)
			if ttfb != tc.wantTTFB || processing != tc.wantProcessing || usage != tc.wantUsage {
				t.Fatalf("ttfb=%d processing=%d usage=%d; frames %v", ttfb, processing, usage, names(fs))
			}
			if ttfb > 0 && firstTTFB > firstText {
				t.Fatalf("TTFB metrics after first text: %v", names(fs))
			}
			for _, m := range ofType[*frames.MetricsFrame](fs) {
				for _, d := range m.Data {
					if d.Base().Processor != "fake" || d.Base().Model != "mod" {
						t.Fatalf("metrics base = %+v", d.Base())
					}
					if u, ok := d.(frames.LLMUsageMetricsData); ok && u.Value.TotalTokens != 7 {
						t.Fatalf("usage = %+v", u.Value)
					}
				}
			}
		})
	}
}

func TestInterruptionMidStreamStopsOutput(t *testing.T) {
	for i := 0; i < 100; i++ {
		runInterruptionMidStream(t)
		if t.Failed() {
			t.Fatalf("failed at iteration %d", i)
		}
	}
}

func runInterruptionMidStream(t *testing.T) {
	done := make(chan struct{})
	h := newHarness(t, func(ctx context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
		defer close(done)
		for n := 0; ; n++ {
			emit(llm.TextDelta{Text: fmt.Sprint(n)})
			if n > 100000 {
				return errors.New("never interrupted")
			}
			select {
			case <-ctx.Done():
				emit(llm.TextDelta{Text: "late"})
				return ctx.Err()
			default:
				runtime.Gosched()
			}
		}
	}, nil)
	h.generate(t, llm.NewContext(llm.User("hi")))
	h.down.waitForType(t, is[*frames.LLMTextFrame], "first LLMTextFrame")
	h.queue(t, frames.NewInterruptionFrame())
	select {
	case <-done:
	case <-time.After(waitTimeout):
		t.Fatal("provider stream did not observe cancellation")
	}
	h.down.waitForType(t, is[*frames.InterruptionFrame], "InterruptionFrame")
	fs := h.drain(t)
	ii := indexOf(fs, is[*frames.InterruptionFrame])
	if ii < 0 {
		t.Fatalf("no InterruptionFrame downstream: %v", names(fs))
	}
	for _, f := range fs[ii+1:] {
		if is[*frames.LLMTextFrame](f) || is[*frames.LLMFullResponseEndFrame](f) {
			t.Fatalf("%s after InterruptionFrame: %v", f.Name(), names(fs[ii:]))
		}
	}
	if up := ofType[*frames.ErrorFrame](h.up.all()); len(up) != 0 {
		t.Fatalf("unexpected error upstream: %v", up[0].Error)
	}
	_ = h.svc.Stop()
}

func TestProviderErrorPushesNonFatalErrorUpstream(t *testing.T) {
	calls := 0
	h := newHarness(t, func(_ context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
		calls++
		if calls == 1 {
			return errors.New("boom")
		}
		emit(llm.TextDelta{Text: "ok"})
		return nil
	}, nil)
	c := llm.NewContext(llm.User("hi"))
	h.generate(t, c)
	ef := h.up.waitForType(t, is[*frames.ErrorFrame], "ErrorFrame").(*frames.ErrorFrame)
	if ef.Fatal || ef.Error == nil || !strings.Contains(ef.Error.Error(), "boom") {
		t.Fatalf("error frame = %+v", ef)
	}
	h.generate(t, c)
	fs := h.drain(t)
	if len(ofType[*frames.LLMTextFrame](fs)) != 1 || len(ofType[*frames.LLMFullResponseEndFrame](fs)) != 2 {
		t.Fatalf("service did not recover: %v", names(fs))
	}
}

func toolCallStream(calls ...llm.ToolCall) streamFunc {
	return func(_ context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
		for _, c := range calls {
			emit(c)
		}
		return nil
	}
}

func resultFor(id string) func(frames.Frame) bool {
	return func(f frames.Frame) bool {
		r, ok := f.(*frames.FunctionCallResultFrame)
		return ok && r.ToolCallID == id
	}
}

func cancelFor(id string) func(frames.Frame) bool {
	return func(f frames.Frame) bool {
		r, ok := f.(*frames.FunctionCallCancelFrame)
		return ok && r.ToolCallID == id
	}
}

func TestFunctionCallFramesBothDirections(t *testing.T) {
	ctxRef := llm.NewContext(llm.User("weather?"))
	var gotParams llm.FunctionCallParams
	h := newHarness(t, toolCallStream(llm.ToolCall{ID: "c1", Name: "weather", Arguments: []byte(`{"city":"Paris"}`)}), nil)
	h.svc.RegisterFunction("weather", func(_ context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		gotParams = p
		return llm.FunctionResult{Value: "sunny"}, nil
	})
	if !h.svc.HasFunction("weather") || h.svc.HasFunction("other") {
		t.Fatal("HasFunction mismatch")
	}
	h.generate(t, ctxRef)
	for _, side := range []*capture{h.down, h.up} {
		fs := side.waitFor(t, "result", func(fs []frames.Frame) bool { return indexOf(fs, resultFor("c1")) >= 0 })
		si := indexOf(fs, is[*frames.FunctionCallsStartedFrame])
		pi := indexOf(fs, is[*frames.FunctionCallInProgressFrame])
		ri := indexOf(fs, resultFor("c1"))
		if !(si >= 0 && si < pi && pi < ri) {
			t.Fatalf("%s order: %v", side.name, names(fs))
		}
		started := fs[si].(*frames.FunctionCallsStartedFrame)
		if len(started.FunctionCalls) != 1 || started.FunctionCalls[0].FunctionName != "weather" {
			t.Fatalf("started = %+v", started.FunctionCalls)
		}
		ip := fs[pi].(*frames.FunctionCallInProgressFrame)
		if ip.ToolCallID != "c1" || ip.Arguments["city"] != "Paris" || ip.CancelOnInterruption || !ip.SurvivesInterruption() {
			t.Fatalf("in progress = %+v", ip)
		}
		r := fs[ri].(*frames.FunctionCallResultFrame)
		if r.Result != "sunny" || r.Error != "" || r.Arguments["city"] != "Paris" || !r.SurvivesInterruption() {
			t.Fatalf("result = %+v", r)
		}
	}
	if gotParams.ToolCallID != "c1" || gotParams.FunctionName != "weather" || gotParams.Context != ctxRef || string(gotParams.RawArguments) != `{"city":"Paris"}` {
		t.Fatalf("params = %+v", gotParams)
	}
	if ctxRef.Len() != 1 {
		t.Fatalf("service mutated context: len=%d", ctxRef.Len())
	}
}

func TestFunctionCallErrorResults(t *testing.T) {
	cases := []struct {
		name    string
		call    llm.ToolCall
		handler llm.FunctionHandler
		wantErr string
	}{
		{"handler error", llm.ToolCall{ID: "e1", Name: "f", Arguments: []byte(`{}`)},
			func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
				return llm.FunctionResult{}, errors.New("db down")
			}, "db down"},
		{"unregistered", llm.ToolCall{ID: "e2", Name: "missing", Arguments: []byte(`{}`)}, nil, "not registered"},
		{"bad json", llm.ToolCall{ID: "e3", Name: "f", Arguments: []byte(`{"a":`)},
			func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
				return llm.FunctionResult{Value: "should not run"}, nil
			}, "invalid arguments"},
		{"panic", llm.ToolCall{ID: "e4", Name: "f", Arguments: []byte(`{}`)},
			func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
				panic("kaboom")
			}, "kaboom"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, toolCallStream(tc.call), nil)
			if tc.handler != nil {
				h.svc.RegisterFunction("f", tc.handler)
			}
			h.generate(t, llm.NewContext())
			for _, side := range []*capture{h.down, h.up} {
				r := side.waitForType(t, resultFor(tc.call.ID), "result").(*frames.FunctionCallResultFrame)
				if !strings.Contains(r.Error, tc.wantErr) {
					t.Fatalf("%s: error = %q want %q", side.name, r.Error, tc.wantErr)
				}
				if m, ok := r.Result.(map[string]any); !ok || m["error"] != r.Error {
					t.Fatalf("result = %#v", r.Result)
				}
			}
			fs := h.drain(t)
			if indexOf(fs, cancelFor(tc.call.ID)) >= 0 || indexOf(h.up.all(), cancelFor(tc.call.ID)) >= 0 {
				t.Fatal("unexpected cancel frame")
			}
			if len(ofType[*frames.ErrorFrame](h.up.all())) != 0 {
				t.Fatal("function error escalated to ErrorFrame")
			}
		})
	}
}

func TestCatchAllFunction(t *testing.T) {
	h := newHarness(t, toolCallStream(llm.ToolCall{ID: "x", Name: "anything"}), nil)
	h.svc.RegisterFunction("", func(_ context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		return llm.FunctionResult{Value: p.FunctionName}, nil
	})
	h.generate(t, llm.NewContext())
	r := h.down.waitForType(t, resultFor("x"), "result").(*frames.FunctionCallResultFrame)
	if r.Result != "anything" {
		t.Fatalf("result = %v", r.Result)
	}
}

func TestFunctionTimeoutIgnoringCtx(t *testing.T) {
	cases := []struct {
		name string
		opts []Option
		fo   []llm.FunctionOptions
	}{
		{"service timeout", []Option{WithFunctionCallTimeout(50 * time.Millisecond)}, nil},
		{"per function timeout", nil, []llm.FunctionOptions{{CancelOnInterruption: true, Timeout: 50 * time.Millisecond}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			release := make(chan struct{})
			t.Cleanup(func() { close(release) })
			h := newHarness(t, toolCallStream(llm.ToolCall{ID: "t1", Name: "slow"}), nil, tc.opts...)
			h.svc.RegisterFunction("slow", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
				<-release
				return llm.FunctionResult{Value: "late"}, nil
			}, tc.fo...)
			begin := time.Now()
			h.generate(t, llm.NewContext())
			r := h.down.waitForType(t, resultFor("t1"), "timeout result").(*frames.FunctionCallResultFrame)
			if r.Error == "" || !strings.Contains(r.Error, "deadline") {
				t.Fatalf("error = %q", r.Error)
			}
			if el := time.Since(begin); el > time.Second {
				t.Fatalf("timeout took %v", el)
			}
			if indexOf(h.down.all(), cancelFor("t1")) >= 0 {
				t.Fatal("timeout produced cancel frame")
			}
		})
	}
}

func TestParallelFunctionCallsRunConcurrently(t *testing.T) {
	h := newHarness(t, toolCallStream(
		llm.ToolCall{ID: "a", Name: "f"},
		llm.ToolCall{ID: "b", Name: "f"},
	), nil)
	var wg sync.WaitGroup
	wg.Add(2)
	both := make(chan struct{})
	go func() { wg.Wait(); close(both) }()
	h.svc.RegisterFunction("f", func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		wg.Done()
		select {
		case <-both:
			return llm.FunctionResult{Value: p.ToolCallID}, nil
		case <-time.After(waitTimeout):
			return llm.FunctionResult{}, errors.New("not concurrent")
		}
	})
	h.generate(t, llm.NewContext())
	for _, id := range []string{"a", "b"} {
		r := h.down.waitForType(t, resultFor(id), "result "+id).(*frames.FunctionCallResultFrame)
		if r.Error != "" {
			t.Fatalf("%s: %s", id, r.Error)
		}
	}
	fs := h.down.all()
	started := indexOf(fs, is[*frames.FunctionCallsStartedFrame])
	ips := 0
	for _, f := range fs[started:] {
		if is[*frames.FunctionCallInProgressFrame](f) {
			ips++
		}
		if is[*frames.FunctionCallResultFrame](f) && ips != 2 {
			t.Fatalf("result before all in-progress frames: %v", names(fs))
		}
	}
}

func TestSequentialFunctionCallsRunInOrder(t *testing.T) {
	h := newHarness(t, toolCallStream(
		llm.ToolCall{ID: "a", Name: "f"},
		llm.ToolCall{ID: "b", Name: "f"},
		llm.ToolCall{ID: "c", Name: "f"},
	), nil, WithSequentialFunctionCalls())
	var mu sync.Mutex
	running := 0
	var order []string
	h.svc.RegisterFunction("f", func(_ context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		mu.Lock()
		running++
		order = append(order, p.ToolCallID)
		overlap := running > 1
		mu.Unlock()
		time.Sleep(5 * time.Millisecond)
		mu.Lock()
		running--
		mu.Unlock()
		if overlap {
			return llm.FunctionResult{}, errors.New("overlap")
		}
		return llm.FunctionResult{Value: p.ToolCallID}, nil
	})
	h.generate(t, llm.NewContext())
	fs := h.down.waitFor(t, "3 results", func(fs []frames.Frame) bool {
		return len(ofType[*frames.FunctionCallResultFrame](fs)) == 3
	})
	var got []string
	for _, r := range ofType[*frames.FunctionCallResultFrame](fs) {
		if r.Error != "" {
			t.Fatalf("%s: %s", r.ToolCallID, r.Error)
		}
		got = append(got, r.ToolCallID)
	}
	mu.Lock()
	defer mu.Unlock()
	if strings.Join(got, "") != "abc" || strings.Join(order, "") != "abc" {
		t.Fatalf("results %v execution %v", got, order)
	}
}

func TestInterruptionCancelsOnlyInterruptibleCalls(t *testing.T) {
	h := newHarness(t, toolCallStream(
		llm.ToolCall{ID: "soft", Name: "soft"},
		llm.ToolCall{ID: "hard", Name: "hard"},
	), nil)
	releaseHard := make(chan struct{})
	softCtxDone := make(chan struct{})
	var startedWG sync.WaitGroup
	startedWG.Add(2)
	h.svc.RegisterFunction("soft", func(ctx context.Context, _ llm.FunctionCallParams) (llm.FunctionResult, error) {
		startedWG.Done()
		<-ctx.Done()
		close(softCtxDone)
		return llm.FunctionResult{}, ctx.Err()
	}, llm.FunctionOptions{CancelOnInterruption: true})
	h.svc.RegisterFunction("hard", func(ctx context.Context, _ llm.FunctionCallParams) (llm.FunctionResult, error) {
		startedWG.Done()
		select {
		case <-releaseHard:
			return llm.FunctionResult{Value: "kept"}, nil
		case <-ctx.Done():
			return llm.FunctionResult{}, ctx.Err()
		}
	}, llm.FunctionOptions{CancelOnInterruption: false})
	h.generate(t, llm.NewContext())
	startedWG.Wait()
	h.queue(t, frames.NewInterruptionFrame())
	for _, side := range []*capture{h.down, h.up} {
		side.waitForType(t, cancelFor("soft"), "soft cancel")
	}
	select {
	case <-softCtxDone:
	case <-time.After(waitTimeout):
		t.Fatal("soft handler ctx not cancelled")
	}
	close(releaseHard)
	for _, side := range []*capture{h.down, h.up} {
		r := side.waitForType(t, resultFor("hard"), "hard result").(*frames.FunctionCallResultFrame)
		if r.Result != "kept" || r.Error != "" {
			t.Fatalf("hard result = %+v", r)
		}
	}
	all := append(h.down.all(), h.up.all()...)
	if indexOf(all, cancelFor("hard")) >= 0 || indexOf(all, resultFor("soft")) >= 0 {
		t.Fatalf("wrong frames: %v", names(all))
	}
}

func TestDefaultFunctionSurvivesInterruption(t *testing.T) {
	h := newHarness(t, toolCallStream(llm.ToolCall{ID: "c1", Name: "book"}), nil)
	entered := make(chan struct{})
	release := make(chan struct{})
	h.svc.RegisterFunction("book", func(ctx context.Context, _ llm.FunctionCallParams) (llm.FunctionResult, error) {
		close(entered)
		select {
		case <-release:
			return llm.FunctionResult{Value: "booked"}, nil
		case <-ctx.Done():
			return llm.FunctionResult{}, ctx.Err()
		}
	})
	h.generate(t, llm.NewContext())
	select {
	case <-entered:
	case <-time.After(waitTimeout):
		t.Fatal("handler never started")
	}
	h.queue(t, frames.NewInterruptionFrame())
	h.down.waitForType(t, is[*frames.InterruptionFrame], "interruption")
	h.queue(t, frames.NewInterruptionFrame())
	h.down.waitFor(t, "second interruption", func(fs []frames.Frame) bool {
		return len(ofType[*frames.InterruptionFrame](fs)) == 2
	})
	close(release)
	for _, side := range []*capture{h.down, h.up} {
		r := side.waitForType(t, resultFor("c1"), "result").(*frames.FunctionCallResultFrame)
		if r.Result != "booked" || r.Error != "" {
			t.Fatalf("%s result = %+v", side.name, r)
		}
	}
	all := append(h.drain(t), h.up.all()...)
	if indexOf(all, cancelFor("c1")) >= 0 {
		t.Fatalf("default function was cancelled by interruption: %v", names(all))
	}
}

func TestCancelledCallSettlesOnceWhenHandlerFinishesLate(t *testing.T) {
	h := newHarness(t, toolCallStream(llm.ToolCall{ID: "c1", Name: "book"}), nil)
	entered := make(chan struct{})
	release := make(chan struct{})
	returned := make(chan struct{})
	h.svc.RegisterFunction("book", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
		defer close(returned)
		close(entered)
		<-release
		return llm.FunctionResult{Value: "booked"}, nil
	}, llm.FunctionOptions{CancelOnInterruption: true})
	h.generate(t, llm.NewContext())
	select {
	case <-entered:
	case <-time.After(waitTimeout):
		t.Fatal("handler never started")
	}
	h.queue(t, frames.NewInterruptionFrame())
	for _, side := range []*capture{h.down, h.up} {
		side.waitForType(t, cancelFor("c1"), "cancel")
	}
	close(release)
	select {
	case <-returned:
	case <-time.After(waitTimeout):
		t.Fatal("handler never returned")
	}
	h.svc.callsWG.Wait()
	down := h.drain(t)
	for _, side := range [][]frames.Frame{down, h.up.all()} {
		cancels := 0
		for _, f := range side {
			if cancelFor("c1")(f) {
				cancels++
			}
		}
		if cancels != 1 || indexOf(side, resultFor("c1")) >= 0 {
			t.Fatalf("expected one cancel and no result, got %v", names(side))
		}
	}
}

func TestStopReturnsPromptlyWithStuckHandler(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	entered := make(chan struct{})
	h := newHarness(t, toolCallStream(llm.ToolCall{ID: "s", Name: "stuck"}), nil)
	h.svc.RegisterFunction("stuck", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
		close(entered)
		<-release
		return llm.FunctionResult{}, nil
	}, llm.FunctionOptions{CancelOnInterruption: false})
	h.generate(t, llm.NewContext())
	select {
	case <-entered:
	case <-time.After(waitTimeout):
		t.Fatal("handler never started")
	}
	begin := time.Now()
	if err := h.svc.Stop(); err != nil {
		t.Fatal(err)
	}
	if el := time.Since(begin); el > time.Second {
		t.Fatalf("Stop took %v", el)
	}
}

type namedRef string

func (n namedRef) Name() string { return string(n) }

func TestUpdateSettingsFrame(t *testing.T) {
	cases := []struct {
		name        string
		target      frames.ProcessorRef
		wantModel   string
		wantForward bool
	}{
		{"untargeted", nil, "m2", false},
		{"targeted at self", namedRef("fake"), "m2", false},
		{"targeted at other", namedRef("other"), "m1", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var seen llm.Settings
			h := newHarness(t, func(_ context.Context, _ llm.Snapshot, s llm.Settings, _ func(llm.Event)) error {
				seen = s
				return nil
			}, nil, WithSettings(llm.Settings{Model: "m1", Extra: map[string]any{"a": 1}}))
			f := frames.NewLLMUpdateSettingsFrame(llm.Settings{Model: "m2", Extra: map[string]any{"b": 2}})
			f.Service = tc.target
			h.queue(t, f)
			h.generate(t, llm.NewContext())
			fs := h.drain(t)
			if seen.Model != tc.wantModel || h.svc.Settings().Model != tc.wantModel {
				t.Fatalf("model seen=%q settings=%q want %q", seen.Model, h.svc.Settings().Model, tc.wantModel)
			}
			if !tc.wantForward && seen.Extra["a"] != 1 {
				t.Fatalf("extra lost: %v", seen.Extra)
			}
			if forwarded := indexOf(fs, is[*frames.LLMUpdateSettingsFrame]) >= 0; forwarded != tc.wantForward {
				t.Fatalf("forwarded=%v want %v", forwarded, tc.wantForward)
			}
		})
	}
}

func TestRunInference(t *testing.T) {
	var seen llm.Settings
	var seenSnap llm.Snapshot
	svc := New(&fakeProvider{name: "p", fn: func(_ context.Context, snap llm.Snapshot, s llm.Settings, emit func(llm.Event)) error {
		seen, seenSnap = s, snap
		emit(llm.ThoughtDelta{Text: "hmm"})
		emit(llm.TextDelta{Text: "foo"})
		emit(llm.UsageEvent{})
		emit(llm.TextDelta{Text: "bar"})
		return nil
	}}, WithSettings(llm.Settings{Model: "base", SystemInstruction: "sys"}))
	snap := llm.NewContext(llm.User("q")).Snapshot()
	out, err := svc.RunInference(context.Background(), snap, llm.Settings{Model: "override"})
	if err != nil || out != "foobar" {
		t.Fatalf("out=%q err=%v", out, err)
	}
	if seen.Model != "override" || seen.SystemInstruction != "sys" || len(seenSnap.Messages) != 1 {
		t.Fatalf("settings=%+v snap=%+v", seen, seenSnap)
	}
	if svc.Settings().Model != "base" {
		t.Fatal("RunInference mutated service settings")
	}

	boom := errors.New("boom")
	failing := New(&fakeProvider{name: "p", fn: func(_ context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
		emit(llm.TextDelta{Text: "partial"})
		return boom
	}})
	out, err = failing.RunInference(context.Background(), snap, llm.Settings{})
	if !errors.Is(err, boom) || out != "partial" {
		t.Fatalf("out=%q err=%v", out, err)
	}
}

func TestThoughtFrames(t *testing.T) {
	h := newHarness(t, scripted(
		llm.ThoughtDelta{Text: "think"},
		llm.ThoughtDelta{Signature: "sig"},
		llm.TextDelta{Text: "answer"},
	), nil)
	h.generate(t, llm.NewContext())
	got := names(h.drain(t))
	want := "StartFrame,LLMFullResponseStartFrame,LLMThoughtStartFrame,LLMThoughtTextFrame,LLMThoughtEndFrame,LLMTextFrame(answer),LLMFullResponseEndFrame,EndFrame"
	if strings.Join(got, ",") != want {
		t.Fatalf("got %v", got)
	}
	end := ofType[*frames.LLMThoughtEndFrame](h.down.all())[0]
	if end.Signature != "sig" {
		t.Fatalf("signature = %q", end.Signature)
	}
}

func TestTTFATMetricsAfterThinking(t *testing.T) {
	ttfat := func(fs []frames.Frame) []frames.TTFATMetricsData {
		var out []frames.TTFATMetricsData
		for _, m := range ofType[*frames.MetricsFrame](fs) {
			for _, d := range m.Data {
				if v, ok := d.(frames.TTFATMetricsData); ok {
					out = append(out, v)
				}
			}
		}
		return out
	}
	start := frames.NewStartFrame()
	start.EnableMetrics = true

	h := newHarness(t, func(_ context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
		emit(llm.ThoughtDelta{Text: "think"})
		time.Sleep(20 * time.Millisecond)
		emit(llm.TextDelta{Text: "answer"})
		emit(llm.TextDelta{Text: " more"})
		return nil
	}, start, WithSettings(llm.Settings{Model: "mod"}))
	h.generate(t, llm.NewContext(llm.User("hi")))
	got := ttfat(h.drain(t))
	if len(got) != 1 {
		t.Fatalf("ttfat = %+v", got)
	}
	m := got[0]
	if m.Processor != "fake" || m.Model != "mod" || m.ThinkingTime < 20*time.Millisecond || m.TTFAT != m.TTFB+m.ThinkingTime {
		t.Fatalf("ttfat = %+v", m)
	}

	plain := newHarness(t, scripted(llm.TextDelta{Text: "a"}), start)
	plain.generate(t, llm.NewContext(llm.User("hi")))
	if got := ttfat(plain.drain(t)); len(got) != 0 {
		t.Fatalf("response without thinking reported ttfat: %+v", got)
	}
}

func TestRefusalBecomesTypedNonFatalError(t *testing.T) {
	cases := []struct {
		name string
		fn   streamFunc
	}{
		{"provider returns typed error", func(_ context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
			r := llm.Refusal{Category: llm.RefusalContentFilter, Provider: "fake", Reason: "content_filter", Detail: "hate"}
			emit(llm.TextDelta{Text: "partial"})
			emit(r)
			emit(llm.Done{StopReason: llm.StopReasonContentFilter, RawStopReason: "content_filter"})
			return &llm.RefusalError{Refusal: r}
		}},
		{"provider only emits the event", scripted(
			llm.Refusal{Category: llm.RefusalContentFilter, Provider: "fake", Reason: "content_filter", Detail: "hate"},
			llm.Done{StopReason: llm.StopReasonContentFilter},
		)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, tc.fn, nil)
			h.generate(t, llm.NewContext(llm.User("hi")))
			ef := h.up.waitForType(t, is[*frames.ErrorFrame], "ErrorFrame").(*frames.ErrorFrame)
			r, ok := llm.AsRefusal(ef.Error)
			if ef.Fatal || !ok || r.Category != llm.RefusalContentFilter || r.Detail != "hate" {
				t.Fatalf("error frame = %+v refusal = %+v ok=%v", ef, r, ok)
			}
			if !strings.Contains(ef.Error.Error(), "response refused") {
				t.Fatalf("error text = %q", ef.Error)
			}
			fs := h.drain(t)
			if len(ofType[*frames.LLMFullResponseEndFrame](fs)) != 1 {
				t.Fatalf("frames %v", names(fs))
			}
		})
	}
}

func TestResponseMetricsCarryIDsAndStopReason(t *testing.T) {
	start := frames.NewStartFrame()
	start.EnableMetrics = true
	h := newHarness(t, scripted(
		llm.ResponseInfo{ResponseID: "resp_1", Model: "served-model"},
		llm.TextDelta{Text: "a"},
		llm.ResponseInfo{RequestID: "req_1"},
		llm.Done{StopReason: llm.StopReasonLength, RawStopReason: "max_tokens"},
	), start, WithSettings(llm.Settings{Model: "asked-model"}))
	h.generate(t, llm.NewContext(llm.User("hi")))
	fs := h.drain(t)
	var got []frames.LLMResponseMetricsData
	for _, m := range ofType[*frames.MetricsFrame](fs) {
		for _, d := range m.Data {
			if r, ok := d.(frames.LLMResponseMetricsData); ok {
				got = append(got, r)
			}
		}
	}
	want := frames.LLMResponseMetricsData{
		MetricsBase:   frames.MetricsBase{Processor: "fake", Model: "asked-model"},
		ResponseID:    "resp_1",
		RequestID:     "req_1",
		ServedModel:   "served-model",
		StopReason:    llm.StopReasonLength,
		RawStopReason: "max_tokens",
	}
	if len(got) != 1 || got[0] != want {
		t.Fatalf("response metrics = %+v", got)
	}
	if up := h.up.all(); len(up) != 0 {
		t.Fatalf("truncation must not raise an error frame: %v", names(up))
	}
}

func TestToolCallSignatureReachesInProgressFrame(t *testing.T) {
	h := newHarness(t, toolCallStream(llm.ToolCall{ID: "c1", Name: "weather", Arguments: []byte(`{}`), Signature: "c2ln"}), nil)
	h.svc.RegisterFunction("weather", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
		return llm.FunctionResult{Value: "ok"}, nil
	})
	h.generate(t, llm.NewContext(llm.User("hi")))
	for _, side := range []*capture{h.down, h.up} {
		ip := side.waitForType(t, is[*frames.FunctionCallInProgressFrame], "in progress").(*frames.FunctionCallInProgressFrame)
		if ip.Signature != "c2ln" {
			t.Fatalf("%s signature = %q", side.name, ip.Signature)
		}
	}
}
