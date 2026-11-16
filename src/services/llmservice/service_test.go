package llmservice

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
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
	h.up.waitForType(t, is[*frames.LLMServiceMetadataFrame], "LLMServiceMetadataFrame")
	h.up.mu.Lock()
	h.up.frames = nil
	h.up.mu.Unlock()
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

type httpStatusError struct{ status int }

func (e *httpStatusError) Error() string       { return fmt.Sprintf("HTTP %d", e.status) }
func (e *httpStatusError) HTTPStatusCode() int { return e.status }

func TestProviderErrorCarriesCategoryFromHTTPStatus(t *testing.T) {
	cases := []struct {
		status int
		want   frames.ErrorCategory
	}{
		{http.StatusUnauthorized, frames.ErrorCategoryAuthentication},
		{http.StatusTooManyRequests, frames.ErrorCategoryRateLimit},
	}
	for _, tc := range cases {
		t.Run(http.StatusText(tc.status), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "rejected", tc.status)
			}))
			defer srv.Close()
			h := newHarness(t, func(ctx context.Context, _ llm.Snapshot, _ llm.Settings, _ func(llm.Event)) error {
				req, err := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL, nil)
				if err != nil {
					return err
				}
				resp, err := srv.Client().Do(req)
				if err != nil {
					return err
				}
				defer resp.Body.Close()
				if resp.StatusCode != http.StatusOK {
					return fmt.Errorf("fake: %w", &httpStatusError{status: resp.StatusCode})
				}
				return nil
			}, nil)
			h.generate(t, llm.NewContext(llm.User("hi")))
			ef := h.up.waitForType(t, is[*frames.ErrorFrame], "ErrorFrame").(*frames.ErrorFrame)
			if ef.Fatal || ef.Category != tc.want {
				t.Fatalf("error frame fatal=%v category=%q, want %q", ef.Fatal, ef.Category, tc.want)
			}
			if ef.Processor == "" || !strings.Contains(ef.Error.Error(), "generation failed") {
				t.Fatalf("error frame = %+v", ef)
			}
		})
	}
}

func TestUnclassifiedProviderErrorIsUnknown(t *testing.T) {
	h := newHarness(t, func(context.Context, llm.Snapshot, llm.Settings, func(llm.Event)) error {
		return errors.New("boom")
	}, nil)
	h.generate(t, llm.NewContext(llm.User("hi")))
	ef := h.up.waitForType(t, is[*frames.ErrorFrame], "ErrorFrame").(*frames.ErrorFrame)
	if ef.Category != frames.ErrorCategoryUnknown {
		t.Fatalf("category = %q, want unknown", ef.Category)
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
			if ef.Category != frames.ErrorCategoryContentFiltered {
				t.Fatalf("category = %q, want content_filtered", ef.Category)
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

func TestAnnouncesCascadedLLMAtStartAndOnSwitch(t *testing.T) {
	svc := New(&fakeProvider{name: "fake"})
	up, down := newCapture("up"), newCapture("down")
	svc.SetPrev(up)
	svc.Link(down)
	if err := svc.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = svc.Stop() })
	_ = svc.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = svc.QueueFrame(frames.NewServiceSwitcherRequestMetadataFrame(svc), frames.Downstream)
	fs := up.waitFor(t, "two announcements", func(fs []frames.Frame) bool {
		n := 0
		for _, f := range fs {
			if m, ok := f.(*frames.LLMServiceMetadataFrame); ok && !m.IsRealtimeService && m.ServiceName == svc.Name() {
				n++
			}
		}
		return n == 2
	})
	if len(fs) != 2 {
		t.Fatalf("upstream = %v", names(fs))
	}
}

type countedStream struct {
	mu    sync.Mutex
	calls int
	done  chan struct{}
}

func newCountedStream() *countedStream {
	return &countedStream{done: make(chan struct{}, 8)}
}

func (c *countedStream) wrap(fn streamFunc) streamFunc {
	return func(ctx context.Context, snap llm.Snapshot, s llm.Settings, emit func(llm.Event)) error {
		defer func() { c.done <- struct{}{} }()
		c.mu.Lock()
		c.calls++
		c.mu.Unlock()
		return fn(ctx, snap, s, emit)
	}
}

func (c *countedStream) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

func (c *countedStream) wait(t *testing.T) {
	t.Helper()
	select {
	case <-c.done:
	case <-time.After(waitTimeout):
		t.Fatal("provider was not called")
	}
}

func speculativeFrame(c *llm.Context) *frames.LLMContextFrame {
	f := frames.NewLLMContextFrame(c)
	f.Speculation = true
	return f
}

func responseFrames(fs []frames.Frame) int {
	n := 0
	for _, f := range fs {
		switch f.(type) {
		case *frames.LLMFullResponseStartFrame, *frames.LLMTextFrame, *frames.LLMFullResponseEndFrame:
			n++
		}
	}
	return n
}

func assertNoResponseYet(t *testing.T, h *harness) {
	t.Helper()
	time.Sleep(30 * time.Millisecond)
	if n := responseFrames(h.down.all()); n != 0 {
		t.Fatalf("speculative response leaked before confirmation: %v", names(h.down.all()))
	}
}

func TestSpeculativeResponseHeldUntilConfirmed(t *testing.T) {
	stream := newCountedStream()
	h := newHarness(t, stream.wrap(scripted(llm.TextDelta{Text: "Sure."})), nil)
	spec := llm.NewContext(llm.User("book a table"))
	h.queue(t, speculativeFrame(spec))
	stream.wait(t)
	assertNoResponseYet(t, h)

	confirm := frames.NewUserTurnSpeculationConfirmedFrame(spec, "Book a table.")
	h.queue(t, confirm)
	h.down.waitForType(t, is[*frames.LLMFullResponseEndFrame], "released response")
	fs := h.drain(t)
	ci := indexOf(fs, func(f frames.Frame) bool { return f == frames.Frame(confirm) })
	si := indexOf(fs, is[*frames.LLMFullResponseStartFrame])
	ti := indexOf(fs, is[*frames.LLMTextFrame])
	ei := indexOf(fs, is[*frames.LLMFullResponseEndFrame])
	if !(ci >= 0 && ci < si && si < ti && ti < ei) {
		t.Fatalf("order: %v", names(fs))
	}
	if stream.count() != 1 {
		t.Fatalf("LLM ran %d times, want 1", stream.count())
	}
}

func TestSpeculationConfirmedBeforeInferenceIsNotHeld(t *testing.T) {
	stream := newCountedStream()
	h := newHarness(t, stream.wrap(scripted(llm.TextDelta{Text: "Sure."})), nil)
	spec := llm.NewContext(llm.User("book a table"))
	h.queue(t, frames.NewUserTurnSpeculationConfirmedFrame(spec, "book a table"))
	h.down.waitForType(t, is[*frames.UserTurnSpeculationConfirmedFrame], "confirmation")
	h.queue(t, speculativeFrame(spec))
	h.down.waitForType(t, is[*frames.LLMFullResponseEndFrame], "released response")
}

func TestSpeculativeResponseDiscardedOnCancel(t *testing.T) {
	stream := newCountedStream()
	h := newHarness(t, stream.wrap(scripted(llm.TextDelta{Text: "Sure."})), nil)
	spec := llm.NewContext(llm.User("book a table"))
	h.queue(t, speculativeFrame(spec))
	stream.wait(t)
	h.queue(t, frames.NewUserTurnResponseCancelFrame())
	h.down.waitForType(t, is[*frames.UserTurnResponseCancelFrame], "cancel")
	h.queue(t, frames.NewUserTurnSpeculationConfirmedFrame(spec, "book a table"))
	assertNoResponseYet(t, h)

	h.generate(t, llm.NewContext(llm.User("book a table for six")))
	stream.wait(t)
	fs := h.drain(t)
	if n := len(ofType[*frames.LLMTextFrame](fs)); n != 1 {
		t.Fatalf("want only the committed answer, got %v", names(fs))
	}
}

func TestSpeculativeCancelStopsGeneration(t *testing.T) {
	started := make(chan struct{})
	stopped := make(chan struct{})
	h := newHarness(t, func(ctx context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
		emit(llm.TextDelta{Text: "partial"})
		close(started)
		<-ctx.Done()
		emit(llm.TextDelta{Text: "late"})
		close(stopped)
		return ctx.Err()
	}, nil)
	h.queue(t, speculativeFrame(llm.NewContext(llm.User("book"))))
	<-started
	h.queue(t, frames.NewUserTurnResponseCancelFrame())
	select {
	case <-stopped:
	case <-time.After(waitTimeout):
		t.Fatal("speculative generation kept running after the cancel")
	}
	fs := h.drain(t)
	if n := responseFrames(fs); n != 0 {
		t.Fatalf("discarded response reached downstream: %v", names(fs))
	}
}

func TestSpeculativeToolCallWithdrawsSpeculation(t *testing.T) {
	stream := newCountedStream()
	h := newHarness(t, stream.wrap(toolCallStream(llm.ToolCall{ID: "c1", Name: "book", Arguments: []byte(`{}`)})), nil)
	var mu sync.Mutex
	ran := 0
	h.svc.RegisterFunction("book", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
		mu.Lock()
		ran++
		mu.Unlock()
		return llm.FunctionResult{Value: "booked"}, nil
	})
	spec := llm.NewContext(llm.User("book a table"))
	h.queue(t, speculativeFrame(spec))
	h.up.waitForType(t, is[*frames.UserTurnResponseCancelFrame], "withdrawal upstream")
	h.queue(t, frames.NewUserTurnSpeculationConfirmedFrame(spec, "book a table"))
	fs := h.drain(t)
	if responseFrames(fs) != 0 || len(ofType[*frames.FunctionCallsStartedFrame](fs)) != 0 || len(ofType[*frames.FunctionCallInProgressFrame](fs)) != 0 {
		t.Fatalf("speculative tool call leaked: %v", names(fs))
	}
	if indexOf(fs, is[*frames.UserTurnResponseCancelFrame]) < 0 {
		t.Fatalf("no withdrawal downstream: %v", names(fs))
	}
	mu.Lock()
	defer mu.Unlock()
	if ran != 0 {
		t.Fatalf("speculative tool call executed %d times", ran)
	}
}

func TestConfirmedSpeculationRunsToolCalls(t *testing.T) {
	release := make(chan struct{})
	h := newHarness(t, func(ctx context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
		<-release
		emit(llm.ToolCall{ID: "c1", Name: "book", Arguments: []byte(`{}`)})
		return nil
	}, nil)
	h.svc.RegisterFunction("book", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
		return llm.FunctionResult{Value: "booked"}, nil
	})
	spec := llm.NewContext(llm.User("book a table"))
	h.queue(t, speculativeFrame(spec))
	h.queue(t, frames.NewUserTurnSpeculationConfirmedFrame(spec, "book a table"))
	h.down.waitForType(t, is[*frames.UserTurnSpeculationConfirmedFrame], "confirmation")
	close(release)
	h.up.waitForType(t, resultFor("c1"), "tool result")
	if indexOf(h.up.all(), is[*frames.UserTurnResponseCancelFrame]) >= 0 {
		t.Fatal("a confirmed speculation must not be withdrawn")
	}
}

func TestInterruptionDiscardsHeldResponse(t *testing.T) {
	stream := newCountedStream()
	h := newHarness(t, stream.wrap(scripted(llm.TextDelta{Text: "Sure."})), nil)
	spec := llm.NewContext(llm.User("book a table"))
	h.queue(t, speculativeFrame(spec))
	stream.wait(t)
	assertNoResponseYet(t, h)
	h.queue(t, frames.NewInterruptionFrame())
	h.down.waitForType(t, is[*frames.InterruptionFrame], "interruption")
	h.queue(t, frames.NewUserTurnSpeculationConfirmedFrame(spec, "book a table"))
	fs := h.drain(t)
	if n := responseFrames(fs); n != 0 {
		t.Fatalf("held response survived the interruption: %v", names(fs))
	}
}

func TestNewInferenceSupersedesHeldResponse(t *testing.T) {
	stream := newCountedStream()
	h := newHarness(t, stream.wrap(scripted(llm.TextDelta{Text: "Sure."})), nil)
	h.queue(t, speculativeFrame(llm.NewContext(llm.User("book a table"))))
	stream.wait(t)
	h.generate(t, llm.NewContext(llm.User("book a table for six")))
	stream.wait(t)
	fs := h.drain(t)
	if n := len(ofType[*frames.LLMFullResponseStartFrame](fs)); n != 1 {
		t.Fatalf("want only the new response, got %v", names(fs))
	}
	var withdrawn *frames.UserTurnResponseCancelFrame
	for _, f := range h.up.all() {
		if c, ok := f.(*frames.UserTurnResponseCancelFrame); ok {
			withdrawn = c
		}
	}
	if withdrawn == nil || withdrawn.Context == nil {
		t.Fatalf("superseding a held speculation must tell the user aggregator: %v", names(h.up.all()))
	}
}

func TestPassThroughFramesAreNotHeldDuringSpeculation(t *testing.T) {
	stream := newCountedStream()
	h := newHarness(t, stream.wrap(scripted(llm.TextDelta{Text: "Sure."})), nil)
	spec := llm.NewContext(llm.User("book a table"))
	h.queue(t, speculativeFrame(spec))
	stream.wait(t)
	speak := frames.NewTTSSpeakFrame("please hold")
	h.queue(t, speak)
	h.down.waitForType(t, func(f frames.Frame) bool { return f == frames.Frame(speak) }, "pass-through TTSSpeakFrame")
	assertNoResponseYet(t, h)
	cancel := frames.NewUserTurnResponseCancelFrame()
	cancel.Context = spec
	h.queue(t, cancel)
	h.down.waitForType(t, is[*frames.UserTurnResponseCancelFrame], "cancel")
	fs := h.drain(t)
	if n := len(ofType[*frames.TTSSpeakFrame](fs)); n != 1 || responseFrames(fs) != 0 {
		t.Fatalf("TTSSpeakFrame delivered %d times, response frames %d: %v", n, responseFrames(fs), names(fs))
	}
}

func TestStopFrameDiscardsHeldSpeculation(t *testing.T) {
	stream := newCountedStream()
	h := newHarness(t, stream.wrap(scripted(llm.TextDelta{Text: "Sure."})), nil)
	h.queue(t, speculativeFrame(llm.NewContext(llm.User("book a table"))))
	stream.wait(t)
	assertNoResponseYet(t, h)
	h.queue(t, frames.NewStopFrame())
	fs := h.down.waitFor(t, "StopFrame", func(fs []frames.Frame) bool { return indexOf(fs, is[*frames.StopFrame]) >= 0 })
	if responseFrames(fs) != 0 {
		t.Fatalf("held response released by StopFrame: %v", names(fs))
	}
}

func TestCancelBeforeSpeculativeFrameSkipsIt(t *testing.T) {
	for _, named := range []bool{true, false} {
		t.Run(map[bool]string{true: "cancel names the speculation", false: "context-less cancel"}[named], func(t *testing.T) {
			release := make(chan struct{})
			var mu sync.Mutex
			calls := 0
			h := newHarness(t, func(ctx context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
				mu.Lock()
				calls++
				n := calls
				mu.Unlock()
				if n == 1 {
					<-release
				}
				emit(llm.TextDelta{Text: "x"})
				return nil
			}, nil)
			h.generate(t, llm.NewContext(llm.User("earlier")))
			time.Sleep(30 * time.Millisecond)
			spec := llm.NewContext(llm.User("book"))
			h.queue(t, speculativeFrame(spec))
			cancel := frames.NewUserTurnResponseCancelFrame()
			if named {
				cancel.Context = spec
			}
			h.queue(t, cancel)
			h.down.waitForType(t, is[*frames.UserTurnResponseCancelFrame], "cancel")
			close(release)
			up := h.up.waitForType(t, func(f frames.Frame) bool {
				c, ok := f.(*frames.UserTurnResponseCancelFrame)
				return ok && c.Context == spec
			}, "skipped speculation reported upstream")
			if up == nil {
				t.Fatal("no upstream cancel")
			}
			speak := frames.NewTTSSpeakFrame("unrelated")
			h.queue(t, speak)
			h.down.waitForType(t, func(f frames.Frame) bool { return f == frames.Frame(speak) }, "unrelated TTSSpeakFrame")
			h.drain(t)
			mu.Lock()
			defer mu.Unlock()
			if calls != 1 {
				t.Fatalf("withdrawn speculation still ran: %d inferences", calls)
			}
		})
	}
}

func TestSpeculativeMetricsFollowTheVerdict(t *testing.T) {
	for _, confirm := range []bool{true, false} {
		t.Run(map[bool]string{true: "confirmed", false: "withdrawn"}[confirm], func(t *testing.T) {
			start := frames.NewStartFrame()
			start.EnableMetrics = true
			start.EnableUsageMetrics = true
			stream := newCountedStream()
			h := newHarness(t, stream.wrap(scripted(llm.TextDelta{Text: "Sure."}, llm.UsageEvent{Usage: llm.Usage{PromptTokens: 7}})), start)
			spec := llm.NewContext(llm.User("book a table"))
			h.queue(t, speculativeFrame(spec))
			stream.wait(t)
			time.Sleep(50 * time.Millisecond)
			if ttfb, processing, usage, _, _ := metricsData(h.down.all()); ttfb+processing+usage != 0 {
				t.Fatalf("metrics of an unconfirmed speculation leaked: %v", names(h.down.all()))
			}
			if confirm {
				h.queue(t, frames.NewUserTurnSpeculationConfirmedFrame(spec, "book a table"))
				h.down.waitForType(t, is[*frames.LLMFullResponseEndFrame], "released response")
			} else {
				cancel := frames.NewUserTurnResponseCancelFrame()
				cancel.Context = spec
				h.queue(t, cancel)
				h.down.waitForType(t, is[*frames.UserTurnResponseCancelFrame], "cancel")
			}
			fs := h.drain(t)
			ttfb, processing, usage, _, _ := metricsData(fs)
			want := [3]int{0, 0, 1}
			if confirm {
				want = [3]int{1, 1, 1}
			}
			if got := [3]int{ttfb, processing, usage}; got != want {
				t.Fatalf("ttfb/processing/usage = %v, want %v: %v", got, want, names(fs))
			}
		})
	}
}

func TestSpeculativeToolCallNeverRunsAfterDiscard(t *testing.T) {
	var ran atomic.Int32
	for i := 0; i < 100; i++ {
		ended := make(chan struct{})
		h := newHarness(t, func(ctx context.Context, _ llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
			emit(llm.ToolCall{ID: "c1", Name: "book", Arguments: []byte(`{}`)})
			close(ended)
			return nil
		}, nil)
		h.svc.RegisterFunction("book", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
			ran.Add(1)
			return llm.FunctionResult{Value: "booked"}, nil
		})
		spec := llm.NewContext(llm.User("book"))
		h.queue(t, speculativeFrame(spec))
		<-ended
		cancel := frames.NewUserTurnResponseCancelFrame()
		cancel.Context = spec
		h.queue(t, cancel)
		fs := h.drain(t)
		if len(ofType[*frames.FunctionCallInProgressFrame](fs)) != 0 {
			t.Fatalf("iteration %d: discarded speculative tool call dispatched: %v", i, names(fs))
		}
		_ = h.svc.Stop()
	}
	if n := ran.Load(); n != 0 {
		t.Fatalf("discarded speculative tool call ran %d times", n)
	}
}

type turnScript struct {
	mu    sync.Mutex
	tools [][]string
	turns [][]llm.Event
}

func (s *turnScript) add(events ...llm.Event) {
	s.mu.Lock()
	s.turns = append(s.turns, events)
	s.mu.Unlock()
}

func (s *turnScript) stream(_ context.Context, snap llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
	s.mu.Lock()
	var names []string
	for _, f := range snap.Tools.Functions {
		names = append(names, f.Name)
	}
	s.tools = append(s.tools, names)
	var events []llm.Event
	if len(s.turns) > 0 {
		events, s.turns = s.turns[0], s.turns[1:]
	}
	s.mu.Unlock()
	for _, e := range events {
		emit(e)
	}
	return nil
}

func (s *turnScript) toolsAt(i int) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.tools[i]
}

func (h *harness) generateAndWait(t *testing.T, c *llm.Context, n int) {
	t.Helper()
	h.generate(t, c)
	h.down.waitFor(t, fmt.Sprintf("%d responses", n), func(fs []frames.Frame) bool {
		return len(ofType[*frames.LLMFullResponseEndFrame](fs)) >= n
	})
}

func blockingHandler(entered chan<- string, cancelled chan<- string) llm.FunctionHandler {
	return func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		entered <- p.ToolCallID
		<-ctx.Done()
		cancelled <- p.ToolCallID
		return llm.FunctionResult{}, ctx.Err()
	}
}

func receive(t *testing.T, ch <-chan string, what string) string {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(waitTimeout):
		t.Fatalf("timed out waiting for %s", what)
	}
	return ""
}

func TestCancelToolOfferedOnlyWhileCallRuns(t *testing.T) {
	script := &turnScript{}
	h := newHarness(t, script.stream, nil)
	entered, cancelled := make(chan string, 4), make(chan string, 4)
	h.svc.RegisterFunction("report", blockingHandler(entered, cancelled), llm.FunctionOptions{CancellableByLLM: true})
	h.svc.RegisterFunction("lookup", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
		return llm.FunctionResult{Value: "ok"}, nil
	})
	c := llm.NewContext(llm.User("write the report"))
	c.SetTools(llm.Tools{Functions: []llm.FunctionSchema{{Name: "report"}, {Name: "lookup"}}})

	script.add()
	h.generateAndWait(t, c, 1)
	if got := script.toolsAt(0); len(got) != 2 {
		t.Fatalf("cancel tool offered before any call ran: %v", got)
	}

	script.add(llm.ToolCall{ID: "c1", Name: "report"})
	h.generateAndWait(t, c, 2)
	if id := receive(t, entered, "report to start"); id != "c1" {
		t.Fatalf("entered %s", id)
	}

	script.add()
	h.generateAndWait(t, c, 3)
	if got := script.toolsAt(2); strings.Join(got, ",") != "report,lookup,cancel_report" {
		t.Fatalf("tools while report runs = %v", got)
	}
	if n := len(c.Tools().Functions); n != 2 {
		t.Fatalf("service changed the context's tools: %d", n)
	}

	script.add(llm.ToolCall{ID: "c2", Name: "cancel_report"})
	h.generateAndWait(t, c, 4)
	if id := receive(t, cancelled, "report handler to see cancellation"); id != "c1" {
		t.Fatalf("cancelled %s", id)
	}
	for _, side := range []*capture{h.down, h.up} {
		fs := side.waitFor(t, "cancel tool result", func(fs []frames.Frame) bool { return indexOf(fs, resultFor("c2")) >= 0 })
		ci := indexOf(fs, cancelFor("c1"))
		ri := indexOf(fs, resultFor("c2"))
		if ci < 0 || ci > ri {
			t.Fatalf("%s: cancelled call must settle before the cancel tool answers: %v", side.name, names(fs))
		}
		if got := fs[ci].(*frames.FunctionCallCancelFrame); got.Result != llm.ToolResultCancelled || got.RunLLM {
			t.Fatalf("cancel frame = %+v", got)
		}
		value, _ := fs[ri].(*frames.FunctionCallResultFrame).Result.(map[string]any)
		if value["cancelled"] != "c1" || value["function_name"] != "report" {
			t.Fatalf("cancel tool result = %v", value)
		}
		if indexOf(fs, resultFor("c1")) >= 0 {
			t.Fatalf("%s: cancelled call also produced a result", side.name)
		}
	}

	script.add()
	h.generateAndWait(t, c, 5)
	if got := script.toolsAt(4); len(got) != 2 {
		t.Fatalf("cancel tool still offered after the call ended: %v", got)
	}
}

func TestCancelToolAsksWhichCallWhenSeveralRun(t *testing.T) {
	script := &turnScript{}
	h := newHarness(t, script.stream, nil)
	entered, cancelled := make(chan string, 4), make(chan string, 4)
	h.svc.RegisterFunction("report", blockingHandler(entered, cancelled), llm.FunctionOptions{CancellableByLLM: true})
	c := llm.NewContext(llm.User("two reports"))

	script.add(
		llm.ToolCall{ID: "r1", Name: "report", Arguments: []byte(`{"topic":"a"}`)},
		llm.ToolCall{ID: "r2", Name: "report", Arguments: []byte(`{"topic":"b"}`)},
	)
	h.generateAndWait(t, c, 1)
	receive(t, entered, "first report")
	receive(t, entered, "second report")

	script.add(llm.ToolCall{ID: "k1", Name: "cancel_report"})
	h.generateAndWait(t, c, 2)
	r := h.down.waitForType(t, resultFor("k1"), "refusal").(*frames.FunctionCallResultFrame)
	value := r.Result.(map[string]any)
	reason, _ := value["reason"].(string)
	if value["cancelled"] != nil || !strings.Contains(reason, "r1") || !strings.Contains(reason, "r2") || !strings.Contains(reason, `{"topic":"b"}`) {
		t.Fatalf("refusal = %v", value)
	}

	script.add(llm.ToolCall{ID: "k2", Name: "cancel_report", Arguments: []byte(`{"tool_call_id":"r2"}`)})
	h.generateAndWait(t, c, 3)
	h.down.waitForType(t, resultFor("k2"), "cancel result")
	if id := receive(t, cancelled, "r2 cancellation"); id != "r2" {
		t.Fatalf("cancelled %s", id)
	}
	if indexOf(h.down.all(), cancelFor("r1")) >= 0 {
		t.Fatal("cancelled the wrong call")
	}

	script.add(llm.ToolCall{ID: "k3", Name: "cancel_report", Arguments: []byte(`{"tool_call_id":"r9"}`)})
	h.generateAndWait(t, c, 4)
	r = h.down.waitForType(t, resultFor("k3"), "unknown id").(*frames.FunctionCallResultFrame)
	if v := r.Result.(map[string]any); v["cancelled"] != nil {
		t.Fatalf("unknown id cancelled something: %v", v)
	}
}

func TestModelCancelAfterHandlerReturnedKeepsResult(t *testing.T) {
	script := &turnScript{}
	h := newHarness(t, script.stream, nil)
	entered := make(chan string, 1)
	finish := make(chan struct{})
	h.svc.RegisterFunction("report", func(_ context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		entered <- p.ToolCallID
		<-finish
		return llm.FunctionResult{Value: "sent"}, nil
	}, llm.FunctionOptions{CancellableByLLM: true})
	script.add(llm.ToolCall{ID: "c1", Name: "report"})
	h.generateAndWait(t, llm.NewContext(llm.User("send it")), 1)
	receive(t, entered, "report to start")

	h.svc.callsMu.Lock()
	h.svc.calls["c1"].cancelledByLLM = true
	h.svc.callsMu.Unlock()
	close(finish)

	for _, side := range []*capture{h.down, h.up} {
		r := side.waitForType(t, resultFor("c1"), "result").(*frames.FunctionCallResultFrame)
		if r.Result != "sent" || r.Error != "" {
			t.Fatalf("%s: completed result = %+v", side.name, r)
		}
	}
	h.svc.callsWG.Wait()
	all := append(h.drain(t), h.up.all()...)
	if indexOf(all, cancelFor("c1")) >= 0 {
		t.Fatalf("a call that finished was reported cancelled: %v", names(all))
	}
}

func TestCancelToolReportsCallThatFinishedFirst(t *testing.T) {
	h := newHarness(t, scripted(), nil)
	h.svc.RegisterFunction("report", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
		return llm.FunctionResult{}, nil
	}, llm.FunctionOptions{CancellableByLLM: true})
	rc := &runningCall{name: "report", id: "c1", cancellableByLLM: true, done: make(chan struct{})}
	rc.cancel = func() {
		go func() {
			h.svc.callsMu.Lock()
			delete(h.svc.calls, "c1")
			rc.returned = true
			h.svc.callsMu.Unlock()
			close(rc.done)
		}()
	}
	h.svc.callsMu.Lock()
	h.svc.calls["c1"] = rc
	h.svc.callsMu.Unlock()

	out, err := h.svc.cancelToolHandler("report")(context.Background(), llm.FunctionCallParams{FunctionName: "cancel_report", Arguments: map[string]any{}})
	if err != nil {
		t.Fatal(err)
	}
	value := out.Value.(map[string]any)
	reason, _ := value["reason"].(string)
	if value["cancelled"] != nil || !strings.Contains(reason, "finished before it could be cancelled") {
		t.Fatalf("cancel tool result = %v", value)
	}
}

func TestCancellableByLLMNeedsCallThatSurvivesInterruption(t *testing.T) {
	script := &turnScript{}
	h := newHarness(t, script.stream, nil)
	entered, cancelled := make(chan string, 1), make(chan string, 1)
	h.svc.RegisterFunction("report", blockingHandler(entered, cancelled), llm.FunctionOptions{CancellableByLLM: true, CancelOnInterruption: true})
	c := llm.NewContext(llm.User("write"))
	script.add(llm.ToolCall{ID: "c1", Name: "report"})
	h.generateAndWait(t, c, 1)
	receive(t, entered, "report to start")
	script.add()
	h.generateAndWait(t, c, 2)
	if got := script.toolsAt(1); len(got) != 0 {
		t.Fatalf("cancel tool offered for a call that cancels on interruption: %v", got)
	}
}

func TestResultDeliveryReachesResultFrame(t *testing.T) {
	yes := true
	cases := []struct {
		name    string
		opts    llm.FunctionOptions
		result  llm.FunctionResult
		want    llm.ResultDelivery
		wantRun *bool
		silent  bool
	}{
		{name: "default", want: llm.ResultDeliveryDefault},
		{name: "function silent", opts: llm.FunctionOptions{Delivery: llm.ResultDeliverySilent}, want: llm.ResultDeliverySilent, silent: true},
		{name: "result overrides", opts: llm.FunctionOptions{Delivery: llm.ResultDeliveryWhenIdle}, result: llm.FunctionResult{Delivery: llm.ResultDeliveryInterrupt}, want: llm.ResultDeliveryInterrupt},
		{name: "result unsilences", opts: llm.FunctionOptions{Delivery: llm.ResultDeliverySilent}, result: llm.FunctionResult{Delivery: llm.ResultDeliveryWhenIdle}, want: llm.ResultDeliveryWhenIdle},
		{name: "silent beats run llm", result: llm.FunctionResult{Delivery: llm.ResultDeliverySilent, RunLLM: &yes}, want: llm.ResultDeliverySilent, silent: true},
		{name: "run llm kept", result: llm.FunctionResult{RunLLM: &yes}, want: llm.ResultDeliveryDefault, wantRun: &yes},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, toolCallStream(llm.ToolCall{ID: "c1", Name: "book"}), nil)
			h.svc.RegisterFunction("book", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
				r := tc.result
				r.Value = "booked"
				return r, nil
			}, tc.opts)
			h.generate(t, llm.NewContext())
			r := h.down.waitForType(t, resultFor("c1"), "result").(*frames.FunctionCallResultFrame)
			if r.Delivery != tc.want {
				t.Fatalf("delivery = %q, want %q", r.Delivery, tc.want)
			}
			switch {
			case tc.silent:
				if r.RunLLM == nil || *r.RunLLM {
					t.Fatalf("silent result must carry RunLLM=false, got %v", r.RunLLM)
				}
			case tc.wantRun != nil:
				if r.RunLLM == nil || *r.RunLLM != *tc.wantRun {
					t.Fatalf("RunLLM = %v", r.RunLLM)
				}
			default:
				if r.RunLLM != nil {
					t.Fatalf("default result must leave RunLLM unset, got %v", *r.RunLLM)
				}
			}
		})
	}
}

func TestReportProgressSendsIntermediateResults(t *testing.T) {
	h := newHarness(t, toolCallStream(llm.ToolCall{ID: "c1", Name: "report"}), nil)
	yes := true
	reporters := make(chan func(llm.FunctionResult) bool, 1)
	accepted := make(chan []bool, 1)
	h.svc.RegisterFunction("report", func(_ context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		reporters <- p.ReportProgress
		accepted <- []bool{
			p.ReportProgress(llm.FunctionResult{Value: "half"}),
			p.ReportProgress(llm.FunctionResult{Value: "almost", RunLLM: &yes}),
		}
		return llm.FunctionResult{Value: "done"}, nil
	}, llm.FunctionOptions{Delivery: llm.ResultDeliveryWhenIdle})
	h.generate(t, llm.NewContext())
	report := <-reporters
	if got := <-accepted; !got[0] || !got[1] {
		t.Fatalf("progress rejected while running: %v", got)
	}
	for _, side := range []*capture{h.down, h.up} {
		side.waitForType(t, func(f frames.Frame) bool {
			r, ok := f.(*frames.FunctionCallResultFrame)
			return ok && r.Properties == nil
		}, "final result")
	}
	if report(llm.FunctionResult{Value: "late"}) {
		t.Fatal("progress accepted after the call settled")
	}
	for _, side := range []*capture{h.down, h.up} {
		results := ofType[*frames.FunctionCallResultFrame](side.all())
		if len(results) != 3 {
			t.Fatalf("%s: expected two progress frames and one result, got %d", side.name, len(results))
		}
		silent, spoken, final := results[0], results[1], results[2]
		if silent.Result != "half" || silent.Properties == nil || silent.Properties.IsFinal || *silent.Properties.RunLLM || silent.Delivery != llm.ResultDeliverySilent {
			t.Fatalf("silent progress = %+v", silent)
		}
		if spoken.Result != "almost" || spoken.Properties == nil || spoken.Properties.IsFinal || !*spoken.Properties.RunLLM || spoken.Delivery != llm.ResultDeliveryWhenIdle {
			t.Fatalf("spoken progress = %+v", spoken)
		}
		if final.Result != "done" || final.Properties != nil || final.Delivery != llm.ResultDeliveryWhenIdle {
			t.Fatalf("final = %+v", final)
		}
	}
}
