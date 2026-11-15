package flows_test

import (
	"context"
	"errors"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/flows"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

const waitLimit = 5 * time.Second

type turn struct {
	snapshot llm.Snapshot
	settings llm.Settings
}

type fakeProvider struct {
	mu            sync.Mutex
	turns         []turn
	summaries     []turn
	changed       chan struct{}
	script        func(n int) []llm.Event
	summaryPrompt string
	summary       func() (string, error)
}

func newFakeProvider(script func(n int) []llm.Event) *fakeProvider {
	return &fakeProvider{changed: make(chan struct{}), script: script}
}

func (p *fakeProvider) Name() string { return "fake" }

func (p *fakeProvider) Stream(ctx context.Context, snapshot llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error {
	p.mu.Lock()
	if p.summaryPrompt != "" && settings.SystemInstruction == p.summaryPrompt {
		p.summaries = append(p.summaries, turn{snapshot, settings})
		fn := p.summary
		p.mu.Unlock()
		text, err := fn()
		if err != nil {
			return err
		}
		emit(llm.TextDelta{Text: text})
		return nil
	}
	n := len(p.turns)
	p.turns = append(p.turns, turn{snapshot, settings})
	close(p.changed)
	p.changed = make(chan struct{})
	p.mu.Unlock()
	if p.script != nil {
		for _, ev := range p.script(n) {
			emit(ev)
		}
	}
	return nil
}

func (p *fakeProvider) count() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.turns)
}

func (p *fakeProvider) waitTurns(t *testing.T, n int) []turn {
	t.Helper()
	deadline := time.After(waitLimit)
	for {
		p.mu.Lock()
		if len(p.turns) >= n {
			out := append([]turn(nil), p.turns...)
			p.mu.Unlock()
			return out
		}
		ch := p.changed
		p.mu.Unlock()
		select {
		case <-ch:
		case <-deadline:
			t.Fatalf("timed out waiting for %d LLM turns, have %d", n, p.count())
		}
	}
}

func (p *fakeProvider) summaryCalls() []turn {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]turn(nil), p.summaries...)
}

type harness struct {
	provider *fakeProvider
	task     *pipeline.PipelineTask
	manager  *flows.Manager
	conv     *llm.Context
	done     chan error
}

func newHarness(t *testing.T, p *fakeProvider, mutate func(*flows.Config)) *harness {
	t.Helper()
	conv := llm.NewContext()
	pair := aggregators.NewContextAggregatorPair(conv, aggregators.ContextAggregatorPairParams{})
	svc := llmservice.New(p)
	proc := flows.NewProcessor()
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{pair.User(), svc, proc, pair.Assistant()})
	cfg := pipeline.DefaultPipelineTaskConfig()
	cfg.CancelTimeout = 2 * time.Second
	task := pipeline.NewPipelineTaskWithConfig(pipe, cfg)
	fc := flows.Config{LLM: svc, Task: task, Context: conv, Processor: proc}
	if mutate != nil {
		mutate(&fc)
	}
	m, err := flows.NewManager(fc)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	h := &harness{provider: p, task: task, manager: m, conv: conv, done: make(chan error, 1)}
	ctx, cancel := context.WithCancel(context.Background())
	go func() { h.done <- task.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-h.done:
		case <-time.After(waitLimit):
			t.Errorf("pipeline did not stop")
		}
		m.Close()
	})
	return h
}

func (h *harness) initialize(t *testing.T, node *flows.Node) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), waitLimit)
	defer cancel()
	if err := h.manager.Initialize(ctx, node); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
}

func (h *harness) waitRun(t *testing.T) {
	t.Helper()
	select {
	case err := <-h.done:
		if err != nil {
			t.Fatalf("Run returned %v", err)
		}
		h.done <- err
	case <-time.After(waitLimit):
		t.Fatal("pipeline did not finish")
	}
}

func toolNames(s llm.Snapshot) []string {
	out := make([]string, 0, len(s.Tools.Functions))
	for _, f := range s.Tools.Functions {
		out = append(out, f.Name)
	}
	sort.Strings(out)
	return out
}

func texts(msgs []llm.Message) []string {
	var out []string
	for _, m := range msgs {
		if t := m.Text(); t != "" {
			out = append(out, string(m.Role)+":"+t)
		}
	}
	return out
}

func toolResult(msgs []llm.Message, id string) (llm.ToolResultPart, bool) {
	for _, m := range msgs {
		if r, ok := m.ToolResult(); ok && r.ToolCallID == id {
			return r, true
		}
	}
	return llm.ToolResultPart{}, false
}

func equal(a, b []string) bool {
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

func waitClosed(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(waitLimit):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func TestThreeNodeFlowWithParallelCallsAndStrategies(t *testing.T) {
	p := newFakeProvider(func(n int) []llm.Event {
		switch n {
		case 0:
			return []llm.Event{
				llm.ToolCall{ID: "call-a", Name: "collect_a", Arguments: []byte(`{"v":1}`)},
				llm.ToolCall{ID: "call-b", Name: "collect_b", Arguments: []byte(`{}`)},
			}
		case 1:
			return []llm.Event{llm.ToolCall{ID: "call-c", Name: "go_c", Arguments: []byte(`{}`)}}
		}
		return []llm.Event{llm.TextDelta{Text: "bye"}}
	})
	p.summaryPrompt = "SUMMARIZE"
	p.summary = func() (string, error) { return "the summary", nil }

	release := make(chan struct{})
	var gotArg any
	nodeC := &flows.Node{
		Name:            "C",
		TaskMessages:    []llm.Message{llm.Developer("task-C")},
		ContextStrategy: &flows.ContextStrategy{Kind: flows.ResetWithSummary, SummaryPrompt: "SUMMARIZE"},
		PostActions:     []flows.Action{flows.EndConversation{}},
	}
	nodeB := &flows.Node{
		Name:            "B",
		RoleMessage:     "role-B",
		TaskMessages:    []llm.Message{llm.Developer("task-B")},
		ContextStrategy: &flows.ContextStrategy{Kind: flows.Reset},
		Functions:       []flows.Function{{Name: "go_c", Description: "to C", Next: nodeC}},
	}
	nodeA := &flows.Node{
		Name:         "A",
		RoleMessage:  "role-A",
		TaskMessages: []llm.Message{llm.Developer("task-A")},
		Functions: []flows.Function{
			{Name: "collect_a", Handler: func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
				gotArg = args["v"]
				select {
				case <-release:
				case <-ctx.Done():
					return nil, nil, ctx.Err()
				}
				return map[string]any{"a": true}, nil, nil
			}},
			{Name: "collect_b", Handler: func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
				return map[string]any{"b": true}, nodeB, nil
			}},
		},
	}
	global := flows.Function{Name: "help", Handler: func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
		return "ok", nil, nil
	}}

	h := newHarness(t, p, func(c *flows.Config) { c.GlobalFunctions = []flows.Function{global} })
	bDone := make(chan struct{})
	var once sync.Once
	h.task.OnFrameReachedUpstream(func(f frames.Frame) {
		if r, ok := f.(*frames.FunctionCallResultFrame); ok && r.ToolCallID == "call-b" {
			once.Do(func() { close(bDone) })
		}
	}, (*frames.FunctionCallResultFrame)(nil))

	h.initialize(t, nodeA)
	turns := h.provider.waitTurns(t, 1)
	if got, want := toolNames(turns[0].snapshot), []string{"collect_a", "collect_b", "help"}; !equal(got, want) {
		t.Fatalf("node A tools = %v, want %v", got, want)
	}
	if got := texts(turns[0].snapshot.Messages); !equal(got, []string{"developer:task-A"}) {
		t.Fatalf("node A messages = %v", got)
	}
	if turns[0].settings.SystemInstruction != "role-A" {
		t.Fatalf("node A system instruction = %q", turns[0].settings.SystemInstruction)
	}

	waitClosed(t, bDone, "collect_b result")
	time.Sleep(100 * time.Millisecond)
	if n := h.provider.count(); n != 1 {
		t.Fatalf("LLM ran %d times before every parallel call finished", n)
	}
	if cur := h.manager.CurrentNode(); cur != "A" {
		t.Fatalf("transitioned to %q before every parallel call finished", cur)
	}
	close(release)

	turns = h.provider.waitTurns(t, 2)
	if got, want := toolNames(turns[1].snapshot), []string{"go_c", "help"}; !equal(got, want) {
		t.Fatalf("node B tools = %v, want %v", got, want)
	}
	if got := texts(turns[1].snapshot.Messages); !equal(got, []string{"developer:task-B"}) {
		t.Fatalf("node B (reset) messages = %v", got)
	}
	if turns[1].settings.SystemInstruction != "role-B" {
		t.Fatalf("node B system instruction = %q", turns[1].settings.SystemInstruction)
	}
	if gotArg != float64(1) {
		t.Fatalf("collect_a argument v = %v", gotArg)
	}

	turns = h.provider.waitTurns(t, 3)
	if got, want := toolNames(turns[2].snapshot), []string{"help"}; !equal(got, want) {
		t.Fatalf("node C tools = %v, want %v", got, want)
	}
	want := []string{"developer:Here's a summary of the conversation:\nthe summary", "developer:task-C"}
	if got := texts(turns[2].snapshot.Messages); !equal(got, want) {
		t.Fatalf("node C (reset_with_summary) messages = %q, want %q", got, want)
	}
	sums := h.provider.summaryCalls()
	if len(sums) != 1 {
		t.Fatalf("summary inference ran %d times", len(sums))
	}
	if hist := strings.Join(texts(sums[0].snapshot.Messages), "\n"); !strings.Contains(hist, "task-B") || !strings.Contains(hist, "go_c") {
		t.Fatalf("summary request lacks the history: %q", hist)
	}

	h.waitRun(t)
	if cur := h.manager.CurrentNode(); cur != "C" {
		t.Fatalf("current node = %q, want C", cur)
	}
	if n := h.provider.count(); n != 3 {
		t.Fatalf("LLM turns = %d, want 3", n)
	}
}

func TestContextStrategies(t *testing.T) {
	cases := []struct {
		name       string
		global     flows.ContextStrategy
		node       *flows.ContextStrategy
		summaryErr error
		check      func(t *testing.T, msgs []llm.Message)
	}{
		{
			name: "append",
			node: &flows.ContextStrategy{Kind: flows.Append},
			check: func(t *testing.T, msgs []llm.Message) {
				assertAppended(t, msgs)
			},
		},
		{
			name: "reset",
			node: &flows.ContextStrategy{Kind: flows.Reset},
			check: func(t *testing.T, msgs []llm.Message) {
				if got := texts(msgs); len(msgs) != 1 || !equal(got, []string{"developer:task-Y"}) {
					t.Fatalf("messages = %v", got)
				}
			},
		},
		{
			name:   "global reset",
			global: flows.ContextStrategy{Kind: flows.Reset},
			check: func(t *testing.T, msgs []llm.Message) {
				if got := texts(msgs); len(msgs) != 1 || !equal(got, []string{"developer:task-Y"}) {
					t.Fatalf("messages = %v", got)
				}
			},
		},
		{
			name: "reset with summary",
			node: &flows.ContextStrategy{Kind: flows.ResetWithSummary, SummaryPrompt: "SUM"},
			check: func(t *testing.T, msgs []llm.Message) {
				want := []string{"developer:Here's a summary of the conversation:\nS", "developer:task-Y"}
				if got := texts(msgs); len(msgs) != 2 || !equal(got, want) {
					t.Fatalf("messages = %q", got)
				}
			},
		},
		{
			name:       "reset with failing summary appends",
			node:       &flows.ContextStrategy{Kind: flows.ResetWithSummary, SummaryPrompt: "SUM"},
			summaryErr: errors.New("boom"),
			check: func(t *testing.T, msgs []llm.Message) {
				assertAppended(t, msgs)
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := newFakeProvider(func(n int) []llm.Event {
				if n == 0 {
					return []llm.Event{llm.ToolCall{ID: "t1", Name: "next", Arguments: []byte(`{}`)}}
				}
				return []llm.Event{llm.TextDelta{Text: "ok"}}
			})
			p.summaryPrompt = "SUM"
			p.summary = func() (string, error) { return "S", tc.summaryErr }
			y := &flows.Node{Name: "Y", TaskMessages: []llm.Message{llm.Developer("task-Y")}, ContextStrategy: tc.node}
			x := &flows.Node{
				Name:         "X",
				TaskMessages: []llm.Message{llm.Developer("task-X")},
				Functions:    []flows.Function{{Name: "next", Next: y}},
			}
			h := newHarness(t, p, func(c *flows.Config) { c.ContextStrategy = tc.global })
			h.initialize(t, x)
			turns := h.provider.waitTurns(t, 2)
			tc.check(t, turns[1].snapshot.Messages)
			if len(turns[1].snapshot.Tools.Functions) != 0 {
				t.Fatalf("node Y tools = %v", toolNames(turns[1].snapshot))
			}
		})
	}
}

func assertAppended(t *testing.T, msgs []llm.Message) {
	t.Helper()
	got := texts(msgs)
	if len(got) != 2 || got[0] != "developer:task-X" || got[1] != "developer:task-Y" {
		t.Fatalf("text messages = %v", got)
	}
	if msgs[len(msgs)-1].Text() != "task-Y" {
		t.Fatalf("last message = %v", msgs[len(msgs)-1])
	}
	r, ok := toolResult(msgs, "t1")
	if !ok {
		t.Fatalf("tool result missing from appended context")
	}
	if !strings.Contains(r.Content, "acknowledged") {
		t.Fatalf("tool result content = %q", r.Content)
	}
}

func TestPreAndPostActionsOrder(t *testing.T) {
	p := newFakeProvider(func(n int) []llm.Event { return []llm.Event{llm.TextDelta{Text: "hi"}} })
	var mu sync.Mutex
	var log []string
	record := func(s string) {
		mu.Lock()
		log = append(log, s)
		mu.Unlock()
	}
	finished := make(chan struct{})
	rec := func(ctx context.Context, a flows.Custom, m *flows.Manager) error {
		id, _ := a.Params["id"].(string)
		record(id + "@" + m.CurrentNode())
		if id == "post2" {
			close(finished)
		}
		return nil
	}
	fn := func(id string) flows.Action {
		return flows.FunctionAction{Fn: func(ctx context.Context, m *flows.Manager) error {
			record(id + "@" + m.CurrentNode())
			return nil
		}}
	}
	node := &flows.Node{
		Name:         "N",
		TaskMessages: []llm.Message{llm.Developer("task")},
		PreActions:   []flows.Action{flows.Custom{Type: "rec", Params: map[string]any{"id": "pre1"}}, fn("pre2")},
		PostActions:  []flows.Action{flows.TTSSay{Text: "hello"}, fn("post1"), flows.Custom{Type: "rec", Params: map[string]any{"id": "post2"}}},
	}
	h := newHarness(t, p, func(c *flows.Config) { c.Actions = map[string]flows.ActionHandler{"rec": rec} })
	h.task.OnFrameReachedDownstream(func(f frames.Frame) {
		record("tts:" + f.(*frames.TTSSpeakFrame).Text)
	}, (*frames.TTSSpeakFrame)(nil))
	h.initialize(t, node)
	waitClosed(t, finished, "post actions")
	h.provider.waitTurns(t, 1)

	mu.Lock()
	got := append([]string(nil), log...)
	mu.Unlock()
	want := []string{"pre1@", "pre2@", "tts:hello", "post1@N", "post2@N"}
	if !equal(got, want) {
		t.Fatalf("action order = %v, want %v", got, want)
	}
}

func TestWaitForUserDefersPostActionsUntilBotStopsSpeaking(t *testing.T) {
	p := newFakeProvider(nil)
	ran := make(chan struct{})
	node := &flows.Node{
		Name:         "W",
		TaskMessages: []llm.Message{llm.Developer("task")},
		WaitForUser:  true,
		PostActions:  []flows.Action{flows.Custom{Type: "mark"}},
	}
	h := newHarness(t, p, func(c *flows.Config) {
		c.Actions = map[string]flows.ActionHandler{"mark": func(ctx context.Context, a flows.Custom, m *flows.Manager) error {
			close(ran)
			return nil
		}}
	})
	h.initialize(t, node)
	select {
	case <-ran:
		t.Fatal("post action ran before the bot stopped speaking")
	case <-time.After(100 * time.Millisecond):
	}
	if err := h.task.QueueFrame(frames.NewBotStoppedSpeakingFrame()); err != nil {
		t.Fatal(err)
	}
	waitClosed(t, ran, "deferred post action")
	if n := h.provider.count(); n != 0 {
		t.Fatalf("LLM ran %d times for a WaitForUser node", n)
	}
}

func TestFunctionFailuresBecomeErrorResults(t *testing.T) {
	p := newFakeProvider(func(n int) []llm.Event {
		if n == 0 {
			return []llm.Event{
				llm.ToolCall{ID: "e1", Name: "fails", Arguments: []byte(`{}`)},
				llm.ToolCall{ID: "e2", Name: "panics", Arguments: []byte(`{}`)},
			}
		}
		return []llm.Event{llm.TextDelta{Text: "ok"}}
	})
	node := &flows.Node{
		Name:         "F",
		TaskMessages: []llm.Message{llm.Developer("task")},
		Functions: []flows.Function{
			{Name: "fails", Handler: func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
				return nil, nil, errors.New("nope")
			}},
			{Name: "panics", Handler: func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
				panic("kaboom")
			}},
		},
	}
	h := newHarness(t, p, nil)
	h.initialize(t, node)
	turns := h.provider.waitTurns(t, 2)
	msgs := turns[1].snapshot.Messages
	for id, want := range map[string]string{"e1": "nope", "e2": "kaboom"} {
		r, ok := toolResult(msgs, id)
		if !ok {
			t.Fatalf("no tool result for %s", id)
		}
		if !strings.Contains(r.Content, `"status":"error"`) || !strings.Contains(r.Content, want) {
			t.Fatalf("tool result %s = %q", id, r.Content)
		}
	}
	if cur := h.manager.CurrentNode(); cur != "F" {
		t.Fatalf("current node = %q", cur)
	}
}

func TestManagerLifecycleErrors(t *testing.T) {
	p := newFakeProvider(nil)
	h := newHarness(t, p, nil)
	ctx := context.Background()
	if err := h.manager.SetNode(ctx, &flows.Node{Name: "x"}); !errors.Is(err, flows.ErrNotInitialized) {
		t.Fatalf("SetNode before Initialize = %v", err)
	}
	if err := h.manager.Initialize(ctx, nil); err != nil {
		t.Fatal(err)
	}
	if err := h.manager.Initialize(ctx, nil); !errors.Is(err, flows.ErrAlreadyInitialized) {
		t.Fatalf("second Initialize = %v", err)
	}
	if err := h.manager.SetNode(ctx, nil); err == nil {
		t.Fatal("SetNode(nil) succeeded")
	}
	if err := h.manager.SetNode(ctx, &flows.Node{Name: "bad", TaskMessages: []llm.Message{llm.Developer("{{ missing }}")}}); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("SetNode with unknown placeholder = %v", err)
	}
	if err := h.manager.SetNode(ctx, &flows.Node{Name: "bad", PreActions: []flows.Action{flows.Custom{Type: "nope"}}}); err == nil || !strings.Contains(err.Error(), "not registered") {
		t.Fatalf("SetNode with unregistered action = %v", err)
	}
	h.manager.Close()
	if err := h.manager.SetNode(ctx, &flows.Node{Name: "x"}); !errors.Is(err, flows.ErrClosed) {
		t.Fatalf("SetNode after Close = %v", err)
	}
}

func TestNewManagerValidation(t *testing.T) {
	conv := llm.NewContext()
	svc := llmservice.New(newFakeProvider(nil))
	task := pipeline.NewPipelineTask(pipeline.NewPipeline([]processors.FrameProcessor{processors.NewBaseProcessor("p", nil)}))
	handler := func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
		return nil, nil, nil
	}
	cases := []struct {
		name string
		cfg  func() flows.Config
		want string
	}{
		{"no llm", func() flows.Config { return flows.Config{Task: task, Context: conv, Processor: flows.NewProcessor()} }, "LLM is required"},
		{"no task", func() flows.Config { return flows.Config{LLM: svc, Context: conv, Processor: flows.NewProcessor()} }, "Task is required"},
		{"no context", func() flows.Config { return flows.Config{LLM: svc, Task: task, Processor: flows.NewProcessor()} }, "Context is required"},
		{"no processor", func() flows.Config { return flows.Config{LLM: svc, Task: task, Context: conv} }, "Processor is required"},
		{"summary without prompt", func() flows.Config {
			return flows.Config{LLM: svc, Task: task, Context: conv, Processor: flows.NewProcessor(), ContextStrategy: flows.ContextStrategy{Kind: flows.ResetWithSummary}}
		}, "summary prompt"},
		{"duplicate global", func() flows.Config {
			f := flows.Function{Name: "g", Handler: handler}
			return flows.Config{LLM: svc, Task: task, Context: conv, Processor: flows.NewProcessor(), GlobalFunctions: []flows.Function{f, f}}
		}, "duplicate global"},
		{"global without handler", func() flows.Config {
			return flows.Config{LLM: svc, Task: task, Context: conv, Processor: flows.NewProcessor(), GlobalFunctions: []flows.Function{{Name: "g"}}}
		}, "needs a Handler or a Next"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, err := flows.NewManager(tc.cfg())
			if err == nil {
				m.Close()
				t.Fatal("NewManager succeeded")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not mention %q", err, tc.want)
			}
		})
	}

	proc := flows.NewProcessor()
	m, err := flows.NewManager(flows.Config{LLM: svc, Task: task, Context: conv, Processor: proc})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, err := flows.NewManager(flows.Config{LLM: svc, Task: task, Context: conv, Processor: proc}); err == nil || !strings.Contains(err.Error(), "already bound") {
		t.Fatalf("reusing a processor = %v", err)
	}
}

func TestLoadJSONValidationErrors(t *testing.T) {
	noop := func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
		return nil, nil, nil
	}
	reg := flows.Registry{Handlers: map[string]flows.Handler{"f": noop, "g": noop}}
	cases := []struct {
		name string
		json string
		want []string
	}{
		{"no nodes", `{"initial_node":"x","nodes":{}}`, []string{"flow has no nodes", `initial_node "x" is not a defined node`}},
		{"no initial", `{"nodes":{"a":{"task_messages":[]}}}`, []string{"initial_node is required"}},
		{"bad role", `{"initial_node":"a","nodes":{"a":{"task_messages":[{"role":"tool","content":"x"}]}}}`, []string{`unsupported role "tool"`}},
		{"bad strategy", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"context_strategy":"forget"}}}`, []string{`unknown context_strategy "forget"`}},
		{"summary without prompt", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"context_strategy":"reset_with_summary"}}}`, []string{"requires summary_prompt"}},
		{"stray summary prompt", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"summary_prompt":"s"}}}`, []string{"summary_prompt is only used"}},
		{"missing handler", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"functions":[{"name":"zz"}]}}}`, []string{`handler "zz" is not in the registry`}},
		{"unknown target", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"functions":[{"name":"f","transition_to":"nowhere"}]}}}`, []string{`transitions to unknown node "nowhere"`}},
		{"duplicate function", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"functions":[{"name":"f"},{"name":"f"}]}}}`, []string{"duplicate function name"}},
		{"node shadows global", `{"initial_node":"a","global_functions":[{"name":"g"}],"nodes":{"a":{"task_messages":[],"functions":[{"name":"g"}]}}}`, []string{"also declared in global_functions"}},
		{"unreachable", `{"initial_node":"a","nodes":{"a":{"task_messages":[]},"b":{"task_messages":[]}}}`, []string{`node "b": node is unreachable`}},
		{"transition only without description", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"functions":[{"name":"t","transition_only":true,"transition_to":"a"}]}}}`, []string{"needs a description"}},
		{"transition only with handler", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"functions":[{"name":"t","description":"d","handler":"f","transition_only":true,"transition_to":"a"}]}}}`, []string{"must not name a handler"}},
		{"branch without cases", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"functions":[{"name":"f","transition_to":{"field":"x","cases":{}}}]}}}`, []string{"has no cases"}},
		{"negative timeout", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"functions":[{"name":"f","timeout_secs":-1}]}}}`, []string{"timeout_secs must not be negative"}},
		{"tts without text", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"pre_actions":[{"type":"tts_say"}]}}}`, []string{"tts_say needs text"}},
		{"action without type", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"post_actions":[{"text":"x"}]}}}`, []string{"has no type"}},
		{"unknown action", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"post_actions":[{"type":"jump"}]}}}`, []string{`action "jump" has no handler`}},
		{"function action unknown handler", `{"initial_node":"a","nodes":{"a":{"task_messages":[],"post_actions":[{"type":"function","handler":"h"}]}}}`, []string{`function action handler "h" is not in the registry`}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			flow, err := flows.LoadJSON([]byte(tc.json), reg)
			if err == nil {
				t.Fatalf("LoadJSON succeeded: %v", flow.Initial())
			}
			var verr *flows.ValidationError
			if !errors.As(err, &verr) {
				t.Fatalf("error is %T (%v), want *ValidationError", err, err)
			}
			msg := err.Error()
			for _, w := range tc.want {
				if !strings.Contains(msg, w) {
					t.Fatalf("error %q does not mention %q", msg, w)
				}
			}
		})
	}
}

func TestLoadJSONRejectsMalformedDocuments(t *testing.T) {
	for _, doc := range []string{
		`{"initial_node":"a","nodes":{"a":{"task_messages":[],"bogus":1}}}`,
		`{"initial_node":"a","nodes":{"a":{"task_messages":[],"functions":[{"name":"f","transition_to":{"field":"x","cases":{},"extra":1}}]}}}`,
		`{"initial_node":"a","nodes":{"a":{"task_messages":[],"pre_actions":[{"type":5}]}}}`,
		`not json`,
	} {
		_, err := flows.LoadJSON([]byte(doc), flows.Registry{})
		if err == nil {
			t.Fatalf("LoadJSON(%s) succeeded", doc)
		}
		var verr *flows.ValidationError
		if errors.As(err, &verr) {
			t.Fatalf("LoadJSON(%s) returned a ValidationError for a parse failure: %v", doc, err)
		}
	}
}

func TestLoadJSONBuildsBranchingFlow(t *testing.T) {
	doc := `{
	  "initial_node": "start",
	  "nodes": {
	    "start": {
	      "role_message": "role",
	      "task_messages": [{"role": "developer", "content": "pick"}],
	      "functions": [
	        {"name": "choose", "handler": "echo", "transition_to": {"field": "kind", "cases": {"a": "a", "true": "b"}, "default": "end"}},
	        {"name": "skip", "description": "skip ahead", "transition_only": true, "transition_to": "end"}
	      ],
	      "respond_immediately": false
	    },
	    "a": {"task_messages": [], "context_strategy": "reset"},
	    "b": {"task_messages": [], "context_strategy": "reset_with_summary", "summary_prompt": "sum"},
	    "end": {"task_messages": [], "post_actions": [{"type": "end_conversation", "text": "bye"}]}
	  }
	}`
	echo := func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
		return map[string]any{"kind": args["kind"]}, nil, nil
	}
	flow, err := flows.LoadJSON([]byte(doc), flows.Registry{Handlers: map[string]flows.Handler{"echo": echo}})
	if err != nil {
		t.Fatalf("LoadJSON: %v", err)
	}
	start := flow.Initial()
	if start.Name != "start" || start.RoleMessage != "role" || !start.WaitForUser {
		t.Fatalf("initial node = %+v", start)
	}
	if b, _ := flow.Node("b"); b.ContextStrategy == nil || b.ContextStrategy.Kind != flows.ResetWithSummary || b.ContextStrategy.SummaryPrompt != "sum" {
		t.Fatalf("node b strategy = %+v", b.ContextStrategy)
	}
	var choose, skip flows.Function
	for _, f := range start.Functions {
		switch f.Name {
		case "choose":
			choose = f
		case "skip":
			skip = f
		}
	}
	if skip.Next == nil || skip.Next.Name != "end" || skip.Handler != nil {
		t.Fatalf("transition_only function = %+v", skip)
	}
	for _, tc := range []struct {
		kind any
		want string
	}{{"a", "a"}, {true, "b"}, {"TRUE", "b"}, {"other", "end"}} {
		_, next, err := choose.Handler(context.Background(), flows.Args{"kind": tc.kind}, nil)
		if err != nil {
			t.Fatalf("choose(%v): %v", tc.kind, err)
		}
		if next == nil || next.Name != tc.want {
			t.Fatalf("choose(%v) -> %v, want %s", tc.kind, next, tc.want)
		}
	}
	if mm := flow.Mermaid(); !strings.Contains(mm, "flowchart") {
		t.Fatalf("mermaid = %q", mm)
	}
}

func TestTransitionRunsExactlyOneInference(t *testing.T) {
	p := newFakeProvider(func(n int) []llm.Event {
		if n == 0 {
			return []llm.Event{
				llm.ToolCall{ID: "call-plain", Name: "note", Arguments: []byte(`{}`)},
				llm.ToolCall{ID: "call-next", Name: "go_b", Arguments: []byte(`{}`)},
			}
		}
		return []llm.Event{llm.TextDelta{Text: "hello from B"}}
	})
	nodeB := &flows.Node{
		Name:         "B",
		TaskMessages: []llm.Message{llm.Developer("task-B")},
		Functions: []flows.Function{{Name: "finish", Handler: func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
			return nil, nil, nil
		}}},
	}
	nodeA := &flows.Node{
		Name:         "A",
		TaskMessages: []llm.Message{llm.Developer("task-A")},
		Functions: []flows.Function{
			{Name: "note", Handler: func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
				return "noted", nil, nil
			}},
			{Name: "go_b", Description: "to B", Next: nodeB},
		},
	}
	h := newHarness(t, p, nil)
	h.initialize(t, nodeA)

	turns := h.provider.waitTurns(t, 2)
	if got, want := toolNames(turns[1].snapshot), []string{"finish"}; !equal(got, want) {
		t.Fatalf("second inference tools = %v, want %v", got, want)
	}
	if got := texts(turns[1].snapshot.Messages); len(got) == 0 || got[len(got)-1] != "developer:task-B" {
		t.Fatalf("second inference messages = %v", got)
	}
	time.Sleep(300 * time.Millisecond)
	if n := h.provider.count(); n != 2 {
		t.Fatalf("LLM ran %d times, want exactly one inference after the transition", n)
	}
}

func TestSetNodeInsideHandlerRunsExactlyOneInference(t *testing.T) {
	p := newFakeProvider(func(n int) []llm.Event {
		if n == 0 {
			return []llm.Event{llm.ToolCall{ID: "call-1", Name: "go_b", Arguments: []byte(`{}`)}}
		}
		return []llm.Event{llm.TextDelta{Text: "hello from B"}}
	})
	nodeB := &flows.Node{
		Name:         "B",
		TaskMessages: []llm.Message{llm.Developer("task-B")},
		Functions: []flows.Function{{Name: "finish", Handler: func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
			return nil, nil, nil
		}}},
	}
	nodeA := &flows.Node{
		Name:         "A",
		TaskMessages: []llm.Message{llm.Developer("task-A")},
		Functions: []flows.Function{{Name: "go_b", Handler: func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
			return "moving on", nil, m.SetNode(ctx, nodeB)
		}}},
	}
	h := newHarness(t, p, nil)
	h.initialize(t, nodeA)

	turns := h.provider.waitTurns(t, 2)
	if got, want := toolNames(turns[1].snapshot), []string{"finish"}; !equal(got, want) {
		t.Fatalf("second inference tools = %v, want %v", got, want)
	}
	if r, ok := toolResult(turns[1].snapshot.Messages, "call-1"); !ok || !strings.Contains(r.Content, "moving on") {
		t.Fatalf("second inference lacks the function result: %+v", r)
	}
	time.Sleep(300 * time.Millisecond)
	if n := h.provider.count(); n != 2 {
		t.Fatalf("LLM ran %d times, want exactly one inference after the transition", n)
	}
}
