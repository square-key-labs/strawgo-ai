package aggregators

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type mockSummaryLLM struct {
	summary       string
	waitForCancel bool
	cancelled     atomic.Bool
	calls         atomic.Int32
	lastRequest   atomic.Value
}

func (m *mockSummaryLLM) Summarize(ctx context.Context, snapshot llm.Snapshot) (string, error) {
	m.calls.Add(1)
	m.lastRequest.Store(snapshot)
	if m.waitForCancel {
		<-ctx.Done()
		m.cancelled.Store(true)
		return "", ctx.Err()
	}
	return m.summary, nil
}

func TestContextSummarizer_AutoTokenThreshold(t *testing.T) {
	c := llm.NewContext(
		llm.User(strings.Repeat("a", 200)),
		llm.Assistant(strings.Repeat("b", 200)),
		llm.User(strings.Repeat("c", 200)),
	)
	mock := &mockSummaryLLM{summary: "compressed context"}
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{
		MaxContextTokens: 80,
		SummaryConfig:    LLMContextSummaryConfig{MinMessagesAfterSummary: 1},
	}, mock.Summarize)

	if !summarizer.ShouldAutoSummarize(c.Snapshot()) {
		t.Fatal("expected auto summarization trigger on token threshold")
	}
	applied, err := summarizer.Summarize(context.Background(), c)
	if err != nil || !applied {
		t.Fatalf("expected summarization to be applied, err=%v", err)
	}
	msgs := c.Messages()
	if len(msgs) >= 3 {
		t.Fatalf("expected summarized context to be shorter, got %d messages", len(msgs))
	}
	if msgs[0].Role != llm.RoleUser || !strings.Contains(msgs[0].Text(), "compressed context") {
		t.Fatalf("expected user summary message first, got %+v", msgs[0])
	}
	req := mock.lastRequest.Load().(llm.Snapshot)
	if len(req.Messages) != 2 || req.Messages[0].Role != llm.RoleSystem {
		t.Fatalf("expected a system prompt plus transcript request, got %+v", req.Messages)
	}
}

func TestContextSummarizer_AutoMessageThreshold(t *testing.T) {
	c := llm.NewContext(llm.User("1"), llm.Assistant("2"), llm.User("3"), llm.Assistant("4"), llm.User("5"))
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{
		MaxContextTokens:        10000,
		MaxUnsummarizedMessages: 3,
		SummaryConfig:           LLMContextSummaryConfig{MinMessagesAfterSummary: 1},
	}, (&mockSummaryLLM{summary: "message-threshold summary"}).Summarize)

	if !summarizer.ShouldAutoSummarize(c.Snapshot()) {
		t.Fatal("expected auto summarization trigger on message threshold")
	}
	if applied, err := summarizer.Summarize(context.Background(), c); err != nil || !applied {
		t.Fatalf("expected summarization to be applied, err=%v", err)
	}
}

func TestContextSummarizer_KeepsSystemAndToolPairs(t *testing.T) {
	call := llm.ToolCallPart{ID: "call-1", Name: "get_weather", Arguments: []byte(`{"city":"SF"}`)}
	c := llm.NewContext(
		llm.System("be nice"),
		llm.User(strings.Repeat("old chat ", 30)),
		llm.Assistant(strings.Repeat("old reply ", 30)),
		llm.ToolCalls(call),
		llm.ToolResult("call-1", "get_weather", `{"temp":20}`),
		llm.Assistant("It is 20 degrees"),
	)
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{
		SummaryConfig: LLMContextSummaryConfig{MinMessagesAfterSummary: 2},
	}, (&mockSummaryLLM{summary: "weather chat"}).Summarize)

	if applied, err := summarizer.Summarize(context.Background(), c); err != nil || !applied {
		t.Fatalf("expected summarization to be applied, err=%v", err)
	}
	msgs := c.Messages()
	if msgs[0].Role != llm.RoleSystem {
		t.Fatalf("expected system prompt preserved, got %+v", msgs[0])
	}
	for i, m := range msgs {
		if _, ok := m.ToolResult(); ok && (i == 0 || len(msgs[i-1].ToolCalls()) == 0) {
			t.Fatalf("tool result at %d lost its tool call: %+v", i, msgs)
		}
	}
}

func TestContextSummarizer_FunctionCallInProgressPreserved(t *testing.T) {
	c := llm.NewContext(
		llm.User("Earlier chat details that should be summarized"),
		llm.Assistant("Ack"),
		llm.ToolCalls(llm.ToolCallPart{ID: "call-1", Name: "get_weather", Arguments: []byte(`{}`)}),
		llm.ToolResult("call-1", "get_weather", "IN_PROGRESS"),
	)
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{
		MaxContextTokens: 30,
		SummaryConfig:    LLMContextSummaryConfig{MinMessagesAfterSummary: 1},
	}, (&mockSummaryLLM{summary: "function-aware summary"}).Summarize)

	if applied, err := summarizer.Summarize(context.Background(), c); err != nil || !applied {
		t.Fatalf("expected summarization to be applied, err=%v", err)
	}
	msgs := c.Messages()
	last := msgs[len(msgs)-1]
	prev := msgs[len(msgs)-2]
	if calls := prev.ToolCalls(); len(calls) == 0 || calls[0].ID != "call-1" {
		t.Fatal("expected trailing incomplete tool call to be preserved")
	}
	if r, ok := last.ToolResult(); !ok || r.ToolCallID != "call-1" || r.Content != "IN_PROGRESS" {
		t.Fatal("expected trailing incomplete tool response to be preserved")
	}
}

func TestContextSummarizer_Timeout(t *testing.T) {
	c := llm.NewContext(llm.User(strings.Repeat("timeout", 30)), llm.Assistant(strings.Repeat("test", 30)))
	hung := &mockSummaryLLM{waitForCancel: true}
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{MaxContextTokens: 20}, hung.Summarize)
	summarizer.SetTimeout(30 * time.Millisecond)

	start := time.Now()
	applied, err := summarizer.Summarize(context.Background(), c)
	if applied || err == nil {
		t.Fatal("expected timeout to skip summarization")
	}
	if time.Since(start) > 300*time.Millisecond {
		t.Fatalf("expected timeout cancellation to be fast")
	}
	if !hung.cancelled.Load() {
		t.Fatal("expected summarization context to be cancelled on timeout")
	}
}

func TestContextSummarizer_StaleResultNotApplied(t *testing.T) {
	c := llm.NewContext(llm.User("a"), llm.Assistant("b"), llm.User("c"))
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{}, (&mockSummaryLLM{summary: "s"}).Summarize)

	done := make(chan SummaryResult, 1)
	if !summarizer.Start(context.Background(), c.Snapshot(), func(r SummaryResult) { done <- r }) {
		t.Fatal("expected summarization to start")
	}
	res := <-done
	c.Set([]llm.Message{llm.User("replaced")})
	if summarizer.Apply(c, res) {
		t.Fatal("expected a summary of replaced history to be discarded")
	}
	if c.Len() != 1 {
		t.Fatalf("expected context untouched, got %d messages", c.Len())
	}
}

func TestContextSummarizer_OnDemandIsAsync(t *testing.T) {
	c := llm.NewContext(
		llm.User(strings.Repeat("hello", 40)),
		llm.Assistant(strings.Repeat("world", 40)),
		llm.User(strings.Repeat("again", 40)),
	)
	release := make(chan struct{})
	assistant := NewLLMAssistantAggregator(c, &AssistantAggregatorParams{
		AutoSummarizationConfig: LLMAutoContextSummarizationConfig{
			MaxContextTokens: 30,
			SummaryConfig:    LLMContextSummaryConfig{MinMessagesAfterSummary: 1},
		},
		Summarize: func(ctx context.Context, _ llm.Snapshot) (string, error) {
			select {
			case <-release:
				return "forced on-demand summary", nil
			case <-ctx.Done():
				return "", ctx.Err()
			}
		},
	})
	down := &captureProc{}
	assistant.Link(down)
	if err := assistant.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer assistant.Stop()

	_ = assistant.QueueFrame(frames.NewLLMSummarizeContextFrame(), frames.Downstream)
	_ = assistant.QueueFrame(frames.NewTextFrame("after"), frames.Downstream)
	down.waitFor(t, "TextFrame", time.Second)
	if c.Len() != 3 {
		t.Fatal("expected summarization not to block or apply before the summary returns")
	}

	close(release)
	deadline := time.Now().Add(time.Second)
	for c.Len() >= 3 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	msgs := c.Messages()
	if len(msgs) >= 3 || msgs[0].Role != llm.RoleUser {
		t.Fatalf("expected context to be summarized on-demand, got %+v", msgs)
	}
}

func keepMarked(m llm.Message) bool {
	if strings.HasPrefix(m.Text(), "PIN") {
		return true
	}
	r, ok := m.ToolResult()
	return ok && strings.HasPrefix(r.Content, "PIN")
}

func messageTexts(msgs []llm.Message) []string {
	out := make([]string, len(msgs))
	for i, m := range msgs {
		out[i] = m.Text()
	}
	return out
}

func TestContextSummarizer_KeepPredicateLeavesMessagesInPlace(t *testing.T) {
	c := llm.NewContext(
		llm.System("be nice"),
		llm.User("PIN account 42"),
		llm.Assistant("old reply one"),
		llm.User("old question two"),
		llm.Assistant("PIN refund approved"),
		llm.User("old question three"),
		llm.Assistant("recent reply"),
	)
	mock := &mockSummaryLLM{summary: "S"}
	var before, after int
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{
		SummaryConfig: LLMContextSummaryConfig{
			MinMessagesAfterSummary: 1,
			SummaryMessageTemplate:  "SUMMARY %s",
			Keep:                    keepMarked,
		},
	}, mock.Summarize)
	summarizer.OnSummaryApplied = func(b, a int) { before, after = b, a }

	if applied, err := summarizer.Summarize(context.Background(), c); err != nil || !applied {
		t.Fatalf("expected summarization to be applied, err=%v", err)
	}
	want := []string{"be nice", "PIN account 42", "SUMMARY S", "PIN refund approved", "recent reply"}
	if got := messageTexts(c.Messages()); strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("messages = %q, want %q", got, want)
	}
	if before != 7 || after != 5 {
		t.Fatalf("OnSummaryApplied(%d, %d), want (7, 5)", before, after)
	}
	transcript := mock.lastRequest.Load().(llm.Snapshot).Messages[1].Text()
	if strings.Contains(transcript, "PIN") || !strings.Contains(transcript, "old question two") || !strings.Contains(transcript, "old question three") {
		t.Fatalf("transcript must hold only the summarized messages, got %q", transcript)
	}
}

func TestContextSummarizer_KeepNeverSplitsToolPairs(t *testing.T) {
	c := llm.NewContext(
		llm.User("old question"),
		llm.ToolCalls(
			llm.ToolCallPart{ID: "a", Name: "lookup", Arguments: []byte(`{}`)},
			llm.ToolCallPart{ID: "b", Name: "lookup", Arguments: []byte(`{}`)},
		),
		llm.ToolResult("a", "lookup", "PIN order 7"),
		llm.ToolResult("b", "lookup", "nothing"),
		llm.Assistant("old reply"),
		llm.User("recent"),
	)
	mock := &mockSummaryLLM{summary: "S"}
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{
		SummaryConfig: LLMContextSummaryConfig{MinMessagesAfterSummary: 1, Keep: keepMarked},
	}, mock.Summarize)

	if applied, err := summarizer.Summarize(context.Background(), c); err != nil || !applied {
		t.Fatalf("expected summarization to be applied, err=%v", err)
	}
	msgs := c.Messages()
	if len(msgs) != 5 {
		t.Fatalf("expected summary, call, two results and the recent message, got %d: %+v", len(msgs), msgs)
	}
	if !strings.Contains(msgs[0].Text(), "S") || len(msgs[1].ToolCalls()) != 2 {
		t.Fatalf("unexpected layout: %+v", msgs)
	}
	for i, id := range []string{"a", "b"} {
		if r, ok := msgs[2+i].ToolResult(); !ok || r.ToolCallID != id {
			t.Fatalf("tool result %s separated from its call: %+v", id, msgs)
		}
	}
	transcript := mock.lastRequest.Load().(llm.Snapshot).Messages[1].Text()
	if strings.Contains(transcript, "lookup") {
		t.Fatalf("kept tool exchange leaked into the transcript: %q", transcript)
	}
}

func TestContextSummarizer_AllKeptSkipsSummarization(t *testing.T) {
	c := llm.NewContext(llm.User("PIN a"), llm.Assistant("PIN b"), llm.User("PIN c"), llm.Assistant("recent"))
	mock := &mockSummaryLLM{summary: "S"}
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{
		MaxUnsummarizedMessages: 2,
		SummaryConfig:           LLMContextSummaryConfig{MinMessagesAfterSummary: 1, Keep: keepMarked},
	}, mock.Summarize)

	if summarizer.ShouldAutoSummarize(c.Snapshot()) {
		t.Fatal("kept messages must not count towards the thresholds")
	}
	if applied, err := summarizer.Summarize(context.Background(), c); err != nil || applied {
		t.Fatalf("nothing to summarize, got applied=%v err=%v", applied, err)
	}
	if mock.calls.Load() != 0 || c.Len() != 4 {
		t.Fatalf("summarizer ran with nothing to summarize: calls=%d len=%d", mock.calls.Load(), c.Len())
	}
}

func TestContextSummarizer_ShouldSummarizeGate(t *testing.T) {
	c := llm.NewContext(llm.User("1"), llm.Assistant("2"), llm.User("3"), llm.Assistant("4"))
	allow := false
	consulted := 0
	gate := func(snapshot llm.Snapshot) bool {
		consulted++
		if len(snapshot.Messages) != 4 {
			t.Errorf("gate saw %d messages", len(snapshot.Messages))
		}
		return allow
	}
	over := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{MaxUnsummarizedMessages: 3, ShouldSummarize: gate}, nil)
	if over.ShouldAutoSummarize(c.Snapshot()) || consulted != 1 {
		t.Fatalf("gate must veto a tripped threshold (consulted %d)", consulted)
	}
	allow = true
	if !over.ShouldAutoSummarize(c.Snapshot()) {
		t.Fatal("gate allowed summarization but it was not triggered")
	}

	consulted = 0
	under := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{MaxUnsummarizedMessages: 10, ShouldSummarize: gate}, nil)
	if under.ShouldAutoSummarize(c.Snapshot()) || consulted != 0 {
		t.Fatalf("gate must not be consulted below the thresholds (consulted %d)", consulted)
	}

	unbounded := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{ShouldSummarize: gate}, nil)
	if !unbounded.ShouldAutoSummarize(c.Snapshot()) {
		t.Fatal("gate alone must be able to trigger summarization")
	}
	if NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{}, nil).ShouldAutoSummarize(c.Snapshot()) {
		t.Fatal("no thresholds and no gate must never trigger")
	}
}

func TestContextSummarizer_GateVetoesAutomaticButNotManual(t *testing.T) {
	c := llm.NewContext(llm.User("1"), llm.Assistant("2"), llm.User("3"), llm.Assistant("4"))
	var runs atomic.Int32
	applied := make(chan [2]int, 1)
	a := NewLLMAssistantAggregator(c, &AssistantAggregatorParams{
		EnableAutoContextSummarization: true,
		AutoSummarizationConfig: LLMAutoContextSummarizationConfig{
			MaxUnsummarizedMessages: 2,
			SummaryConfig:           LLMContextSummaryConfig{MinMessagesAfterSummary: 1},
			ShouldSummarize:         func(llm.Snapshot) bool { return false },
		},
		Summarize: func(context.Context, llm.Snapshot) (string, error) {
			runs.Add(1)
			return "S", nil
		},
		OnSummaryApplied: func(before, after int) { applied <- [2]int{before, after} },
	})
	down := &captureProc{}
	a.Link(down)
	if err := a.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer a.Stop()

	_ = a.QueueFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)
	_ = a.QueueFrame(frames.NewLLMTextFrame("five"), frames.Downstream)
	_ = a.QueueFrame(frames.NewLLMFullResponseEndFrame(), frames.Downstream)
	dataBarrier(t, a, down)
	if c.Len() != 5 || runs.Load() != 0 || a.Summarizer().InFlight() {
		t.Fatalf("vetoed automatic summarization still ran: len=%d runs=%d", c.Len(), runs.Load())
	}

	_ = a.QueueFrame(frames.NewLLMSummarizeContextFrame(), frames.Downstream)
	select {
	case got := <-applied:
		if got != [2]int{5, 2} {
			t.Fatalf("OnSummaryApplied%v, want [5 2]", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("manual summarization frame did not summarize")
	}
}

func TestContextSummarizer_CustomTokenEstimator(t *testing.T) {
	c := llm.NewContext(llm.User(strings.Repeat("a", 400)), llm.Assistant(strings.Repeat("b", 400)), llm.User("c"))
	config := LLMAutoContextSummarizationConfig{MaxContextTokens: 50}
	if !NewLLMContextSummarizer(config, nil).ShouldAutoSummarize(c.Snapshot()) {
		t.Fatal("built-in estimator should trip the token threshold")
	}
	config.EstimateTokens = func(llm.Message) int { return 1 }
	if NewLLMContextSummarizer(config, nil).ShouldAutoSummarize(c.Snapshot()) {
		t.Fatal("custom estimator reports 3 tokens, threshold is 50")
	}
	config.EstimateTokens = func(llm.Message) int { return 20 }
	if !NewLLMContextSummarizer(config, nil).ShouldAutoSummarize(c.Snapshot()) {
		t.Fatal("custom estimator reports 60 tokens, threshold is 50")
	}
}
