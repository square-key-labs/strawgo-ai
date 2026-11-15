package aggregators

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
)

func startAssistantAggregator(t *testing.T, c *llm.Context) (*LLMAssistantAggregator, *captureProc, *captureProc) {
	t.Helper()
	a := NewLLMAssistantAggregator(c, nil)
	down, up := &captureProc{}, &captureProc{}
	a.Link(down)
	a.SetPrev(up)
	if err := a.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = a.Stop() })
	return a, down, up
}

func toolResultOf(c *llm.Context, id string) string {
	for _, m := range c.Messages() {
		for _, p := range m.Parts {
			if r, ok := p.(llm.ToolResultPart); ok && r.ToolCallID == id {
				return r.Content
			}
		}
	}
	return ""
}

func systemBarrier(t *testing.T, a *LLMAssistantAggregator, down *captureProc) {
	t.Helper()
	before := countNamed(down.get(), "MetricsFrame")
	_ = a.QueueFrame(frames.NewMetricsFrame(), frames.Downstream)
	deadline := time.Now().Add(time.Second)
	for countNamed(down.get(), "MetricsFrame") == before {
		if time.Now().After(deadline) {
			t.Fatal("system barrier timed out")
		}
		time.Sleep(time.Millisecond)
	}
}

func dataBarrier(t *testing.T, a *LLMAssistantAggregator, down *captureProc) {
	t.Helper()
	before := countNamed(down.get(), "TextFrame")
	_ = a.QueueFrame(frames.NewTextFrame(""), frames.Downstream)
	deadline := time.Now().Add(time.Second)
	for countNamed(down.get(), "TextFrame") == before {
		if time.Now().After(deadline) {
			t.Fatal("data barrier timed out")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestAssistantAggregator_CancelBeforeInProgressRecordsCancelled(t *testing.T) {
	c := llm.NewContext(llm.User("hi"))
	a, down, up := startAssistantAggregator(t, c)

	_ = a.QueueFrame(frames.NewFunctionCallsStartedFrame([]frames.FunctionCallInfo{{ToolCallID: "1", FunctionName: "lookup"}}), frames.Downstream)
	cancel := frames.NewFunctionCallCancelFrame("1", "lookup")
	cancel.RunLLM = true
	_ = a.QueueFrame(cancel, frames.Downstream)
	systemBarrier(t, a, down)
	_ = a.QueueFrame(frames.NewFunctionCallInProgressFrame("1", "lookup", nil, true), frames.Downstream)
	dataBarrier(t, a, down)

	if a.HasFunctionCallsInProgress() {
		t.Fatal("cancelled call must not stay in progress")
	}
	if got := toolResultOf(c, "1"); got != llm.ToolResultInterrupted {
		t.Fatalf("expected %s, got %q", llm.ToolResultInterrupted, got)
	}
	up.waitFor(t, "LLMContextFrame", time.Second)
}

func TestAssistantAggregator_CancelAfterInProgressRecordsCancelled(t *testing.T) {
	c := llm.NewContext(llm.User("hi"))
	a, down, _ := startAssistantAggregator(t, c)

	_ = a.QueueFrame(frames.NewFunctionCallsStartedFrame([]frames.FunctionCallInfo{{ToolCallID: "1", FunctionName: "lookup"}}), frames.Downstream)
	systemBarrier(t, a, down)
	_ = a.QueueFrame(frames.NewFunctionCallInProgressFrame("1", "lookup", nil, true), frames.Downstream)
	dataBarrier(t, a, down)
	_ = a.QueueFrame(frames.NewFunctionCallCancelFrame("1", "lookup"), frames.Downstream)
	systemBarrier(t, a, down)
	_ = a.QueueFrame(frames.NewFunctionCallInProgressFrame("1", "lookup", nil, true), frames.Downstream)
	dataBarrier(t, a, down)

	if a.HasFunctionCallsInProgress() {
		t.Fatal("cancelled call must not stay in progress")
	}
	calls := 0
	for _, m := range c.Messages() {
		for _, p := range m.Parts {
			if _, ok := p.(llm.ToolCallPart); ok {
				calls++
			}
		}
	}
	if got := toolResultOf(c, "1"); got != llm.ToolResultInterrupted || calls != 1 {
		t.Fatalf("expected one cancelled tool call, got %q with %d calls", got, calls)
	}
}

func startRunningCall(t *testing.T, a *LLMAssistantAggregator, down *captureProc, cancelOnInterruption bool) {
	t.Helper()
	_ = a.QueueFrame(frames.NewFunctionCallsStartedFrame([]frames.FunctionCallInfo{{ToolCallID: "1", FunctionName: "book"}}), frames.Downstream)
	systemBarrier(t, a, down)
	_ = a.QueueFrame(frames.NewFunctionCallInProgressFrame("1", "book", nil, cancelOnInterruption), frames.Downstream)
	dataBarrier(t, a, down)
}

func toolResultCount(c *llm.Context, id string) int {
	n := 0
	for _, m := range c.Messages() {
		if r, ok := m.ToolResult(); ok && r.ToolCallID == id {
			n++
		}
	}
	return n
}

func TestAssistantAggregator_InterruptedCallRecordsUnknownOutcome(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	a, down, _ := startAssistantAggregator(t, c)
	startRunningCall(t, a, down, true)

	_ = a.QueueFrame(frames.NewFunctionCallCancelFrame("1", "book"), frames.Downstream)
	systemBarrier(t, a, down)

	got := toolResultOf(c, "1")
	if got != llm.ToolResultInterrupted {
		t.Fatalf("expected interrupted marker, got %q", got)
	}
	if got == "CANCELLED" || !strings.Contains(got, "outcome is unknown") {
		t.Fatalf("interrupted marker must say the outcome is unknown, got %q", got)
	}
}

func TestAssistantAggregator_LateResultAfterCancelIsIgnored(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	a, down, up := startAssistantAggregator(t, c)
	startRunningCall(t, a, down, true)

	_ = a.QueueFrame(frames.NewFunctionCallCancelFrame("1", "book"), frames.Downstream)
	systemBarrier(t, a, down)
	before := c.Len()
	_ = a.QueueFrame(frames.NewFunctionCallResultFrame("1", "book", "booked", nil), frames.Downstream)
	dataBarrier(t, a, down)

	if got := toolResultOf(c, "1"); got != llm.ToolResultInterrupted {
		t.Fatalf("late result overwrote the interrupted marker: %q", got)
	}
	if c.Len() != before || toolResultCount(c, "1") != 1 {
		t.Fatalf("late result duplicated the tool result: %d messages, %d results", c.Len(), toolResultCount(c, "1"))
	}
	if n := countNamed(up.get(), "LLMContextFrame"); n != 0 {
		t.Fatalf("late result triggered %d inferences", n)
	}
}

func TestAssistantAggregator_DuplicateResultIsIgnored(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	a, down, up := startAssistantAggregator(t, c)
	startRunningCall(t, a, down, false)

	_ = a.QueueFrame(frames.NewFunctionCallResultFrame("1", "book", "booked", nil), frames.Downstream)
	dataBarrier(t, a, down)
	_ = a.QueueFrame(frames.NewFunctionCallResultFrame("1", "book", "booked twice", nil), frames.Downstream)
	dataBarrier(t, a, down)

	if got := toolResultOf(c, "1"); got != `"booked"` {
		t.Fatalf("duplicate result overwrote the first: %q", got)
	}
	if n := countNamed(up.get(), "LLMContextFrame"); n != 1 {
		t.Fatalf("expected one inference, got %d", n)
	}
}

func TestAssistantAggregator_UserSpeechDuringToolKeepsResultAndDefersInference(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	a, down, up := startAssistantAggregator(t, c)
	startRunningCall(t, a, down, false)

	_ = a.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	_ = a.QueueFrame(frames.NewInterruptionFrame(), frames.Downstream)
	systemBarrier(t, a, down)
	if !a.HasFunctionCallsInProgress() {
		t.Fatal("user speech must not settle a running call")
	}
	if got := toolResultOf(c, "1"); got != toolResultInProgress {
		t.Fatalf("running call changed by user speech: %q", got)
	}

	_ = a.QueueFrame(frames.NewFunctionCallResultFrame("1", "book", "booked", nil), frames.Downstream)
	dataBarrier(t, a, down)
	if got := toolResultOf(c, "1"); got != `"booked"` {
		t.Fatalf("result lost while user was speaking: %q", got)
	}
	if n := countNamed(up.get(), "LLMContextFrame"); n != 0 {
		t.Fatalf("inference ran while user was speaking: %d", n)
	}

	_ = a.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	up.waitFor(t, "LLMContextFrame", time.Second)
	systemBarrier(t, a, down)
	dataBarrier(t, a, down)
	if n := countNamed(up.get(), "LLMContextFrame"); n != 1 {
		t.Fatalf("expected exactly one follow-up inference, got %d", n)
	}
}

func TestAssistantAggregator_ToolResultRunsInferenceImmediatelyWhenUserSilent(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	a, down, up := startAssistantAggregator(t, c)
	startRunningCall(t, a, down, false)

	_ = a.QueueFrame(frames.NewFunctionCallResultFrame("1", "book", "booked", nil), frames.Downstream)
	dataBarrier(t, a, down)
	if got := toolResultOf(c, "1"); got != `"booked"` {
		t.Fatalf("result not recorded: %q", got)
	}
	if n := countNamed(up.get(), "LLMContextFrame"); n != 1 {
		t.Fatalf("expected one immediate inference, got %d", n)
	}
}

func TestAssistantAggregator_NilResultFollowsRunLLM(t *testing.T) {
	no := false
	yes := true
	cases := []struct {
		name   string
		runLLM *bool
		want   int
	}{
		{"default", nil, 1},
		{"true", &yes, 1},
		{"false", &no, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := llm.NewContext(llm.User("hang up"))
			a, down, up := startAssistantAggregator(t, c)
			startRunningCall(t, a, down, false)

			_ = a.QueueFrame(frames.NewFunctionCallResultFrame("1", "book", nil, tc.runLLM), frames.Downstream)
			dataBarrier(t, a, down)
			if got := toolResultOf(c, "1"); got != toolResultCompleted {
				t.Fatalf("nil result recorded as %q, want %q", got, toolResultCompleted)
			}
			if n := countNamed(up.get(), "LLMContextFrame"); n != tc.want {
				t.Fatalf("expected %d inferences, got %d", tc.want, n)
			}
		})
	}
}
