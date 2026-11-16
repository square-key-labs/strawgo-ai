package aggregators

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/turns"
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

func startAssistantWith(t *testing.T, c *llm.Context, params *AssistantAggregatorParams) (*LLMAssistantAggregator, *captureProc, *captureProc) {
	t.Helper()
	a := NewLLMAssistantAggregator(c, params)
	down, up := &captureProc{}, &captureProc{}
	a.Link(down)
	a.SetPrev(up)
	if err := a.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = a.Stop() })
	_ = a.QueueFrame(frames.NewStartFrameWithConfig(true, turns.UserTurnStrategies{}), frames.Downstream)
	settle(t, a, down)
	return a, down, up
}

func settle(t *testing.T, a *LLMAssistantAggregator, down *captureProc) {
	t.Helper()
	systemBarrier(t, a, down)
	dataBarrier(t, a, down)
}

func deliverResult(t *testing.T, a *LLMAssistantAggregator, down *captureProc, value any, delivery llm.ResultDelivery) {
	t.Helper()
	f := frames.NewFunctionCallResultFrame("1", "book", value, nil)
	f.Delivery = delivery
	_ = a.QueueFrame(f, frames.Downstream)
	settle(t, a, down)
}

func sendAsst(t *testing.T, a *LLMAssistantAggregator, down *captureProc, f frames.Frame) {
	t.Helper()
	_ = a.QueueFrame(f, frames.Downstream)
	settle(t, a, down)
}

func TestAssistantAggregator_ResultDeliveryModes(t *testing.T) {
	cases := []struct {
		name          string
		delivery      llm.ResultDelivery
		botSpeaking   bool
		wantNow       int
		wantInterrupt int
		wantTotal     int
	}{
		{"default idle", llm.ResultDeliveryDefault, false, 1, 0, 1},
		{"default speaking", llm.ResultDeliveryDefault, true, 0, 0, 1},
		{"interrupt idle", llm.ResultDeliveryInterrupt, false, 1, 0, 1},
		{"interrupt speaking", llm.ResultDeliveryInterrupt, true, 0, 1, 1},
		{"when idle idle", llm.ResultDeliveryWhenIdle, false, 1, 0, 1},
		{"when idle speaking", llm.ResultDeliveryWhenIdle, true, 0, 0, 1},
		{"silent idle", llm.ResultDeliverySilent, false, 0, 0, 0},
		{"silent speaking", llm.ResultDeliverySilent, true, 0, 0, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := llm.NewContext(llm.User("book it"))
			a, down, up := startAssistantWith(t, c, nil)
			startRunningCall(t, a, down, false)
			if tc.botSpeaking {
				sendAsst(t, a, down, frames.NewBotStartedSpeakingFrame())
			}

			deliverResult(t, a, down, "booked", tc.delivery)
			if got := toolResultOf(c, "1"); got != `"booked"` {
				t.Fatalf("result not recorded: %q", got)
			}
			if n := countNamed(up.get(), "LLMContextFrame"); n != tc.wantNow {
				t.Fatalf("expected %d inferences on arrival, got %d", tc.wantNow, n)
			}
			if n := countNamed(up.get(), "InterruptionTaskFrame"); n != tc.wantInterrupt {
				t.Fatalf("expected %d interruption requests, got %d", tc.wantInterrupt, n)
			}
			if tc.wantInterrupt > 0 {
				sendAsst(t, a, down, frames.NewInterruptionFrame())
				if n := countNamed(up.get(), "LLMContextFrame"); n != 1 {
					t.Fatalf("interrupting result must run inference once the bot is cut off, got %d", n)
				}
			}
			if tc.botSpeaking {
				sendAsst(t, a, down, frames.NewBotStoppedSpeakingFrame())
			}
			if n := countNamed(up.get(), "LLMContextFrame"); n != tc.wantTotal {
				t.Fatalf("expected %d inferences in total, got %d", tc.wantTotal, n)
			}
			if n := toolResultCount(c, "1"); n != 1 {
				t.Fatalf("expected one tool result, got %d", n)
			}
		})
	}
}

func TestAssistantAggregator_InterruptWithoutInterruptionsWaitsForBot(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	a, down, up := startAssistantAggregator(t, c)
	startRunningCall(t, a, down, false)
	sendAsst(t, a, down, frames.NewBotStartedSpeakingFrame())

	deliverResult(t, a, down, "booked", llm.ResultDeliveryInterrupt)
	if n := countNamed(up.get(), "InterruptionTaskFrame") + countNamed(up.get(), "LLMContextFrame"); n != 0 {
		t.Fatalf("interruptions are off, expected nothing yet, got %d frames", n)
	}
	sendAsst(t, a, down, frames.NewBotStoppedSpeakingFrame())
	if n := countNamed(up.get(), "LLMContextFrame"); n != 1 {
		t.Fatalf("expected one inference after the bot finished, got %d", n)
	}
}

func TestAssistantAggregator_WhenIdleWaitsForUserTurn(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	a, down, up := startAssistantWith(t, c, &AssistantAggregatorParams{WhenIdleUserTurnTimeout: 40 * time.Millisecond})
	startRunningCall(t, a, down, false)

	sendAsst(t, a, down, frames.NewUserStartedSpeakingFrame())
	deliverResult(t, a, down, "booked", llm.ResultDeliveryWhenIdle)
	sendAsst(t, a, down, frames.NewUserStoppedSpeakingFrame())
	if n := countNamed(up.get(), "LLMContextFrame"); n != 0 {
		t.Fatalf("result ran inference before the user turn ended: %d", n)
	}
	if got := toolResultOf(c, "1"); got != `"booked"` {
		t.Fatalf("result not written during the user turn: %q", got)
	}

	sendAsst(t, a, down, frames.NewLLMFullResponseStartFrame())
	sendAsst(t, a, down, frames.NewLLMTextFrame("Booked, and noted."))
	sendAsst(t, a, down, frames.NewLLMFullResponseEndFrame())
	time.Sleep(120 * time.Millisecond)
	settle(t, a, down)
	if n := countNamed(up.get(), "LLMContextFrame"); n != 0 {
		t.Fatalf("the user turn's reply already carried the result, got %d extra inferences", n)
	}
}

func TestAssistantAggregator_WhenIdleRunsAfterEmptyUserTurn(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	a, down, up := startAssistantWith(t, c, &AssistantAggregatorParams{WhenIdleUserTurnTimeout: 40 * time.Millisecond})
	startRunningCall(t, a, down, false)

	sendAsst(t, a, down, frames.NewUserStartedSpeakingFrame())
	deliverResult(t, a, down, "booked", llm.ResultDeliveryWhenIdle)
	sendAsst(t, a, down, frames.NewUserStoppedSpeakingFrame())
	if n := countNamed(up.get(), "LLMContextFrame"); n != 0 {
		t.Fatalf("result ran inference before the user turn ended: %d", n)
	}
	up.waitFor(t, "LLMContextFrame", time.Second)
	settle(t, a, down)
	if n := countNamed(up.get(), "LLMContextFrame"); n != 1 {
		t.Fatalf("expected one inference after the empty turn, got %d", n)
	}
}

func TestAssistantAggregator_WhenIdleResultBetweenSpeechAndReplyWaits(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	a, down, up := startAssistantWith(t, c, &AssistantAggregatorParams{WhenIdleUserTurnTimeout: 60 * time.Millisecond})
	startRunningCall(t, a, down, false)

	sendAsst(t, a, down, frames.NewUserStartedSpeakingFrame())
	sendAsst(t, a, down, frames.NewUserStoppedSpeakingFrame())
	deliverResult(t, a, down, "booked", llm.ResultDeliveryWhenIdle)
	if n := countNamed(up.get(), "LLMContextFrame"); n != 0 {
		t.Fatalf("result ran inference while the user turn was pending: %d", n)
	}
	up.waitFor(t, "LLMContextFrame", time.Second)
	settle(t, a, down)
	if n := countNamed(up.get(), "LLMContextFrame"); n != 1 {
		t.Fatalf("expected one inference once the turn produced no reply, got %d", n)
	}
}

func TestAssistantAggregator_SilentProgressKeepsCallRunning(t *testing.T) {
	c := llm.NewContext(llm.User("write the report"))
	a, down, up := startAssistantWith(t, c, nil)
	startRunningCall(t, a, down, false)

	no := false
	progress := frames.NewFunctionCallResultFrame("1", "book", map[string]any{"percent": 50}, &no)
	progress.Properties = &frames.FunctionCallResultProperties{IsFinal: false, RunLLM: &no}
	sendAsst(t, a, down, progress)

	if !a.HasFunctionCallsInProgress() {
		t.Fatal("progress settled the call")
	}
	if got := toolResultOf(c, "1"); got != toolResultInProgress {
		t.Fatalf("progress replaced the placeholder: %q", got)
	}
	found := false
	for _, m := range c.Messages() {
		if m.Role == llm.RoleDeveloper && strings.Contains(m.Text(), `{"percent":50}`) {
			found = true
		}
	}
	if !found {
		t.Fatal("progress not written to the context")
	}
	if n := countNamed(up.get(), "LLMContextFrame"); n != 0 {
		t.Fatalf("silent progress ran inference: %d", n)
	}

	deliverResult(t, a, down, "done", llm.ResultDeliveryDefault)
	if n := countNamed(up.get(), "LLMContextFrame"); n != 1 {
		t.Fatalf("final result must run inference once, got %d", n)
	}
}

func TestAssistantAggregator_SurvivingCallDeliversOnceAfterInterruption(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	a, down, up := startAssistantWith(t, c, nil)
	startRunningCall(t, a, down, false)

	sendAsst(t, a, down, frames.NewBotStartedSpeakingFrame())
	sendAsst(t, a, down, frames.NewUserStartedSpeakingFrame())
	sendAsst(t, a, down, frames.NewInterruptionFrame())
	sendAsst(t, a, down, frames.NewBotStoppedSpeakingFrame())
	sendAsst(t, a, down, frames.NewUserStoppedSpeakingFrame())
	if !a.HasFunctionCallsInProgress() {
		t.Fatal("interruption settled a call that survives interruptions")
	}

	deliverResult(t, a, down, "booked", llm.ResultDeliveryDefault)
	deliverResult(t, a, down, "booked again", llm.ResultDeliveryDefault)
	sendAsst(t, a, down, frames.NewBotStoppedSpeakingFrame())
	if got := toolResultOf(c, "1"); got != `"booked"` || toolResultCount(c, "1") != 1 {
		t.Fatalf("expected one booked result, got %q x%d", got, toolResultCount(c, "1"))
	}
	if n := countNamed(up.get(), "LLMContextFrame"); n != 1 {
		t.Fatalf("expected exactly one follow-up inference, got %d", n)
	}
}

func TestAssistantAggregator_LLMCancelledCallRecordsCancelled(t *testing.T) {
	c := llm.NewContext(llm.User("write the report"))
	a, down, up := startAssistantWith(t, c, nil)
	startRunningCall(t, a, down, false)

	cancel := frames.NewFunctionCallCancelFrame("1", "book")
	cancel.Result = llm.ToolResultCancelled
	sendAsst(t, a, down, cancel)
	if got := toolResultOf(c, "1"); got != llm.ToolResultCancelled {
		t.Fatalf("expected cancelled result, got %q", got)
	}
	if a.HasFunctionCallsInProgress() || countNamed(up.get(), "LLMContextFrame") != 0 {
		t.Fatal("cancelled call must settle without inference")
	}
}

func TestAssistantAggregator_NewOptionsOffByDefault(t *testing.T) {
	if got := *DefaultAssistantAggregatorParams(); got.AddToolChangeMessages || got.WhenIdleUserTurnTimeout != 0 {
		t.Fatalf("new options must be off by default: %+v", got)
	}
	if opts := llm.DefaultFunctionOptions(); opts.CancellableByLLM || opts.Delivery != llm.ResultDeliveryDefault {
		t.Fatalf("new function options must be off by default: %+v", opts)
	}
}

func TestToolChangeMessages(t *testing.T) {
	tools := func(names ...string) llm.Tools {
		var out llm.Tools
		for _, n := range names {
			out.Functions = append(out.Functions, llm.FunctionSchema{Name: n})
		}
		return out
	}
	added, _ := ToolChangeMessage(tools("a"), tools("a", "c", "b"))
	if want := "The following function(s) have just been added and may now be called: `b`, `c`. Any previously available functions remain available."; added != want {
		t.Fatalf("added message = %q", added)
	}
	both, _ := ToolChangeMessage(tools("a", "b"), tools("b", "c"))
	if want := "The following function(s) have just been added and may now be called: `c`. Any previously available functions remain available. The following function(s) have just been removed and should not be called: `a`. Any previously available functions remain available. The removed function(s) may become available again later, in which case you will be informed."; both != want {
		t.Fatalf("change message = %q", both)
	}
	if _, ok := ToolChangeMessage(tools("a", "b"), tools("b", "a")); ok {
		t.Fatal("same tools must not produce a message")
	}

	for _, enabled := range []bool{false, true} {
		c := llm.NewContext(llm.User("hi"))
		c.SetTools(tools("lookup"))
		pair := NewContextAggregatorPair(c, ContextAggregatorPairParams{Assistant: AssistantAggregatorParams{AddToolChangeMessages: enabled}})
		u, a := pair.User(), pair.Assistant()
		userDown, asstDown := &captureProc{}, &captureProc{}
		u.Link(userDown)
		u.SetPrev(&captureProc{})
		a.Link(asstDown)
		a.SetPrev(&captureProc{})
		for _, p := range []interface{ Start(context.Context) error }{u, a} {
			if err := p.Start(context.Background()); err != nil {
				t.Fatal(err)
			}
		}
		_ = u.QueueFrame(frames.NewLLMSetToolsFrame(tools("book")), frames.Downstream)
		userDown.waitFor(t, "LLMSetToolsFrame", time.Second)
		_ = a.QueueFrame(frames.NewLLMSetToolsFrame(tools("book")), frames.Downstream)
		dataBarrier(t, a, asstDown)
		_ = u.Stop()
		_ = a.Stop()

		var notes []string
		for _, m := range c.Messages() {
			if m.Role == llm.RoleDeveloper {
				notes = append(notes, m.Text())
			}
		}
		if !enabled {
			if len(notes) != 0 {
				t.Fatalf("tool change messages are opt-in, got %q", notes)
			}
			continue
		}
		if len(notes) != 1 || !strings.Contains(notes[0], "added and may now be called: `book`") || !strings.Contains(notes[0], "removed and should not be called: `lookup`") {
			t.Fatalf("expected one tool change message, got %q", notes)
		}
	}
}
