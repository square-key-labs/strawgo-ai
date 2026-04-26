package aggregators

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// setupAssistantAggregator wires up an LLMAssistantAggregator with capture
// processors on both ends. Returns aggregator, downstream capture, upstream
// capture (where LLM-trigger context frames go), and the LLMContext for
// inspection.
func setupAssistantAggregator(t *testing.T, params *AssistantAggregatorParams) (*LLMAssistantAggregator, *captureProc, *captureProc, *services.LLMContext) {
	t.Helper()
	llmCtx := services.NewLLMContext("system")
	agg := NewLLMAssistantAggregator(llmCtx, params)
	down := &captureProc{}
	up := &captureProc{}
	agg.Link(down)
	agg.SetPrev(up)
	if err := agg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = agg.Stop() })
	return agg, down, up, llmCtx
}

// sendAndWait queues a frame and gives the aggregator goroutine time to handle it.
func sendAndWait(t *testing.T, a *LLMAssistantAggregator, f frames.Frame) {
	t.Helper()
	if err := a.QueueFrame(f, frames.Downstream); err != nil {
		t.Fatalf("QueueFrame: %v", err)
	}
	time.Sleep(20 * time.Millisecond)
}

// countByName returns how many frames in the slice share the given Name().
func countByName(fs []frames.Frame, name string) int {
	n := 0
	for _, f := range fs {
		if f.Name() == name {
			n++
		}
	}
	return n
}

// TestGroupParallelToolsSingleTrigger verifies that for a 2-call batch the
// LLM is triggered exactly once — after the second result lands — not after
// each individual result.
func TestGroupParallelToolsSingleTrigger(t *testing.T) {
	agg, _, up, _ := setupAssistantAggregator(t, &AssistantAggregatorParams{})

	calls := []frames.FunctionCallInfo{
		{ToolCallID: "c1", FunctionName: "fn_a"},
		{ToolCallID: "c2", FunctionName: "fn_b"},
	}
	sendAndWait(t, agg, frames.NewFunctionCallsStartedFrame(calls))
	sendAndWait(t, agg, frames.NewFunctionCallInProgressFrame("c1", "fn_a", map[string]interface{}{}, true))
	sendAndWait(t, agg, frames.NewFunctionCallInProgressFrame("c2", "fn_b", map[string]interface{}{}, true))

	// First result — must NOT trigger LLM.
	sendAndWait(t, agg, frames.NewFunctionCallResultFrame("c1", "fn_a", "ok", nil))
	if got := countByName(up.get(), "LLMContextFrame"); got != 0 {
		t.Fatalf("expected no LLM trigger after first result, got %d upstream context frames", got)
	}

	// Second result — must trigger exactly once.
	sendAndWait(t, agg, frames.NewFunctionCallResultFrame("c2", "fn_b", "ok", nil))
	if got := countByName(up.get(), "LLMContextFrame"); got != 1 {
		t.Fatalf("expected single LLM trigger after group complete, got %d", got)
	}
}

// TestAsyncFunctionCallInjectsDeveloperMessage verifies that when
// FunctionRegistry says CancelOnInterruption=false, the result is injected
// as a developer message and the LLM is triggered immediately.
func TestAsyncFunctionCallInjectsDeveloperMessage(t *testing.T) {
	reg := services.NewInMemoryFunctionRegistry()
	_ = reg.Register(services.RegisteredFunction{
		Name:                 "search",
		CancelOnInterruption: false, // async
	})

	agg, _, up, llmCtx := setupAssistantAggregator(t, &AssistantAggregatorParams{
		FunctionRegistry: reg,
	})

	sendAndWait(t, agg, frames.NewFunctionCallsStartedFrame([]frames.FunctionCallInfo{
		{ToolCallID: "c1", FunctionName: "search"},
	}))
	sendAndWait(t, agg, frames.NewFunctionCallInProgressFrame("c1", "search", map[string]interface{}{"q": "go"}, true /* will be overridden by registry */))
	sendAndWait(t, agg, frames.NewFunctionCallResultFrame("c1", "search", []string{"hit-1", "hit-2"}, nil))

	// LLM should be triggered immediately for an async result.
	if got := countByName(up.get(), "LLMContextFrame"); got != 1 {
		t.Fatalf("expected one LLM trigger after async result, got %d", got)
	}

	// Developer message must be appended to the context.
	var found bool
	for _, m := range llmCtx.Messages {
		if m.Role == "developer" && strings.Contains(m.Content, "search") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected developer message describing search result, got %+v", llmCtx.Messages)
	}
}

// TestPerToolTimeoutCancelsCall verifies that a registered function with a
// short TimeoutSecs causes the aggregator to push a FunctionCallCancelFrame
// upstream when no result arrives in time.
func TestPerToolTimeoutCancelsCall(t *testing.T) {
	timeout := 50 * time.Millisecond
	reg := services.NewInMemoryFunctionRegistry()
	_ = reg.Register(services.RegisteredFunction{
		Name:                 "slow_lookup",
		TimeoutSecs:          &timeout,
		CancelOnInterruption: true,
	})

	agg, _, up, _ := setupAssistantAggregator(t, &AssistantAggregatorParams{
		FunctionRegistry: reg,
	})

	sendAndWait(t, agg, frames.NewFunctionCallsStartedFrame([]frames.FunctionCallInfo{
		{ToolCallID: "c1", FunctionName: "slow_lookup"},
	}))
	sendAndWait(t, agg, frames.NewFunctionCallInProgressFrame("c1", "slow_lookup", map[string]interface{}{}, true))

	// Wait past the timeout. Don't send a result.
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if countByName(up.get(), "FunctionCallCancelFrame") > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if got := countByName(up.get(), "FunctionCallCancelFrame"); got == 0 {
		t.Fatalf("expected FunctionCallCancelFrame upstream after timeout")
	}
}

// TestLegacyPathStillTriggersLLM verifies backwards compat: when no
// FunctionCallsStartedFrame is sent (legacy LLM service path), a result
// frame still triggers the LLM via the empty-pending fallback.
func TestLegacyPathStillTriggersLLM(t *testing.T) {
	agg, _, up, _ := setupAssistantAggregator(t, &AssistantAggregatorParams{})

	// Skip FunctionCallsStartedFrame — go straight to in-progress + result.
	sendAndWait(t, agg, frames.NewFunctionCallInProgressFrame("c1", "ping", map[string]interface{}{}, true))
	sendAndWait(t, agg, frames.NewFunctionCallResultFrame("c1", "ping", "pong", nil))

	if got := countByName(up.get(), "LLMContextFrame"); got != 1 {
		t.Fatalf("expected legacy single trigger, got %d", got)
	}
}

// TestExplicitRunLLMOverrides checks that an explicit RunLLM=false on the
// result frame suppresses the trigger even when the group is complete.
func TestExplicitRunLLMOverrides(t *testing.T) {
	agg, _, up, _ := setupAssistantAggregator(t, &AssistantAggregatorParams{})

	sendAndWait(t, agg, frames.NewFunctionCallsStartedFrame([]frames.FunctionCallInfo{
		{ToolCallID: "c1", FunctionName: "fn"},
	}))
	sendAndWait(t, agg, frames.NewFunctionCallInProgressFrame("c1", "fn", map[string]interface{}{}, true))

	no := false
	sendAndWait(t, agg, frames.NewFunctionCallResultFrame("c1", "fn", "ok", &no))

	if got := countByName(up.get(), "LLMContextFrame"); got != 0 {
		t.Fatalf("expected RunLLM=false to suppress trigger, got %d", got)
	}
}
