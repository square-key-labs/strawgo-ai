package aggregators

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

func TestContextSummarizer_AutoTokenThreshold(t *testing.T) {
	ctx := &services.LLMContext{Messages: []services.LLMMessage{
		{Role: "user", Content: strings.Repeat("a", 200)},
		{Role: "assistant", Content: strings.Repeat("b", 200)},
		{Role: "user", Content: strings.Repeat("c", 200)},
	}}

	mock := &mockSummaryLLM{summary: "compressed context"}
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{
		MaxContextTokens: 80,
		SummaryConfig: LLMContextSummaryConfig{
			MinMessagesAfterSummary: 1,
		},
	}, mock)

	if !summarizer.ShouldAutoSummarize(ctx) {
		t.Fatal("expected auto summarization trigger on token threshold")
	}

	if !summarizer.SummarizeContext(context.Background(), ctx, nil) {
		t.Fatal("expected summarization to be applied")
	}

	if len(ctx.Messages) >= 3 {
		t.Fatalf("expected summarized context to be shorter, got %d messages", len(ctx.Messages))
	}
	if ctx.Messages[0].Role != "user" {
		t.Fatalf("expected summary role to be user, got %s", ctx.Messages[0].Role)
	}
}

func TestContextSummarizer_AutoMessageThreshold(t *testing.T) {
	ctx := &services.LLMContext{Messages: []services.LLMMessage{
		{Role: "user", Content: "1"},
		{Role: "assistant", Content: "2"},
		{Role: "user", Content: "3"},
		{Role: "assistant", Content: "4"},
		{Role: "user", Content: "5"},
	}}

	mock := &mockSummaryLLM{summary: "message-threshold summary"}
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{
		MaxContextTokens:        10000,
		MaxUnsummarizedMessages: 3,
		SummaryConfig: LLMContextSummaryConfig{
			MinMessagesAfterSummary: 1,
		},
	}, mock)

	if !summarizer.ShouldAutoSummarize(ctx) {
		t.Fatal("expected auto summarization trigger on message threshold")
	}

	if !summarizer.SummarizeContext(context.Background(), ctx, nil) {
		t.Fatal("expected summarization to be applied")
	}
}

func TestContextSummarizer_FunctionCallPreserved(t *testing.T) {
	ctx := &services.LLMContext{Messages: []services.LLMMessage{
		{Role: "user", Content: "Earlier chat details that should be summarized"},
		{Role: "assistant", Content: "Ack"},
		{Role: "assistant", ToolCalls: []services.ToolCall{{
			ID:   "call-1",
			Type: "function",
			Function: services.FunctionCall{
				Name:      "get_weather",
				Arguments: `{"city":"SF"}`,
			},
		}}},
		{Role: "tool", ToolCallID: "call-1", Content: "IN_PROGRESS"},
	}}

	mock := &mockSummaryLLM{summary: "function-aware summary"}
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{
		MaxContextTokens: 30,
		SummaryConfig: LLMContextSummaryConfig{
			MinMessagesAfterSummary: 1,
		},
	}, mock)

	if !summarizer.SummarizeContext(context.Background(), ctx, nil) {
		t.Fatal("expected summarization to be applied")
	}

	if len(ctx.Messages) < 3 {
		t.Fatalf("expected summary plus preserved function messages, got %d messages", len(ctx.Messages))
	}

	lastAssistant := ctx.Messages[len(ctx.Messages)-2]
	lastTool := ctx.Messages[len(ctx.Messages)-1]
	if len(lastAssistant.ToolCalls) == 0 || lastAssistant.ToolCalls[0].ID != "call-1" {
		t.Fatal("expected trailing incomplete tool call to be preserved")
	}
	if lastTool.Role != "tool" || lastTool.ToolCallID != "call-1" || lastTool.Content != "IN_PROGRESS" {
		t.Fatal("expected trailing incomplete tool response to be preserved")
	}
}

func TestContextSummarizer_DedicatedLLMTimeout(t *testing.T) {
	ctx := &services.LLMContext{Messages: []services.LLMMessage{
		{Role: "user", Content: strings.Repeat("timeout", 30)},
		{Role: "assistant", Content: strings.Repeat("test", 30)},
	}}

	hung := &mockSummaryLLM{waitForCancel: true}
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{MaxContextTokens: 20}, hung)
	summarizer.SetTimeout(30 * time.Millisecond)

	start := time.Now()
	applied := summarizer.SummarizeContext(context.Background(), ctx, nil)
	duration := time.Since(start)

	if applied {
		t.Fatal("expected timeout to skip summarization")
	}
	if duration > 300*time.Millisecond {
		t.Fatalf("expected timeout cancellation to be fast, took %s", duration)
	}
	if !hung.cancelled.Load() {
		t.Fatal("expected dedicated summarization context to be cancelled on timeout")
	}
}

func TestContextSummarizer_SummaryRoleIsUser(t *testing.T) {
	ctx := &services.LLMContext{Messages: []services.LLMMessage{
		{Role: "user", Content: "a"},
		{Role: "assistant", Content: "b"},
	}}

	mock := &mockSummaryLLM{summary: "role test"}
	summarizer := NewLLMContextSummarizer(LLMAutoContextSummarizationConfig{MaxContextTokens: 10}, mock)

	if !summarizer.SummarizeContext(context.Background(), ctx, nil) {
		t.Fatal("expected summarization to be applied")
	}

	if ctx.Messages[0].Role != "user" {
		t.Fatalf("expected summary role user, got %s", ctx.Messages[0].Role)
	}
}

func TestContextSummarizer_OnDemand(t *testing.T) {
	ctx := &services.LLMContext{Messages: []services.LLMMessage{
		{Role: "user", Content: strings.Repeat("hello", 40)},
		{Role: "assistant", Content: strings.Repeat("world", 40)},
		{Role: "user", Content: strings.Repeat("again", 40)},
	}}

	assistant := NewLLMAssistantAggregator(ctx, &AssistantAggregatorParams{
		EnableAutoContextSummarization: false,
		AutoSummarizationConfig: LLMAutoContextSummarizationConfig{
			MaxContextTokens: 30,
			SummaryConfig: LLMContextSummaryConfig{
				MinMessagesAfterSummary: 1,
			},
		},
		SummaryLLM: &mockSummaryLLM{summary: "forced on-demand summary"},
	})

	if err := assistant.HandleFrame(context.Background(), frames.NewLLMSummarizeContextFrame(), frames.Downstream); err != nil {
		t.Fatalf("expected on-demand frame to be handled, got error: %v", err)
	}

	if len(ctx.Messages) >= 3 {
		t.Fatalf("expected context to be summarized on-demand, got %d messages", len(ctx.Messages))
	}
	if ctx.Messages[0].Role != "user" {
		t.Fatalf("expected summary role user on-demand, got %s", ctx.Messages[0].Role)
	}
}

type mockSummaryLLM struct {
	summary       string
	waitForCancel bool
	cancelled     atomic.Bool
}

func (m *mockSummaryLLM) SummarizeContext(ctx context.Context, _ string, _ *services.LLMContext) (string, error) {
	if m.waitForCancel {
		<-ctx.Done()
		m.cancelled.Store(true)
		return "", ctx.Err()
	}
	return m.summary, nil
}

func (m *mockSummaryLLM) ProcessFrame(context.Context, frames.Frame, frames.FrameDirection) error {
	return nil
}

func (m *mockSummaryLLM) QueueFrame(frames.Frame, frames.FrameDirection) error {
	return nil
}

func (m *mockSummaryLLM) PushFrame(frames.Frame, frames.FrameDirection) error {
	return nil
}

func (m *mockSummaryLLM) Link(processors.FrameProcessor) {}

func (m *mockSummaryLLM) SetPrev(processors.FrameProcessor) {}

func (m *mockSummaryLLM) Start(context.Context) error { return nil }

func (m *mockSummaryLLM) Stop() error { return nil }

func (m *mockSummaryLLM) Name() string { return "mockSummaryLLM" }

func (m *mockSummaryLLM) Initialize(context.Context) error { return nil }

func (m *mockSummaryLLM) Cleanup() error { return nil }

func (m *mockSummaryLLM) SetModel(string) {}

func (m *mockSummaryLLM) SetSystemPrompt(string) {}

func (m *mockSummaryLLM) SetTemperature(float64) {}

func (m *mockSummaryLLM) AddMessage(string, string) {}

func (m *mockSummaryLLM) ClearContext() {}
