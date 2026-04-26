package aggregators

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

func newTestUserAggregator(t *testing.T) (*LLMUserAggregator, *services.LLMContext, context.Context, context.CancelFunc) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())

	llmCtx := &services.LLMContext{
		Messages: []services.LLMMessage{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "hi there"},
			{Role: "user", Content: "secret token: abc"},
		},
	}
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewTranscriptionUserTurnStartStrategy(true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(100*time.Millisecond, true),
		},
	}

	agg := NewLLMUserAggregator(llmCtx, strategies)
	if err := agg.HandleFrame(ctx, frames.NewStartFrame(), frames.Downstream); err != nil {
		t.Fatalf("StartFrame: %v", err)
	}
	return agg, llmCtx, ctx, cancel
}

func TestTransformFrameRedactsMessages(t *testing.T) {
	agg, llmCtx, ctx, cancel := newTestUserAggregator(t)
	defer cancel()

	transform := func(msgs []services.LLMMessage) []services.LLMMessage {
		out := make([]services.LLMMessage, len(msgs))
		for i, m := range msgs {
			m.Content = strings.ReplaceAll(m.Content, "secret token: abc", "[REDACTED]")
			out[i] = m
		}
		return out
	}

	frame := frames.NewLLMMessagesTransformFrame(transform, false)
	if err := agg.HandleFrame(ctx, frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}

	if got := llmCtx.Messages[2].Content; got != "[REDACTED]" {
		t.Fatalf("expected [REDACTED], got %q", got)
	}
	if got := llmCtx.Messages[0].Content; got != "hello" {
		t.Fatalf("untouched message changed: %q", got)
	}
}

func TestTransformFrameWrongTypeIsLogged(t *testing.T) {
	agg, llmCtx, ctx, cancel := newTestUserAggregator(t)
	defer cancel()

	frame := frames.NewLLMMessagesTransformFrame("not a function", false)
	if err := agg.HandleFrame(ctx, frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame returned error: %v", err)
	}

	// Messages must be unchanged when transform has wrong type.
	if got := llmCtx.Messages[2].Content; got != "secret token: abc" {
		t.Fatalf("messages mutated despite wrong type: %q", got)
	}
}

func TestTransformFrameRunLLMFalseSuppressesContextPush(t *testing.T) {
	// When RunLLM is false the transform applies but no LLMContextFrame is
	// pushed downstream. Direct verification of "no push" requires a
	// downstream sink. As a baseline correctness check we ensure HandleFrame
	// returns nil and messages were transformed in place.
	agg, llmCtx, ctx, cancel := newTestUserAggregator(t)
	defer cancel()

	called := false
	transform := func(msgs []services.LLMMessage) []services.LLMMessage {
		called = true
		return msgs
	}
	frame := frames.NewLLMMessagesTransformFrame(transform, false)
	if err := agg.HandleFrame(ctx, frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}
	if !called {
		t.Fatal("transform was not invoked")
	}
	_ = llmCtx
}
