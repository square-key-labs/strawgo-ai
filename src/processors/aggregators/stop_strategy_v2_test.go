package aggregators

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

// countingStopper tracks how many times ShouldStop / ShouldStopV2 was called.
type countingStopper struct {
	calls atomic.Int64
	want  user_stop.StopResult // V2 result; mirrored as bool for V1 callers.
	v2    bool                 // when true, exposes ShouldStopV2 too.
}

func (c *countingStopper) ShouldStop(_ any) bool {
	c.calls.Add(1)
	return c.want != user_stop.StopResultContinue
}
func (c *countingStopper) Reset() {}

type countingStopperV2 struct {
	*countingStopper
}

func (c countingStopperV2) ShouldStopV2(_ any) user_stop.StopResult {
	c.calls.Add(1)
	return c.want
}

// activateTurn pushes a frame sequence that puts the user aggregator into
// userTurnActive=true so handleTurnStop actually evaluates strategies.
// The aggregator is set up with a transcription-based start strategy.
func activateTurn(t *testing.T, agg *LLMUserAggregator) {
	t.Helper()
	ctx := context.Background()
	if err := agg.HandleFrame(ctx, frames.NewStartFrame(), frames.Downstream); err != nil {
		t.Fatalf("StartFrame: %v", err)
	}
	if err := agg.HandleFrame(ctx, frames.NewTranscriptionFrame("hi", false), frames.Downstream); err != nil {
		t.Fatalf("Transcription start: %v", err)
	}
}

func TestV2StopShortCircuitsLaterStrategies(t *testing.T) {
	llmCtx := services.NewLLMContext("")
	first := &countingStopper{want: user_stop.StopResultStopShortCircuit, v2: true}
	second := &countingStopper{want: user_stop.StopResultStop}

	agg := NewLLMUserAggregator(llmCtx, turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewTranscriptionUserTurnStartStrategy(true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			countingStopperV2{first},
			second,
		},
	})

	activateTurn(t, agg)

	final := frames.NewTranscriptionFrame("hi", true)
	_ = agg.HandleFrame(context.Background(), final, frames.Downstream)
	time.Sleep(10 * time.Millisecond)

	if got := first.calls.Load(); got == 0 {
		t.Fatal("first strategy was not consulted")
	}
	if got := second.calls.Load(); got != 0 {
		t.Fatalf("second strategy was called despite ShortCircuit; got %d", got)
	}
}

func TestV2StopAllowsLaterStrategies(t *testing.T) {
	llmCtx := services.NewLLMContext("")
	first := &countingStopper{want: user_stop.StopResultStop, v2: true}
	second := &countingStopper{want: user_stop.StopResultContinue}

	agg := NewLLMUserAggregator(llmCtx, turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewTranscriptionUserTurnStartStrategy(true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			countingStopperV2{first},
			second,
		},
	})

	activateTurn(t, agg)

	final := frames.NewTranscriptionFrame("hi", true)
	_ = agg.HandleFrame(context.Background(), final, frames.Downstream)
	time.Sleep(10 * time.Millisecond)

	if got := first.calls.Load(); got == 0 {
		t.Fatal("first strategy was not consulted")
	}
	if got := second.calls.Load(); got == 0 {
		t.Fatal("second strategy must still be evaluated when first returns Stop")
	}
}

func TestV1BoolPreservesShortCircuit(t *testing.T) {
	llmCtx := services.NewLLMContext("")
	first := &countingStopper{want: user_stop.StopResultStop} // returns true via bool
	second := &countingStopper{want: user_stop.StopResultStop}

	agg := NewLLMUserAggregator(llmCtx, turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewTranscriptionUserTurnStartStrategy(true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{first, second},
	})

	activateTurn(t, agg)

	final := frames.NewTranscriptionFrame("hi", true)
	_ = agg.HandleFrame(context.Background(), final, frames.Downstream)
	time.Sleep(10 * time.Millisecond)

	// Legacy bool strategy returning true short-circuits — second never called.
	if got := first.calls.Load(); got == 0 {
		t.Fatal("first strategy was not consulted")
	}
	if got := second.calls.Load(); got != 0 {
		t.Fatalf("V1 bool true should still short-circuit (legacy behavior); got %d", got)
	}
}
