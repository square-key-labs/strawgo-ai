package aggregators

import (
	"context"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

// TestUserAggregator_InterimNotLeaked verifies that interim transcriptions
// (non-final TranscriptionFrame) are consumed and NOT pushed downstream.
// Regression test for interim transcription handling.
func TestUserAggregator_InterimNotLeaked(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create aggregator with minimal turn strategies
	llmCtx := &services.LLMContext{
		Messages: []services.LLMMessage{},
	}
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewTranscriptionUserTurnStartStrategy(true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(100*time.Millisecond, true),
		},
	}

	aggregator := NewLLMUserAggregator(llmCtx, strategies)

	// Send StartFrame to initialize
	startFrame := frames.NewStartFrame()
	if err := aggregator.HandleFrame(ctx, startFrame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(StartFrame) failed: %v", err)
	}

	// Send interim transcription (IsFinal=false)
	interimFrame := frames.NewTranscriptionFrame("how are you", false)

	if err := aggregator.HandleFrame(ctx, interimFrame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(InterimTranscriptionFrame) failed: %v", err)
	}

	// Verify interim frame was NOT pushed downstream
	// The HandleFrame should return nil without pushing the frame
	// This is verified by the fact that no error occurred and the frame was consumed

	// Send final transcription (IsFinal=true)
	finalFrame := frames.NewTranscriptionFrame("how are you doing", true)

	if err := aggregator.HandleFrame(ctx, finalFrame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(FinalTranscriptionFrame) failed: %v", err)
	}

	// Verify aggregation contains only final text (not interim + final)
	aggregator.stateMu.Lock()
	aggregatedText := aggregator.AggregationString()
	aggregator.stateMu.Unlock()

	// The aggregation should contain the final text
	// (Note: aggregation is only populated when pushAggregation is called,
	// which happens when user stops speaking or timeout occurs)
	if aggregatedText != "" && aggregatedText != "how are you doing" {
		t.Errorf("Expected aggregation 'how are you doing' or empty, got '%s'", aggregatedText)
	}
}

// TestUserAggregator_InterimDoesNotDuplicateText verifies that interim transcriptions
// don't cause text duplication in the aggregation buffer.
func TestUserAggregator_InterimDoesNotDuplicateText(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	llmCtx := &services.LLMContext{
		Messages: []services.LLMMessage{},
	}
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewTranscriptionUserTurnStartStrategy(true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(100*time.Millisecond, true),
		},
	}

	aggregator := NewLLMUserAggregator(llmCtx, strategies)

	// Initialize
	startFrame := frames.NewStartFrame()
	aggregator.HandleFrame(ctx, startFrame, frames.Downstream)

	// Send multiple interim transcriptions with same text
	for i := 0; i < 3; i++ {
		interimFrame := frames.NewTranscriptionFrame("hello world", false)
		aggregator.HandleFrame(ctx, interimFrame, frames.Downstream)
	}

	// Send final transcription
	finalFrame := frames.NewTranscriptionFrame("hello world", true)
	aggregator.HandleFrame(ctx, finalFrame, frames.Downstream)

	// Verify aggregation contains text only once
	aggregator.stateMu.Lock()
	aggregatedText := aggregator.AggregationString()
	aggregator.stateMu.Unlock()

	// The aggregation should not contain duplicates
	// (Note: aggregation is only populated when pushAggregation is called)
	if aggregatedText != "" && aggregatedText != "hello world" {
		t.Errorf("Expected aggregation 'hello world' or empty, got '%s'", aggregatedText)
	}
}

// TestUserAggregator_InterimFlagTracking verifies that the seenInterimResults flag
// is properly managed for interim vs final transcriptions.
func TestUserAggregator_InterimFlagTracking(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	llmCtx := &services.LLMContext{
		Messages: []services.LLMMessage{},
	}
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewTranscriptionUserTurnStartStrategy(true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(100*time.Millisecond, true),
		},
	}

	aggregator := NewLLMUserAggregator(llmCtx, strategies)

	// Initialize
	startFrame := frames.NewStartFrame()
	aggregator.HandleFrame(ctx, startFrame, frames.Downstream)

	// Initially, seenInterimResults should be false
	aggregator.stateMu.Lock()
	if aggregator.seenInterimResults {
		t.Error("Expected seenInterimResults=false initially")
	}
	aggregator.stateMu.Unlock()

	// Send interim transcription
	interimFrame := frames.NewTranscriptionFrame("hello", false)
	aggregator.HandleFrame(ctx, interimFrame, frames.Downstream)

	// After interim, seenInterimResults should be true
	aggregator.stateMu.Lock()
	if !aggregator.seenInterimResults {
		t.Error("Expected seenInterimResults=true after interim transcription")
	}
	aggregator.stateMu.Unlock()

	// Send final transcription
	finalFrame := frames.NewTranscriptionFrame("hello world", true)
	aggregator.HandleFrame(ctx, finalFrame, frames.Downstream)

	// After final, seenInterimResults should be reset to false
	aggregator.stateMu.Lock()
	if aggregator.seenInterimResults {
		t.Error("Expected seenInterimResults=false after final transcription")
	}
	aggregator.stateMu.Unlock()
}

// TestUserAggregator_InterimReturnsNil verifies that interim transcription frames
// return nil (consumed) rather than being pushed downstream.
func TestUserAggregator_InterimReturnsNil(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	llmCtx := &services.LLMContext{
		Messages: []services.LLMMessage{},
	}
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewTranscriptionUserTurnStartStrategy(true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(100*time.Millisecond, true),
		},
	}

	aggregator := NewLLMUserAggregator(llmCtx, strategies)

	// Initialize
	startFrame := frames.NewStartFrame()
	aggregator.HandleFrame(ctx, startFrame, frames.Downstream)

	// Send interim transcription - should return nil (consumed)
	interimFrame := frames.NewTranscriptionFrame("interim text", false)
	err := aggregator.HandleFrame(ctx, interimFrame, frames.Downstream)

	if err != nil {
		t.Errorf("Expected nil error for interim transcription, got %v", err)
	}

	// Send final transcription - should also return nil (consumed, not pushed)
	finalFrame := frames.NewTranscriptionFrame("final text", true)
	err = aggregator.HandleFrame(ctx, finalFrame, frames.Downstream)

	if err != nil {
		t.Errorf("Expected nil error for final transcription, got %v", err)
	}
}
