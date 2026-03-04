package src

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/turns"
)

// BenchmarkUserAggregatorConcurrency tests the concurrent access patterns
// in LLMUserAggregator between HandleFrame and aggregationTaskHandler goroutine
func BenchmarkUserAggregatorConcurrency(b *testing.B) {
	ctx := context.Background()
	llmCtx := services.NewLLMContext("You are a helpful assistant")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agg := aggregators.NewLLMUserAggregator(llmCtx, turns.UserTurnStrategies{})

		// Simulate StartFrame to start the goroutine
		startFrame := frames.NewStartFrame()
		agg.HandleFrame(ctx, startFrame, frames.Downstream)

		// Concurrent operations - simulating real usage
		var wg sync.WaitGroup

		// Simulate rapid state changes (like in real voice call)
		wg.Add(3)

		// Goroutine 1: Rapid UserStarted/Stopped speaking
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				agg.HandleFrame(ctx, frames.NewUserStartedSpeakingFrame(), frames.Downstream)
				time.Sleep(time.Microsecond)
				agg.HandleFrame(ctx, frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
			}
		}()

		// Goroutine 2: Rapid transcriptions
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				frame := frames.NewTranscriptionFrame("hello world", true)
				agg.HandleFrame(ctx, frame, frames.Downstream)
				time.Sleep(time.Microsecond)
			}
		}()

		// Goroutine 3: Bot speaking state changes
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				agg.HandleFrame(ctx, frames.NewBotStartedSpeakingFrame(), frames.Upstream)
				time.Sleep(time.Microsecond)
				agg.HandleFrame(ctx, frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
			}
		}()

		wg.Wait()

		// Cleanup
		agg.HandleFrame(ctx, frames.NewEndFrame(), frames.Downstream)
	}
}

// BenchmarkTextAggregatorConcurrency tests SimpleTextAggregator
func BenchmarkTextAggregatorConcurrency(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agg := aggregators.NewSimpleTextAggregator()

		var wg sync.WaitGroup
		wg.Add(2)

		// Writer goroutine
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				agg.Aggregate("Hello world. This is a test. ")
			}
		}()

		// Reader goroutine (simulating flush)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				agg.Flush()
				time.Sleep(time.Microsecond)
			}
		}()

		wg.Wait()
	}
}

// TestRaceDetection is a test specifically for race detection
// Run with: go test -race -run TestRaceDetection ./src/...
func TestRaceDetection(t *testing.T) {
	ctx := context.Background()
	llmCtx := services.NewLLMContext("Test")

	agg := aggregators.NewLLMUserAggregator(llmCtx, turns.UserTurnStrategies{})

	// Start the aggregator
	startFrame := frames.NewStartFrame()
	agg.HandleFrame(ctx, startFrame, frames.Downstream)

	// Run concurrent operations that would trigger races without proper locking
	var wg sync.WaitGroup
	iterations := 1000

	wg.Add(4)

	// Simulate UserStarted/Stopped
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			agg.HandleFrame(ctx, frames.NewUserStartedSpeakingFrame(), frames.Downstream)
			agg.HandleFrame(ctx, frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
		}
	}()

	// Simulate transcriptions
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			frame := frames.NewTranscriptionFrame("test message", true)
			agg.HandleFrame(ctx, frame, frames.Downstream)
		}
	}()

	// Simulate bot speaking
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			agg.HandleFrame(ctx, frames.NewBotStartedSpeakingFrame(), frames.Upstream)
			agg.HandleFrame(ctx, frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
		}
	}()

	// Simulate interruptions
	go func() {
		defer wg.Done()
		for i := 0; i < iterations/10; i++ {
			// Note: InterruptionFrame needs to be created properly
			time.Sleep(time.Millisecond)
		}
	}()

	wg.Wait()

	// Cleanup
	agg.HandleFrame(ctx, frames.NewEndFrame(), frames.Downstream)

	t.Log("Race detection test completed - if no races detected, locks are working")
}
