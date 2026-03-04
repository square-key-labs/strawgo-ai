package aggregators

import (
	"context"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// TestTextAggregationMode_Token verifies that TOKEN mode passes text through immediately
// without buffering for sentence boundaries.
func TestTextAggregationMode_Token(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create SentenceAggregator in TOKEN mode
	aggregator := NewSentenceAggregator(TextAggregationModeToken)

	// Verify mode is set correctly
	if aggregator.mode != TextAggregationModeToken {
		t.Errorf("Expected mode to be TextAggregationModeToken, got %v", aggregator.mode)
	}

	// Send StartFrame (should pass through)
	startFrame := frames.NewStartFrame()
	if err := aggregator.HandleFrame(ctx, startFrame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(StartFrame) failed: %v", err)
	}

	// Test 1: Send partial text "Hello" - should be processed immediately
	llmFrame1 := frames.NewLLMTextFrame("Hello")
	if err := aggregator.HandleFrame(ctx, llmFrame1, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(LLMTextFrame 'Hello') failed: %v", err)
	}

	// Test 2: Send partial text " world" - should be processed immediately
	llmFrame2 := frames.NewLLMTextFrame(" world")
	if err := aggregator.HandleFrame(ctx, llmFrame2, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(LLMTextFrame ' world') failed: %v", err)
	}

	// Test 3: Send text with sentence boundary "!" - should be processed immediately
	llmFrame3 := frames.NewLLMTextFrame("!")
	if err := aggregator.HandleFrame(ctx, llmFrame3, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(LLMTextFrame '!') failed: %v", err)
	}

	// Verify buffer is empty (all text passed through immediately)
	if aggregator.buffer.Len() > 0 {
		t.Errorf("Expected empty buffer in TOKEN mode, got %d bytes", aggregator.buffer.Len())
	}
}

// TestTextAggregationMode_Sentence verifies that SENTENCE mode buffers text
// until sentence boundaries are detected.
func TestTextAggregationMode_Sentence(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create SentenceAggregator in SENTENCE mode (default)
	aggregator := NewSentenceAggregator(TextAggregationModeSentence)

	// Verify mode is set correctly
	if aggregator.mode != TextAggregationModeSentence {
		t.Errorf("Expected mode to be TextAggregationModeSentence, got %v", aggregator.mode)
	}

	// Send StartFrame
	startFrame := frames.NewStartFrame()
	if err := aggregator.HandleFrame(ctx, startFrame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(StartFrame) failed: %v", err)
	}

	// Send partial text (no sentence boundary yet)
	llmFrame1 := frames.NewLLMTextFrame("Hello")
	if err := aggregator.HandleFrame(ctx, llmFrame1, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(LLMTextFrame 'Hello') failed: %v", err)
	}

	// Verify text is buffered (not passed through)
	if aggregator.buffer.Len() == 0 {
		t.Errorf("Expected buffered text in SENTENCE mode after 'Hello', got empty buffer")
	}

	// Send text with sentence boundary
	llmFrame2 := frames.NewLLMTextFrame(" world.")
	if err := aggregator.HandleFrame(ctx, llmFrame2, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(LLMTextFrame ' world.') failed: %v", err)
	}

	// Verify buffer is cleared after sentence boundary
	// (text was extracted and pushed downstream)
	if aggregator.buffer.Len() > 0 {
		t.Errorf("Expected empty buffer after sentence boundary, got %d bytes: %q",
			aggregator.buffer.Len(), aggregator.buffer.String())
	}
}

// TestTextAggregationMode_Default verifies that default mode is SENTENCE
func TestTextAggregationMode_Default(t *testing.T) {
	// Create SentenceAggregator without specifying mode
	aggregator := NewSentenceAggregator()

	if aggregator.mode != TextAggregationModeSentence {
		t.Errorf("Expected default mode to be TextAggregationModeSentence, got %v", aggregator.mode)
	}
}

// TestTextAggregationMode_TokenMultipleChunks verifies TOKEN mode with multiple chunks
func TestTextAggregationMode_TokenMultipleChunks(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create SentenceAggregator in TOKEN mode
	aggregator := NewSentenceAggregator(TextAggregationModeToken)

	// Send StartFrame
	startFrame := frames.NewStartFrame()
	if err := aggregator.HandleFrame(ctx, startFrame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(StartFrame) failed: %v", err)
	}

	// Send 3 chunks that would normally be buffered in SENTENCE mode
	chunks := []string{"The quick", " brown", " fox."}
	for _, chunk := range chunks {
		llmFrame := frames.NewLLMTextFrame(chunk)
		if err := aggregator.HandleFrame(ctx, llmFrame, frames.Downstream); err != nil {
			t.Fatalf("HandleFrame(LLMTextFrame %q) failed: %v", chunk, err)
		}
	}

	// Verify buffer is empty (all chunks passed through immediately)
	if aggregator.buffer.Len() > 0 {
		t.Errorf("Expected empty buffer in TOKEN mode after 3 chunks, got %d bytes: %q",
			aggregator.buffer.Len(), aggregator.buffer.String())
	}
}

// TestTextAggregationMode_SentenceMultipleSentences verifies SENTENCE mode with multiple sentences
func TestTextAggregationMode_SentenceMultipleSentences(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create SentenceAggregator in SENTENCE mode
	aggregator := NewSentenceAggregator(TextAggregationModeSentence)

	// Send StartFrame
	startFrame := frames.NewStartFrame()
	if err := aggregator.HandleFrame(ctx, startFrame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(StartFrame) failed: %v", err)
	}

	// Send text with multiple sentences
	llmFrame := frames.NewLLMTextFrame("First sentence. Second sentence. Third")
	if err := aggregator.HandleFrame(ctx, llmFrame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(LLMTextFrame) failed: %v", err)
	}

	// Verify buffer contains only the incomplete sentence
	buffered := aggregator.buffer.String()
	if buffered != " Third" && buffered != "Third" {
		t.Errorf("Expected buffer to contain 'Third' (with or without leading space), got %q", buffered)
	}
}
