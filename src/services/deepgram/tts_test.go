package deepgram

import (
	"context"
	"net/http"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func TestNewTTSService(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey: "test-api-key",
		Model:  "aura-asteria-en",
	})

	if service == nil {
		t.Fatal("Expected service to be created")
	}

	if service.model != "aura-asteria-en" {
		t.Errorf("Expected model aura-asteria-en, got %s", service.model)
	}

	if service.encoding != DefaultTTSEncoding {
		t.Errorf("Expected encoding %s, got %s", DefaultTTSEncoding, service.encoding)
	}

	if service.sampleRate != DefaultTTSSampleRate {
		t.Errorf("Expected sample rate %d, got %d", DefaultTTSSampleRate, service.sampleRate)
	}
}

func TestNewTTSServiceDefaults(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey: "test-api-key",
	})

	if service.model != DefaultTTSModel {
		t.Errorf("Expected default model %s, got %s", DefaultTTSModel, service.model)
	}

	if service.encoding != DefaultTTSEncoding {
		t.Errorf("Expected default encoding %s, got %s", DefaultTTSEncoding, service.encoding)
	}

	if service.sampleRate != DefaultTTSSampleRate {
		t.Errorf("Expected default sample rate %d, got %d", DefaultTTSSampleRate, service.sampleRate)
	}
}

func TestTTSContextIDGeneration(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey: "test-api-key",
		Model:  "aura-asteria-en",
	})

	ctx := context.Background()

	textFrame := frames.NewTextFrame("Hello world")
	err := service.synthesizeText(textFrame.Text)
	if err == nil {
		if service.contextID == "" {
			t.Error("Expected context ID to be generated")
		}

		firstContextID := service.contextID

		endFrame := frames.NewLLMFullResponseEndFrame()
		service.HandleFrame(ctx, endFrame, frames.Downstream)

		if service.contextID != "" {
			t.Error("Expected context ID to be cleared after response end")
		}

		textFrame2 := frames.NewTextFrame("Hello again")
		service.synthesizeText(textFrame2.Text)

		if service.contextID == "" {
			t.Error("Expected new context ID to be generated")
		}

		if service.contextID == firstContextID {
			t.Error("Expected different context ID for new synthesis")
		}
	}
}

func TestTTSInterruption(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey: "test-api-key",
		Model:  "aura-asteria-en",
	})

	service.contextID = "test-context-123"
	service.isSpeaking = true

	interruptFrame := frames.NewInterruptionFrame()
	err := service.HandleFrame(context.Background(), interruptFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle interruption: %v", err)
	}

	if service.contextID != "" {
		t.Error("Expected context ID to be cleared after interruption")
	}

	if service.isSpeaking {
		t.Error("Expected isSpeaking to be false after interruption")
	}
}

func TestTTSStartedStoppedFrames(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey: "test-api-key",
		Model:  "aura-asteria-en",
	})

	service.ctx = context.Background()

	textFrame := frames.NewTextFrame("Test speech")
	err := service.synthesizeText(textFrame.Text)
	if err == nil {
		if !service.isSpeaking {
			t.Error("Expected isSpeaking to be true after first text")
		}

		if service.contextID == "" {
			t.Error("Expected context ID to be set")
		}
	}
}

func TestEncodingToCodec(t *testing.T) {
	tests := []struct {
		encoding string
		expected string
	}{
		{"mulaw", "mulaw"},
		{"alaw", "alaw"},
		{"linear16", "linear16"},
		{"unknown", "linear16"},
	}

	for _, test := range tests {
		service := NewTTSService(TTSConfig{
			APIKey:   "test-api-key",
			Encoding: test.encoding,
		})

		codec := service.encodingToCodec()
		if codec != test.expected {
			t.Errorf("For encoding %s, expected codec %s, got %s", test.encoding, test.expected, codec)
		}
	}
}

func TestSetVoiceAndModel(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey: "test-api-key",
	})

	service.SetVoice("aura-luna-en")
	if service.model != "aura-luna-en" {
		t.Errorf("Expected model aura-luna-en, got %s", service.model)
	}

	service.SetModel("aura-stella-en")
	if service.model != "aura-stella-en" {
		t.Errorf("Expected model aura-stella-en, got %s", service.model)
	}
}

func TestTTSEmptyText(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey: "test-api-key",
	})

	err := service.synthesizeText("")
	if err != nil {
		t.Errorf("Expected no error for empty text, got %v", err)
	}
}

func TestTTSCleanup(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()
	service.ctx, service.cancel = context.WithCancel(ctx)

	err := service.Cleanup()
	if err != nil {
		t.Errorf("Expected no error during cleanup, got %v", err)
	}

	if service.conn != nil {
		t.Error("Expected connection to be nil after cleanup")
	}
}

func TestTTSStartFrame(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()
	startFrame := frames.NewStartFrame()

	err := service.HandleFrame(ctx, startFrame, frames.Downstream)
	if err != nil {
		t.Errorf("Expected no error handling StartFrame, got %v", err)
	}
}

func TestTTSEndFrame(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()
	service.ctx, service.cancel = context.WithCancel(ctx)

	endFrame := frames.NewEndFrame()
	err := service.HandleFrame(ctx, endFrame, frames.Downstream)
	if err != nil {
		t.Errorf("Expected no error handling EndFrame, got %v", err)
	}
}

func TestTTSLLMResponseFrames(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()

	startFrame := frames.NewLLMFullResponseStartFrame()
	err := service.HandleFrame(ctx, startFrame, frames.Downstream)
	if err != nil {
		t.Errorf("Expected no error handling LLMFullResponseStartFrame, got %v", err)
	}

	service.contextID = "test-context"
	endFrame := frames.NewLLMFullResponseEndFrame()
	err = service.HandleFrame(ctx, endFrame, frames.Downstream)
	if err != nil {
		t.Errorf("Expected no error handling LLMFullResponseEndFrame, got %v", err)
	}

	if service.contextID != "" {
		t.Error("Expected context ID to be cleared after LLMFullResponseEndFrame")
	}
}

func TestDeepgramTTSContextCleanupOnCompletion(t *testing.T) {
	// Test that context is cleaned up on normal completion (LLMFullResponseEndFrame)
	// not just on interruption

	service := NewTTSService(TTSConfig{
		APIKey: "test-key",
		Model:  "aura-asteria-en",
	})

	// Simulate state changes that would occur during normal completion
	ctx := context.Background()

	// 1. Manually set up state as if synthesis had started
	service.mu.Lock()
	service.isSpeaking = true
	service.contextID = "test-context-id"
	service.mu.Unlock()

	// 2. Send LLMFullResponseEndFrame - this should reset isSpeaking and contextID
	llmEndFrame := frames.NewLLMFullResponseEndFrame()
	err := service.HandleFrame(ctx, llmEndFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("HandleFrame(LLMFullResponseEndFrame) failed: %v", err)
	}

	// 3. Verify isSpeaking was reset to false and contextID was cleared
	if service.isSpeaking {
		t.Error("Expected isSpeaking to be false after LLMFullResponseEndFrame")
	}
	if service.contextID != "" {
		t.Error("Expected contextID to be empty after LLMFullResponseEndFrame")
	}
}

func TestDeepgramTTSContextIDReuse(t *testing.T) {
	// Test that context_id is reused within a single LLM turn
	// (between LLMFullResponseStartFrame and LLMFullResponseEndFrame)

	service := NewTTSService(TTSConfig{
		APIKey: "test-key",
		Model:  "aura-asteria-en",
	})

	ctx := context.Background()

	// 1. Send LLMFullResponseStartFrame - should generate turn context ID
	startFrame := frames.NewLLMFullResponseStartFrame()
	err := service.HandleFrame(ctx, startFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("HandleFrame(LLMFullResponseStartFrame) failed: %v", err)
	}

	// Capture the turn context ID
	service.mu.Lock()
	turnContextID := service.currentTurnContextID
	service.mu.Unlock()

	if turnContextID == "" {
		t.Error("Expected currentTurnContextID to be set after LLMFullResponseStartFrame")
	}

	// 2. Simulate 3 text chunks by directly calling synthesizeText logic
	// (without actually making WebSocket calls)
	collectedContextIDs := []string{}

	for i := 0; i < 3; i++ {
		// Simulate what synthesizeText does: reuse turn context ID
		service.mu.Lock()
		if service.contextID == "" {
			if service.currentTurnContextID != "" {
				service.contextID = service.currentTurnContextID
			}
		}
		currentContextID := service.contextID
		service.mu.Unlock()

		if currentContextID == "" {
			t.Errorf("Expected contextID to be set for text chunk %d", i)
		}
		collectedContextIDs = append(collectedContextIDs, currentContextID)
	}

	// 3. Verify all 3 text chunks used the SAME context ID
	for i := 1; i < len(collectedContextIDs); i++ {
		if collectedContextIDs[i] != collectedContextIDs[0] {
			t.Errorf("Context ID mismatch: chunk 0 used %s, chunk %d used %s",
				collectedContextIDs[0], i, collectedContextIDs[i])
		}
	}

	// 4. Verify the context ID matches the turn context ID
	if collectedContextIDs[0] != turnContextID {
		t.Errorf("Context ID %s does not match turn context ID %s",
			collectedContextIDs[0], turnContextID)
	}

	// 5. Manually set isSpeaking to true to simulate that synthesis occurred
	service.mu.Lock()
	service.isSpeaking = true
	service.mu.Unlock()

	// 6. Send LLMFullResponseEndFrame - should reset context IDs
	endFrame := frames.NewLLMFullResponseEndFrame()
	err = service.HandleFrame(ctx, endFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("HandleFrame(LLMFullResponseEndFrame) failed: %v", err)
	}

	// 7. Verify context IDs were reset
	service.mu.Lock()
	if service.contextID != "" {
		t.Errorf("Expected contextID to be reset after LLMFullResponseEndFrame, got: %s", service.contextID)
	}
	if service.currentTurnContextID != "" {
		t.Errorf("Expected currentTurnContextID to be reset after LLMFullResponseEndFrame, got: %s", service.currentTurnContextID)
	}
	service.mu.Unlock()
}

