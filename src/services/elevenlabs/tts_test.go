package elevenlabs

import (
	"context"
	"strings"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

func TestElevenLabsTTSContextIDGeneration(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey:       "test-key",
		VoiceID:      "test-voice",
		Model:        "eleven_turbo_v2_5",
		UseStreaming: false,
	})

	if service.HasActiveAudioContext() {
		t.Errorf("Expected no active audio context before initialization, got: %s", service.GetActiveAudioContextID())
	}

	contextID1 := services.GenerateContextID()
	contextID2 := services.GenerateContextID()

	if contextID1 == "" || contextID2 == "" {
		t.Error("GenerateContextID returned empty string")
	}

	if contextID1 == contextID2 {
		t.Errorf("GenerateContextID should generate unique IDs, got: %s == %s", contextID1, contextID2)
	}

	if len(contextID1) != 36 {
		t.Errorf("Expected UUID length 36, got: %d", len(contextID1))
	}
	if contextID1[8] != '-' || contextID1[13] != '-' || contextID1[18] != '-' || contextID1[23] != '-' {
		t.Errorf("Invalid UUID format: %s", contextID1)
	}
}

func TestElevenLabsTTSContextIDConsistency(t *testing.T) {
	contextID1 := services.GenerateContextID()
	contextID2 := services.GenerateContextID()
	contextID3 := services.GenerateContextID()

	contextIDs := []string{contextID1, contextID2, contextID3}

	for i, id := range contextIDs {
		if id == "" {
			t.Errorf("Context ID %d is empty", i)
		}

		if len(id) != 36 {
			t.Errorf("Context ID %d has wrong length: expected 36, got %d", i, len(id))
		}

		for j := i + 1; j < len(contextIDs); j++ {
			if id == contextIDs[j] {
				t.Errorf("Context IDs %d and %d are identical: %s", i, j, id)
			}
		}
	}
}

func TestElevenLabsTTSContextIDPattern(t *testing.T) {
	contextIDs := make([]string, 10)
	for i := 0; i < 10; i++ {
		contextIDs[i] = services.GenerateContextID()
	}

	for i, id := range contextIDs {
		if len(id) != 36 {
			t.Errorf("Expected UUID length 36, got: %d", len(id))
		}
		hyphenCount := strings.Count(id, "-")
		if hyphenCount != 4 {
			t.Errorf("Expected 4 hyphens in UUID, got: %d", hyphenCount)
		}

		for j := i + 1; j < len(contextIDs); j++ {
			if id == contextIDs[j] {
				t.Errorf("Duplicate context IDs found at positions %d and %d: %s", i, j, id)
			}
		}
	}
}
func TestElevenLabsTTSContextCleanupOnCompletion(t *testing.T) {
	// Test that context is cleaned up on normal completion (LLMFullResponseEndFrame)
	// not just on interruption

	service := NewTTSService(TTSConfig{
		APIKey:       "test-key",
		VoiceID:      "test-voice",
		Model:        "eleven_turbo_v2_5",
		UseStreaming: false, // Non-streaming mode for simplicity
	})

	// Simulate state changes that would occur during normal completion
	ctx := context.Background()

	// 1. Manually set isSpeaking to true (simulating that synthesis started)
	service.mu.Lock()
	service.isSpeaking = true
	service.mu.Unlock()

	// 2. Send LLMFullResponseEndFrame - this should reset isSpeaking
	llmEndFrame := frames.NewLLMFullResponseEndFrame()
	err := service.HandleFrame(ctx, llmEndFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("HandleFrame(LLMFullResponseEndFrame) failed: %v", err)
	}

	// 3. Verify isSpeaking was reset to false
	if service.isSpeaking {
		t.Error("Expected isSpeaking to be false after LLMFullResponseEndFrame")
	}
}

func TestElevenLabsTTSContextIDReuse(t *testing.T) {
	// Test that context_id is reused within a single LLM turn
	// (between LLMFullResponseStartFrame and LLMFullResponseEndFrame)

	service := NewTTSService(TTSConfig{
		APIKey:       "test-key",
		VoiceID:      "test-voice",
		Model:        "eleven_turbo_v2_5",
		UseStreaming: false, // Non-streaming mode for simplicity
	})

	ctx := context.Background()

	// 1. Send LLMFullResponseStartFrame - should generate turn context ID
	startFrame := frames.NewLLMFullResponseStartFrame()
	err := service.HandleFrame(ctx, startFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("HandleFrame(LLMFullResponseStartFrame) failed: %v", err)
	}

	// Capture the turn context ID via AudioContextManager
	turnContextID := service.GetTurnContextID()

	if turnContextID == "" {
		t.Error("Expected currentTurnContextID to be set after LLMFullResponseStartFrame")
	}

	// 2. Simulate 3 text chunks by calling GetOrCreateContextID (same as synthesizeText)
	collectedContextIDs := []string{}

	for i := 0; i < 3; i++ {
		// Use AudioContextManager method (same as refactored synthesizeText)
		currentContextID := service.GetOrCreateContextID()

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

	// 7. Verify context IDs were reset via AudioContextManager
	if service.HasActiveAudioContext() {
		t.Errorf("Expected contextID to be reset after LLMFullResponseEndFrame, got: %s", service.GetActiveAudioContextID())
	}
	if service.GetTurnContextID() != "" {
		t.Errorf("Expected currentTurnContextID to be reset after LLMFullResponseEndFrame, got: %s", service.GetTurnContextID())
	}
}
