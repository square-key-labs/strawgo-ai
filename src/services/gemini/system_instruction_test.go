package gemini

import "testing"

// Gemini's chat LLM hardcodes its endpoint URL today. End-to-end behavior
// is exercised across the OpenAI-compatible services. For Gemini we verify
// the constructor wires the config into the service struct, and that
// SystemInstruction precedence logic happens in code by inspecting the
// service struct after construction.

func TestGeminiSystemInstructionFieldWired(t *testing.T) {
	svc := NewLLMService(LLMConfig{
		APIKey:            "test",
		Model:             "gemini-2.5-flash",
		SystemPrompt:      "context-level",
		SystemInstruction: "service-level",
	})

	if svc.systemInstruction != "service-level" {
		t.Fatalf("expected systemInstruction='service-level', got %q", svc.systemInstruction)
	}
	if svc.context.SystemPrompt != "context-level" {
		t.Fatalf("expected context.SystemPrompt='context-level', got %q", svc.context.SystemPrompt)
	}
}

func TestGeminiSystemInstructionEmpty(t *testing.T) {
	svc := NewLLMService(LLMConfig{
		APIKey:       "test",
		Model:        "gemini-2.5-flash",
		SystemPrompt: "context-only",
	})
	if svc.systemInstruction != "" {
		t.Fatalf("expected empty systemInstruction, got %q", svc.systemInstruction)
	}
	if svc.context.SystemPrompt != "context-only" {
		t.Fatalf("expected context.SystemPrompt='context-only', got %q", svc.context.SystemPrompt)
	}
}
