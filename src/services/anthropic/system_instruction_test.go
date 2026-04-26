package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// captureSystem spins up a fake Anthropic server and returns the "system"
// field from the request body sent by the service.
func captureSystem(t *testing.T, cfg LLMConfig, ctxPrompt string) interface{} {
	t.Helper()
	var captured map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		writeSSE(w, w.(http.Flusher), "message_stop", map[string]interface{}{"type": "message_stop"})
	}))
	t.Cleanup(server.Close)

	cfg.BaseURL = server.URL
	cfg.APIKey = "test-key"
	svc := NewLLMService(cfg)
	ctx := context.Background()
	if err := svc.Initialize(ctx); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	t.Cleanup(func() { _ = svc.Cleanup() })

	svc.Link(&frameCapturer{})

	llmCtx := services.NewLLMContext(ctxPrompt)
	llmCtx.AddUserMessage("hi")
	if err := svc.HandleFrame(ctx, frames.NewLLMContextFrame(llmCtx), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}
	return captured["system"]
}

func TestSystemInstructionWinsOverContextPrompt(t *testing.T) {
	got := captureSystem(t, LLMConfig{SystemInstruction: "service-level"}, "context-level")
	if got != "service-level" {
		t.Fatalf("expected SystemInstruction to win, got %v", got)
	}
}

func TestSystemInstructionWithoutContextPrompt(t *testing.T) {
	got := captureSystem(t, LLMConfig{SystemInstruction: "service-only"}, "")
	if got != "service-only" {
		t.Fatalf("expected service-only, got %v", got)
	}
}

func TestContextPromptWithoutSystemInstruction(t *testing.T) {
	got := captureSystem(t, LLMConfig{}, "context-only")
	if got != "context-only" {
		t.Fatalf("expected context-only, got %v", got)
	}
}

func TestNoSystemPromptAtAll(t *testing.T) {
	got := captureSystem(t, LLMConfig{}, "")
	if got != nil {
		t.Fatalf("expected absent system field, got %v", got)
	}
}
