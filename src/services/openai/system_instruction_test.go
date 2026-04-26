package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

type testCapturer struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (c *testCapturer) QueueFrame(frame frames.Frame, _ frames.FrameDirection) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.frames = append(c.frames, frame)
	return nil
}
func (c *testCapturer) ProcessFrame(_ context.Context, _ frames.Frame, _ frames.FrameDirection) error {
	return nil
}
func (c *testCapturer) PushFrame(_ frames.Frame, _ frames.FrameDirection) error { return nil }
func (c *testCapturer) Link(_ processors.FrameProcessor)                        {}
func (c *testCapturer) SetPrev(_ processors.FrameProcessor)                     {}
func (c *testCapturer) Start(_ context.Context) error                           { return nil }
func (c *testCapturer) Stop() error                                             { return nil }
func (c *testCapturer) Name() string                                            { return "TestCapturer" }

// captureFirstSystemMessage runs the service against a fake server and
// returns the system message content from the request body, or "" if no
// system message was sent.
func captureFirstSystemMessage(t *testing.T, cfg LLMConfig, ctxPrompt string) string {
	t.Helper()
	var captured map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		// Stream a [DONE] terminator so the service exits cleanly.
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		w.(http.Flusher).Flush()
	}))
	t.Cleanup(server.Close)

	cfg.BaseURL = server.URL
	cfg.APIKey = "test-key"
	if cfg.Model == "" {
		cfg.Model = "gpt-4o-mini"
	}
	svc := NewLLMService(cfg)
	ctx := context.Background()
	if err := svc.Initialize(ctx); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	t.Cleanup(func() { _ = svc.Cleanup() })

	svc.Link(&testCapturer{})

	llmCtx := services.NewLLMContext(ctxPrompt)
	llmCtx.AddUserMessage("hi")
	if err := svc.HandleFrame(ctx, frames.NewLLMContextFrame(llmCtx), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}

	msgs, ok := captured["messages"].([]interface{})
	if !ok || len(msgs) == 0 {
		return ""
	}
	first := msgs[0].(map[string]interface{})
	if first["role"] != "system" {
		return ""
	}
	return first["content"].(string)
}

func TestSystemInstructionWinsOverContextPrompt(t *testing.T) {
	got := captureFirstSystemMessage(t, LLMConfig{SystemInstruction: "service-level"}, "context-level")
	if got != "service-level" {
		t.Fatalf("expected SystemInstruction to win, got %q", got)
	}
}

func TestSystemInstructionWithoutContextPrompt(t *testing.T) {
	got := captureFirstSystemMessage(t, LLMConfig{SystemInstruction: "service-only"}, "")
	if got != "service-only" {
		t.Fatalf("expected service-only, got %q", got)
	}
}

func TestContextPromptWithoutSystemInstruction(t *testing.T) {
	got := captureFirstSystemMessage(t, LLMConfig{}, "context-only")
	if got != "context-only" {
		t.Fatalf("expected context-only, got %q", got)
	}
}

func TestNoSystemPromptAtAll(t *testing.T) {
	got := captureFirstSystemMessage(t, LLMConfig{}, "")
	if got != "" {
		t.Fatalf("expected no system message, got %q", got)
	}
}
