package sarvam

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

func TestNewLLMServiceDefaults(t *testing.T) {
	s := NewLLMService(LLMConfig{APIKey: "k"})
	if s.model != DefaultSarvamModel {
		t.Errorf("expected default model %s, got %s", DefaultSarvamModel, s.model)
	}
	if s.baseURL != DefaultSarvamBaseURL {
		t.Errorf("expected default base URL, got %s", s.baseURL)
	}
}

func TestNewLLMServiceOverrides(t *testing.T) {
	s := NewLLMService(LLMConfig{
		APIKey:       "k",
		Model:        "sarvam-30b",
		SystemPrompt: "act helpful",
		Temperature:  0.5,
		BaseURL:      "https://example.local/v1",
	})
	if s.model != "sarvam-30b" {
		t.Errorf("model override broken: %s", s.model)
	}
	if s.baseURL != "https://example.local/v1" {
		t.Errorf("baseURL override broken: %s", s.baseURL)
	}
	if s.temperature != 0.5 {
		t.Errorf("temperature override broken: %f", s.temperature)
	}
	if s.context.SystemPrompt != "act helpful" {
		t.Errorf("system prompt broken: %s", s.context.SystemPrompt)
	}
}

func TestLLMServiceMessageManagement(t *testing.T) {
	s := NewLLMService(LLMConfig{APIKey: "k"})
	s.AddMessage("user", "hi")
	s.AddMessage("assistant", "hello")
	if len(s.context.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(s.context.Messages))
	}
	s.ClearContext()
	if len(s.context.Messages) != 0 {
		t.Fatalf("expected empty after Clear")
	}
}

func TestLLMServiceInitializeCleanup(t *testing.T) {
	s := NewLLMService(LLMConfig{APIKey: "k"})
	if err := s.Initialize(context.Background()); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if s.ctx == nil || s.cancel == nil {
		t.Fatal("ctx/cancel should be set")
	}
	if err := s.Cleanup(); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
}

func TestLLMServiceStreamsContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			http.NotFound(w, r)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("missing/incorrect bearer: %q", got)
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("body decode: %v", err)
		}
		if body["stream"] != true {
			t.Errorf("expected stream=true")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		writeChunk := func(content string) {
			payload := map[string]interface{}{
				"choices": []map[string]interface{}{
					{"delta": map[string]interface{}{"content": content}},
				},
			}
			b, _ := json.Marshal(payload)
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
		}
		writeChunk("Hello ")
		writeChunk("world")
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	s := NewLLMService(LLMConfig{APIKey: "test-key", BaseURL: srv.URL})
	if err := s.Initialize(context.Background()); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	defer s.Cleanup()

	llmCtx := services.NewLLMContext("system")
	llmCtx.AddUserMessage("Hi")

	if err := s.generateResponseFromContext(llmCtx); err != nil {
		t.Fatalf("generateResponseFromContext: %v", err)
	}

	if len(llmCtx.Messages) < 2 || llmCtx.Messages[len(llmCtx.Messages)-1].Role != "assistant" {
		t.Fatalf("expected assistant message appended, got %#v", llmCtx.Messages)
	}
	if llmCtx.Messages[len(llmCtx.Messages)-1].Content != "Hello world" {
		t.Errorf("expected concatenated SSE content, got %q", llmCtx.Messages[len(llmCtx.Messages)-1].Content)
	}
}

func TestLLMServiceMapsDeveloperToUser(t *testing.T) {
	captured := make(chan map[string]interface{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)
		captured <- body
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	s := NewLLMService(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	if err := s.Initialize(context.Background()); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	defer s.Cleanup()

	llmCtx := services.NewLLMContext("")
	llmCtx.Messages = append(llmCtx.Messages, services.LLMMessage{Role: "developer", Content: "tool said x"})

	if err := s.generateResponseFromContext(llmCtx); err != nil {
		t.Fatalf("generateResponseFromContext: %v", err)
	}

	body := <-captured
	msgs := body["messages"].([]interface{})
	last := msgs[len(msgs)-1].(map[string]interface{})
	if last["role"] != "user" {
		t.Fatalf("developer role must be mapped to user, got %v", last["role"])
	}
}

func TestLLMServiceInterruptionCancelsStream(t *testing.T) {
	s := NewLLMService(LLMConfig{APIKey: "k"})
	if err := s.Initialize(context.Background()); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	defer s.Cleanup()

	s.streamMu.Lock()
	s.isGenerating = true
	s.requestCtx, s.requestCancel = context.WithCancel(context.Background())
	s.streamMu.Unlock()

	if err := s.HandleFrame(context.Background(), frames.NewInterruptionFrame(), frames.Downstream); err != nil {
		t.Fatalf("interruption: %v", err)
	}

	s.streamMu.Lock()
	stillGenerating := s.isGenerating
	s.streamMu.Unlock()
	if stillGenerating {
		t.Fatal("isGenerating must be false after interruption")
	}
}
