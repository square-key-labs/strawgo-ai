package openai_responses

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

func TestNewLLMServiceDefaults(t *testing.T) {
	s := NewLLMService(LLMConfig{APIKey: "k"})
	if s.model != DefaultResponsesModel {
		t.Errorf("expected default model %s, got %s", DefaultResponsesModel, s.model)
	}
	if s.baseURL != DefaultResponsesBaseURL {
		t.Errorf("expected default base URL, got %s", s.baseURL)
	}
}

func TestStreamingEmitsTextDeltasAndCapturesResponseID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/responses") {
			http.NotFound(w, r)
			return
		}
		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["stream"] != true {
			t.Errorf("expected stream=true")
		}
		if body["instructions"] != "act helpful" {
			t.Errorf("expected instructions to carry system prompt, got %v", body["instructions"])
		}
		if _, hasMessages := body["messages"]; hasMessages {
			t.Errorf("Responses API must use input items, not messages")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		writeEvent := func(payload map[string]interface{}) {
			b, _ := json.Marshal(payload)
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
		}
		writeEvent(map[string]interface{}{"type": "response.output_text.delta", "delta": "Hello "})
		writeEvent(map[string]interface{}{"type": "response.output_text.delta", "delta": "world"})
		writeEvent(map[string]interface{}{
			"type":     "response.completed",
			"response": map[string]interface{}{"id": "resp_123"},
		})
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	s := NewLLMService(LLMConfig{APIKey: "k", BaseURL: srv.URL, SystemPrompt: "act helpful"})
	if err := s.Initialize(context.Background()); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	defer s.Cleanup()

	llmCtx := services.NewLLMContext("act helpful")
	llmCtx.AddUserMessage("Hi")

	if err := s.generateResponseFromContext(llmCtx); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if got := s.PreviousResponseID(); got != "resp_123" {
		t.Errorf("expected previous_response_id to be captured, got %q", got)
	}
	if last := llmCtx.Messages[len(llmCtx.Messages)-1]; last.Role != "assistant" || last.Content != "Hello world" {
		t.Errorf("expected assistant message 'Hello world', got %+v", last)
	}
}

func TestIncrementalUsesPreviousResponseID(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&calls, 1)
		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)

		if count == 1 {
			// First request — no previous_response_id; ships full context.
			if _, hasPrev := body["previous_response_id"]; hasPrev {
				t.Errorf("first request must not include previous_response_id")
			}
			input := body["input"].([]interface{})
			if len(input) != 1 {
				t.Errorf("first request expected 1 input item, got %d", len(input))
			}
		} else {
			// Second request — must use previous_response_id and ship ONLY
			// the latest user turn.
			if got := body["previous_response_id"]; got != "resp_first" {
				t.Errorf("second request must reference resp_first, got %v", got)
			}
			input := body["input"].([]interface{})
			if len(input) != 1 {
				t.Errorf("second request expected 1 input item (latest only), got %d", len(input))
			}
			last := input[0].(map[string]interface{})
			if last["content"] != "second turn" {
				t.Errorf("second request must ship the new user turn only, got %v", last["content"])
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: %s\n\n", `{"type":"response.output_text.delta","delta":"ok"}`)
		flusher.Flush()
		respID := "resp_first"
		if count == 2 {
			respID = "resp_second"
		}
		fmt.Fprintf(w, "data: %s\n\n", fmt.Sprintf(`{"type":"response.completed","response":{"id":"%s"}}`, respID))
		flusher.Flush()
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
	llmCtx.AddUserMessage("first turn")
	if err := s.generateResponseFromContext(llmCtx); err != nil {
		t.Fatalf("first generate: %v", err)
	}
	if got := s.PreviousResponseID(); got != "resp_first" {
		t.Fatalf("expected resp_first captured, got %q", got)
	}

	llmCtx.AddUserMessage("second turn")
	if err := s.generateResponseFromContext(llmCtx); err != nil {
		t.Fatalf("second generate: %v", err)
	}
	if got := s.PreviousResponseID(); got != "resp_second" {
		t.Fatalf("expected resp_second captured, got %q", got)
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("expected 2 server calls, got %d", calls)
	}
}

func TestPreviousResponseIDFallbackOnStaleID(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&calls, 1)
		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)

		if count == 1 {
			// First incremental attempt sends previous_response_id and is rejected.
			if _, hasPrev := body["previous_response_id"]; !hasPrev {
				t.Errorf("first attempt should include previous_response_id")
			}
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"error":{"message":"previous_response_id no longer available"}}`)
			return
		}
		// Second attempt is the fallback full-context replay; must NOT
		// carry previous_response_id and must ship every message.
		if _, hasPrev := body["previous_response_id"]; hasPrev {
			t.Errorf("fallback must not include previous_response_id")
		}
		input := body["input"].([]interface{})
		if len(input) < 2 {
			t.Errorf("fallback should ship full context, got %d items", len(input))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, `data: {"type":"response.output_text.delta","delta":"recovered"}`+"\n\n")
		fmt.Fprint(w, `data: {"type":"response.completed","response":{"id":"resp_recovered"}}`+"\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	s := NewLLMService(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	s.SetPreviousResponseID("resp_stale")
	if err := s.Initialize(context.Background()); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	defer s.Cleanup()

	llmCtx := services.NewLLMContext("")
	llmCtx.AddUserMessage("first")
	llmCtx.AddAssistantMessage("ok")
	llmCtx.AddUserMessage("second")

	if err := s.generateResponseFromContext(llmCtx); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("expected 2 calls (first 400 then fallback), got %d", calls)
	}
	if got := s.PreviousResponseID(); got != "resp_recovered" {
		t.Errorf("expected resp_recovered after fallback, got %q", got)
	}
}

func TestClearContextResetsResponseID(t *testing.T) {
	s := NewLLMService(LLMConfig{APIKey: "k"})
	s.SetPreviousResponseID("resp_xyz")
	s.ClearContext()
	if got := s.PreviousResponseID(); got != "" {
		t.Errorf("ClearContext must wipe previous_response_id, got %q", got)
	}
}

func TestBuildInputItemsDropsSystemAndMapsDeveloper(t *testing.T) {
	in := []services.LLMMessage{
		{Role: "system", Content: "ignore me"},
		{Role: "developer", Content: "tool said hi"},
		{Role: "user", Content: "user msg"},
	}
	out := buildInputItems(in)
	if len(out) != 2 {
		t.Fatalf("expected 2 items (system dropped), got %d", len(out))
	}
	if out[0]["role"] != "user" {
		t.Errorf("developer must be remapped to user, got %v", out[0]["role"])
	}
}

func TestIsPreviousResponseIDError(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"plain string", fmt.Errorf("oops"), false},
		{"500 not previous_response_id", &apiError{Status: 500, Body: "server boom"}, false},
		{"400 previous_response_id text", &apiError{Status: 400, Body: "Invalid previous_response_id"}, true},
		{"404 expired", &apiError{Status: 404, Body: "response expired"}, true},
		{"400 unrelated", &apiError{Status: 400, Body: "missing api key"}, false},
		// Structured envelope, code-only — no human-readable hint.
		{"400 structured code-only", &apiError{Status: 400, Body: `{"error":{"code":"previous_response_id_expired","type":"invalid_request_error","message":"x"}}`}, true},
		// Structured envelope, param-only.
		{"400 structured param-only", &apiError{Status: 400, Body: `{"error":{"code":"missing","param":"previous_response_id","message":"y"}}`}, true},
		// Structured envelope, totally unrelated error.
		{"400 structured unrelated", &apiError{Status: 400, Body: `{"error":{"code":"invalid_api_key","type":"authentication_error","message":"bad key"}}`}, false},
	}
	for _, tc := range cases {
		if got := isPreviousResponseIDError(tc.err); got != tc.want {
			t.Errorf("%s: want %v got %v", tc.name, tc.want, got)
		}
	}
}

func TestIncrementalShipsAllUnsentMessages(t *testing.T) {
	// After turn 1 the checkpoint covers (user1 + assistant1). Then the
	// caller bundles a tool-result and a follow-up user message before
	// the next LLMContextFrame; the next incremental request must ship
	// BOTH — sending only the trailing user message would silently
	// drop the tool result.
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&calls, 1)
		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)
		input := body["input"].([]interface{})

		if count == 1 {
			if len(input) != 1 {
				t.Errorf("turn 1: expected 1 input item, got %d", len(input))
			}
		} else {
			if got := body["previous_response_id"]; got != "resp_t1" {
				t.Errorf("turn 2: expected previous_response_id=resp_t1, got %v", got)
			}
			if len(input) != 2 {
				t.Errorf("turn 2: expected 2 input items (tool+user), got %d", len(input))
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, `data: {"type":"response.output_text.delta","delta":"ok"}`+"\n\n")
		respID := "resp_t1"
		if count == 2 {
			respID = "resp_t2"
		}
		fmt.Fprintf(w, `data: {"type":"response.completed","response":{"id":"%s"}}`+"\n\n", respID)
		flusher.Flush()
	}))
	defer srv.Close()

	s := NewLLMService(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	if err := s.Initialize(context.Background()); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	defer s.Cleanup()

	llmCtx := services.NewLLMContext("")
	llmCtx.AddUserMessage("first")
	if err := s.generateResponseFromContext(llmCtx); err != nil {
		t.Fatalf("turn 1: %v", err)
	}

	// Caller appends a tool result + a fresh user follow-up to the same
	// context before the next generation.
	llmCtx.AddToolMessage("tool_call_42", "tool result")
	llmCtx.AddUserMessage("based on that, do X")
	if err := s.generateResponseFromContext(llmCtx); err != nil {
		t.Fatalf("turn 2: %v", err)
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("expected 2 calls, got %d", calls)
	}
}

func TestInterruptionCancelsStream(t *testing.T) {
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
