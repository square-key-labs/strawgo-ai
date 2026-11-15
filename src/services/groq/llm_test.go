package groq

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

func collect(t *testing.T, p llm.Provider, snap llm.Snapshot, settings llm.Settings) ([]llm.Event, error) {
	t.Helper()
	var events []llm.Event
	err := p.Stream(context.Background(), snap, settings, func(ev llm.Event) { events = append(events, ev) })
	return events, err
}

func TestNewLLMServiceDefaults(t *testing.T) {
	svc := NewLLMService(LLMConfig{APIKey: "k"})
	if got := svc.Settings().Model; got != DefaultModel {
		t.Fatalf("model = %q, want %q", got, DefaultModel)
	}
	if svc.Provider().Name() != "Groq" {
		t.Fatalf("provider name = %q", svc.Provider().Name())
	}
}

func TestNewLLMServiceOverrides(t *testing.T) {
	temp := 0.7
	svc := NewLLMService(LLMConfig{
		APIKey:       "k",
		Model:        "llama-3.3-70b-versatile",
		SystemPrompt: "be brief",
		Settings:     llm.Settings{Temperature: &temp},
	})
	s := svc.Settings()
	if s.Model != "llama-3.3-70b-versatile" || s.SystemInstruction != "be brief" || s.Temperature == nil || *s.Temperature != 0.7 {
		t.Fatalf("unexpected settings: %+v", s)
	}
}

func TestGroqSendsToolsAndAuth(t *testing.T) {
	var body map[string]any
	var auth, path string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth, path = r.Header.Get("Authorization"), r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"SF\\\"}\"}}]}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	p := NewProvider(LLMConfig{APIKey: "secret", BaseURL: server.URL})
	snap := llm.Snapshot{
		Messages: []llm.Message{llm.User("weather?")},
		Tools: llm.Tools{Functions: []llm.FunctionSchema{{
			Name:       "get_weather",
			Properties: map[string]any{"city": map[string]any{"type": "string"}},
			Required:   []string{"city"},
		}}},
		ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceAuto},
	}
	events, err := collect(t, p, snap, llm.Settings{Model: DefaultModel})
	if err != nil {
		t.Fatal(err)
	}
	if auth != "Bearer secret" || path != "/chat/completions" {
		t.Fatalf("auth=%q path=%q", auth, path)
	}
	tools, _ := body["tools"].([]any)
	if len(tools) != 1 || body["tool_choice"] != "auto" {
		t.Fatalf("tools not sent: %v", body)
	}
	var calls []llm.ToolCall
	for _, ev := range events {
		if c, ok := ev.(llm.ToolCall); ok {
			calls = append(calls, c)
		}
	}
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Name != "get_weather" || string(calls[0].Arguments) != `{"city":"SF"}` {
		t.Fatalf("unexpected calls: %+v", calls)
	}
}
