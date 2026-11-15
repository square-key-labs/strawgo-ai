package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

func sseServer(t *testing.T, check func(r *http.Request, body map[string]any), lines ...string) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if check != nil {
			check(r, body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, l := range lines {
			fmt.Fprint(w, l)
		}
	}))
	t.Cleanup(s.Close)
	return s
}

func stream(ctx context.Context, p llm.Provider, snap llm.Snapshot, s llm.Settings) (string, []llm.ToolCall, *llm.Usage, error) {
	var text strings.Builder
	var calls []llm.ToolCall
	var usage *llm.Usage
	err := p.Stream(ctx, snap, s, func(ev llm.Event) {
		switch e := ev.(type) {
		case llm.TextDelta:
			text.WriteString(e.Text)
		case llm.ToolCall:
			calls = append(calls, e)
		case llm.UsageEvent:
			u := e.Usage
			usage = &u
		}
	})
	return text.String(), calls, usage, err
}

func userSnap(text string) llm.Snapshot {
	return llm.Snapshot{Messages: []llm.Message{llm.User(text)}}
}

func TestNewLLMServiceDefaults(t *testing.T) {
	svc := NewLLMService(LLMConfig{})
	if got := svc.Settings().Model; got != DefaultModel {
		t.Fatalf("model = %q, want %q", got, DefaultModel)
	}
	if svc.Provider().Name() != "Ollama" {
		t.Fatalf("name = %q", svc.Provider().Name())
	}
}

func TestCustomBaseURLAndNoAuthHeader(t *testing.T) {
	var path, auth string
	srv := sseServer(t, func(r *http.Request, _ map[string]any) {
		path, auth = r.URL.Path, r.Header.Get("Authorization")
	}, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n", "data: [DONE]\n\n")
	text, _, _, err := stream(context.Background(), NewProvider(LLMConfig{BaseURL: srv.URL + "/v1/"}), userSnap("hi"), llm.Settings{Model: DefaultModel})
	if err != nil || text != "ok" {
		t.Fatalf("text=%q err=%v", text, err)
	}
	if path != "/v1/chat/completions" || auth != "" {
		t.Fatalf("path=%q auth=%q", path, auth)
	}
}

func TestSSEParsingAndRequestShape(t *testing.T) {
	var body map[string]any
	srv := sseServer(t, func(_ *http.Request, b map[string]any) { body = b },
		"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\n",
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3,\"total_tokens\":13,\"prompt_tokens_details\":{\"cached_tokens\":4},\"completion_tokens_details\":{\"reasoning_tokens\":1}}}\n\n",
		"data: [DONE]\n\n")
	temp := 0.2
	snap := llm.Snapshot{Messages: []llm.Message{llm.Developer("dev note"), llm.User("Say hello")}}
	text, _, usage, err := stream(context.Background(), NewProvider(LLMConfig{BaseURL: srv.URL}), snap, llm.Settings{
		Model:             "llama3.2",
		SystemInstruction: "You are a pirate",
		Temperature:       &temp,
		Extra:             map[string]any{"num_ctx": 4096},
	})
	if err != nil || text != "Hello world!" {
		t.Fatalf("text=%q err=%v", text, err)
	}
	if usage == nil || usage.PromptTokens != 10 || usage.CacheReadTokens != 4 || usage.ReasoningTokens != 1 || usage.TotalTokens != 13 {
		t.Fatalf("usage = %+v", usage)
	}
	msgs := body["messages"].([]any)
	first := msgs[0].(map[string]any)
	second := msgs[1].(map[string]any)
	if first["role"] != "system" || first["content"] != "You are a pirate" || second["role"] != "user" {
		t.Fatalf("messages = %v", msgs)
	}
	if body["stream"] != true || body["temperature"] != 0.2 || body["num_ctx"] != float64(4096) {
		t.Fatalf("body = %v", body)
	}
	if so, _ := body["stream_options"].(map[string]any); so["include_usage"] != true {
		t.Fatalf("stream_options = %v", body["stream_options"])
	}
	if _, ok := body["tools"]; ok {
		t.Fatal("tools sent without any defined")
	}
}

func TestToolCallsAccumulatedByIndex(t *testing.T) {
	srv := sseServer(t, nil,
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"arguments\":\"{}\"}}]}}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}]}}]}\n\n",
		"data: [DONE]\n\n")
	_, calls, _, err := stream(context.Background(), NewProvider(LLMConfig{BaseURL: srv.URL}), userSnap("q"), llm.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 || calls[0].ID != "call_a" || string(calls[0].Arguments) != `{"city":"SF"}` || calls[1].Name != "get_time" {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestOllamaCompleteToolCallsWithoutIndex(t *testing.T) {
	srv := sseServer(t, nil,
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_1\",\"function\":{\"name\":\"a\",\"arguments\":\"{\\\"x\\\":1}\"}},{\"id\":\"call_2\",\"function\":{\"name\":\"b\",\"arguments\":\"{}\"}}]}}]}\n\n",
		"data: [DONE]\n\n")
	_, calls, _, err := stream(context.Background(), NewProvider(LLMConfig{BaseURL: srv.URL}), userSnap("q"), llm.Settings{})
	if err != nil || len(calls) != 2 || calls[0].Name != "a" || calls[1].Name != "b" {
		t.Fatalf("calls=%+v err=%v", calls, err)
	}
}

func TestToolHistoryAndToolsOnWire(t *testing.T) {
	var body map[string]any
	srv := sseServer(t, func(_ *http.Request, b map[string]any) { body = b }, "data: [DONE]\n\n")
	snap := llm.Snapshot{
		Messages: []llm.Message{
			llm.User("weather?"),
			llm.ToolCalls(llm.ToolCallPart{ID: "c1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"SF"}`)}),
			llm.ToolResult("c1", "get_weather", `{"temp":20}`),
		},
		Tools:      llm.Tools{Functions: []llm.FunctionSchema{{Name: "get_weather"}}},
		ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceFunction, Name: "get_weather"},
	}
	if _, _, _, err := stream(context.Background(), NewProvider(LLMConfig{BaseURL: srv.URL}), snap, llm.Settings{}); err != nil {
		t.Fatal(err)
	}
	msgs := body["messages"].([]any)
	asst := msgs[1].(map[string]any)
	tool := msgs[2].(map[string]any)
	tc := asst["tool_calls"].([]any)[0].(map[string]any)
	if tc["id"] != "c1" || tc["function"].(map[string]any)["arguments"] != `{"city":"SF"}` {
		t.Fatalf("assistant = %v", asst)
	}
	if tool["role"] != "tool" || tool["tool_call_id"] != "c1" || tool["content"] != `{"temp":20}` {
		t.Fatalf("tool = %v", tool)
	}
	choice := body["tool_choice"].(map[string]any)
	if choice["function"].(map[string]any)["name"] != "get_weather" || len(body["tools"].([]any)) != 1 {
		t.Fatalf("tool_choice=%v tools=%v", body["tool_choice"], body["tools"])
	}
}

func TestMultimodalParts(t *testing.T) {
	var body map[string]any
	srv := sseServer(t, func(_ *http.Request, b map[string]any) { body = b }, "data: [DONE]\n\n")
	snap := llm.Snapshot{Messages: []llm.Message{{Role: llm.RoleUser, Parts: []llm.Part{
		llm.TextPart{Text: "what is this"},
		llm.ImagePart{Data: []byte{1, 2, 3}, MIME: "image/png"},
		llm.AudioPart{Data: []byte{0, 0}, SampleRate: 16000},
	}}}}
	if _, _, _, err := stream(context.Background(), NewProvider(LLMConfig{BaseURL: srv.URL}), snap, llm.Settings{}); err != nil {
		t.Fatal(err)
	}
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	img := content[1].(map[string]any)["image_url"].(map[string]any)["url"].(string)
	audio := content[2].(map[string]any)["input_audio"].(map[string]any)
	if !strings.HasPrefix(img, "data:image/png;base64,") || audio["format"] != "wav" {
		t.Fatalf("content = %v", content)
	}
}

func TestHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"error":{"message":"model \"x\" not found","type":"api_error"}}`)
	}))
	defer srv.Close()
	_, _, _, err := stream(context.Background(), NewProvider(LLMConfig{BaseURL: srv.URL}), userSnap("hi"), llm.Settings{Model: "x"})
	var apiErr *openaicompat.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 404 || !strings.Contains(err.Error(), "Ollama API error") || !strings.Contains(apiErr.Message, "not found") {
		t.Fatalf("err = %v", err)
	}
}

func TestMidStreamError(t *testing.T) {
	srv := sseServer(t, nil,
		"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
		"data: {\"error\":{\"message\":\"overloaded\",\"type\":\"server_error\"}}\n\n")
	text, _, _, err := stream(context.Background(), NewProvider(LLMConfig{BaseURL: srv.URL}), userSnap("hi"), llm.Settings{})
	if text != "partial" || err == nil || !strings.Contains(err.Error(), "overloaded") {
		t.Fatalf("text=%q err=%v", text, err)
	}
}

func TestStreamCancellation(t *testing.T) {
	started := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n")
		w.(http.Flusher).Flush()
		close(started)
		<-r.Context().Done()
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, _, _, err := stream(ctx, NewProvider(LLMConfig{BaseURL: srv.URL}), userSnap("long story"), llm.Settings{})
		done <- err
	}()
	<-started
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("stream did not stop after cancellation")
	}
}

func TestMalformedAndLongSSE(t *testing.T) {
	long := strings.Repeat("x", 200<<10)
	srv := sseServer(t, nil,
		"data: not-json\n\n",
		": keep-alive\n\n",
		"event: ping\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"works\"}}]}\r\n\r\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\""+long+"\"}}]}\n\n",
		"data: [DONE]\n\n")
	text, _, _, err := stream(context.Background(), NewProvider(LLMConfig{BaseURL: srv.URL}), userSnap("t"), llm.Settings{})
	if err != nil || text != "works"+long {
		t.Fatalf("len(text)=%d err=%v", len(text), err)
	}
}

func TestRunInferenceDoesNotWriteContext(t *testing.T) {
	srv := sseServer(t, nil, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n", "data: [DONE]\n\n")
	svc := NewLLMService(LLMConfig{BaseURL: srv.URL})
	c := llm.NewContext(llm.User("hello"))
	out, err := svc.RunInference(context.Background(), c.Snapshot(), llm.Settings{})
	if err != nil || out != "hi" {
		t.Fatalf("out=%q err=%v", out, err)
	}
	if c.Len() != 1 {
		t.Fatalf("context mutated: %d messages", c.Len())
	}
}
