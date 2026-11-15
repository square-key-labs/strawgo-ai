package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type capture struct {
	mu      sync.Mutex
	body    map[string]any
	headers http.Header
	path    string
}

func (c *capture) get() (map[string]any, http.Header, string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.body, c.headers, c.path
}

func sseServer(t *testing.T, events ...string) (*httptest.Server, *capture) {
	t.Helper()
	c := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		c.mu.Lock()
		c.body, c.headers, c.path = body, r.Header.Clone(), r.URL.RequestURI()
		c.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		for _, e := range events {
			_, _ = io.WriteString(w, e)
			if fl != nil {
				fl.Flush()
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv, c
}

func data(v string) string { return "data: " + v + "\n\n" }

func newProvider(srv *httptest.Server, opts ...Option) *Provider {
	return New(Config{Name: "Test", BaseURL: srv.URL + "/v1/", APIKey: "sk-test", Headers: map[string]string{"X-Extra": "yes"}, HTTPClient: srv.Client()}, opts...)
}

func collect(t *testing.T, p *Provider, snap llm.Snapshot, settings llm.Settings) ([]llm.Event, error) {
	t.Helper()
	var events []llm.Event
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err := p.Stream(ctx, snap, settings, func(e llm.Event) { events = append(events, e) })
	return events, err
}

func textOf(events []llm.Event) string {
	var b strings.Builder
	for _, e := range events {
		if d, ok := e.(llm.TextDelta); ok {
			b.WriteString(d.Text)
		}
	}
	return b.String()
}

func toolCallsOf(events []llm.Event) []llm.ToolCall {
	var out []llm.ToolCall
	for _, e := range events {
		if c, ok := e.(llm.ToolCall); ok {
			out = append(out, c)
		}
	}
	return out
}

func ptr[T any](v T) *T { return &v }

func TestRequestBodyShape(t *testing.T) {
	srv, c := sseServer(t, data(`[DONE]`))
	p := newProvider(srv)
	ctx := llm.NewContext(
		llm.Developer("dev rules"),
		llm.User("weather?"),
		llm.ToolCalls(llm.ToolCallPart{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Paris"}`)}, llm.ToolCallPart{ID: "call_2", Name: "now"}),
		llm.ToolResult("call_1", "get_weather", "sunny"),
		llm.ToolResult("call_2", "now", "noon"),
	)
	ctx.SetTools(llm.Tools{Functions: []llm.FunctionSchema{{
		Name:        "get_weather",
		Description: "weather",
		Properties:  map[string]any{"city": map[string]any{"type": "string"}},
		Required:    []string{"city"},
	}}})
	ctx.SetToolChoice(llm.ToolChoice{Mode: llm.ToolChoiceFunction, Name: "get_weather"})
	settings := llm.Settings{
		Model:             "gpt-test",
		SystemInstruction: "be brief",
		Temperature:       ptr(0.5),
		MaxTokens:         ptr(42),
		Seed:              ptr(7),
		Extra:             map[string]any{"user": "u1", "extra_body": map[string]any{"top_k": 3}},
	}
	if _, err := collect(t, p, ctx.Snapshot(), settings); err != nil {
		t.Fatalf("stream: %v", err)
	}
	body, headers, path := c.get()

	if path != "/v1/chat/completions" {
		t.Errorf("path = %q", path)
	}
	if got := headers.Get("Authorization"); got != "Bearer sk-test" {
		t.Errorf("auth header = %q", got)
	}
	if got := headers.Get("X-Extra"); got != "yes" {
		t.Errorf("custom header = %q", got)
	}
	if got := headers.Get("Accept"); got != "text/event-stream" {
		t.Errorf("accept header = %q", got)
	}
	if body["model"] != "gpt-test" || body["stream"] != true {
		t.Errorf("model/stream = %v/%v", body["model"], body["stream"])
	}
	if body["temperature"] != 0.5 || body["max_tokens"] != float64(42) || body["seed"] != float64(7) {
		t.Errorf("sampling fields = %v %v %v", body["temperature"], body["max_tokens"], body["seed"])
	}
	if _, ok := body["top_p"]; ok {
		t.Errorf("top_p present though unset")
	}
	if body["user"] != "u1" || body["top_k"] != float64(3) {
		t.Errorf("extra fields = %v %v", body["user"], body["top_k"])
	}
	if _, ok := body["extra_body"]; ok {
		t.Errorf("extra_body should be flattened")
	}
	so, _ := body["stream_options"].(map[string]any)
	if so["include_usage"] != true {
		t.Errorf("stream_options = %v", body["stream_options"])
	}

	msgs, _ := body["messages"].([]any)
	want := []struct{ role, content string }{
		{"system", "be brief"},
		{"developer", "dev rules"},
		{"user", "weather?"},
		{"assistant", ""},
		{"tool", "sunny"},
		{"tool", "noon"},
	}
	if len(msgs) != len(want) {
		t.Fatalf("messages = %d, want %d: %v", len(msgs), len(want), msgs)
	}
	for i, w := range want {
		m := msgs[i].(map[string]any)
		if m["role"] != w.role {
			t.Errorf("msg %d role = %v, want %s", i, m["role"], w.role)
		}
		if w.content != "" && m["content"] != w.content {
			t.Errorf("msg %d content = %v, want %s", i, m["content"], w.content)
		}
	}
	asst := msgs[3].(map[string]any)
	calls, _ := asst["tool_calls"].([]any)
	if len(calls) != 2 {
		t.Fatalf("tool_calls = %v", asst["tool_calls"])
	}
	c0 := calls[0].(map[string]any)
	fn0 := c0["function"].(map[string]any)
	if c0["id"] != "call_1" || c0["type"] != "function" || fn0["name"] != "get_weather" || fn0["arguments"] != `{"city":"Paris"}` {
		t.Errorf("tool call 0 = %v", c0)
	}
	fn1 := calls[1].(map[string]any)["function"].(map[string]any)
	if fn1["arguments"] != "{}" {
		t.Errorf("empty arguments should become {}, got %v", fn1["arguments"])
	}
	if msgs[4].(map[string]any)["tool_call_id"] != "call_1" || msgs[5].(map[string]any)["tool_call_id"] != "call_2" {
		t.Errorf("tool_call_id mismatch: %v %v", msgs[4], msgs[5])
	}

	tools, _ := body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v", body["tools"])
	}
	tool := tools[0].(map[string]any)
	tfn := tool["function"].(map[string]any)
	params := tfn["parameters"].(map[string]any)
	if tool["type"] != "function" || tfn["name"] != "get_weather" || params["type"] != "object" {
		t.Errorf("tool = %v", tool)
	}
	if req, _ := params["required"].([]any); len(req) != 1 || req[0] != "city" {
		t.Errorf("required = %v", params["required"])
	}
	choice, _ := body["tool_choice"].(map[string]any)
	if choice["type"] != "function" || choice["function"].(map[string]any)["name"] != "get_weather" {
		t.Errorf("tool_choice = %v", body["tool_choice"])
	}
}

func TestToolChoiceModes(t *testing.T) {
	tools := llm.Tools{Functions: []llm.FunctionSchema{{Name: "f"}}}
	cases := []struct {
		name   string
		tools  llm.Tools
		choice llm.ToolChoice
		want   any
		absent bool
	}{
		{"auto", tools, llm.ToolChoice{Mode: llm.ToolChoiceAuto}, "auto", false},
		{"none", tools, llm.ToolChoice{Mode: llm.ToolChoiceNone}, "none", false},
		{"required", tools, llm.ToolChoice{Mode: llm.ToolChoiceRequired}, "required", false},
		{"unset", tools, llm.ToolChoice{}, nil, true},
		{"no tools", llm.Tools{}, llm.ToolChoice{Mode: llm.ToolChoiceAuto}, nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := New(Config{BaseURL: "http://x"})
			body, err := p.BuildRequestBody(llm.Snapshot{Tools: tc.tools, ToolChoice: tc.choice}, llm.Settings{Model: "m"})
			if err != nil {
				t.Fatal(err)
			}
			got, ok := body["tool_choice"]
			if tc.absent {
				if ok {
					t.Errorf("tool_choice present: %v", got)
				}
				return
			}
			if got != tc.want {
				t.Errorf("tool_choice = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestTextStreaming(t *testing.T) {
	srv, _ := sseServer(t,
		": keepalive\n\n",
		data(`{"choices":[{"delta":{"role":"assistant","content":""}}]}`),
		data(`{"choices":[{"delta":{"content":"Hel"}}]}`),
		"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\r\n\r\n",
		data(`{"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}`),
		data(`[DONE]`),
	)
	events, err := collect(t, newProvider(srv), llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{Model: "m"})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if got := textOf(events); got != "Hello world" {
		t.Errorf("text = %q", got)
	}
	last, ok := events[len(events)-1].(llm.Done)
	if !ok || last.StopReason != "stop" {
		t.Errorf("last event = %#v", events[len(events)-1])
	}
	for _, e := range events {
		if d, ok := e.(llm.TextDelta); ok && d.Text == "" {
			t.Errorf("empty text delta emitted")
		}
	}
}

func TestToolCallDeltas(t *testing.T) {
	cases := []struct {
		name   string
		events []string
		want   []llm.ToolCall
	}{
		{
			name: "split across chunks",
			events: []string{
				data(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`),
				data(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]}}]}`),
				data(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Paris\"}"}}]}}]}`),
				data(`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`),
				data(`[DONE]`),
			},
			want: []llm.ToolCall{{ID: "call_a", Name: "get_weather", Arguments: []byte(`{"city":"Paris"}`)}},
		},
		{
			name: "multiple indexed interleaved",
			events: []string{
				data(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"a","arguments":"{\"x\":"}}]}}]}`),
				data(`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c1","function":{"name":"b","arguments":"{\"y\":"}}]}}]}`),
				data(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}},{"index":1,"function":{"arguments":"2}"}}]}}]}`),
				data(`[DONE]`),
			},
			want: []llm.ToolCall{
				{ID: "c0", Name: "a", Arguments: []byte(`{"x":1}`)},
				{ID: "c1", Name: "b", Arguments: []byte(`{"y":2}`)},
			},
		},
		{
			name: "no arguments becomes empty object",
			events: []string{
				data(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"now"}}]}}]}`),
				data(`[DONE]`),
			},
			want: []llm.ToolCall{{ID: "c0", Name: "now", Arguments: []byte(`{}`)}},
		},
		{
			name: "reused index with new id is a new call",
			events: []string{
				data(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"a","arguments":"{}"}}]}}]}`),
				data(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"b","arguments":"{}"}}]}}]}`),
				data(`[DONE]`),
			},
			want: []llm.ToolCall{
				{ID: "c0", Name: "a", Arguments: []byte(`{}`)},
				{ID: "c1", Name: "b", Arguments: []byte(`{}`)},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := sseServer(t, tc.events...)
			events, err := collect(t, newProvider(srv), llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{Model: "m"})
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			got := toolCallsOf(events)
			if len(got) != len(tc.want) {
				t.Fatalf("calls = %+v, want %+v", got, tc.want)
			}
			for i := range got {
				if got[i].ID != tc.want[i].ID || got[i].Name != tc.want[i].Name || string(got[i].Arguments) != string(tc.want[i].Arguments) {
					t.Errorf("call %d = {%s %s %s}, want {%s %s %s}", i, got[i].ID, got[i].Name, got[i].Arguments, tc.want[i].ID, tc.want[i].Name, tc.want[i].Arguments)
				}
			}
			if _, ok := events[len(events)-1].(llm.Done); !ok {
				t.Errorf("last event is %T, want Done", events[len(events)-1])
			}
		})
	}
}

func TestToolCallWithoutIDGetsGeneratedID(t *testing.T) {
	srv, _ := sseServer(t,
		data(`{"choices":[{"delta":{"tool_calls":[{"function":{"name":"a","arguments":"{}"}}]}}]}`),
		data(`[DONE]`),
	)
	events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	calls := toolCallsOf(events)
	if len(calls) != 1 || !strings.HasPrefix(calls[0].ID, "call_") {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestUsage(t *testing.T) {
	srv, _ := sseServer(t,
		data(`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`),
		data(`{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":64,"cache_write_tokens":8},"completion_tokens_details":{"reasoning_tokens":12}}}`),
		data(`[DONE]`),
	)
	events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	var usage *llm.Usage
	for _, e := range events {
		if u, ok := e.(llm.UsageEvent); ok {
			if usage != nil {
				t.Errorf("usage emitted twice")
			}
			usage = &u.Usage
		}
	}
	want := llm.Usage{PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120, CacheReadTokens: 64, CacheCreationTokens: 8, ReasoningTokens: 12}
	if usage == nil || *usage != want {
		t.Fatalf("usage = %+v, want %+v", usage, want)
	}
	if _, ok := events[len(events)-1].(llm.Done); !ok {
		t.Errorf("Done must be last")
	}
}

func TestUsageHook(t *testing.T) {
	srv, _ := sseServer(t,
		data(`{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`),
		data(`[DONE]`),
	)
	p := newProvider(srv, WithUsageHook(func(u *llm.Usage) { u.TotalTokens = 99 }))
	events, err := collect(t, p, llm.Snapshot{}, llm.Settings{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range events {
		if u, ok := e.(llm.UsageEvent); ok && u.Usage.TotalTokens == 99 {
			return
		}
	}
	t.Fatalf("usage hook not applied: %#v", events)
}

func TestLongSSELine(t *testing.T) {
	big := strings.Repeat("abcdefgh", 10<<10)
	payload, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": big}}}})
	if len(payload) <= 64<<10 {
		t.Fatalf("payload too small: %d", len(payload))
	}
	srv, _ := sseServer(t,
		data(`{"choices":[{"delta":{"content":"<"}}]}`),
		data(string(payload)),
		data(`{"choices":[{"delta":{"content":">"}}]}`),
		data(`[DONE]`),
	)
	events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if got := textOf(events); got != "<"+big+">" {
		t.Fatalf("text length = %d, want %d", len(got), len(big)+2)
	}
}

func TestMidStreamError(t *testing.T) {
	cases := []struct {
		name   string
		events []string
		check  func(*testing.T, *APIError)
	}{
		{
			name: "error object in data",
			events: []string{
				data(`{"choices":[{"delta":{"content":"partial"}}]}`),
				data(`{"error":{"message":"overloaded","type":"server_error","code":"503"}}`),
				data(`{"choices":[{"delta":{"content":"never"}}]}`),
			},
			check: func(t *testing.T, e *APIError) {
				if e.Message != "overloaded" || e.Type != "server_error" || e.Code != "503" {
					t.Errorf("api error = %+v", e)
				}
			},
		},
		{
			name: "error event with non json data",
			events: []string{
				data(`{"choices":[{"delta":{"content":"partial"}}]}`),
				"event: error\ndata: upstream exploded\n\n",
			},
			check: func(t *testing.T, e *APIError) {
				if e.Body != "upstream exploded" {
					t.Errorf("api error body = %q", e.Body)
				}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := sseServer(t, tc.events...)
			events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
			var apiErr *APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %v (%T), want *APIError", err, err)
			}
			tc.check(t, apiErr)
			if got := textOf(events); got != "partial" {
				t.Errorf("text = %q", got)
			}
			for _, e := range events {
				if _, ok := e.(llm.Done); ok {
					t.Errorf("Done emitted after error")
				}
			}
		})
	}
}

func TestNon2xxTypedError(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		body    string
		message string
		code    string
	}{
		{"json envelope", http.StatusTooManyRequests, `{"error":{"message":"rate limited","type":"rate_limit_error","code":"rate_limit_exceeded","param":"model"}}`, "rate limited", "rate_limit_exceeded"},
		{"numeric code", http.StatusBadRequest, `{"error":{"message":"bad","type":"invalid_request_error","code":400}}`, "bad", "400"},
		{"plain text", http.StatusBadGateway, "bad gateway", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer srv.Close()
			events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
			var apiErr *APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("err = %v (%T)", err, err)
			}
			if apiErr.StatusCode != tc.status || apiErr.Message != tc.message || apiErr.Code != tc.code || apiErr.Provider != "Test" {
				t.Errorf("api error = %+v", apiErr)
			}
			if tc.message == "" && apiErr.Body != tc.body {
				t.Errorf("body = %q", apiErr.Body)
			}
			if !strings.Contains(apiErr.Error(), fmt.Sprintf("HTTP %d", tc.status)) {
				t.Errorf("Error() = %q", apiErr.Error())
			}
			if len(events) != 0 {
				t.Errorf("events emitted on error: %v", events)
			}
		})
	}
}

func TestContextCancelStopsPromptly(t *testing.T) {
	release := make(chan struct{})
	sent := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, data(`{"choices":[{"delta":{"content":"first"}}]}`))
		w.(http.Flusher).Flush()
		close(sent)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithCancel(context.Background())
	p := newProvider(srv)
	got := make(chan string, 4)
	errc := make(chan error, 1)
	go func() {
		errc <- p.Stream(ctx, llm.Snapshot{}, llm.Settings{Model: "m"}, func(e llm.Event) {
			if d, ok := e.(llm.TextDelta); ok {
				got <- d.Text
			}
		})
	}()
	select {
	case txt := <-got:
		if txt != "first" {
			t.Fatalf("text = %q", txt)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no first delta")
	}
	<-sent
	start := time.Now()
	cancel()
	select {
	case err := <-errc:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
		if d := time.Since(start); d > time.Second {
			t.Errorf("stream took %v to stop", d)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Stream did not return after cancel")
	}
}

func TestContextAlreadyCancelled(t *testing.T) {
	srv, _ := sseServer(t, data(`[DONE]`))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := newProvider(srv).Stream(ctx, llm.Snapshot{}, llm.Settings{Model: "m"}, func(llm.Event) { t.Error("emit called") })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
}

func TestQuirkOptions(t *testing.T) {
	srv, c := sseServer(t, data(`[DONE]`))
	p := newProvider(srv,
		WithoutDeveloperRole(),
		WithoutStreamOptions(),
		WithMaxTokensField("max_completion_tokens"),
		WithAuthHeader("api-key", ""),
		WithPath("/chat"),
		WithQuery("api-version", "2024"),
	)
	snap := llm.NewContext(llm.Developer("rules"), llm.User("hi")).Snapshot()
	if _, err := collect(t, p, snap, llm.Settings{Model: "m", MaxTokens: ptr(5)}); err != nil {
		t.Fatal(err)
	}
	body, headers, path := c.get()
	if _, ok := body["stream_options"]; ok {
		t.Errorf("stream_options present")
	}
	if _, ok := body["max_tokens"]; ok || body["max_completion_tokens"] != float64(5) {
		t.Errorf("max tokens fields = %v / %v", body["max_tokens"], body["max_completion_tokens"])
	}
	msgs := body["messages"].([]any)
	if r := msgs[0].(map[string]any)["role"]; r != "user" {
		t.Errorf("developer role not rewritten: %v", r)
	}
	if headers.Get("api-key") != "sk-test" || headers.Get("Authorization") != "" {
		t.Errorf("auth headers = %v", headers)
	}
	if path != "/v1/chat?api-version=2024" {
		t.Errorf("path = %q", path)
	}
}

func TestWithoutToolsAndBodyHook(t *testing.T) {
	p := New(Config{BaseURL: "http://x"}, WithoutTools(), WithBodyHook(func(b map[string]any, s llm.Settings) { b["hooked"] = s.Model }))
	body, err := p.BuildRequestBody(llm.Snapshot{Tools: llm.Tools{Functions: []llm.FunctionSchema{{Name: "f"}}}, ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceAuto}}, llm.Settings{Model: "m", Extra: map[string]any{"hooked": "override"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := body["tools"]; ok {
		t.Errorf("tools present")
	}
	if _, ok := body["tool_choice"]; ok {
		t.Errorf("tool_choice present")
	}
	if body["hooked"] != "override" {
		t.Errorf("Settings.Extra should win over body hook, got %v", body["hooked"])
	}
}

func TestSkipAnsweredToolCalls(t *testing.T) {
	srv, _ := sseServer(t,
		data(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"old","function":{"name":"a","arguments":"{}"}},{"index":1,"id":"new","function":{"name":"b","arguments":"{}"}}]}}]}`),
		data(`[DONE]`),
	)
	snap := llm.NewContext(llm.ToolResult("old", "a", "done")).Snapshot()
	events, err := collect(t, newProvider(srv, WithSkipAnsweredToolCalls()), snap, llm.Settings{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	calls := toolCallsOf(events)
	if len(calls) != 1 || calls[0].ID != "new" {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestReasoningDeltas(t *testing.T) {
	srv, _ := sseServer(t,
		data(`{"choices":[{"delta":{"reasoning_content":"think"}}]}`),
		data(`{"choices":[{"delta":{"reasoning":"more"}}]}`),
		data(`{"choices":[{"delta":{"content":"answer"}}]}`),
		data(`[DONE]`),
	)
	events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	var thoughts []string
	for _, e := range events {
		if d, ok := e.(llm.ThoughtDelta); ok {
			thoughts = append(thoughts, d.Text)
		}
	}
	if strings.Join(thoughts, "|") != "think|more" || textOf(events) != "answer" {
		t.Errorf("thoughts = %v text = %q", thoughts, textOf(events))
	}
}

func TestStreamWithoutDoneMarker(t *testing.T) {
	srv, _ := sseServer(t, "data: {\"choices\":[{\"delta\":{\"content\":\"tail\"}}]}")
	events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if textOf(events) != "tail" {
		t.Errorf("text = %q", textOf(events))
	}
	if _, ok := events[len(events)-1].(llm.Done); !ok {
		t.Errorf("missing Done")
	}
}

func TestSSEReaderMultilineData(t *testing.T) {
	r := NewSSEReader(strings.NewReader("event: x\ndata: a\ndata: b\nid: 1\n\ndata:c\n\n"))
	ev, d, err := r.Next()
	if err != nil || string(ev) != "x" || string(d) != "a\nb" {
		t.Fatalf("first = %q %q %v", ev, d, err)
	}
	ev, d, err = r.Next()
	if err != nil || len(ev) != 0 || string(d) != "c" {
		t.Fatalf("second = %q %q %v", ev, d, err)
	}
	if _, _, err = r.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("third err = %v", err)
	}
}
