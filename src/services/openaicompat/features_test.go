package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
)

func doneOf(t *testing.T, events []llm.Event) llm.Done {
	t.Helper()
	for _, e := range events {
		if d, ok := e.(llm.Done); ok {
			return d
		}
	}
	t.Fatalf("no Done event in %v", events)
	return llm.Done{}
}

func thoughtOf(events []llm.Event) string {
	var b strings.Builder
	for _, e := range events {
		if d, ok := e.(llm.ThoughtDelta); ok {
			b.WriteString(d.Text)
		}
	}
	return b.String()
}

func TestStopReasonsAreNormalised(t *testing.T) {
	cases := map[string]string{
		"stop":          llm.StopReasonStop,
		"length":        llm.StopReasonLength,
		"tool_calls":    llm.StopReasonToolCalls,
		"function_call": llm.StopReasonToolCalls,
		"weird":         llm.StopReasonOther,
	}
	for raw, want := range cases {
		t.Run(raw, func(t *testing.T) {
			srv, _ := sseServer(t,
				data(`{"choices":[{"delta":{"content":"hi"}}]}`),
				data(`{"choices":[{"delta":{},"finish_reason":"`+raw+`"}]}`),
				data("[DONE]"),
			)
			events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
			if err != nil {
				t.Fatal(err)
			}
			if d := doneOf(t, events); d.StopReason != want || d.RawStopReason != raw {
				t.Fatalf("done = %+v", d)
			}
		})
	}
}

func TestContentFilterFinishReason(t *testing.T) {
	srv, _ := sseServer(t,
		data(`{"id":"chatcmpl-1","model":"gpt-x","choices":[{"delta":{"content":"par"}}]}`),
		data(`{"id":"chatcmpl-1","model":"gpt-x","choices":[{"delta":{},"finish_reason":"content_filter","content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"violence":{"filtered":true,"severity":"medium"},"sexual":{"filtered":true,"severity":"high"}}}]}`),
		data(`{"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`),
		data("[DONE]"),
	)
	events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
	r, ok := llm.AsRefusal(err)
	if !ok || r.Category != llm.RefusalContentFilter || r.Provider != "Test" || r.Detail != "sexual,violence" {
		t.Fatalf("err = %v refusal = %+v", err, r)
	}
	if textOf(events) != "par" {
		t.Fatalf("partial text lost: %q", textOf(events))
	}
	if d := doneOf(t, events); d.StopReason != llm.StopReasonContentFilter || d.RawStopReason != "content_filter" {
		t.Fatalf("done = %+v", d)
	}
	var sawRefusal, sawUsage bool
	for _, e := range events {
		switch e.(type) {
		case llm.Refusal:
			sawRefusal = true
		case llm.UsageEvent:
			sawUsage = true
		}
	}
	if !sawRefusal || !sawUsage {
		t.Fatalf("refusal=%v usage=%v events=%v", sawRefusal, sawUsage, events)
	}
}

func TestRefusalDelta(t *testing.T) {
	srv, _ := sseServer(t,
		data(`{"choices":[{"delta":{"role":"assistant","refusal":"I'm sorry, "}}]}`),
		data(`{"choices":[{"delta":{"refusal":"I can't help with that."}}]}`),
		data(`{"choices":[{"delta":{},"finish_reason":"stop"}]}`),
		data("[DONE]"),
	)
	events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
	r, ok := llm.AsRefusal(err)
	if !ok || r.Category != llm.RefusalModel || r.Message != "I'm sorry, I can't help with that." {
		t.Fatalf("err = %v refusal = %+v", err, r)
	}
	if d := doneOf(t, events); d.StopReason != llm.StopReasonRefusal || d.RawStopReason != "stop" {
		t.Fatalf("done = %+v", d)
	}
	if textOf(events) != "" {
		t.Fatalf("refusal text leaked into the answer: %q", textOf(events))
	}
}

func TestAzureContentFilterErrorAndRequestID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-request-id", "req_abc")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"The response was filtered due to the prompt triggering Azure OpenAI's content management policy.","type":null,"param":"prompt","code":"content_filter","status":400,"innererror":{"code":"ResponsibleAIPolicyViolation","content_filter_result":{"hate":{"filtered":false,"severity":"safe"},"jailbreak":{"filtered":true,"detected":true},"self_harm":{"filtered":true,"severity":"high"}}}}}`)
	}))
	defer srv.Close()
	events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
	r, ok := llm.AsRefusal(err)
	if !ok || r.Category != llm.RefusalPromptBlocked || r.Detail != "jailbreak,self_harm" {
		t.Fatalf("err = %v refusal = %+v", err, r)
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 400 || apiErr.Code != "content_filter" || apiErr.RequestID != "req_abc" {
		t.Fatalf("api error = %+v", apiErr)
	}
	if got := frames.ClassifyError(err); got != frames.ErrorCategoryContentFiltered {
		t.Fatalf("category = %q, want content_filtered", got)
	}
	if !strings.Contains(err.Error(), "req_abc") || !strings.Contains(err.Error(), "content management policy") {
		t.Fatalf("error text = %q", err.Error())
	}
	if len(events) != 0 {
		t.Fatalf("events = %v", events)
	}
}

func TestRequestIDInOrdinaryAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-request-id", "req_500")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"message":"boom","type":"server_error"}}`)
	}))
	defer srv.Close()
	_, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
	if _, refused := llm.AsRefusal(err); refused {
		t.Fatalf("plain API error classified as refusal: %v", err)
	}
	if err == nil || !strings.Contains(err.Error(), "request id req_500") {
		t.Fatalf("err = %v", err)
	}
}

func TestResponseInfoEvent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("x-request-id", "req_xyz")
		_, _ = io.WriteString(w, data(`{"id":"chatcmpl-9","model":"gpt-4.1-2025-04-14","choices":[{"delta":{"content":"a"}}]}`))
		_, _ = io.WriteString(w, data(`{"id":"chatcmpl-9","model":"gpt-4.1-2025-04-14","choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}`))
		_, _ = io.WriteString(w, data("[DONE]"))
	}))
	defer srv.Close()
	events, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "gpt-4.1"})
	if err != nil {
		t.Fatal(err)
	}
	var infos []llm.ResponseInfo
	for _, e := range events {
		if i, ok := e.(llm.ResponseInfo); ok {
			infos = append(infos, i)
		}
	}
	want := llm.ResponseInfo{ResponseID: "chatcmpl-9", RequestID: "req_xyz", Model: "gpt-4.1-2025-04-14"}
	if len(infos) != 1 || infos[0] != want {
		t.Fatalf("infos = %+v", infos)
	}
	if _, first := events[0].(llm.ResponseInfo); !first {
		t.Fatalf("response info should precede content: %v", events)
	}
}

func TestResponseFormatBody(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}, "additionalProperties": false}
	srv, c := sseServer(t, data("[DONE]"))
	p := newProvider(srv)

	format := llm.JSONSchemaFormat("weather", schema)
	format.Description = "a city"
	if _, err := collect(t, p, llm.Snapshot{}, llm.Settings{Model: "m", ResponseFormat: format}); err != nil {
		t.Fatal(err)
	}
	body, _, _ := c.get()
	rf, _ := body["response_format"].(map[string]any)
	js, _ := rf["json_schema"].(map[string]any)
	if rf["type"] != "json_schema" || js["name"] != "weather" || js["strict"] != true || js["description"] != "a city" {
		t.Fatalf("response_format = %v", rf)
	}
	if got, _ := js["schema"].(map[string]any); got["additionalProperties"] != false || got["type"] != "object" {
		t.Fatalf("schema = %v", js["schema"])
	}

	if _, err := collect(t, p, llm.Snapshot{}, llm.Settings{Model: "m", ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject}}); err != nil {
		t.Fatal(err)
	}
	body, _, _ = c.get()
	if rf, _ := body["response_format"].(map[string]any); len(rf) != 1 || rf["type"] != "json_object" {
		t.Fatalf("response_format = %v", body["response_format"])
	}

	for _, s := range []llm.Settings{{Model: "m"}, {Model: "m", ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatText}}} {
		if _, err := collect(t, p, llm.Snapshot{}, s); err != nil {
			t.Fatal(err)
		}
		body, _, _ = c.get()
		if _, present := body["response_format"]; present {
			t.Fatalf("response_format sent for %+v", s.ResponseFormat)
		}
	}
}

func TestInlineThoughtFiltering(t *testing.T) {
	cases := []struct {
		name          string
		chunks        []string
		text, thought string
	}{
		{"single chunk", []string{"<think>plan it</think>\n\nHello there"}, "Hello there", "plan it"},
		{"tags split across chunks", []string{"<th", "ink>pl", "an</thi", "nk>", "\n", "Hel", "lo"}, "Hello", "plan"},
		{"byte at a time", strings.Split("<think>a<b</think>x < y", ""), "x < y", "a<b"},
		{"no thought", []string{"plain ", "answer <b>"}, "plain answer <b>", ""},
		{"near miss tag", []string{"a <thin", "g> b"}, "a <thing> b", ""},
		{"unterminated thought", []string{"<think>still going"}, "", "still going"},
		{"dangling partial tag at end", []string{"value <thi"}, "value <thi", ""},
		{"second thought block", []string{"<think>one</think>A<think>two</think>B"}, "AB", "onetwo"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			events := make([]string, 0, len(tc.chunks)+2)
			for _, c := range tc.chunks {
				events = append(events, data(`{"choices":[{"delta":{"content":`+quote(c)+`}}]}`))
			}
			events = append(events, data(`{"choices":[{"delta":{},"finish_reason":"stop"}]}`), data("[DONE]"))
			srv, _ := sseServer(t, events...)
			got, err := collect(t, newProvider(srv, WithInlineThoughts()), llm.Snapshot{}, llm.Settings{Model: "m"})
			if err != nil {
				t.Fatal(err)
			}
			if textOf(got) != tc.text || thoughtOf(got) != tc.thought {
				t.Fatalf("text = %q thought = %q", textOf(got), thoughtOf(got))
			}
			for _, e := range got {
				if d, ok := e.(llm.TextDelta); ok && d.Text == "" {
					t.Fatal("empty text delta emitted")
				}
			}
		})
	}

	srv, _ := sseServer(t, data(`{"choices":[{"delta":{"content":"<think>kept</think>"}}]}`), data("[DONE]"))
	got, err := collect(t, newProvider(srv), llm.Snapshot{}, llm.Settings{Model: "m"})
	if err != nil || textOf(got) != "<think>kept</think>" {
		t.Fatalf("filter must be opt-in: text = %q err = %v", textOf(got), err)
	}
}

func quote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestRetryOnFirstByteTimeout(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		if hits.Add(1) == 1 {
			w.(http.Flusher).Flush()
			select {
			case <-r.Context().Done():
			case <-time.After(5 * time.Second):
			}
			return
		}
		_, _ = io.WriteString(w, data(`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`)+data("[DONE]"))
	}))
	defer srv.Close()

	policy := RetryPolicy{MaxAttempts: 2, FirstByteTimeout: 80 * time.Millisecond, Backoff: time.Millisecond}
	for name, p := range map[string]*Provider{
		"option": New(Config{Name: "Test", BaseURL: srv.URL, HTTPClient: srv.Client()}, WithRetry(policy)),
		"config": New(Config{Name: "Test", BaseURL: srv.URL, HTTPClient: srv.Client(), Retry: policy}),
	} {
		hits.Store(0)
		events, err := collect(t, p, llm.Snapshot{}, llm.Settings{Model: "m"})
		if err != nil || textOf(events) != "ok" || hits.Load() != 2 {
			t.Fatalf("%s: text = %q err = %v attempts = %d", name, textOf(events), err, hits.Load())
		}
	}

	hits.Store(0)
	off := New(Config{Name: "Test", BaseURL: srv.URL, HTTPClient: srv.Client()})
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	var emitted int
	err := off.Stream(ctx, llm.Snapshot{}, llm.Settings{Model: "m"}, func(llm.Event) { emitted++ })
	if err == nil || hits.Load() != 1 || emitted != 0 {
		t.Fatalf("retry must be off by default: err = %v attempts = %d events = %d", err, hits.Load(), emitted)
	}
}

func TestNoRetryOnceStreaming(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, data(`{"choices":[{"delta":{"content":"partial"}}]}`))
		w.(http.Flusher).Flush()
		time.Sleep(200 * time.Millisecond)
		hj, _ := w.(http.Hijacker)
		conn, _, _ := hj.Hijack()
		conn.Close()
	}))
	defer srv.Close()
	p := New(Config{Name: "Test", BaseURL: srv.URL, HTTPClient: srv.Client(), Retry: RetryPolicy{MaxAttempts: 3, FirstByteTimeout: 50 * time.Millisecond, Backoff: time.Millisecond}})
	events, err := collect(t, p, llm.Snapshot{}, llm.Settings{Model: "m"})
	if err == nil {
		t.Fatalf("expected the broken stream to fail; events = %v", events)
	}
	if hits.Load() != 1 || textOf(events) != "partial" {
		t.Fatalf("attempts = %d text = %q", hits.Load(), textOf(events))
	}
}
