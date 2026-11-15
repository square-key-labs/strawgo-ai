package anthropic

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

func messageStart(w http.ResponseWriter) {
	writeSSE(w, "message_start", map[string]any{"type": "message_start", "message": map[string]any{
		"id": "msg_01", "model": "claude-sonnet-4-6", "usage": map[string]any{"input_tokens": 5, "output_tokens": 1},
	}})
}

func finish(w http.ResponseWriter, delta map[string]any) {
	writeSSE(w, "message_delta", map[string]any{"type": "message_delta", "delta": delta, "usage": map[string]any{"output_tokens": 3}})
	writeSSE(w, "message_stop", map[string]any{"type": "message_stop"})
}

func textBlock(w http.ResponseWriter, text string) {
	writeSSE(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": text}})
}

func finalDone(t *testing.T, events []llm.Event) llm.Done {
	t.Helper()
	for _, e := range events {
		if d, ok := e.(llm.Done); ok {
			return d
		}
	}
	t.Fatalf("no Done event: %v", events)
	return llm.Done{}
}

func TestStopReasonsNormalised(t *testing.T) {
	cases := map[string]string{
		"end_turn":      llm.StopReasonStop,
		"stop_sequence": llm.StopReasonStop,
		"max_tokens":    llm.StopReasonLength,
		"tool_use":      llm.StopReasonToolCalls,
		"pause_turn":    llm.StopReasonOther,
	}
	for raw, want := range cases {
		t.Run(raw, func(t *testing.T) {
			srv, _ := newServer(t, func(w http.ResponseWriter) {
				messageStart(w)
				textBlock(w, "hi")
				finish(w, map[string]any{"stop_reason": raw})
			})
			p := NewProvider(LLMConfig{APIKey: "k", BaseURL: srv.URL})
			events, err := collect(t, p, llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{})
			if err != nil {
				t.Fatal(err)
			}
			if d := finalDone(t, events); d.StopReason != want || d.RawStopReason != raw {
				t.Fatalf("done = %+v", d)
			}
		})
	}
}

func TestRefusalStopReason(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("request-id", "req_011")
		messageStart(w)
		textBlock(w, "partial")
		finish(w, map[string]any{"stop_reason": "refusal", "stop_details": map[string]any{"type": "refusal", "category": "cyber"}})
	}))
	defer srv.Close()
	p := NewProvider(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	events, err := collect(t, p, llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{})
	r, ok := llm.AsRefusal(err)
	if !ok || r.Category != llm.RefusalModel || r.Reason != "refusal" || r.Detail != "cyber" || r.Provider != "Anthropic" {
		t.Fatalf("err = %v refusal = %+v", err, r)
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Type != "refusal" || apiErr.RequestID != "req_011" {
		t.Fatalf("api error = %+v", apiErr)
	}
	if d := finalDone(t, events); d.StopReason != llm.StopReasonRefusal || d.RawStopReason != "refusal" {
		t.Fatalf("done = %+v", d)
	}
	var info llm.ResponseInfo
	var sawRefusal, sawUsage bool
	for _, e := range events {
		switch ev := e.(type) {
		case llm.ResponseInfo:
			info = ev
		case llm.Refusal:
			sawRefusal = true
		case llm.UsageEvent:
			sawUsage = true
		}
	}
	if want := (llm.ResponseInfo{ResponseID: "msg_01", RequestID: "req_011", Model: "claude-sonnet-4-6"}); info != want {
		t.Fatalf("info = %+v", info)
	}
	if !sawRefusal || !sawUsage {
		t.Fatalf("refusal=%v usage=%v", sawRefusal, sawUsage)
	}
}

func TestHTTPErrorCarriesRequestID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("request-id", "req_err")
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprint(w, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
	}))
	defer srv.Close()
	p := NewProvider(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	_, err := collect(t, p, llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.RequestID != "req_err" || !strings.Contains(err.Error(), "request id req_err") {
		t.Fatalf("err = %v", err)
	}
	if _, refused := llm.AsRefusal(err); refused {
		t.Fatal("overload classified as refusal")
	}
}

func TestStructuredOutputFormat(t *testing.T) {
	srv, reqs := newServer(t, func(w http.ResponseWriter) {
		messageStart(w)
		finish(w, map[string]any{"stop_reason": "end_turn"})
	})
	p := NewProvider(LLMConfig{APIKey: "k", BaseURL: srv.URL, Effort: "low"})
	schema := map[string]any{"type": "object", "properties": map[string]any{"n": map[string]any{"type": "integer"}}}
	if _, err := collect(t, p, llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{ResponseFormat: llm.JSONSchemaFormat("n", schema)}); err != nil {
		t.Fatal(err)
	}
	oc, _ := (<-reqs).body["output_config"].(map[string]any)
	format, _ := oc["format"].(map[string]any)
	if oc["effort"] != "low" || format["type"] != "json_schema" || len(format) != 2 {
		t.Fatalf("output_config = %v", oc)
	}
	if s, _ := format["schema"].(map[string]any); s["type"] != "object" {
		t.Fatalf("schema = %v", format["schema"])
	}

	_, err := collect(t, p, llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject}})
	if !errors.Is(err, llm.ErrResponseFormatUnsupported) {
		t.Fatalf("schema-less JSON mode must be rejected, got %v", err)
	}
	select {
	case r := <-reqs:
		t.Fatalf("unsupported format still sent a request: %v", r.body)
	default:
	}
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
		messageStart(w)
		textBlock(w, "ok")
		finish(w, map[string]any{"stop_reason": "end_turn"})
	}))
	defer srv.Close()
	p := NewProvider(LLMConfig{APIKey: "k", BaseURL: srv.URL, Retry: RetryPolicy{MaxAttempts: 2, FirstByteTimeout: 80 * time.Millisecond, Backoff: time.Millisecond}})
	events, err := collect(t, p, llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{})
	if err != nil || hits.Load() != 2 {
		t.Fatalf("err = %v attempts = %d", err, hits.Load())
	}
	texts := 0
	for _, e := range events {
		if d, ok := e.(llm.TextDelta); ok && d.Text == "ok" {
			texts++
		}
	}
	if texts != 1 {
		t.Fatalf("text emitted %d times: %v", texts, events)
	}
}
