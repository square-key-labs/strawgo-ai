package openai

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

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

type responsesServer struct {
	*httptest.Server
	body atomic.Pointer[map[string]any]
}

func newResponsesServer(t *testing.T, status int, events ...string) *responsesServer {
	t.Helper()
	s := &responsesServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		s.body.Store(&body)
		w.Header().Set("x-request-id", "req_resp")
		if status != http.StatusOK {
			w.WriteHeader(status)
			_, _ = io.WriteString(w, strings.Join(events, ""))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			_, _ = io.WriteString(w, "data: "+e+"\n\n")
		}
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *responsesServer) provider(retry openaicompat.RetryPolicy) *ResponsesProvider {
	return NewResponsesProvider(LLMConfig{APIKey: "k", BaseURL: s.URL, HTTPClient: s.Client(), Retry: retry})
}

func run(t *testing.T, p *ResponsesProvider, settings llm.Settings) ([]llm.Event, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var events []llm.Event
	err := p.Stream(ctx, llm.NewContext(llm.User("hi")).Snapshot(), settings, func(e llm.Event) { events = append(events, e) })
	return events, err
}

func lastDone(t *testing.T, events []llm.Event) llm.Done {
	t.Helper()
	if len(events) > 0 {
		if d, ok := events[len(events)-1].(llm.Done); ok {
			return d
		}
	}
	t.Fatalf("stream did not end with Done: %v", events)
	return llm.Done{}
}

func TestResponsesInfoAndStopReasons(t *testing.T) {
	created := `{"type":"response.created","response":{"id":"resp_1","model":"gpt-4.1-2025-04-14","status":"in_progress"}}`
	cases := []struct {
		name      string
		events    []string
		stop, raw string
	}{
		{"completed", []string{created, `{"type":"response.output_text.delta","delta":"hi"}`, `{"type":"response.completed","response":{"id":"resp_1","model":"gpt-4.1-2025-04-14","status":"completed","usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}}`}, llm.StopReasonStop, "completed"},
		{"truncated", []string{created, `{"type":"response.output_text.delta","delta":"hi"}`, `{"type":"response.incomplete","response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`}, llm.StopReasonLength, "max_output_tokens"},
		{"tool call", []string{created, `{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"f","arguments":"{}"}}`, `{"type":"response.completed","response":{"id":"resp_1","status":"completed"}}`}, llm.StopReasonToolCalls, "completed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := newResponsesServer(t, http.StatusOK, tc.events...)
			events, err := run(t, srv.provider(openaicompat.RetryPolicy{}), llm.Settings{Model: "gpt-4.1"})
			if err != nil {
				t.Fatal(err)
			}
			want := llm.ResponseInfo{ResponseID: "resp_1", RequestID: "req_resp", Model: "gpt-4.1-2025-04-14"}
			if info, ok := events[0].(llm.ResponseInfo); !ok || info != want {
				t.Fatalf("first event = %+v", events[0])
			}
			if d := lastDone(t, events); d.StopReason != tc.stop || d.RawStopReason != tc.raw {
				t.Fatalf("done = %+v", d)
			}
		})
	}
}

func TestResponsesRefusals(t *testing.T) {
	t.Run("content filter", func(t *testing.T) {
		srv := newResponsesServer(t, http.StatusOK,
			`{"type":"response.output_text.delta","delta":"par"}`,
			`{"type":"response.incomplete","response":{"id":"resp_2","status":"incomplete","incomplete_details":{"reason":"content_filter"}}}`,
		)
		events, err := run(t, srv.provider(openaicompat.RetryPolicy{}), llm.Settings{Model: "m"})
		r, ok := llm.AsRefusal(err)
		if !ok || r.Category != llm.RefusalContentFilter || r.Provider != "OpenAIResponses" {
			t.Fatalf("err = %v refusal = %+v", err, r)
		}
		if d := lastDone(t, events); d.StopReason != llm.StopReasonContentFilter {
			t.Fatalf("done = %+v", d)
		}
	})
	t.Run("model refusal", func(t *testing.T) {
		srv := newResponsesServer(t, http.StatusOK,
			`{"type":"response.refusal.delta","delta":"I can't "}`,
			`{"type":"response.refusal.delta","delta":"help with that."}`,
			`{"type":"response.completed","response":{"id":"resp_3","status":"completed"}}`,
		)
		events, err := run(t, srv.provider(openaicompat.RetryPolicy{}), llm.Settings{Model: "m"})
		r, ok := llm.AsRefusal(err)
		if !ok || r.Category != llm.RefusalModel || r.Message != "I can't help with that." {
			t.Fatalf("err = %v refusal = %+v", err, r)
		}
		if d := lastDone(t, events); d.StopReason != llm.StopReasonRefusal {
			t.Fatalf("done = %+v", d)
		}
	})
}

func TestResponsesErrorsCarryRequestID(t *testing.T) {
	srv := newResponsesServer(t, http.StatusTooManyRequests, `{"error":{"message":"slow down","type":"rate_limit_error","code":"rate_limit_exceeded"}}`)
	_, err := run(t, srv.provider(openaicompat.RetryPolicy{}), llm.Settings{Model: "m"})
	var apiErr *openaicompat.APIError
	if !errors.As(err, &apiErr) || apiErr.RequestID != "req_resp" || !strings.Contains(err.Error(), "request id req_resp") {
		t.Fatalf("err = %v", err)
	}

	failed := newResponsesServer(t, http.StatusOK, `{"type":"response.failed","response":{"id":"resp_4","status":"failed","error":{"code":"server_error","message":"exploded"}}}`)
	_, err = run(t, failed.provider(openaicompat.RetryPolicy{}), llm.Settings{Model: "m"})
	if !errors.As(err, &apiErr) || apiErr.Message != "exploded" || apiErr.RequestID != "req_resp" {
		t.Fatalf("err = %v", err)
	}
}

func TestResponsesTextFormat(t *testing.T) {
	srv := newResponsesServer(t, http.StatusOK, `{"type":"response.completed","response":{"id":"resp_5","status":"completed"}}`)
	p := srv.provider(openaicompat.RetryPolicy{})
	schema := map[string]any{"type": "object", "properties": map[string]any{"n": map[string]any{"type": "integer"}}}
	if _, err := run(t, p, llm.Settings{Model: "m", ResponseFormat: llm.JSONSchemaFormat("", schema)}); err != nil {
		t.Fatal(err)
	}
	text, _ := (*srv.body.Load())["text"].(map[string]any)
	format, _ := text["format"].(map[string]any)
	if format["type"] != "json_schema" || format["name"] != "response" || format["strict"] != true {
		t.Fatalf("text.format = %v", format)
	}
	if s, _ := format["schema"].(map[string]any); s["type"] != "object" {
		t.Fatalf("schema = %v", format["schema"])
	}
	if _, err := run(t, p, llm.Settings{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	if _, present := (*srv.body.Load())["text"]; present {
		t.Fatal("text.format sent without a response format")
	}
}

func TestResponsesRetryOnFirstByteTimeout(t *testing.T) {
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
		_, _ = io.WriteString(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"status\":\"completed\"}}\n\n")
	}))
	defer srv.Close()
	p := NewResponsesProvider(LLMConfig{BaseURL: srv.URL, HTTPClient: srv.Client(), Retry: openaicompat.RetryPolicy{MaxAttempts: 2, FirstByteTimeout: 80 * time.Millisecond, Backoff: time.Millisecond}})
	events, err := run(t, p, llm.Settings{Model: "m"})
	if err != nil || hits.Load() != 2 {
		t.Fatalf("err = %v attempts = %d", err, hits.Load())
	}
	text := ""
	for _, e := range events {
		if d, ok := e.(llm.TextDelta); ok {
			text += d.Text
		}
	}
	if text != "ok" {
		t.Fatalf("text = %q", text)
	}
}
