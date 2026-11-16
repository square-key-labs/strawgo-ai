package gemini

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
)

func geminiProvider(srv *httptest.Server) *Provider {
	return NewProvider(ProviderConfig{Model: "m", URL: func(string) string { return srv.URL }})
}

func geminiDone(t *testing.T, events []llm.Event) llm.Done {
	t.Helper()
	for _, e := range events {
		if d, ok := e.(llm.Done); ok {
			return d
		}
	}
	t.Fatalf("no Done event: %v", events)
	return llm.Done{}
}

func TestGeminiStopReasonsAndResponseInfo(t *testing.T) {
	cases := []struct {
		name, chunk, stop, raw string
	}{
		{"stop", `{"candidates":[{"content":{"parts":[{"text":"a"}]},"finishReason":"STOP"}],"responseId":"resp-g1","modelVersion":"gemini-2.5-flash-001"}`, llm.StopReasonStop, "STOP"},
		{"max tokens", `{"candidates":[{"content":{"parts":[{"text":"a"}]},"finishReason":"MAX_TOKENS"}],"responseId":"resp-g1","modelVersion":"gemini-2.5-flash-001"}`, llm.StopReasonLength, "MAX_TOKENS"},
		{"tool call", `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":"STOP"}],"responseId":"resp-g1","modelVersion":"gemini-2.5-flash-001"}`, llm.StopReasonToolCalls, "STOP"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _, _ := newGeminiSSEServer(t, tc.chunk)
			events, err := streamAll(t, geminiProvider(srv), llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{})
			if err != nil {
				t.Fatal(err)
			}
			if d := geminiDone(t, events); d.StopReason != tc.stop || d.RawStopReason != tc.raw {
				t.Fatalf("done = %+v", d)
			}
			want := llm.ResponseInfo{ResponseID: "resp-g1", Model: "gemini-2.5-flash-001"}
			if info, ok := events[0].(llm.ResponseInfo); !ok || info != want {
				t.Fatalf("first event = %+v", events[0])
			}
		})
	}
}

func TestGeminiRefusalsAreTyped(t *testing.T) {
	cases := []struct {
		name, chunk string
		category    llm.RefusalCategory
		reason      string
		detail      string
		code        string
	}{
		{"prompt blocked", `{"promptFeedback":{"blockReason":"PROHIBITED_CONTENT","blockReasonMessage":"not allowed"},"usageMetadata":{"promptTokenCount":4,"totalTokenCount":4}}`, llm.RefusalPromptBlocked, "PROHIBITED_CONTENT", "", "BLOCKED_PROHIBITED_CONTENT"},
		{"safety", `{"candidates":[{"content":{"parts":[{"text":"a"}]},"finishReason":"SAFETY","safetyRatings":[{"category":"HARM_CATEGORY_HARASSMENT","probability":"NEGLIGIBLE"},{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"HIGH","blocked":true}]}]}`, llm.RefusalContentFilter, "SAFETY", "HARM_CATEGORY_DANGEROUS_CONTENT", "SAFETY"},
		{"recitation", `{"candidates":[{"finishReason":"RECITATION"}]}`, llm.RefusalRecitation, "RECITATION", "", "RECITATION"},
		{"blocklist", `{"candidates":[{"finishReason":"BLOCKLIST"}]}`, llm.RefusalContentFilter, "BLOCKLIST", "", "BLOCKLIST"},
		{"prohibited content", `{"candidates":[{"finishReason":"PROHIBITED_CONTENT"}]}`, llm.RefusalContentFilter, "PROHIBITED_CONTENT", "", "PROHIBITED_CONTENT"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _, _ := newGeminiSSEServer(t, tc.chunk)
			events, err := streamAll(t, geminiProvider(srv), llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{})
			r, ok := llm.AsRefusal(err)
			if !ok || r.Category != tc.category || r.Reason != tc.reason || r.Detail != tc.detail || r.Provider != "Gemini" {
				t.Fatalf("err = %v refusal = %+v", err, r)
			}
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.Code != tc.code {
				t.Fatalf("api error = %+v", apiErr)
			}
			if d := geminiDone(t, events); d.StopReason != llm.StopReasonContentFilter || d.RawStopReason != tc.reason {
				t.Fatalf("done = %+v", d)
			}
		})
	}

	srv, _, _ := newGeminiSSEServer(t, `{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL"}]}`)
	_, err := streamAll(t, geminiProvider(srv), llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{})
	if _, refused := llm.AsRefusal(err); err == nil || refused {
		t.Fatalf("malformed function call is an error but not a refusal: %v", err)
	}
}

func TestGeminiStructuredOutput(t *testing.T) {
	srv, _, bodies := newGeminiSSEServer(t, `{"candidates":[{"content":{"parts":[{"text":"{}"}]},"finishReason":"STOP"}]}`)
	p := geminiProvider(srv)
	schema := map[string]any{"type": "object", "properties": map[string]any{"n": map[string]any{"type": "integer"}}}
	if _, err := streamAll(t, p, llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{ResponseFormat: llm.JSONSchemaFormat("n", schema)}); err != nil {
		t.Fatal(err)
	}
	gen := mustMap(t, (<-bodies)["generationConfig"], "generationConfig")
	if gen["responseMimeType"] != "application/json" || mustMap(t, gen["responseJsonSchema"], "schema")["type"] != "object" {
		t.Fatalf("generationConfig = %v", gen)
	}
	if _, err := streamAll(t, p, llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject}}); err != nil {
		t.Fatal(err)
	}
	gen = mustMap(t, (<-bodies)["generationConfig"], "generationConfig")
	if _, hasSchema := gen["responseJsonSchema"]; gen["responseMimeType"] != "application/json" || hasSchema {
		t.Fatalf("generationConfig = %v", gen)
	}
	if _, err := streamAll(t, p, llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{}); err != nil {
		t.Fatal(err)
	}
	if _, present := (<-bodies)["generationConfig"]; present {
		t.Fatal("generationConfig sent without settings")
	}
}

func TestGeminiToolCallSignatureSurvivesRestart(t *testing.T) {
	srv, _, _ := newGeminiSSEServer(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Pune"}},"thoughtSignature":"c2lnLTE="}]},"finishReason":"STOP"}]}`,
	)
	events, err := streamAll(t, geminiProvider(srv), llm.NewContext(llm.User("weather?")).Snapshot(), llm.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	var call llm.ToolCall
	for _, e := range events {
		if tc, ok := e.(llm.ToolCall); ok {
			call = tc
		}
	}
	if call.Signature != "c2lnLTE=" {
		t.Fatalf("call = %+v", call)
	}

	restarted := geminiProvider(srv)
	persisted := llm.NewContext(
		llm.User("weather?"),
		llm.ToolCalls(llm.ToolCallPart{ID: call.ID, Name: call.Name, Arguments: call.Arguments, Signature: call.Signature}),
		llm.ToolResult(call.ID, call.Name, `{"temp":21}`),
	)
	_, contents := restarted.ConvertMessages(persisted.Snapshot().Messages, "")
	if got := contents[1].Parts[0].ThoughtSignature; got != "c2lnLTE=" {
		t.Fatalf("signature from context not sent after restart: %q", got)
	}

	unsigned := llm.NewContext(llm.User("weather?"), llm.ToolCalls(llm.ToolCallPart{ID: call.ID, Name: call.Name, Arguments: call.Arguments}))
	_, contents = restarted.ConvertMessages(unsigned.Snapshot().Messages, "")
	if got := contents[1].Parts[0].ThoughtSignature; got != placeholderSignature {
		t.Fatalf("restarted provider without a context signature should fall back to the placeholder, got %q", got)
	}
}

func TestGeminiRetryOnFirstByteTimeout(t *testing.T) {
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
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n")
	}))
	defer srv.Close()
	p := NewProvider(ProviderConfig{Model: "m", URL: func(string) string { return srv.URL }, Retry: RetryPolicy{MaxAttempts: 2, FirstByteTimeout: 80 * time.Millisecond, Backoff: time.Millisecond}})
	events, err := streamAll(t, p, llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{})
	if err != nil || hits.Load() != 2 {
		t.Fatalf("err = %v attempts = %d", err, hits.Load())
	}
	if d := geminiDone(t, events); d.StopReason != llm.StopReasonStop {
		t.Fatalf("done = %+v", d)
	}
}

func TestNewLLMServiceDefaultModel(t *testing.T) {
	if got := NewLLMService(LLMConfig{APIKey: "k"}).Settings().Model; got != DefaultModel {
		t.Fatalf("model = %q, want %q", got, DefaultModel)
	}
	if got := NewLLMService(LLMConfig{APIKey: "k", Model: "gemini-custom"}).Settings().Model; got != "gemini-custom" {
		t.Fatalf("model = %q, want gemini-custom", got)
	}
}

func TestGeminiHTTPErrorIsClassified(t *testing.T) {
	cases := []struct {
		name     string
		status   int
		body     string
		code     string
		category frames.ErrorCategory
	}{
		{"not found", http.StatusNotFound, `{"error":{"code":404,"message":"models/nope is not found","status":"NOT_FOUND"}}`, "NOT_FOUND", frames.ErrorCategoryConfiguration},
		{"unauthenticated", http.StatusUnauthorized, `{"error":{"code":401,"message":"missing credentials","status":"UNAUTHENTICATED"}}`, "UNAUTHENTICATED", frames.ErrorCategoryAuthentication},
		{"permission denied", http.StatusForbidden, `{"error":{"code":403,"message":"API key not valid","status":"PERMISSION_DENIED"}}`, "PERMISSION_DENIED", frames.ErrorCategoryAuthorization},
		{"resource exhausted", http.StatusTooManyRequests, `{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}`, "RESOURCE_EXHAUSTED", frames.ErrorCategoryRateLimit},
		{"invalid argument", http.StatusBadRequest, `{"error":{"code":400,"message":"bad field","status":"INVALID_ARGUMENT"}}`, "INVALID_ARGUMENT", frames.ErrorCategoryInvalidRequest},
		{"plain 404", http.StatusNotFound, "no such page", "Not Found", frames.ErrorCategoryInvalidRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer srv.Close()
			_, err := streamAll(t, geminiProvider(srv), llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{})
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.Status != tc.status || apiErr.Code != tc.code {
				t.Fatalf("err = %v", err)
			}
			if got := frames.ClassifyError(err); got != tc.category {
				t.Fatalf("category = %q, want %q", got, tc.category)
			}
		})
	}
}
