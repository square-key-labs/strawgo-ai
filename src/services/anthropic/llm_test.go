package anthropic

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

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type frameCapturer struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (c *frameCapturer) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.frames = append(c.frames, frame)
	return nil
}

func (c *frameCapturer) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}
func (c *frameCapturer) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}
func (c *frameCapturer) Link(next processors.FrameProcessor)    {}
func (c *frameCapturer) SetPrev(prev processors.FrameProcessor) {}
func (c *frameCapturer) Start(ctx context.Context) error        { return nil }
func (c *frameCapturer) Stop() error                            { return nil }
func (c *frameCapturer) Name() string                           { return "TestCapturer" }

func (c *frameCapturer) getFrames() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]frames.Frame(nil), c.frames...)
}

func writeSSE(w http.ResponseWriter, eventType string, data any) {
	b, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, b)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

type capturedRequest struct {
	header http.Header
	url    string
	body   map[string]any
}

func newServer(t *testing.T, handler func(w http.ResponseWriter)) (*httptest.Server, chan capturedRequest) {
	t.Helper()
	reqs := make(chan capturedRequest, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		reqs <- capturedRequest{header: r.Header.Clone(), url: r.URL.String(), body: body}
		w.Header().Set("Content-Type", "text/event-stream")
		handler(w)
	}))
	t.Cleanup(srv.Close)
	return srv, reqs
}

func collect(t *testing.T, p *Provider, snap llm.Snapshot, settings llm.Settings) ([]llm.Event, error) {
	t.Helper()
	var events []llm.Event
	err := p.Stream(context.Background(), snap, settings, func(e llm.Event) { events = append(events, e) })
	return events, err
}

func TestNewLLMServiceDefaults(t *testing.T) {
	svc := NewLLMService(LLMConfig{APIKey: "k", SystemPrompt: "be brief"})
	s := svc.Settings()
	if s.Model != DefaultModel {
		t.Errorf("model = %q, want %q", s.Model, DefaultModel)
	}
	if s.MaxTokens == nil || *s.MaxTokens != DefaultMaxTokens {
		t.Errorf("max tokens = %v", s.MaxTokens)
	}
	if s.SystemInstruction != "be brief" {
		t.Errorf("system = %q", s.SystemInstruction)
	}
	if s.Temperature != nil {
		t.Errorf("temperature should be unset")
	}
	if svc.Name() != "Anthropic" {
		t.Errorf("name = %q", svc.Name())
	}
}

func TestStreamTextUsageAndThinking(t *testing.T) {
	srv, reqs := newServer(t, func(w http.ResponseWriter) {
		writeSSE(w, "message_start", map[string]any{"type": "message_start", "message": map[string]any{"usage": map[string]any{"input_tokens": 10, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 2, "output_tokens": 1}}})
		writeSSE(w, "content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "thinking", "thinking": ""}})
		writeSSE(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "thinking_delta", "thinking": "hmm"}})
		writeSSE(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "signature_delta", "signature": "sig-1"}})
		writeSSE(w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		fmt.Fprint(w, ": keepalive\n\n")
		writeSSE(w, "ping", map[string]any{"type": "ping"})
		writeSSE(w, "content_block_start", map[string]any{"type": "content_block_start", "index": 1, "content_block": map[string]any{"type": "text", "text": ""}})
		writeSSE(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 1, "delta": map[string]any{"type": "text_delta", "text": "Hello"}})
		writeSSE(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 1, "delta": map[string]any{"type": "text_delta", "text": " world"}})
		writeSSE(w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 1})
		writeSSE(w, "message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "end_turn"}, "usage": map[string]any{"output_tokens": 7}})
		writeSSE(w, "message_stop", map[string]any{"type": "message_stop"})
	})
	p := NewProvider(LLMConfig{APIKey: "secret", BaseURL: srv.URL, Thinking: &Thinking{Type: "adaptive"}})
	snap := llm.NewContext(llm.System("sys a"), llm.Developer("dev b"), llm.User("hi")).Snapshot()
	events, err := collect(t, p, snap, llm.Settings{SystemInstruction: "settings sys"})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	req := <-reqs
	if req.header.Get("x-api-key") != "secret" || strings.Contains(req.url, "secret") {
		t.Errorf("api key must be in header only: url=%q", req.url)
	}
	if req.header.Get("anthropic-version") != APIVersion {
		t.Errorf("missing anthropic-version")
	}
	system, _ := req.body["system"].([]any)
	if len(system) != 3 {
		t.Fatalf("system blocks = %v", req.body["system"])
	}
	for i, want := range []string{"settings sys", "sys a", "dev b"} {
		if system[i].(map[string]any)["text"] != want {
			t.Errorf("system[%d] = %v, want %q", i, system[i], want)
		}
	}
	msgs := req.body["messages"].([]any)
	if len(msgs) != 1 || msgs[0].(map[string]any)["role"] != "user" {
		t.Errorf("messages = %v", msgs)
	}
	if req.body["thinking"].(map[string]any)["type"] != "adaptive" {
		t.Errorf("thinking = %v", req.body["thinking"])
	}
	if req.body["stream"] != true || req.body["max_tokens"] == nil {
		t.Errorf("stream/max_tokens missing: %v", req.body)
	}

	var text, thought, sig string
	var usage *llm.Usage
	done := false
	for _, e := range events {
		switch ev := e.(type) {
		case llm.TextDelta:
			text += ev.Text
		case llm.ThoughtDelta:
			thought += ev.Text
			if ev.Signature != "" {
				sig = ev.Signature
			}
		case llm.UsageEvent:
			u := ev.Usage
			usage = &u
		case llm.Done:
			done = ev.StopReason == llm.StopReasonStop && ev.RawStopReason == "end_turn"
		}
	}
	if text != "Hello world" || thought != "hmm" || sig != "sig-1" || !done {
		t.Errorf("text=%q thought=%q sig=%q done=%v", text, thought, sig, done)
	}
	if usage == nil || usage.PromptTokens != 10 || usage.CompletionTokens != 7 || usage.CacheReadTokens != 4 || usage.CacheCreationTokens != 2 {
		t.Errorf("usage = %+v", usage)
	}
}

func TestStreamParallelToolCalls(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter) {
		writeSSE(w, "content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "tool_use", "id": "tu_1", "name": "weather"}})
		writeSSE(w, "content_block_start", map[string]any{"type": "content_block_start", "index": 1, "content_block": map[string]any{"type": "tool_use", "id": "tu_2", "name": "time"}})
		writeSSE(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "input_json_delta", "partial_json": `{"city":`}})
		writeSSE(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "input_json_delta", "partial_json": `"Paris"}`}})
		writeSSE(w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		writeSSE(w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 1})
		writeSSE(w, "message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use"}})
		writeSSE(w, "message_stop", map[string]any{"type": "message_stop"})
	})
	p := NewProvider(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	events, err := collect(t, p, llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	var calls []llm.ToolCall
	for _, e := range events {
		if c, ok := e.(llm.ToolCall); ok {
			calls = append(calls, c)
		}
	}
	if len(calls) != 2 {
		t.Fatalf("calls = %+v", calls)
	}
	if calls[0].ID != "tu_1" || string(calls[0].Arguments) != `{"city":"Paris"}` {
		t.Errorf("call0 = %+v", calls[0])
	}
	if calls[1].ID != "tu_2" || string(calls[1].Arguments) != `{}` {
		t.Errorf("call1 = %+v", calls[1])
	}
}

func TestConvertMessagesToolRoundTrip(t *testing.T) {
	msgs := []llm.Message{
		llm.System("sys"),
		llm.User("weather and time?"),
		{Role: llm.RoleAssistant, Parts: []llm.Part{
			llm.ThoughtPart{Text: "thinking", Signature: "sig"},
			llm.TextPart{Text: "checking"},
			llm.ToolCallPart{ID: "a", Name: "weather", Arguments: []byte(`{"city":"Paris"}`)},
			llm.ToolCallPart{ID: "b", Name: "time"},
		}},
		llm.ToolResult("a", "weather", `{"temp":20}`),
		{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultPart{ToolCallID: "b", Content: "boom", IsError: true}}},
		llm.System("mid-call note"),
		{Role: llm.RoleUser, Parts: []llm.Part{llm.ImagePart{Data: []byte{0x89, 'P', 'N', 'G'}, MIME: "image/png"}}},
	}
	system, out := convertMessages(msgs, "")
	if len(system) != 1 || system[0]["text"] != "sys" {
		t.Fatalf("system = %v", system)
	}
	if len(out) != 3 {
		t.Fatalf("messages = %d, want 3 (user, assistant, merged user)", len(out))
	}
	asst := out[1]
	if asst.Role != "assistant" || len(asst.Content) != 4 {
		t.Fatalf("assistant = %+v", asst)
	}
	if asst.Content[0]["type"] != "thinking" || asst.Content[0]["signature"] != "sig" {
		t.Errorf("thinking not round-tripped: %v", asst.Content[0])
	}
	if asst.Content[2]["type"] != "tool_use" || asst.Content[3]["type"] != "tool_use" {
		t.Errorf("tool_use blocks = %v", asst.Content)
	}
	user := out[2]
	if user.Role != "user" || len(user.Content) != 4 {
		t.Fatalf("merged user = %+v", user)
	}
	if user.Content[0]["type"] != "tool_result" || user.Content[1]["type"] != "tool_result" {
		t.Errorf("tool results must lead the user message: %v", user.Content)
	}
	if user.Content[1]["is_error"] != true {
		t.Errorf("is_error lost: %v", user.Content[1])
	}
	if user.Content[2]["text"] != "mid-call note" || user.Content[3]["type"] != "image" {
		t.Errorf("trailing blocks = %v", user.Content[2:])
	}
}

func TestPromptCachingMarkers(t *testing.T) {
	p := NewProvider(LLMConfig{APIKey: "k", EnablePromptCaching: true, PromptCacheTTL: "1h"})
	ctx := llm.NewContext(llm.System("sys"), llm.User("one"), llm.Assistant("two"), llm.User("three"))
	ctx.SetTools(llm.Tools{Functions: []llm.FunctionSchema{{Name: "f", Properties: map[string]any{"x": map[string]any{"type": "string"}}}}})
	ctx.SetToolChoice(llm.ToolChoice{Mode: llm.ToolChoiceNone})
	body, err := p.buildRequest(ctx.Snapshot(), llm.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(body)
	var decoded map[string]any
	_ = json.Unmarshal(b, &decoded)
	sys := decoded["system"].([]any)[0].(map[string]any)
	if cc := sys["cache_control"].(map[string]any); cc["ttl"] != "1h" {
		t.Errorf("system cache_control = %v", cc)
	}
	tools := decoded["tools"].([]any)
	if tools[0].(map[string]any)["cache_control"] == nil {
		t.Errorf("tools not cached")
	}
	if decoded["tool_choice"].(map[string]any)["type"] != "none" {
		t.Errorf("tool_choice = %v", decoded["tool_choice"])
	}
	msgs := decoded["messages"].([]any)
	for i, wantCached := range []bool{true, false, true} {
		content := msgs[i].(map[string]any)["content"].([]any)
		_, cached := content[len(content)-1].(map[string]any)["cache_control"]
		if cached != wantCached {
			t.Errorf("message %d cached=%v want %v", i, cached, wantCached)
		}
	}

	off, _ := p.buildRequest(ctx.Snapshot(), llm.Settings{Extra: map[string]any{ExtraPromptCaching: false}})
	if strings.Contains(mustJSON(off), "cache_control") || strings.Contains(mustJSON(off), ExtraPromptCaching) {
		t.Errorf("caching should be disabled by settings extra")
	}
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func TestMidStreamErrorEvent(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter) {
		writeSSE(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": "partial"}})
		writeSSE(w, "error", map[string]any{"type": "error", "error": map[string]any{"type": "overloaded_error", "message": "Overloaded"}})
	})
	p := NewProvider(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	_, err := collect(t, p, llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Type != "overloaded_error" {
		t.Fatalf("err = %v", err)
	}
}

func TestHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}`)
	}))
	defer srv.Close()
	p := NewProvider(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	_, err := collect(t, p, llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != 400 || apiErr.Message != "bad" {
		t.Fatalf("err = %v", err)
	}
	if got := frames.ClassifyError(err); got != frames.ErrorCategoryInvalidRequest {
		t.Fatalf("category = %q, want invalid_request", got)
	}
}

func TestLargeSSELine(t *testing.T) {
	big := strings.Repeat("x", 200<<10)
	srv, _ := newServer(t, func(w http.ResponseWriter) {
		writeSSE(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": big}})
		writeSSE(w, "message_stop", map[string]any{"type": "message_stop"})
	})
	p := NewProvider(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	events, err := collect(t, p, llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	if td, ok := events[0].(llm.TextDelta); !ok || len(td.Text) != len(big) {
		t.Fatalf("large delta not delivered intact")
	}
}

func TestStreamCancellation(t *testing.T) {
	release := make(chan struct{})
	srv, _ := newServer(t, func(w http.ResponseWriter) {
		writeSSE(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": "a"}})
		<-release
	})
	defer close(release)
	p := NewProvider(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	ctx, cancel := context.WithCancel(context.Background())
	got := make(chan struct{}, 1)
	errc := make(chan error, 1)
	go func() {
		errc <- p.Stream(ctx, llm.NewContext(llm.User("hi")).Snapshot(), llm.Settings{}, func(llm.Event) {
			select {
			case got <- struct{}{}:
			default:
			}
		})
	}()
	<-got
	cancel()
	select {
	case err := <-errc:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("stream did not stop on cancel")
	}
}

func TestServiceEmitsFramesWithoutTouchingContext(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter) {
		writeSSE(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": "Hi"}})
		writeSSE(w, "message_stop", map[string]any{"type": "message_stop"})
	})
	svc := NewLLMService(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	capt := &frameCapturer{}
	svc.Link(capt)
	c := llm.NewContext(llm.User("hello"))
	if err := svc.HandleFrame(context.Background(), frames.NewLLMContextFrame(c), frames.Downstream); err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, f := range capt.getFrames() {
		names = append(names, f.Name())
	}
	got := strings.Join(names, ",")
	if !strings.HasPrefix(got, "LLMFullResponseStartFrame") || !strings.Contains(got, "LLMTextFrame") || !strings.Contains(got, "LLMFullResponseEndFrame") {
		t.Errorf("frames = %s", got)
	}
	if c.Len() != 1 {
		t.Errorf("service must not mutate the context; len = %d", c.Len())
	}
}
