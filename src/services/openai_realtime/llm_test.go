package openai_realtime

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services/realtime"
)

type llmFrameCollector struct {
	name string
	ch   chan frames.Frame
}

func newLLMFrameCollector(name string) *llmFrameCollector {
	return &llmFrameCollector{name: name, ch: make(chan frames.Frame, 256)}
}

func (c *llmFrameCollector) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return c.QueueFrame(frame, direction)
}
func (c *llmFrameCollector) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	select {
	case c.ch <- frame:
	default:
	}
	return nil
}
func (c *llmFrameCollector) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return c.QueueFrame(frame, direction)
}
func (c *llmFrameCollector) Link(next processors.FrameProcessor)    {}
func (c *llmFrameCollector) SetPrev(prev processors.FrameProcessor) {}
func (c *llmFrameCollector) Start(ctx context.Context) error        { return nil }
func (c *llmFrameCollector) Stop() error                            { return nil }
func (c *llmFrameCollector) Name() string                           { return c.name }

func (c *llmFrameCollector) waitForFrame(timeout time.Duration, predicate func(frames.Frame) bool) (frames.Frame, bool) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		select {
		case frame := <-c.ch:
			if predicate(frame) {
				return frame, true
			}
		case <-timer.C:
			return nil, false
		}
	}
}

type llmMockRealtimeServer struct {
	t *testing.T

	server *httptest.Server

	connMu sync.Mutex
	conn   *websocket.Conn

	msgs          chan map[string]any
	connected     chan struct{}
	connectedOnce sync.Once

	modelQuery string
	authHeader string
	betaHeader string
}

func newLLMMockRealtimeServer(t *testing.T) *llmMockRealtimeServer {
	t.Helper()

	m := &llmMockRealtimeServer{
		t:         t,
		msgs:      make(chan map[string]any, 256),
		connected: make(chan struct{}),
	}

	upgrader := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
	mux := http.NewServeMux()
	mux.HandleFunc("/realtime", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade failed: %v", err)
			return
		}

		m.connMu.Lock()
		m.conn = conn
		m.modelQuery = r.URL.Query().Get("model")
		m.authHeader = r.Header.Get("Authorization")
		m.betaHeader = r.Header.Get("OpenAI-Beta")
		m.connMu.Unlock()

		m.connectedOnce.Do(func() { close(m.connected) })
		go m.readLoop(conn)
	})

	m.server = httptest.NewServer(mux)
	return m
}

func (m *llmMockRealtimeServer) URL() string {
	return "ws" + strings.TrimPrefix(m.server.URL, "http") + "/realtime"
}

func (m *llmMockRealtimeServer) readLoop(conn *websocket.Conn) {
	for {
		var payload map[string]any
		if err := conn.ReadJSON(&payload); err != nil {
			return
		}
		select {
		case m.msgs <- payload:
		default:
		}
	}
}

func (m *llmMockRealtimeServer) waitConnected(timeout time.Duration) bool {
	select {
	case <-m.connected:
		return true
	case <-time.After(timeout):
		return false
	}
}

func (m *llmMockRealtimeServer) waitMessage(timeout time.Duration, predicate func(map[string]any) bool) (map[string]any, bool) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case msg := <-m.msgs:
			if predicate == nil || predicate(msg) {
				return msg, true
			}
		case <-timer.C:
			return nil, false
		}
	}
}

func (m *llmMockRealtimeServer) send(payload map[string]any) {
	m.t.Helper()
	m.connMu.Lock()
	conn := m.conn
	m.connMu.Unlock()
	if conn == nil {
		m.t.Fatal("mock websocket not connected")
	}
	m.connMu.Lock()
	err := conn.WriteJSON(payload)
	m.connMu.Unlock()
	if err != nil {
		m.t.Fatalf("mock websocket write failed: %v", err)
	}
}

func (m *llmMockRealtimeServer) dropConnection() {
	m.connMu.Lock()
	if m.conn != nil {
		_ = m.conn.Close()
	}
	m.connMu.Unlock()
}

func (m *llmMockRealtimeServer) close() {
	m.connMu.Lock()
	if m.conn != nil {
		_ = m.conn.Close()
		m.conn = nil
	}
	m.connMu.Unlock()
	m.server.Close()
}

type llmRealtimeHarness struct {
	t          *testing.T
	server     *llmMockRealtimeServer
	service    *LLMService
	upstream   *llmFrameCollector
	downstream *llmFrameCollector
}

func newLLMRealtimeHarness(t *testing.T, cfg LLMConfig) *llmRealtimeHarness {
	t.Helper()

	server := newLLMMockRealtimeServer(t)
	cfg.Endpoint = server.URL()

	service := NewLLMService(cfg)
	up := newLLMFrameCollector("upstream")
	down := newLLMFrameCollector("downstream")
	service.Link(down)
	service.SetPrev(up)

	if err := service.HandleFrame(context.Background(), frames.NewStartFrame(), frames.Downstream); err != nil {
		t.Fatalf("start handling failed: %v", err)
	}

	if !server.waitConnected(2 * time.Second) {
		t.Fatal("service did not connect")
	}

	if _, ok := server.waitMessage(2*time.Second, hasTypeLLM("session.update")); !ok {
		t.Fatal("did not receive session.update")
	}
	deadline := time.Now().Add(2 * time.Second)
	for !service.socket.Ready() {
		if time.Now().After(deadline) {
			t.Fatal("socket never became ready")
		}
		time.Sleep(5 * time.Millisecond)
	}

	return &llmRealtimeHarness{t: t, server: server, service: service, upstream: up, downstream: down}
}

func (h *llmRealtimeHarness) close() {
	if err := h.service.Stop(); err != nil {
		h.t.Fatalf("stop failed: %v", err)
	}
	h.server.close()
}

func hasTypeLLM(eventType string) func(map[string]any) bool {
	return func(msg map[string]any) bool {
		rawType, ok := msg["type"].(string)
		return ok && rawType == eventType
	}
}

func mustMapLLM(t *testing.T, value any, name string) map[string]any {
	t.Helper()
	mapped, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("%s should be map[string]any", name)
	}
	return mapped
}

func isUUIDLikeLLM(value string) bool {
	return len(value) == 36 && value[8] == '-' && value[13] == '-' && value[18] == '-' && value[23] == '-'
}

func llmIsType[T frames.Frame](f frames.Frame) bool {
	_, ok := f.(T)
	return ok
}

func TestOpenAIRealtimeLLM_SessionSetup(t *testing.T) {
	server := newLLMMockRealtimeServer(t)
	defer server.close()
	svc := NewLLMService(LLMConfig{
		APIKey:            "rt-key",
		Model:             "gpt-test-realtime",
		Endpoint:          server.URL(),
		SystemInstruction: "Be brief.",
		Voice:             "alloy",
		Tools:             llm.Tools{Functions: []llm.FunctionSchema{{Name: "lookup", Description: "look", Properties: map[string]any{"q": map[string]any{"type": "string"}}}}},
		ToolChoice:        llm.ToolChoice{Mode: llm.ToolChoiceAuto},
	})
	if err := svc.HandleFrame(context.Background(), frames.NewStartFrame(), frames.Downstream); err != nil {
		t.Fatal(err)
	}
	defer svc.Stop()
	msg, ok := server.waitMessage(2*time.Second, hasTypeLLM("session.update"))
	if !ok {
		t.Fatal("no session.update")
	}
	server.connMu.Lock()
	model, auth, beta := server.modelQuery, server.authHeader, server.betaHeader
	server.connMu.Unlock()
	if model != "gpt-test-realtime" || auth != "Bearer rt-key" || beta != "realtime=v1" {
		t.Fatalf("model=%q auth=%q beta=%q", model, auth, beta)
	}
	session := mustMapLLM(t, msg["session"], "session")
	if session["instructions"] != "Be brief." || session["voice"] != "alloy" || session["input_audio_format"] != "pcm16" {
		t.Fatalf("session = %v", session)
	}
	if mustMapLLM(t, session["input_audio_transcription"], "transcription")["model"] != DefaultTranscriptionModel {
		t.Fatalf("transcription = %v", session["input_audio_transcription"])
	}
	tools, _ := session["tools"].([]any)
	if len(tools) != 1 || mustMapLLM(t, tools[0], "tool")["name"] != "lookup" || session["tool_choice"] != "auto" {
		t.Fatalf("tools = %v choice = %v", session["tools"], session["tool_choice"])
	}
}

func TestOpenAIRealtimeLLM_AudioRoundTrip(t *testing.T) {
	h := newLLMRealtimeHarness(t, LLMConfig{APIKey: "k"})
	defer h.close()

	in := make([]byte, 320)
	if err := h.service.HandleFrame(context.Background(), frames.NewAudioFrame(in, 16000, 1), frames.Downstream); err != nil {
		t.Fatal(err)
	}
	msg, ok := h.server.waitMessage(2*time.Second, hasTypeLLM("input_audio_buffer.append"))
	if !ok {
		t.Fatal("no audio append")
	}
	decoded, _ := base64.StdEncoding.DecodeString(msg["audio"].(string))
	if len(decoded) != 480 {
		t.Fatalf("16k audio must be resampled to 24k: got %d bytes", len(decoded))
	}

	out := []byte{1, 2, 3, 4}
	h.server.send(map[string]any{"type": "response.created"})
	h.server.send(map[string]any{"type": "response.audio.delta", "item_id": "item-1", "content_index": 0, "delta": base64.StdEncoding.EncodeToString(out)})
	h.server.send(map[string]any{"type": "response.audio_transcript.delta", "delta": "Hi there"})
	h.server.send(map[string]any{"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello bot"})
	h.server.send(map[string]any{"type": "response.done", "response": map[string]any{"status": "completed", "usage": map[string]any{"input_tokens": 5, "output_tokens": 6, "total_tokens": 11}}})

	if _, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.LLMFullResponseStartFrame]); !ok {
		t.Fatal("missing LLMFullResponseStartFrame")
	}
	started, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.TTSStartedFrame])
	if !ok || !isUUIDLikeLLM(started.(*frames.TTSStartedFrame).ContextID) {
		t.Fatal("missing TTSStartedFrame")
	}
	audio, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.TTSAudioFrame])
	if !ok || string(audio.(*frames.TTSAudioFrame).Data) != string(out) || audio.(*frames.TTSAudioFrame).SampleRate != DefaultOutputSampleRate {
		t.Fatal("bad TTSAudioFrame")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(f frames.Frame) bool {
		tf, ok := f.(*frames.TTSTextFrame)
		return ok && tf.Text == "Hi there"
	}); !ok {
		t.Fatal("missing TTSTextFrame")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.TTSStoppedFrame]); !ok {
		t.Fatal("missing TTSStoppedFrame")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.LLMFullResponseEndFrame]); !ok {
		t.Fatal("missing LLMFullResponseEndFrame")
	}
	tr, ok := h.upstream.waitForFrame(2*time.Second, llmIsType[*frames.TranscriptionFrame])
	if !ok || tr.(*frames.TranscriptionFrame).Text != "hello bot" {
		t.Fatal("missing user TranscriptionFrame upstream")
	}
}

func TestOpenAIRealtimeLLM_Interruption(t *testing.T) {
	h := newLLMRealtimeHarness(t, LLMConfig{APIKey: "k"})
	defer h.close()
	h.server.send(map[string]any{"type": "response.created"})
	h.server.send(map[string]any{"type": "response.audio.delta", "item_id": "item-9", "content_index": 0, "delta": base64.StdEncoding.EncodeToString(make([]byte, 4800))})
	if _, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.TTSAudioFrame]); !ok {
		t.Fatal("missing audio")
	}
	if err := h.service.HandleFrame(context.Background(), frames.NewInterruptionFrame(), frames.Downstream); err != nil {
		t.Fatal(err)
	}
	if _, ok := h.server.waitMessage(2*time.Second, hasTypeLLM("response.cancel")); !ok {
		t.Fatal("missing response.cancel")
	}
	tr, ok := h.server.waitMessage(2*time.Second, hasTypeLLM("conversation.item.truncate"))
	if !ok || tr["item_id"] != "item-9" {
		t.Fatalf("missing truncate: %v", tr)
	}
	if ms := tr["audio_end_ms"].(float64); ms < 0 || ms > 100 {
		t.Fatalf("audio_end_ms = %v", ms)
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.TTSStoppedFrame]); !ok {
		t.Fatal("missing TTSStoppedFrame")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.InterruptionFrame]); !ok {
		t.Fatal("InterruptionFrame must be forwarded")
	}
}

func TestOpenAIRealtimeLLM_FunctionCallResult(t *testing.T) {
	h := newLLMRealtimeHarness(t, LLMConfig{APIKey: "k"})
	defer h.close()
	h.service.RegisterFunction("lookup", func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		return llm.FunctionResult{Value: map[string]any{"answer": p.Arguments["q"]}}, nil
	})
	h.server.send(map[string]any{"type": "response.created"})
	h.server.send(map[string]any{"type": "response.function_call_arguments.done", "call_id": "call-7", "name": "lookup", "arguments": `{"q":"42"}`})
	h.server.send(map[string]any{"type": "response.done", "response": map[string]any{"status": "completed"}})

	item, ok := h.server.waitMessage(2*time.Second, func(m map[string]any) bool {
		if m["type"] != "conversation.item.create" {
			return false
		}
		it, _ := m["item"].(map[string]any)
		return it["type"] == "function_call_output"
	})
	if !ok {
		t.Fatal("function_call_output not sent")
	}
	it := mustMapLLM(t, item["item"], "item")
	if it["call_id"] != "call-7" || it["output"] != `{"answer":"42"}` {
		t.Fatalf("item = %v", it)
	}
	if _, ok := h.server.waitMessage(2*time.Second, hasTypeLLM("response.create")); !ok {
		t.Fatal("response.create not sent after tool result")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.FunctionCallInProgressFrame]); !ok {
		t.Fatal("missing FunctionCallInProgressFrame")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.FunctionCallResultFrame]); !ok {
		t.Fatal("missing FunctionCallResultFrame")
	}
}

func TestOpenAIRealtimeLLM_Reconnects(t *testing.T) {
	h := newLLMRealtimeHarness(t, LLMConfig{APIKey: "k", Backoff: realtime.Backoff{Initial: 10 * time.Millisecond, MaxAttempts: 3}})
	defer h.close()
	h.server.dropConnection()
	if _, ok := h.server.waitMessage(3*time.Second, hasTypeLLM("session.update")); !ok {
		t.Fatal("service did not reconnect")
	}
}
