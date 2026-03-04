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
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
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
	if err := conn.WriteJSON(payload); err != nil {
		m.t.Fatalf("mock websocket write failed: %v", err)
	}
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

	return &llmRealtimeHarness{t: t, server: server, service: service, upstream: up, downstream: down}
}

func (h *llmRealtimeHarness) close() {
	if err := h.service.Cleanup(); err != nil {
		h.t.Fatalf("cleanup failed: %v", err)
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

func TestOpenAIRealtimeLLM_SessionSetup(t *testing.T) {
	server := newLLMMockRealtimeServer(t)
	defer server.close()

	service := NewLLMService(LLMConfig{
		APIKey:            "rt-key",
		Model:             "gpt-4o-realtime-preview",
		SystemInstruction: "be concise",
		Voice:             "alloy",
		Tools: []services.Tool{{
			Type: "function",
			Function: services.ToolFunction{
				Name:        "lookup_weather",
				Description: "lookup weather",
				Parameters: map[string]any{
					"type": "object",
				},
			},
		}},
		ToolChoice: "auto",
		Endpoint:   server.URL(),
	})

	if err := service.Initialize(context.Background()); err != nil {
		t.Fatalf("initialize failed: %v", err)
	}
	defer func() {
		if err := service.Cleanup(); err != nil {
			t.Fatalf("cleanup failed: %v", err)
		}
	}()

	if !server.waitConnected(2 * time.Second) {
		t.Fatal("service did not connect")
	}

	msg, ok := server.waitMessage(2*time.Second, hasTypeLLM("session.update"))
	if !ok {
		t.Fatal("did not receive session.update")
	}

	if server.authHeader != "Bearer rt-key" {
		t.Fatalf("unexpected authorization header: %s", server.authHeader)
	}
	if server.betaHeader != "realtime=v1" {
		t.Fatalf("unexpected OpenAI-Beta header: %s", server.betaHeader)
	}
	if server.modelQuery != "gpt-4o-realtime-preview" {
		t.Fatalf("unexpected model query: %s", server.modelQuery)
	}

	session := mustMapLLM(t, msg["session"], "session")
	if session["instructions"] != "be concise" || session["voice"] != "alloy" {
		t.Fatalf("session instructions/voice mismatch: %v", session)
	}
	if session["model"] != "gpt-4o-realtime-preview" {
		t.Fatalf("session model mismatch: %v", session["model"])
	}
	if session["tool_choice"] != "auto" {
		t.Fatalf("session tool_choice mismatch: %v", session["tool_choice"])
	}

	toolsAny, ok := session["tools"].([]any)
	if !ok || len(toolsAny) != 1 {
		t.Fatalf("expected one tool in session, got: %#v", session["tools"])
	}
	tool := mustMapLLM(t, toolsAny[0], "tool")
	if tool["type"] != "function" || tool["name"] != "lookup_weather" {
		t.Fatalf("unexpected tool payload: %v", tool)
	}
}

func TestOpenAIRealtimeLLM_AudioRoundTrip(t *testing.T) {
	h := newLLMRealtimeHarness(t, LLMConfig{APIKey: "live-key", Model: "gpt-4o-realtime-preview", Voice: "alloy"})
	defer h.close()

	inputAudio := []byte{0x01, 0x02, 0x03, 0x04}
	audioFrame := frames.NewAudioFrame(inputAudio, 16000, 1)
	if err := h.service.HandleFrame(context.Background(), audioFrame, frames.Downstream); err != nil {
		t.Fatalf("audio handling failed: %v", err)
	}

	audioInputMsg, ok := h.server.waitMessage(2*time.Second, hasTypeLLM("input_audio_buffer.append"))
	if !ok {
		t.Fatal("did not receive input_audio_buffer.append")
	}
	encoded, ok := audioInputMsg["audio"].(string)
	if !ok {
		t.Fatal("audio field should be base64 string")
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil || string(decoded) != string(inputAudio) {
		t.Fatalf("unexpected encoded audio payload: err=%v decoded=%v", err, decoded)
	}

	assistantAudio := []byte{0x10, 0x20, 0x30}
	h.server.send(map[string]any{"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello openai"})
	h.server.send(map[string]any{"type": "response.text.delta", "delta": "Hello there."})
	h.server.send(map[string]any{"type": "response.audio_transcript.delta", "delta": "spoken hello"})
	h.server.send(map[string]any{"type": "response.audio.delta", "delta": base64.StdEncoding.EncodeToString(assistantAudio)})
	h.server.send(map[string]any{"type": "response.done"})

	transcriptionAny, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		transcription, match := frame.(*frames.TranscriptionFrame)
		return match && transcription.Text == "hello openai"
	})
	if !ok {
		t.Fatal("missing TranscriptionFrame")
	}
	if !transcriptionAny.(*frames.TranscriptionFrame).IsFinal {
		t.Fatal("expected final transcription")
	}

	if _, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		llm, match := frame.(*frames.LLMTextFrame)
		return match && llm.Text == "Hello there."
	}); !ok {
		t.Fatal("missing response.text.delta LLMTextFrame")
	}

	if _, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		llm, match := frame.(*frames.LLMTextFrame)
		return match && llm.Text == "spoken hello"
	}); !ok {
		t.Fatal("missing response.audio_transcript.delta LLMTextFrame")
	}

	startAny, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		_, match := frame.(*frames.TTSStartedFrame)
		return match
	})
	if !ok {
		t.Fatal("missing TTSStartedFrame")
	}
	start := startAny.(*frames.TTSStartedFrame)
	if !isUUIDLikeLLM(start.ContextID) || start.Metadata()["context_id"] != start.ContextID {
		t.Fatalf("invalid started context id: %s metadata=%v", start.ContextID, start.Metadata()["context_id"])
	}

	audioAny, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		audio, match := frame.(*frames.TTSAudioFrame)
		return match && audio.ContextID == start.ContextID
	})
	if !ok {
		t.Fatal("missing TTSAudioFrame")
	}
	audio := audioAny.(*frames.TTSAudioFrame)
	if string(audio.Data) != string(assistantAudio) {
		t.Fatalf("unexpected TTSAudioFrame data: %v", audio.Data)
	}
	if audio.Metadata()["context_id"] != start.ContextID {
		t.Fatalf("audio context metadata mismatch: %v", audio.Metadata()["context_id"])
	}

	if _, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		stop, match := frame.(*frames.TTSStoppedFrame)
		return match && stop.ContextID == start.ContextID
	}); !ok {
		t.Fatal("missing downstream TTSStoppedFrame")
	}
}

func TestOpenAIRealtimeLLM_Interruption(t *testing.T) {
	h := newLLMRealtimeHarness(t, LLMConfig{APIKey: "interrupt-key", Model: "gpt-4o-realtime-preview"})
	defer h.close()

	audio := []byte{0x42, 0x43, 0x44}
	h.server.send(map[string]any{"type": "response.audio.delta", "delta": base64.StdEncoding.EncodeToString(audio)})

	startAny, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		_, match := frame.(*frames.TTSStartedFrame)
		return match
	})
	if !ok {
		t.Fatal("missing initial TTSStartedFrame")
	}
	initialContext := startAny.(*frames.TTSStartedFrame).ContextID

	if err := h.service.HandleFrame(context.Background(), frames.NewInterruptionFrame(), frames.Downstream); err != nil {
		t.Fatalf("interruption handling failed: %v", err)
	}

	cancelMsg, ok := h.server.waitMessage(2*time.Second, hasTypeLLM("response.cancel"))
	if !ok {
		t.Fatal("missing response.cancel message")
	}
	if cancelMsg["type"] != "response.cancel" {
		t.Fatalf("unexpected cancel message: %v", cancelMsg)
	}

	if _, ok := h.upstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		stop, match := frame.(*frames.TTSStoppedFrame)
		return match && stop.ContextID == initialContext
	}); !ok {
		t.Fatal("missing upstream TTSStoppedFrame on interruption")
	}

	downInterruption, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		_, match := frame.(*frames.InterruptionFrame)
		return match
	})
	if !ok {
		t.Fatal("missing broadcast downstream InterruptionFrame")
	}

	upInterruption, ok := h.upstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		_, match := frame.(*frames.InterruptionFrame)
		return match
	})
	if !ok {
		t.Fatal("missing broadcast upstream InterruptionFrame")
	}

	down := downInterruption.(*frames.InterruptionFrame)
	up := upInterruption.(*frames.InterruptionFrame)
	if down.GetBroadcastSiblingID() == "" || up.GetBroadcastSiblingID() == "" {
		t.Fatalf("broadcast sibling ids should be set, down=%q up=%q", down.GetBroadcastSiblingID(), up.GetBroadcastSiblingID())
	}
}

func TestOpenAIRealtimeLLM_FunctionCall(t *testing.T) {
	h := newLLMRealtimeHarness(t, LLMConfig{APIKey: "func-key", Model: "gpt-4o-realtime-preview"})
	defer h.close()

	h.server.send(map[string]any{
		"type":    "response.function_call_arguments.delta",
		"call_id": "call-1",
		"name":    "lookup_weather",
		"delta":   "{\"city\":\"San",
	})
	h.server.send(map[string]any{
		"type":    "response.function_call_arguments.done",
		"call_id": "call-1",
		"name":    "lookup_weather",
		"delta":   " Francisco\"}",
	})

	frameAny, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		fcf, match := frame.(*frames.FunctionCallInProgressFrame)
		if !match {
			return false
		}
		city, _ := fcf.Arguments["city"].(string)
		return fcf.ToolCallID == "call-1" && fcf.FunctionName == "lookup_weather" && city == "San Francisco"
	})
	if !ok {
		t.Fatal("missing FunctionCallInProgressFrame with parsed arguments")
	}

	funcFrame := frameAny.(*frames.FunctionCallInProgressFrame)
	if !funcFrame.CancelOnInterruption {
		t.Fatal("function call frame should be cancelable on interruption")
	}
}
