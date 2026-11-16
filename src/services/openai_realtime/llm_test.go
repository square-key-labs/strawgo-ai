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

	path       string
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
		m.path = r.URL.Path
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
	if model != "gpt-test-realtime" || auth != "Bearer rt-key" || beta != "" {
		t.Fatalf("model=%q auth=%q beta=%q", model, auth, beta)
	}
	session := mustMapLLM(t, msg["session"], "session")
	if session["type"] != "realtime" || session["model"] != "gpt-test-realtime" || session["instructions"] != "Be brief." {
		t.Fatalf("session = %v", session)
	}
	audio := mustMapLLM(t, session["audio"], "audio")
	input := mustMapLLM(t, audio["input"], "input")
	if f := mustMapLLM(t, input["format"], "input format"); f["type"] != "audio/pcm" || f["rate"] != float64(24000) {
		t.Fatalf("input format = %v", f)
	}
	if mustMapLLM(t, input["transcription"], "transcription")["model"] != DefaultTranscriptionModel {
		t.Fatalf("transcription = %v", input["transcription"])
	}
	output := mustMapLLM(t, audio["output"], "output")
	if output["voice"] != "alloy" || mustMapLLM(t, output["format"], "output format")["type"] != "audio/pcm" {
		t.Fatalf("output = %v", output)
	}
	for _, legacy := range []string{"modalities", "voice", "input_audio_format", "output_audio_format", "input_audio_transcription", "turn_detection"} {
		if _, present := session[legacy]; present {
			t.Fatalf("beta field %q must not be sent: %v", legacy, session)
		}
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
	h.server.send(map[string]any{"type": "response.output_audio.delta", "item_id": "item-1", "content_index": 0, "delta": base64.StdEncoding.EncodeToString(out)})
	h.server.send(map[string]any{"type": "response.output_audio_transcript.delta", "delta": "Hi there"})
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
	h.server.send(map[string]any{"type": "response.output_audio.delta", "item_id": "item-9", "content_index": 0, "delta": base64.StdEncoding.EncodeToString(make([]byte, 4800))})
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

func TestOpenAIRealtimeLLM_ConnectURLAndHeaders(t *testing.T) {
	server := newLLMMockRealtimeServer(t)
	defer server.close()
	svc := NewLLMService(LLMConfig{APIKey: "ga-key", Endpoint: server.URL()})
	if err := svc.HandleFrame(context.Background(), frames.NewStartFrame(), frames.Downstream); err != nil {
		t.Fatal(err)
	}
	defer svc.Stop()
	if !server.waitConnected(2 * time.Second) {
		t.Fatal("service did not connect")
	}
	if _, ok := server.waitMessage(2*time.Second, hasTypeLLM("session.update")); !ok {
		t.Fatal("no session.update")
	}
	server.connMu.Lock()
	path, model, auth, beta := server.path, server.modelQuery, server.authHeader, server.betaHeader
	server.connMu.Unlock()
	if path != "/realtime" || model != "gpt-realtime-2.1" || auth != "Bearer ga-key" || beta != "" {
		t.Fatalf("path=%q model=%q auth=%q beta=%q", path, model, auth, beta)
	}
	if OpenAIRealtimeURL != "wss://api.openai.com/v1/realtime" || DefaultRealtimeModel != "gpt-realtime-2.1" {
		t.Fatalf("url=%q model=%q", OpenAIRealtimeURL, DefaultRealtimeModel)
	}
}

func TestOpenAIRealtimeLLM_SessionUpdateOptions(t *testing.T) {
	no := false
	cases := []struct {
		name  string
		cfg   LLMConfig
		check func(t *testing.T, session, input, output map[string]any)
	}{
		{
			name: "semantic vad, noise reduction, transcription, truncation, reasoning",
			cfg: LLMConfig{
				Model:                    "gpt-realtime-2.1",
				Speed:                    1.2,
				OutputAudioFormat:        "audio/pcmu",
				InputAudioFormat:         "g711_alaw",
				TurnDetection:            &TurnDetection{Type: TurnDetectionSemanticVAD, Eagerness: "low", InterruptResponse: &no},
				NoiseReduction:           "far_field",
				TranscriptionModel:       "gpt-transcribe",
				TranscriptionLanguage:    "de",
				TranscriptionPrompt:      "names",
				TruncationRetentionRatio: 0.8,
				ReasoningEffort:          "low",
			},
			check: func(t *testing.T, session, input, output map[string]any) {
				td := mustMapLLM(t, input["turn_detection"], "turn_detection")
				if td["type"] != "semantic_vad" || td["eagerness"] != "low" || td["interrupt_response"] != false {
					t.Fatalf("turn_detection = %v", td)
				}
				if _, present := td["create_response"]; present {
					t.Fatalf("unset create_response must be omitted: %v", td)
				}
				if mustMapLLM(t, input["noise_reduction"], "noise_reduction")["type"] != "far_field" {
					t.Fatalf("noise_reduction = %v", input["noise_reduction"])
				}
				tr := mustMapLLM(t, input["transcription"], "transcription")
				if tr["model"] != "gpt-transcribe" || tr["language"] != "de" || tr["prompt"] != "names" {
					t.Fatalf("transcription = %v", tr)
				}
				if mustMapLLM(t, input["format"], "input format")["type"] != "audio/pcma" {
					t.Fatalf("input format = %v", input["format"])
				}
				if mustMapLLM(t, output["format"], "output format")["type"] != "audio/pcmu" || output["speed"] != 1.2 {
					t.Fatalf("output = %v", output)
				}
				trunc := mustMapLLM(t, session["truncation"], "truncation")
				if trunc["type"] != "retention_ratio" || trunc["retention_ratio"] != 0.8 {
					t.Fatalf("truncation = %v", trunc)
				}
				if mustMapLLM(t, session["reasoning"], "reasoning")["effort"] != "low" {
					t.Fatalf("reasoning = %v", session["reasoning"])
				}
			},
		},
		{
			name: "server vad tuning, whisper drops prompt, reasoning stripped",
			cfg: LLMConfig{
				Model:               "gpt-realtime",
				TurnDetection:       &TurnDetection{Threshold: 0.6, SilenceDurationMs: 700},
				TranscriptionPrompt: "names",
				Truncation:          "disabled",
				ReasoningEffort:     "high",
			},
			check: func(t *testing.T, session, input, output map[string]any) {
				td := mustMapLLM(t, input["turn_detection"], "turn_detection")
				if td["type"] != "server_vad" || td["threshold"] != 0.6 || td["silence_duration_ms"] != float64(700) {
					t.Fatalf("turn_detection = %v", td)
				}
				if _, present := mustMapLLM(t, input["transcription"], "transcription")["prompt"]; present {
					t.Fatalf("gpt-realtime-whisper does not take a prompt: %v", input["transcription"])
				}
				if session["truncation"] != "disabled" {
					t.Fatalf("truncation = %v", session["truncation"])
				}
				if _, present := session["reasoning"]; present {
					t.Fatalf("reasoning must be stripped for non-reasoning model: %v", session)
				}
			},
		},
		{
			name: "server vad disabled",
			cfg:  LLMConfig{DisableServerVAD: true, TurnDetection: &TurnDetection{Type: TurnDetectionSemanticVAD}},
			check: func(t *testing.T, session, input, output map[string]any) {
				if td, present := input["turn_detection"]; !present || td != nil {
					t.Fatalf("disabled VAD must send null turn_detection, got %#v", input["turn_detection"])
				}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.cfg.APIKey = "k"
			h := newLLMRealtimeHarness(t, tc.cfg)
			defer h.close()
			h.service.HandleFrame(context.Background(), frames.NewLLMSetToolsFrame(llm.Tools{}), frames.Downstream)
			msg, ok := h.server.waitMessage(2*time.Second, hasTypeLLM("session.update"))
			if !ok {
				t.Fatal("no session.update")
			}
			session := mustMapLLM(t, msg["session"], "session")
			audio := mustMapLLM(t, session["audio"], "audio")
			tc.check(t, session, mustMapLLM(t, audio["input"], "input"), mustMapLLM(t, audio["output"], "output"))
		})
	}
}

func TestOpenAIRealtimeLLM_AudioAndTextResponseGAEvents(t *testing.T) {
	h := newLLMRealtimeHarness(t, LLMConfig{APIKey: "k"})
	defer h.close()

	h.service.HandleFrame(context.Background(), frames.NewLLMContextFrame(llm.NewContext(llm.User("hi"))), frames.Downstream)
	create, ok := h.server.waitMessage(2*time.Second, hasTypeLLM("response.create"))
	if !ok {
		t.Fatal("no response.create")
	}
	mods, _ := mustMapLLM(t, create["response"], "response")["output_modalities"].([]any)
	if len(mods) != 1 || mods[0] != "audio" {
		t.Fatalf("response.create = %v", create)
	}

	out := []byte{9, 8, 7, 6}
	h.server.send(map[string]any{"type": "response.created"})
	h.server.send(map[string]any{"type": "conversation.item.added", "item": map[string]any{"id": "item-2", "type": "message", "role": "assistant"}})
	h.server.send(map[string]any{"type": "response.audio.delta", "item_id": "old", "content_index": 0, "delta": base64.StdEncoding.EncodeToString([]byte{1, 1})})
	h.server.send(map[string]any{"type": "response.output_audio.delta", "item_id": "item-2", "content_index": 0, "delta": base64.StdEncoding.EncodeToString(out)})
	h.server.send(map[string]any{"type": "response.output_audio_transcript.delta", "item_id": "item-2", "delta": "spoken"})
	h.server.send(map[string]any{"type": "response.output_text.delta", "item_id": "item-2", "delta": "written"})
	h.server.send(map[string]any{"type": "conversation.item.done", "item": map[string]any{"id": "item-2", "type": "message", "role": "assistant"}})
	h.server.send(map[string]any{"type": "response.done", "response": map[string]any{"status": "completed"}})

	audio, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.TTSAudioFrame])
	if !ok || string(audio.(*frames.TTSAudioFrame).Data) != string(out) {
		t.Fatalf("first TTSAudioFrame must come from response.output_audio.delta, got %v", audio)
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(f frames.Frame) bool {
		tf, ok := f.(*frames.TTSTextFrame)
		return ok && tf.Text == "spoken"
	}); !ok {
		t.Fatal("missing TTSTextFrame from response.output_audio_transcript.delta")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(f frames.Frame) bool {
		tf, ok := f.(*frames.LLMTextFrame)
		return ok && tf.Text == "written"
	}); !ok {
		t.Fatal("missing LLMTextFrame from response.output_text.delta")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, llmIsType[*frames.LLMFullResponseEndFrame]); !ok {
		t.Fatal("missing LLMFullResponseEndFrame")
	}
}

func TestOpenAIRealtimeLLM_FunctionCallNameFromItemAdded(t *testing.T) {
	h := newLLMRealtimeHarness(t, LLMConfig{APIKey: "k"})
	defer h.close()
	called := make(chan string, 1)
	h.service.RegisterFunction("lookup", func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		called <- p.FunctionName
		return llm.FunctionResult{Value: "ok"}, nil
	})
	h.server.send(map[string]any{"type": "response.created"})
	h.server.send(map[string]any{"type": "conversation.item.added", "item": map[string]any{"id": "fc-1", "type": "function_call", "call_id": "call-9", "name": "lookup"}})
	h.server.send(map[string]any{"type": "response.function_call_arguments.done", "call_id": "call-9", "arguments": "{}"})
	select {
	case name := <-called:
		if name != "lookup" {
			t.Fatalf("name = %q", name)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("function not called")
	}
}
