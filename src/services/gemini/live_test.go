package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
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

type frameCollector struct {
	name string
	ch   chan frames.Frame
}

func newFrameCollector(name string) *frameCollector {
	return &frameCollector{name: name, ch: make(chan frames.Frame, 256)}
}

func (c *frameCollector) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return c.QueueFrame(frame, direction)
}
func (c *frameCollector) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	select {
	case c.ch <- frame:
	default:
	}
	return nil
}
func (c *frameCollector) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return c.QueueFrame(frame, direction)
}
func (c *frameCollector) Link(next processors.FrameProcessor)    {}
func (c *frameCollector) SetPrev(prev processors.FrameProcessor) {}
func (c *frameCollector) Start(ctx context.Context) error        { return nil }
func (c *frameCollector) Stop() error                            { return nil }
func (c *frameCollector) Name() string                           { return c.name }

func (c *frameCollector) waitForFrame(timeout time.Duration, predicate func(frames.Frame) bool) (frames.Frame, bool) {
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

func (c *frameCollector) expectNoFrame(timeout time.Duration, predicate func(frames.Frame) bool) bool {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		select {
		case frame := <-c.ch:
			if predicate(frame) {
				return false
			}
		case <-timer.C:
			return true
		}
	}
}

type mockLiveServer struct {
	t *testing.T

	server *httptest.Server
	connMu sync.Mutex
	conn   *websocket.Conn

	msgs          chan map[string]any
	connected     chan struct{}
	connectedOnce sync.Once

	apiKeyMu    sync.Mutex
	apiKey      string
	rawQuery    string
	path        string
	connections int
}

func newMockLiveServer(t *testing.T) *mockLiveServer {
	t.Helper()
	m := &mockLiveServer{t: t, msgs: make(chan map[string]any, 256), connected: make(chan struct{})}
	upgrader := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
	mux := http.NewServeMux()
	mux.HandleFunc("/live", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade failed: %v", err)
			return
		}
		m.connMu.Lock()
		m.conn = conn
		m.connMu.Unlock()
		m.apiKeyMu.Lock()
		m.apiKey = r.Header.Get("x-goog-api-key")
		m.rawQuery = r.URL.RawQuery
		m.path = r.URL.Path
		m.connections++
		m.apiKeyMu.Unlock()
		m.connectedOnce.Do(func() { close(m.connected) })
		go m.readLoop(conn)
	})
	m.server = httptest.NewServer(mux)
	return m
}

func (m *mockLiveServer) readLoop(conn *websocket.Conn) {
	for {
		var payload map[string]any
		if err := conn.ReadJSON(&payload); err != nil {
			return
		}
		if _, ok := payload["setup"]; ok {
			m.connMu.Lock()
			_ = conn.WriteJSON(map[string]any{"setupComplete": map[string]any{}})
			m.connMu.Unlock()
		}
		select {
		case m.msgs <- payload:
		default:
		}
	}
}

func (m *mockLiveServer) URL() string {
	return "ws" + strings.TrimPrefix(m.server.URL, "http") + "/live"
}

func (m *mockLiveServer) waitConnected(timeout time.Duration) bool {
	select {
	case <-m.connected:
		return true
	case <-time.After(timeout):
		return false
	}
}

func (m *mockLiveServer) waitMessage(timeout time.Duration, predicate func(map[string]any) bool) (map[string]any, bool) {
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

func (m *mockLiveServer) send(payload map[string]any) {
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

func (m *mockLiveServer) dropConnection() {
	m.connMu.Lock()
	if m.conn != nil {
		_ = m.conn.Close()
	}
	m.connMu.Unlock()
}

func (m *mockLiveServer) connectionCount() int {
	m.apiKeyMu.Lock()
	defer m.apiKeyMu.Unlock()
	return m.connections
}

func (m *mockLiveServer) lastAPIKey() string {
	m.apiKeyMu.Lock()
	defer m.apiKeyMu.Unlock()
	return m.apiKey
}

func (m *mockLiveServer) close() {
	m.connMu.Lock()
	if m.conn != nil {
		_ = m.conn.Close()
		m.conn = nil
	}
	m.connMu.Unlock()
	m.server.Close()
}

type liveHarness struct {
	t          *testing.T
	server     *mockLiveServer
	service    *LiveService
	upstream   *frameCollector
	downstream *frameCollector
}

func newLiveHarness(t *testing.T, cfg LiveConfig) *liveHarness {
	t.Helper()
	server := newMockLiveServer(t)
	cfg.Endpoint = server.URL()
	service := NewLiveService(cfg)
	up := newFrameCollector("upstream")
	down := newFrameCollector("downstream")
	service.Link(down)
	service.SetPrev(up)
	if err := service.HandleFrame(context.Background(), frames.NewStartFrame(), frames.Downstream); err != nil {
		t.Fatalf("start handling failed: %v", err)
	}
	if !server.waitConnected(2 * time.Second) {
		t.Fatal("service did not connect")
	}
	if _, ok := server.waitMessage(2*time.Second, hasKey("setup")); !ok {
		t.Fatal("did not receive setup payload")
	}
	deadline := time.Now().Add(2 * time.Second)
	for !service.socket.Ready() {
		if time.Now().After(deadline) {
			t.Fatal("socket never became ready")
		}
		time.Sleep(5 * time.Millisecond)
	}
	return &liveHarness{t: t, server: server, service: service, upstream: up, downstream: down}
}

func (h *liveHarness) close() {
	if err := h.service.Stop(); err != nil {
		h.t.Fatalf("stop failed: %v", err)
	}
	h.server.close()
}

func hasKey(key string) func(map[string]any) bool {
	return func(msg map[string]any) bool {
		_, ok := msg[key]
		return ok
	}
}

func mustMap(t *testing.T, value any, name string) map[string]any {
	t.Helper()
	mapped, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("%s should be map[string]any", name)
	}
	return mapped
}

func mustSlice(t *testing.T, value any, name string) []any {
	t.Helper()
	switch v := value.(type) {
	case []any:
		return v
	case []string:
		result := make([]any, len(v))
		for i, s := range v {
			result[i] = s
		}
		return result
	default:
		t.Fatalf("%s should be []any or []string, got %T", name, value)
		return nil
	}
}

func isUUIDLike(value string) bool {
	return len(value) == 36 && value[8] == '-' && value[13] == '-' && value[18] == '-' && value[23] == '-'
}

func isType[T frames.Frame](frame frames.Frame) bool {
	_, ok := frame.(T)
	return ok
}

func TestNewGeminiLiveServiceAndConfig(t *testing.T) {
	service := NewGeminiLiveService("test-key", "")
	if service.cfg.APIKey != "test-key" || service.cfg.Model != DefaultLiveModel {
		t.Fatalf("unexpected constructor defaults: %+v", service.cfg)
	}
	if service.cfg.Endpoint != GeminiLiveURL || service.cfg.OutputSampleRate != DefaultLiveOutputSampleRate || service.cfg.OutputChannels != DefaultLiveOutputChannels {
		t.Fatalf("unexpected default output config: %+v", service.cfg)
	}
}

func TestLiveServiceSetupMessage(t *testing.T) {
	service := NewLiveService(LiveConfig{
		Model:             "gemini-3.8-live",
		SystemInstruction: "Be concise.",
		Voice:             "Puck",
		DisableVAD:        true,
		Tools:             llm.Tools{Functions: []llm.FunctionSchema{{Name: "get_weather", Description: "weather", Properties: map[string]any{"city": map[string]any{"type": "string"}}, Required: []string{"city"}}}},
	})
	raw, _ := json.Marshal(service.setupMessage())
	var msg map[string]any
	_ = json.Unmarshal(raw, &msg)
	setup := mustMap(t, msg["setup"], "setup")
	if setup["model"] != "models/gemini-3.8-live" {
		t.Fatalf("unexpected setup model: %v", setup["model"])
	}
	generation := mustMap(t, setup["generationConfig"], "generationConfig")
	if mods := mustSlice(t, generation["responseModalities"], "responseModalities"); len(mods) != 1 || mods[0] != "AUDIO" {
		t.Fatalf("modalities = %v", mods)
	}
	parts := mustSlice(t, mustMap(t, setup["systemInstruction"], "systemInstruction")["parts"], "parts")
	if mustMap(t, parts[0], "part")["text"] != "Be concise." {
		t.Fatalf("system instruction = %v", parts)
	}
	voice := mustMap(t, mustMap(t, mustMap(t, generation["speechConfig"], "speechConfig")["voiceConfig"], "voiceConfig")["prebuiltVoiceConfig"], "prebuilt")
	if voice["voiceName"] != "Puck" {
		t.Fatalf("voice = %v", voice)
	}
	vad := mustMap(t, mustMap(t, setup["realtimeInputConfig"], "realtimeInputConfig")["automaticActivityDetection"], "automaticActivityDetection")
	if vad["disabled"] != true {
		t.Fatalf("vad = %v", vad)
	}
	decls := mustSlice(t, mustMap(t, mustSlice(t, setup["tools"], "tools")[0], "tool")["functionDeclarations"], "decls")
	if decl := mustMap(t, decls[0], "decl"); decl["name"] != "get_weather" || decl["behavior"] != "BLOCKING" {
		t.Fatalf("decls = %v", decls)
	}
	if history := mustMap(t, setup["historyConfig"], "historyConfig"); history["initialHistoryInClientContent"] != true {
		t.Fatalf("historyConfig = %v", history)
	}
	if _, present := setup["contextWindowCompression"]; present {
		t.Fatalf("contextWindowCompression must be off by default: %v", setup)
	}
}

func TestLiveServiceSetupContextWindowCompressionAndToolBehavior(t *testing.T) {
	tools := llm.Tools{Functions: []llm.FunctionSchema{{Name: "lookup", Description: "look"}}}
	setupFor := func(cfg LiveConfig) map[string]any {
		cfg.Tools = tools
		raw, _ := json.Marshal(NewLiveService(cfg).setupMessage())
		var msg map[string]any
		_ = json.Unmarshal(raw, &msg)
		return mustMap(t, msg["setup"], "setup")
	}
	behavior := func(setup map[string]any) any {
		decls := mustSlice(t, mustMap(t, mustSlice(t, setup["tools"], "tools")[0], "tool")["functionDeclarations"], "decls")
		return mustMap(t, decls[0], "decl")["behavior"]
	}

	setup := setupFor(LiveConfig{ContextWindowCompression: true, CompressionTriggerTokens: 25000})
	if setup["model"] != "models/"+DefaultLiveModel {
		t.Fatalf("model = %v", setup["model"])
	}
	cwc := mustMap(t, setup["contextWindowCompression"], "contextWindowCompression")
	if _, ok := cwc["slidingWindow"].(map[string]any); !ok || cwc["triggerTokens"] != float64(25000) {
		t.Fatalf("contextWindowCompression = %v", cwc)
	}
	if behavior(setup) != "BLOCKING" {
		t.Fatalf("3.8 Live must declare BLOCKING: %v", setup["tools"])
	}

	cwc = mustMap(t, setupFor(LiveConfig{ContextWindowCompression: true})["contextWindowCompression"], "contextWindowCompression")
	if _, present := cwc["triggerTokens"]; present {
		t.Fatalf("triggerTokens must be omitted when unset: %v", cwc)
	}
	for _, model := range []string{"gemini-2.5-flash-native-audio-preview-09-2025", "gemini-3.1-flash-live", "gemini-3.8-live-extended-thinking"} {
		if b := behavior(setupFor(LiveConfig{Model: model})); b != nil {
			t.Fatalf("%s must not declare behavior, got %v", model, b)
		}
	}
}

func TestLiveServiceConnectURLAndHeaders(t *testing.T) {
	if GeminiLiveURL != "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent" {
		t.Fatalf("GeminiLiveURL = %q", GeminiLiveURL)
	}
	h := newLiveHarness(t, LiveConfig{APIKey: "url-key"})
	defer h.close()
	h.server.apiKeyMu.Lock()
	path, query, key := h.server.path, h.server.rawQuery, h.server.apiKey
	h.server.apiKeyMu.Unlock()
	if path != "/live" || query != "" || key != "url-key" {
		t.Fatalf("path=%q query=%q key=%q", path, query, key)
	}
}

func TestLiveServiceGemini3ContextTriggersInference(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "g3-key"})
	defer h.close()
	if err := h.service.HandleFrame(context.Background(), frames.NewLLMContextFrame(llm.NewContext(llm.User("hello"))), frames.Downstream); err != nil {
		t.Fatal(err)
	}
	msg, ok := h.server.waitMessage(2*time.Second, hasKey("clientContent"))
	if !ok {
		t.Fatal("no clientContent")
	}
	if mustMap(t, msg["clientContent"], "clientContent")["turnComplete"] != true {
		t.Fatalf("clientContent = %v", msg)
	}
	msg, ok = h.server.waitMessage(2*time.Second, hasKey("realtimeInput"))
	if !ok || mustMap(t, msg["realtimeInput"], "realtimeInput")["text"] != " " {
		t.Fatalf("Gemini 3 must follow turnComplete with a realtime text input, got %v", msg)
	}

	h.server.send(map[string]any{"serverContent": map[string]any{"modelTurn": map[string]any{"parts": []map[string]any{
		{"text": "written reply"},
		{"inlineData": map[string]any{"mimeType": "audio/pcm;rate=24000", "data": base64.StdEncoding.EncodeToString([]byte{7, 7})}},
	}}}})
	h.server.send(map[string]any{"serverContent": map[string]any{"outputTranscription": map[string]any{"text": "spoken reply"}}})
	h.server.send(map[string]any{"serverContent": map[string]any{"turnComplete": true}})
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(f frames.Frame) bool {
		tf, ok := f.(*frames.LLMTextFrame)
		return ok && tf.Text == "written reply"
	}); !ok {
		t.Fatal("missing LLMTextFrame for model text part")
	}
	if a, ok := h.downstream.waitForFrame(2*time.Second, isType[*frames.TTSAudioFrame]); !ok || string(a.(*frames.TTSAudioFrame).Data) != string([]byte{7, 7}) {
		t.Fatal("missing TTSAudioFrame")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(f frames.Frame) bool {
		tf, ok := f.(*frames.TTSTextFrame)
		return ok && tf.Text == "spoken reply"
	}); !ok {
		t.Fatal("missing TTSTextFrame for output transcription")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, isType[*frames.LLMFullResponseEndFrame]); !ok {
		t.Fatal("missing LLMFullResponseEndFrame")
	}
}

func TestLiveServiceGemini25ContextHasNoTextNudge(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "g25-key", Model: "gemini-2.5-flash-native-audio-preview-09-2025"})
	defer h.close()
	if err := h.service.HandleFrame(context.Background(), frames.NewLLMContextFrame(llm.NewContext(llm.User("hello"))), frames.Downstream); err != nil {
		t.Fatal(err)
	}
	if _, ok := h.server.waitMessage(2*time.Second, hasKey("clientContent")); !ok {
		t.Fatal("no clientContent")
	}
	if msg, ok := h.server.waitMessage(300*time.Millisecond, hasKey("realtimeInput")); ok {
		t.Fatalf("Gemini 2.5 must not get a realtime text nudge: %v", msg)
	}
}

func TestLiveServiceAPIKeyInHeaderAndStop(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "init-key"})
	if h.server.lastAPIKey() != "init-key" {
		t.Fatalf("expected api key header, got %q", h.server.lastAPIKey())
	}
	if strings.Contains(h.server.rawQuery, "init-key") {
		t.Fatalf("api key leaked into URL: %q", h.server.rawQuery)
	}
	if err := h.service.HandleFrame(context.Background(), frames.NewEndFrame(), frames.Downstream); err != nil {
		t.Fatalf("EndFrame handling failed: %v", err)
	}
	if h.service.socket.Ready() || h.service.socket.Running() {
		t.Fatal("socket should be stopped after EndFrame")
	}
	if _, ok := h.downstream.waitForFrame(time.Second, isType[*frames.EndFrame]); !ok {
		t.Fatal("EndFrame must be forwarded")
	}
	h.close()
}

func TestLiveServiceBidirectionalAudioAndFrameEmission(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "live-key", SystemInstruction: "Be concise.", Voice: "Puck"})
	defer h.close()

	inputAudio := []byte{0x01, 0x02, 0x03, 0x04}
	audioFrame := frames.NewAudioFrame(inputAudio, 8000, 1)
	audioFrame.SetMetadata("codec", "mulaw")
	if err := h.service.HandleFrame(context.Background(), audioFrame, frames.Downstream); err != nil {
		t.Fatalf("audio handling failed: %v", err)
	}
	msg, ok := h.server.waitMessage(2*time.Second, hasKey("realtimeInput"))
	if !ok {
		t.Fatal("did not receive realtimeInput message")
	}
	chunk := mustMap(t, mustMap(t, msg["realtimeInput"], "realtimeInput")["audio"], "audio")
	if chunk["mimeType"] != "audio/pcm;encoding=mulaw;rate=8000" {
		t.Fatalf("unexpected input mimeType: %v", chunk["mimeType"])
	}
	decoded, err := base64.StdEncoding.DecodeString(chunk["data"].(string))
	if err != nil || string(decoded) != string(inputAudio) {
		t.Fatalf("bad audio payload err=%v bytes=%v", err, decoded)
	}

	assistantAudio := []byte{0x10, 0x20, 0x30}
	h.server.send(map[string]any{"serverContent": map[string]any{"inputTranscription": map[string]any{"text": "hello "}}})
	h.server.send(map[string]any{"serverContent": map[string]any{"inputTranscription": map[string]any{"text": "gemini"}}})
	h.server.send(map[string]any{"serverContent": map[string]any{
		"outputTranscription": map[string]any{"text": "assistant transcript"},
		"modelTurn": map[string]any{"parts": []map[string]any{
			{"inlineData": map[string]any{"mimeType": "audio/pcm;rate=24000", "data": base64.StdEncoding.EncodeToString(assistantAudio)}},
		}},
	}})
	h.server.send(map[string]any{"usageMetadata": map[string]any{"promptTokenCount": 3, "responseTokenCount": 5, "totalTokenCount": 8}})
	h.server.send(map[string]any{"serverContent": map[string]any{"turnComplete": true}})

	tr, ok := h.upstream.waitForFrame(2*time.Second, isType[*frames.TranscriptionFrame])
	if !ok || tr.(*frames.TranscriptionFrame).Text != "hello gemini" || !tr.(*frames.TranscriptionFrame).IsFinal {
		t.Fatalf("missing aggregated user transcription upstream: %v", tr)
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, isType[*frames.LLMFullResponseStartFrame]); !ok {
		t.Fatal("missing LLMFullResponseStartFrame")
	}
	startAny, ok := h.downstream.waitForFrame(2*time.Second, isType[*frames.TTSStartedFrame])
	if !ok {
		t.Fatal("missing TTSStartedFrame")
	}
	start := startAny.(*frames.TTSStartedFrame)
	if !isUUIDLike(start.ContextID) {
		t.Fatalf("invalid context id %q", start.ContextID)
	}
	audioAny, ok := h.downstream.waitForFrame(2*time.Second, isType[*frames.TTSAudioFrame])
	if !ok {
		t.Fatal("missing TTSAudioFrame")
	}
	audio := audioAny.(*frames.TTSAudioFrame)
	if audio.ContextID != start.ContextID || string(audio.Data) != string(assistantAudio) || audio.SampleRate != 24000 || audio.Metadata()["codec"] != "linear16" {
		t.Fatalf("unexpected TTSAudioFrame %+v", audio)
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(f frames.Frame) bool {
		tf, ok := f.(*frames.TTSTextFrame)
		return ok && tf.Text == "assistant transcript"
	}); !ok {
		t.Fatal("missing TTSTextFrame for bot transcript")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(f frames.Frame) bool {
		stop, ok := f.(*frames.TTSStoppedFrame)
		return ok && stop.ContextID == start.ContextID
	}); !ok {
		t.Fatal("missing TTSStoppedFrame")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, isType[*frames.LLMFullResponseEndFrame]); !ok {
		t.Fatal("missing LLMFullResponseEndFrame")
	}
}

func TestLiveServiceInterruptionSuppressesStaleAudio(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "interrupt-key"})
	defer h.close()

	sendAudio := func(b []byte) {
		h.server.send(map[string]any{"serverContent": map[string]any{"modelTurn": map[string]any{"parts": []map[string]any{{
			"inlineData": map[string]any{"mimeType": "audio/pcm;rate=24000", "data": base64.StdEncoding.EncodeToString(b)},
		}}}}})
	}
	sendAudio([]byte{0x55, 0x66})
	startAny, ok := h.downstream.waitForFrame(2*time.Second, isType[*frames.TTSStartedFrame])
	if !ok {
		t.Fatal("missing initial TTSStartedFrame")
	}
	initial := startAny.(*frames.TTSStartedFrame).ContextID
	if _, ok := h.downstream.waitForFrame(2*time.Second, isType[*frames.TTSAudioFrame]); !ok {
		t.Fatal("missing initial audio")
	}

	if err := h.service.HandleFrame(context.Background(), frames.NewInterruptionFrame(), frames.Downstream); err != nil {
		t.Fatalf("interruption failed: %v", err)
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(f frames.Frame) bool {
		stop, ok := f.(*frames.TTSStoppedFrame)
		return ok && stop.ContextID == initial
	}); !ok {
		t.Fatal("missing TTSStoppedFrame after interruption")
	}

	sendAudio([]byte{0x01, 0x02})
	if !h.downstream.expectNoFrame(300*time.Millisecond, isType[*frames.TTSAudioFrame]) {
		t.Fatal("stale audio should be suppressed")
	}

	if err := h.service.HandleFrame(context.Background(), frames.NewUserStoppedSpeakingFrame(), frames.Downstream); err != nil {
		t.Fatalf("user stopped failed: %v", err)
	}
	resumed := []byte{0x90, 0x91}
	sendAudio(resumed)
	newStart, ok := h.downstream.waitForFrame(2*time.Second, func(f frames.Frame) bool {
		s, ok := f.(*frames.TTSStartedFrame)
		return ok && s.ContextID != initial
	})
	if !ok {
		t.Fatal("missing resumed TTSStartedFrame")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(f frames.Frame) bool {
		a, ok := f.(*frames.TTSAudioFrame)
		return ok && a.ContextID == newStart.(*frames.TTSStartedFrame).ContextID && string(a.Data) == string(resumed)
	}); !ok {
		t.Fatal("missing resumed audio")
	}
}

func TestLiveServiceToolCallRoundTrip(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "tool-key"})
	defer h.close()
	h.service.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		return llm.FunctionResult{Value: map[string]any{"temp": 21, "city": p.Arguments["city"]}}, nil
	})
	h.server.send(map[string]any{"toolCall": map[string]any{"functionCalls": []map[string]any{{"id": "call-1", "name": "get_weather", "args": map[string]any{"city": "Pune"}}}}})

	msg, ok := h.server.waitMessage(2*time.Second, hasKey("toolResponse"))
	if !ok {
		t.Fatal("no toolResponse sent")
	}
	resp := mustMap(t, mustSlice(t, mustMap(t, msg["toolResponse"], "toolResponse")["functionResponses"], "functionResponses")[0], "fr")
	if resp["id"] != "call-1" || resp["name"] != "get_weather" {
		t.Fatalf("bad function response %v", resp)
	}
	if body := mustMap(t, resp["response"], "response"); body["city"] != "Pune" {
		t.Fatalf("bad response body %v", body)
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, isType[*frames.FunctionCallResultFrame]); !ok {
		t.Fatal("missing FunctionCallResultFrame downstream")
	}
}

func TestLiveServiceReconnects(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "rc-key", Backoff: realtime.Backoff{Initial: 10 * time.Millisecond, MaxAttempts: 3}})
	defer h.close()
	h.server.dropConnection()
	if _, ok := h.server.waitMessage(3*time.Second, hasKey("setup")); !ok {
		t.Fatal("service did not reconnect")
	}
	if h.server.connectionCount() < 2 {
		t.Fatalf("connections = %d", h.server.connectionCount())
	}
}

func TestLiveServicePassthroughNonAudioFrames(t *testing.T) {
	service := NewLiveService(LiveConfig{APIKey: "test-key"})
	downstream := newFrameCollector("downstream")
	service.Link(downstream)
	textFrame := frames.NewTextFrame("passthrough")
	if err := service.HandleFrame(context.Background(), textFrame, frames.Downstream); err != nil {
		t.Fatalf("passthrough failed: %v", err)
	}
	if _, ok := downstream.waitForFrame(time.Second, func(frame frames.Frame) bool { return frame == textFrame }); !ok {
		t.Fatal("expected passthrough")
	}
	if service.socket.Running() {
		t.Fatal("non-audio frame should not start the websocket")
	}
}

func TestLiveServiceAPIErrorsEmitErrorFrameUpstream(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "error-key"})
	defer h.close()
	h.server.send(map[string]any{"error": map[string]any{"message": "quota exceeded"}})
	errorAny, ok := h.upstream.waitForFrame(2*time.Second, isType[*frames.ErrorFrame])
	if !ok {
		t.Fatal("did not receive ErrorFrame upstream")
	}
	if !strings.Contains(errorAny.(*frames.ErrorFrame).Error.Error(), "Gemini Live API error: quota exceeded") {
		t.Fatalf("unexpected error: %v", errorAny.(*frames.ErrorFrame).Error)
	}
}

func TestParseOutputAudioMIMEType(t *testing.T) {
	codec, sampleRate := parseOutputAudioMIMEType("audio/pcm;encoding=mulaw;rate=8000")
	if codec != "mulaw" || sampleRate != 8000 {
		t.Fatalf("unexpected parse result: codec=%s sampleRate=%d", codec, sampleRate)
	}
}

func newGeminiSSEServer(t *testing.T, chunks ...string) (*httptest.Server, chan *http.Request, chan map[string]any) {
	t.Helper()
	reqs := make(chan *http.Request, 4)
	bodies := make(chan map[string]any, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		reqs <- r
		bodies <- body
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\r\n\r\n", c)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, reqs, bodies
}

func streamAll(t *testing.T, svcProvider *Provider, snap llm.Snapshot, settings llm.Settings) ([]llm.Event, error) {
	t.Helper()
	var events []llm.Event
	err := svcProvider.Stream(context.Background(), snap, settings, func(e llm.Event) { events = append(events, e) })
	return events, err
}

func TestGeminiProviderSystemInstructionEveryTurnAndHeaderKey(t *testing.T) {
	srv, reqs, bodies := newGeminiSSEServer(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":9,"thoughtsTokenCount":3,"cachedContentTokenCount":1}}`,
	)
	svc := NewLLMService(LLMConfig{APIKey: "gem-key", Model: "gemini-2.5-flash", BaseURL: srv.URL, SystemPrompt: "be brief"})
	p := svc.Provider().(*Provider)
	conv := llm.NewContext(llm.User("hi"), llm.Assistant("hello"), llm.User("again"))
	events, err := streamAll(t, p, conv.Snapshot(), svc.Settings())
	if err != nil {
		t.Fatal(err)
	}
	r := <-reqs
	if r.Header.Get("x-goog-api-key") != "gem-key" || strings.Contains(r.URL.RawQuery, "gem-key") {
		t.Fatalf("api key must be header-only: %q", r.URL.String())
	}
	if !strings.Contains(r.URL.Path, "models/gemini-2.5-flash:streamGenerateContent") || r.URL.Query().Get("alt") != "sse" {
		t.Fatalf("bad url %q", r.URL.String())
	}
	body := <-bodies
	si := mustMap(t, body["systemInstruction"], "systemInstruction")
	if mustMap(t, mustSlice(t, si["parts"], "parts")[0], "p")["text"] != "be brief" {
		t.Fatalf("system instruction missing on multi-turn request: %v", si)
	}
	contents := mustSlice(t, body["contents"], "contents")
	if len(contents) != 3 || mustMap(t, contents[1], "c1")["role"] != "model" {
		t.Fatalf("contents = %v", contents)
	}
	text := ""
	var usage llm.Usage
	for _, e := range events {
		switch ev := e.(type) {
		case llm.TextDelta:
			text += ev.Text
		case llm.UsageEvent:
			usage = ev.Usage
		}
	}
	if text != "Hello" || usage.PromptTokens != 4 || usage.CompletionTokens != 2 || usage.ReasoningTokens != 3 || usage.CacheReadTokens != 1 {
		t.Fatalf("text=%q usage=%+v", text, usage)
	}
}

func TestGeminiProviderFunctionCallRoundTripWithSignature(t *testing.T) {
	srv, _, bodies := newGeminiSSEServer(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Pune"}},"thoughtSignature":"c2lnLTE="},{"functionCall":{"name":"get_time","args":{}}}]},"finishReason":"STOP"}]}`,
	)
	p := NewProvider(ProviderConfig{Model: "m", URL: func(string) string { return srv.URL }})
	c := llm.NewContext(llm.User("weather?"))
	c.SetTools(llm.Tools{Functions: []llm.FunctionSchema{{Name: "get_weather", Properties: map[string]any{"city": map[string]any{"type": "string"}}}, {Name: "get_time"}}})
	c.SetToolChoice(llm.ToolChoice{Mode: llm.ToolChoiceRequired})
	events, err := streamAll(t, p, c.Snapshot(), llm.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	body := <-bodies
	if mustMap(t, mustMap(t, body["toolConfig"], "toolConfig")["functionCallingConfig"], "fcc")["mode"] != "ANY" {
		t.Fatalf("toolConfig = %v", body["toolConfig"])
	}
	var calls []llm.ToolCall
	for _, e := range events {
		if tc, ok := e.(llm.ToolCall); ok {
			calls = append(calls, tc)
		}
	}
	if len(calls) != 2 || calls[0].Name != "get_weather" || string(calls[0].Arguments) != `{"city":"Pune"}` || calls[0].ID == "" {
		t.Fatalf("calls = %+v", calls)
	}

	next := llm.NewContext(
		llm.User("weather?"),
		llm.ToolCalls(llm.ToolCallPart{ID: calls[0].ID, Name: calls[0].Name, Arguments: calls[0].Arguments}, llm.ToolCallPart{ID: calls[1].ID, Name: calls[1].Name, Arguments: calls[1].Arguments}),
		llm.ToolResult(calls[0].ID, "", `{"temp":21}`),
		llm.ToolResult(calls[1].ID, "get_time", `"noon"`),
	)
	_, contents := p.ConvertMessages(next.Snapshot().Messages, "")
	if len(contents) != 3 {
		t.Fatalf("contents = %+v", contents)
	}
	model := contents[1]
	if model.Parts[0].ThoughtSignature != "c2lnLTE=" || model.Parts[1].ThoughtSignature != "" {
		t.Fatalf("signature not round-tripped: %+v", model.Parts)
	}
	if model.Parts[0].FunctionCall.ID != "" {
		t.Fatalf("generated ids must not be sent to the API: %+v", model.Parts[0].FunctionCall)
	}
	responses := contents[2]
	if responses.Role != "user" || len(responses.Parts) != 2 {
		t.Fatalf("function responses must be grouped in one user turn: %+v", responses)
	}
	if responses.Parts[0].FunctionResponse.Name != "get_weather" || responses.Parts[0].FunctionResponse.Response["temp"] != float64(21) {
		t.Fatalf("response 0 = %+v", responses.Parts[0].FunctionResponse)
	}
	if responses.Parts[1].FunctionResponse.Response["value"] != "noon" {
		t.Fatalf("response 1 = %+v", responses.Parts[1].FunctionResponse)
	}

	unsigned := llm.NewContext(llm.User("x"), llm.ToolCalls(llm.ToolCallPart{ID: "unknown", Name: "f"}))
	_, contents = p.ConvertMessages(unsigned.Snapshot().Messages, "")
	if contents[1].Parts[0].ThoughtSignature != placeholderSignature {
		t.Fatalf("expected placeholder signature, got %q", contents[1].Parts[0].ThoughtSignature)
	}
}

func TestGeminiProviderBlockedResponses(t *testing.T) {
	srv, _, _ := newGeminiSSEServer(t, `{"promptFeedback":{"blockReason":"SAFETY"}}`)
	p := NewProvider(ProviderConfig{Model: "m", URL: func(string) string { return srv.URL }})
	if _, err := streamAll(t, p, llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{}); err == nil || !strings.Contains(err.Error(), "BLOCKED_SAFETY") {
		t.Fatalf("expected prompt block error, got %v", err)
	}
	srv2, _, _ := newGeminiSSEServer(t, `{"candidates":[{"content":{"parts":[{"text":"a"}]},"finishReason":"SAFETY"}]}`)
	p2 := NewProvider(ProviderConfig{Model: "m", URL: func(string) string { return srv2.URL }})
	if _, err := streamAll(t, p2, llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{}); err == nil || !strings.Contains(err.Error(), "SAFETY") {
		t.Fatalf("expected safety finish error, got %v", err)
	}
	srv3 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}`)
	}))
	defer srv3.Close()
	p3 := NewProvider(ProviderConfig{Model: "m", URL: func(string) string { return srv3.URL }})
	if _, err := streamAll(t, p3, llm.NewContext(llm.User("x")).Snapshot(), llm.Settings{}); err == nil || !strings.Contains(err.Error(), "RESOURCE_EXHAUSTED") {
		t.Fatalf("expected API error, got %v", err)
	}
}

func TestLiveServiceAnnouncesRealtimeUpstream(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "k"})
	defer h.close()
	f, ok := h.upstream.waitForFrame(2*time.Second, isType[*frames.LLMMetadataFrame])
	if !ok || !f.(*frames.LLMMetadataFrame).Realtime {
		t.Fatalf("expected a realtime LLMMetadataFrame upstream, got %v", f)
	}
}
