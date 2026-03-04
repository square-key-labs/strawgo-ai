package gemini

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

	apiKeyMu sync.Mutex
	apiKey   string
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
		m.apiKey = r.URL.Query().Get("key")
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
	if err := conn.WriteJSON(payload); err != nil {
		m.t.Fatalf("mock websocket write failed: %v", err)
	}
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
	return &liveHarness{t: t, server: server, service: service, upstream: up, downstream: down}
}

func (h *liveHarness) close() {
	if err := h.service.Cleanup(); err != nil {
		h.t.Fatalf("cleanup failed: %v", err)
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
	// Handle both []any and []string types
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

func TestNewGeminiLiveServiceAndConfig(t *testing.T) {
	service := NewGeminiLiveService("test-key", "")
	if service.apiKey != "test-key" || service.model != DefaultLiveModel {
		t.Fatalf("unexpected constructor defaults: key=%s model=%s", service.apiKey, service.model)
	}
	if service.endpoint != GeminiLiveURL || service.outputSampleRate != DefaultLiveOutputSampleRate || service.outputChannels != DefaultLiveOutputChannels {
		t.Fatalf("unexpected default output config: endpoint=%s sampleRate=%d channels=%d", service.endpoint, service.outputSampleRate, service.outputChannels)
	}
	custom := NewLiveService(LiveConfig{APIKey: "k", Model: "gemini-2.0-flash-exp", SystemInstruction: "Be brief.", Voice: "Puck", InputMIMEType: "audio/pcm;rate=8000", OutputSampleRate: 16000, OutputChannels: 2})
	if custom.systemInstruction != "Be brief." || custom.voice != "Puck" || custom.inputMIMEType != "audio/pcm;rate=8000" || custom.outputSampleRate != 16000 || custom.outputChannels != 2 {
		t.Fatalf("custom config not applied")
	}
}

func TestLiveServiceSetupMessageIncludesSystemInstructionAndVoice(t *testing.T) {
	service := NewLiveService(LiveConfig{Model: "gemini-2.0-flash-exp", SystemInstruction: "Be concise.", Voice: "Puck"})
	setup := mustMap(t, service.setupMessage()["setup"], "setup")
	if setup["model"] != "models/gemini-2.0-flash-exp" {
		t.Fatalf("unexpected setup model: %v", setup["model"])
	}
	generation := mustMap(t, setup["generationConfig"], "generationConfig")
	switch modalities := generation["responseModalities"].(type) {
	case []string:
		if len(modalities) != 1 || modalities[0] != "AUDIO" {
			t.Fatalf("unexpected response modalities: %v", modalities)
		}
	case []any:
		if len(modalities) != 1 || modalities[0] != "AUDIO" {
			t.Fatalf("unexpected response modalities: %v", modalities)
		}
	default:
		t.Fatalf("responseModalities has unexpected type %T", generation["responseModalities"])
	}
	instruction := mustMap(t, setup["systemInstruction"], "systemInstruction")
	switch parts := instruction["parts"].(type) {
	case []map[string]string:
		if len(parts) != 1 || parts[0]["text"] != "Be concise." {
			t.Fatalf("system instruction text mismatch: %v", parts)
		}
	case []any:
		if len(parts) != 1 || mustMap(t, parts[0], "parts[0]")["text"] != "Be concise." {
			t.Fatalf("system instruction text mismatch: %v", parts)
		}
	default:
		t.Fatalf("parts has unexpected type %T", instruction["parts"])
	}
	prebuilt := mustMap(t, mustMap(t, generation["speechConfig"], "speechConfig")["voiceConfig"], "voiceConfig")["prebuiltVoiceConfig"]
	switch voice := prebuilt.(type) {
	case map[string]string:
		if voice["voiceName"] != "Puck" {
			t.Fatalf("unexpected voiceName: %v", voice["voiceName"])
		}
	case map[string]any:
		if voice["voiceName"] != "Puck" {
			t.Fatalf("unexpected voiceName: %v", voice["voiceName"])
		}
	default:
		t.Fatalf("prebuiltVoiceConfig has unexpected type %T", prebuilt)
	}
}

func TestLiveServiceInitializeAndCleanup(t *testing.T) {
	server := newMockLiveServer(t)
	defer server.close()
	service := NewLiveService(LiveConfig{APIKey: "init-key", Model: "gemini-2.0-flash-exp", Endpoint: server.URL()})
	if err := service.Initialize(context.Background()); err != nil {
		t.Fatalf("initialize failed: %v", err)
	}
	if !server.waitConnected(2 * time.Second) {
		t.Fatal("service did not connect")
	}
	if _, ok := server.waitMessage(2*time.Second, hasKey("setup")); !ok {
		t.Fatal("did not receive setup payload")
	}
	if server.lastAPIKey() != "init-key" {
		t.Fatalf("expected api key init-key, got %s", server.lastAPIKey())
	}
	if err := service.HandleFrame(context.Background(), frames.NewEndFrame(), frames.Downstream); err != nil {
		t.Fatalf("EndFrame handling failed: %v", err)
	}
	if service.getConn() != nil {
		t.Fatal("expected nil connection after EndFrame")
	}
	if err := service.Cleanup(); err != nil {
		t.Fatalf("cleanup failed: %v", err)
	}
	if err := service.Cleanup(); err != nil {
		t.Fatalf("cleanup should be idempotent: %v", err)
	}
	if service.getConn() != nil {
		t.Fatal("expected nil connection after cleanup")
	}
}

func TestLiveServiceBidirectionalAudioAndFrameEmission(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "live-key", Model: "gemini-2.0-flash-exp", SystemInstruction: "Be concise.", Voice: "Puck"})
	defer h.close()

	inputAudio := []byte{0x01, 0x02, 0x03, 0x04}
	audioFrame := frames.NewAudioFrame(inputAudio, 8000, 1)
	audioFrame.SetMetadata("codec", "mulaw")
	if err := h.service.HandleFrame(context.Background(), audioFrame, frames.Downstream); err != nil {
		t.Fatalf("audio handling failed: %v", err)
	}

	audioInputMsg, ok := h.server.waitMessage(2*time.Second, hasKey("realtimeInput"))
	if !ok {
		t.Fatal("did not receive realtimeInput message")
	}
	realtimeInput := mustMap(t, audioInputMsg["realtimeInput"], "realtimeInput")
	chunk := mustMap(t, mustSlice(t, realtimeInput["mediaChunks"], "mediaChunks")[0], "mediaChunks[0]")
	if chunk["mimeType"] != "audio/pcm;encoding=mulaw;rate=8000" {
		t.Fatalf("unexpected input mimeType: %v", chunk["mimeType"])
	}
	encoded, ok := chunk["data"].(string)
	if !ok {
		t.Fatal("expected base64 audio payload")
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil || string(decoded) != string(inputAudio) {
		t.Fatalf("unexpected input audio payload decode error=%v bytes=%v", err, decoded)
	}

	assistantAudio := []byte{0x10, 0x20, 0x30}
	h.server.send(map[string]any{"serverContent": map[string]any{
		"inputTranscription":  map[string]any{"transcript": "hello gemini", "final": false},
		"outputTranscription": map[string]any{"text": "assistant transcript"},
		"modelTurn": map[string]any{"parts": []map[string]any{
			{"text": "Hello there."},
			{"inlineData": map[string]any{"mimeType": "audio/pcm;rate=24000", "data": base64.StdEncoding.EncodeToString(assistantAudio)}},
		}},
	}})
	h.server.send(map[string]any{"serverContent": map[string]any{"turnComplete": true}})

	transcriptionAny, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		transcription, match := frame.(*frames.TranscriptionFrame)
		return match && transcription.Text == "hello gemini"
	})
	if !ok {
		t.Fatal("missing TranscriptionFrame")
	}
	if transcriptionAny.(*frames.TranscriptionFrame).IsFinal {
		t.Fatal("expected non-final transcription")
	}
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		llm, match := frame.(*frames.LLMTextFrame)
		return match && llm.Text == "Hello there."
	}); !ok {
		t.Fatal("missing LLMTextFrame")
	}

	startAny, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		_, match := frame.(*frames.TTSStartedFrame)
		return match
	})
	if !ok {
		t.Fatal("missing TTSStartedFrame")
	}
	start := startAny.(*frames.TTSStartedFrame)
	if !isUUIDLike(start.ContextID) || start.Metadata()["context_id"] != start.ContextID {
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
	if string(audio.Data) != string(assistantAudio) || audio.SampleRate != 24000 || audio.Metadata()["codec"] != "linear16" {
		t.Fatalf("unexpected TTSAudioFrame data=%v sampleRate=%d codec=%v", audio.Data, audio.SampleRate, audio.Metadata()["codec"])
	}

	if _, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		stop, match := frame.(*frames.TTSStoppedFrame)
		return match && stop.ContextID == start.ContextID
	}); !ok {
		t.Fatal("missing downstream TTSStoppedFrame")
	}
}

func TestLiveServiceInterruptionStopsAudioAndResumesOnNextTurn(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "interrupt-key", Model: "gemini-2.0-flash-exp"})
	defer h.close()

	firstAudio := []byte{0x55, 0x66, 0x77}
	h.server.send(map[string]any{"serverContent": map[string]any{"modelTurn": map[string]any{"parts": []map[string]any{{
		"inlineData": map[string]any{"mimeType": "audio/pcm;rate=24000", "data": base64.StdEncoding.EncodeToString(firstAudio)},
	}}}}})

	startAny, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		_, match := frame.(*frames.TTSStartedFrame)
		return match
	})
	if !ok {
		t.Fatal("missing initial TTSStartedFrame")
	}
	initialContext := startAny.(*frames.TTSStartedFrame).ContextID
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		audio, match := frame.(*frames.TTSAudioFrame)
		return match && audio.ContextID == initialContext
	}); !ok {
		t.Fatal("missing initial TTSAudioFrame")
	}

	if err := h.service.HandleFrame(context.Background(), frames.NewInterruptionFrame(), frames.Downstream); err != nil {
		t.Fatalf("interruption handling failed: %v", err)
	}
	if _, ok := h.server.waitMessage(2*time.Second, hasKey("clientContent")); !ok {
		t.Fatal("missing interruption turnComplete signal")
	}
	if _, ok := h.upstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		stop, match := frame.(*frames.TTSStoppedFrame)
		return match && stop.ContextID == initialContext
	}); !ok {
		t.Fatal("missing TTSStoppedFrame after interruption")
	}

	staleAudio := []byte{0x01, 0x02}
	h.server.send(map[string]any{"serverContent": map[string]any{"modelTurn": map[string]any{"parts": []map[string]any{{
		"inlineData": map[string]any{"mimeType": "audio/pcm;rate=24000", "data": base64.StdEncoding.EncodeToString(staleAudio)},
	}}}}})
	if ok := h.downstream.expectNoFrame(300*time.Millisecond, func(frame frames.Frame) bool {
		_, match := frame.(*frames.TTSAudioFrame)
		return match
	}); !ok {
		t.Fatal("stale TTSAudioFrame should be suppressed")
	}

	if err := h.service.HandleFrame(context.Background(), frames.NewUserStoppedSpeakingFrame(), frames.Downstream); err != nil {
		t.Fatalf("user stopped speaking handling failed: %v", err)
	}
	if _, ok := h.server.waitMessage(2*time.Second, hasKey("clientContent")); !ok {
		t.Fatal("missing user-turn-complete signal")
	}

	resumedAudio := []byte{0x90, 0x91, 0x92}
	h.server.send(map[string]any{"serverContent": map[string]any{"modelTurn": map[string]any{"parts": []map[string]any{{
		"inlineData": map[string]any{"mimeType": "audio/pcm;rate=24000", "data": base64.StdEncoding.EncodeToString(resumedAudio)},
	}}}}})

	newStartAny, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		start, match := frame.(*frames.TTSStartedFrame)
		return match && start.ContextID != "" && start.ContextID != initialContext
	})
	if !ok {
		t.Fatal("missing resumed TTSStartedFrame")
	}
	newContext := newStartAny.(*frames.TTSStartedFrame).ContextID
	if _, ok := h.downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		audio, match := frame.(*frames.TTSAudioFrame)
		return match && audio.ContextID == newContext && string(audio.Data) == string(resumedAudio)
	}); !ok {
		t.Fatal("missing resumed TTSAudioFrame")
	}
}

func TestLiveServicePassthroughNonAudioFrames(t *testing.T) {
	service := NewLiveService(LiveConfig{APIKey: "test-key"})
	downstream := newFrameCollector("downstream")
	service.Link(downstream)
	textFrame := frames.NewTextFrame("passthrough")
	if err := service.HandleFrame(context.Background(), textFrame, frames.Downstream); err != nil {
		t.Fatalf("passthrough handling failed: %v", err)
	}
	if _, ok := downstream.waitForFrame(time.Second, func(frame frames.Frame) bool { return frame == textFrame }); !ok {
		t.Fatal("expected non-audio frame passthrough")
	}
	if service.getConn() != nil {
		t.Fatal("non-audio frame should not trigger websocket connection")
	}
}

func TestLiveServiceAPIErrorsEmitErrorFrameUpstream(t *testing.T) {
	h := newLiveHarness(t, LiveConfig{APIKey: "error-key", Model: "gemini-2.0-flash-exp"})
	defer h.close()
	h.server.send(map[string]any{"error": map[string]any{"message": "quota exceeded"}})
	errorAny, ok := h.upstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		_, match := frame.(*frames.ErrorFrame)
		return match
	})
	if !ok {
		t.Fatal("did not receive ErrorFrame upstream")
	}
	errorFrame := errorAny.(*frames.ErrorFrame)
	if !strings.Contains(errorFrame.Error.Error(), "Gemini Live API error: quota exceeded") {
		t.Fatalf("unexpected error message: %v", errorFrame.Error)
	}
}

func TestParseOutputAudioMIMEType(t *testing.T) {
	codec, sampleRate := parseOutputAudioMIMEType("audio/pcm;encoding=mulaw;rate=8000")
	if codec != "mulaw" || sampleRate != 8000 {
		t.Fatalf("unexpected parse result: codec=%s sampleRate=%d", codec, sampleRate)
	}
}
