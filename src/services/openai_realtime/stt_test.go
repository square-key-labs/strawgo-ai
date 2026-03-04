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
	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type sttFrameCollector struct {
	name string
	ch   chan frames.Frame
}

func newSTTFrameCollector(name string) *sttFrameCollector {
	return &sttFrameCollector{name: name, ch: make(chan frames.Frame, 256)}
}

func (c *sttFrameCollector) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return c.QueueFrame(frame, direction)
}

func (c *sttFrameCollector) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	select {
	case c.ch <- frame:
	default:
	}
	return nil
}

func (c *sttFrameCollector) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return c.QueueFrame(frame, direction)
}

func (c *sttFrameCollector) Link(next processors.FrameProcessor)    {}
func (c *sttFrameCollector) SetPrev(prev processors.FrameProcessor) {}
func (c *sttFrameCollector) Start(ctx context.Context) error        { return nil }
func (c *sttFrameCollector) Stop() error                            { return nil }
func (c *sttFrameCollector) Name() string                           { return c.name }

func (c *sttFrameCollector) waitForFrame(timeout time.Duration, predicate func(frames.Frame) bool) (frames.Frame, bool) {
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

type sttMockRealtimeServer struct {
	t *testing.T

	server *httptest.Server

	connMu sync.Mutex
	conn   *websocket.Conn

	msgs chan map[string]any

	headersMu sync.Mutex
	headers   http.Header

	connected     chan struct{}
	connectedOnce sync.Once
}

func newSTTMockRealtimeServer(t *testing.T) *sttMockRealtimeServer {
	t.Helper()

	m := &sttMockRealtimeServer{
		t:         t,
		msgs:      make(chan map[string]any, 256),
		connected: make(chan struct{}),
	}

	upgrader := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/realtime", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("intent") != "transcription" {
			t.Errorf("expected intent=transcription, got %q", r.URL.Query().Get("intent"))
		}

		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade failed: %v", err)
			return
		}

		m.connMu.Lock()
		m.conn = conn
		m.connMu.Unlock()

		m.headersMu.Lock()
		m.headers = r.Header.Clone()
		m.headersMu.Unlock()

		m.connectedOnce.Do(func() { close(m.connected) })
		go m.readLoop(conn)
	})

	m.server = httptest.NewServer(mux)
	return m
}

func (m *sttMockRealtimeServer) endpoint() string {
	return "ws" + strings.TrimPrefix(m.server.URL, "http") + "/v1/realtime?intent=transcription"
}

func (m *sttMockRealtimeServer) readLoop(conn *websocket.Conn) {
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

func (m *sttMockRealtimeServer) waitConnected(timeout time.Duration) bool {
	select {
	case <-m.connected:
		return true
	case <-time.After(timeout):
		return false
	}
}

func (m *sttMockRealtimeServer) waitMessage(timeout time.Duration, predicate func(map[string]any) bool) (map[string]any, bool) {
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

func (m *sttMockRealtimeServer) send(payload map[string]any) {
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

func (m *sttMockRealtimeServer) header(key string) string {
	m.headersMu.Lock()
	defer m.headersMu.Unlock()
	if m.headers == nil {
		return ""
	}
	return m.headers.Get(key)
}

func (m *sttMockRealtimeServer) close() {
	m.connMu.Lock()
	if m.conn != nil {
		_ = m.conn.Close()
		m.conn = nil
	}
	m.connMu.Unlock()
	m.server.Close()
}

func hasSTTType(msgType string) func(map[string]any) bool {
	return func(msg map[string]any) bool {
		rawType, ok := msg["type"].(string)
		return ok && rawType == msgType
	}
}

func TestOpenAIRealtimeSTT_Resampling(t *testing.T) {
	server := newSTTMockRealtimeServer(t)
	defer server.close()

	service := NewSTTService(STTConfig{
		APIKey:   "test-key",
		Endpoint: server.endpoint(),
	})
	defer func() {
		if err := service.Cleanup(); err != nil {
			t.Fatalf("cleanup failed: %v", err)
		}
	}()

	if server.header("Authorization") != "" {
		t.Fatalf("expected no headers before connect")
	}

	pcm16k := make([]int16, 320)
	for i := range pcm16k {
		pcm16k[i] = int16(i * 37)
	}

	frame := frames.NewAudioFrame(audio.PCMToBytes(pcm16k), 16000, 1)
	if err := service.HandleFrame(context.Background(), frame, frames.Downstream); err != nil {
		t.Fatalf("audio handling failed: %v", err)
	}

	if !server.waitConnected(2 * time.Second) {
		t.Fatal("service did not connect")
	}

	if auth := server.header("Authorization"); auth != "Bearer test-key" {
		t.Fatalf("unexpected authorization header: %q", auth)
	}
	if beta := server.header("OpenAI-Beta"); beta != "realtime=v1" {
		t.Fatalf("unexpected OpenAI-Beta header: %q", beta)
	}

	sessionUpdate, ok := server.waitMessage(2*time.Second, hasSTTType("session.update"))
	if !ok {
		t.Fatal("did not receive session.update")
	}

	session, ok := sessionUpdate["session"].(map[string]any)
	if !ok {
		t.Fatal("session.update missing session object")
	}
	transcription, ok := session["input_audio_transcription"].(map[string]any)
	if !ok {
		t.Fatal("session.update missing input_audio_transcription object")
	}
	if transcription["model"] != DefaultSTTModel {
		t.Fatalf("unexpected transcription model: %v", transcription["model"])
	}

	appendMsg, ok := server.waitMessage(2*time.Second, hasSTTType("input_audio_buffer.append"))
	if !ok {
		t.Fatal("did not receive input_audio_buffer.append")
	}

	encoded, ok := appendMsg["audio"].(string)
	if !ok {
		t.Fatal("append message missing base64 audio payload")
	}

	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatalf("failed to decode base64 payload: %v", err)
	}

	resampledPCM, err := audio.BytesToPCM(decoded)
	if err != nil {
		t.Fatalf("resampled payload should be PCM16: %v", err)
	}

	expectedPCM := audio.Resample(pcm16k, 16000, 24000)
	if len(resampledPCM) != len(expectedPCM) {
		t.Fatalf("unexpected sample count after resampling: got=%d expected=%d", len(resampledPCM), len(expectedPCM))
	}
}

func TestOpenAIRealtimeSTT_ServerVAD(t *testing.T) {
	server := newSTTMockRealtimeServer(t)
	defer server.close()

	service := NewSTTService(STTConfig{
		APIKey:        "test-key",
		Endpoint:      server.endpoint(),
		TurnDetection: "server",
	})
	defer func() {
		if err := service.Cleanup(); err != nil {
			t.Fatalf("cleanup failed: %v", err)
		}
	}()

	upstream := newSTTFrameCollector("upstream")
	downstream := newSTTFrameCollector("downstream")
	service.SetPrev(upstream)
	service.Link(downstream)

	if err := service.Initialize(context.Background()); err != nil {
		t.Fatalf("initialize failed: %v", err)
	}

	if !server.waitConnected(2 * time.Second) {
		t.Fatal("service did not connect")
	}

	sessionUpdate, ok := server.waitMessage(2*time.Second, hasSTTType("session.update"))
	if !ok {
		t.Fatal("did not receive session.update")
	}
	session, ok := sessionUpdate["session"].(map[string]any)
	if !ok {
		t.Fatal("session.update missing session object")
	}
	if session["turn_detection"] != nil {
		t.Fatalf("server VAD should set turn_detection to null, got: %#v", session["turn_detection"])
	}

	server.send(map[string]any{"type": "input_audio_buffer.speech_stopped"})

	if _, ok := upstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		_, match := frame.(*frames.InterruptionFrame)
		return match
	}); !ok {
		t.Fatal("missing upstream InterruptionFrame from server VAD speech_stopped")
	}

	if _, ok := downstream.waitForFrame(2*time.Second, func(frame frames.Frame) bool {
		_, match := frame.(*frames.InterruptionFrame)
		return match
	}); !ok {
		t.Fatal("missing downstream InterruptionFrame from server VAD speech_stopped")
	}
}

func TestOpenAIRealtimeSTT_LocalVAD(t *testing.T) {
	server := newSTTMockRealtimeServer(t)
	defer server.close()

	service := NewSTTService(STTConfig{
		APIKey:        "test-key",
		Endpoint:      server.endpoint(),
		TurnDetection: "local",
	})
	defer func() {
		if err := service.Cleanup(); err != nil {
			t.Fatalf("cleanup failed: %v", err)
		}
	}()

	if err := service.Initialize(context.Background()); err != nil {
		t.Fatalf("initialize failed: %v", err)
	}

	if !server.waitConnected(2 * time.Second) {
		t.Fatal("service did not connect")
	}

	sessionUpdate, ok := server.waitMessage(2*time.Second, hasSTTType("session.update"))
	if !ok {
		t.Fatal("did not receive session.update")
	}
	session, ok := sessionUpdate["session"].(map[string]any)
	if !ok {
		t.Fatal("session.update missing session object")
	}
	turnDetection, ok := session["turn_detection"].(bool)
	if !ok || turnDetection {
		t.Fatalf("local VAD should set turn_detection=false, got: %#v", session["turn_detection"])
	}

	if err := service.HandleFrame(context.Background(), frames.NewUserStoppedSpeakingFrame(), frames.Downstream); err != nil {
		t.Fatalf("user stopped handling failed: %v", err)
	}

	if _, ok := server.waitMessage(2*time.Second, hasSTTType("input_audio_buffer.commit")); !ok {
		t.Fatal("did not receive input_audio_buffer.commit for local VAD")
	}
}
