package daily

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pion/webrtc/v3"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type mockCodec struct {
	decode []byte
	encode []byte
}

func (m *mockCodec) DecodeOpusToPCM(payload []byte, sampleRate, channels int) ([]byte, error) {
	if m.decode != nil {
		return append([]byte(nil), m.decode...), nil
	}
	return append([]byte(nil), payload...), nil
}

func (m *mockCodec) EncodePCMToOpus(pcm []byte, sampleRate, channels int) ([]byte, error) {
	if m.encode != nil {
		return append([]byte(nil), m.encode...), nil
	}
	return append([]byte(nil), pcm...), nil
}

type mockPeer struct {
	trackHandler func(remoteAudioTrack)
	stateHandler func(connectionState)
	outgoing     *mockOutgoingTrack
	closed       atomic.Bool
}

func newMockPeer() *mockPeer {
	return &mockPeer{outgoing: &mockOutgoingTrack{}}
}

func (m *mockPeer) OnRemoteTrack(handler func(remoteAudioTrack)) {
	m.trackHandler = handler
}

func (m *mockPeer) OnConnectionStateChange(handler func(connectionState)) {
	m.stateHandler = handler
}

func (m *mockPeer) CreateOutgoingAudioTrack() (outgoingAudioTrack, error) {
	return m.outgoing, nil
}

func (m *mockPeer) Close() error {
	m.closed.Store(true)
	return nil
}

func (m *mockPeer) emitTrack(track remoteAudioTrack) {
	if m.trackHandler != nil {
		m.trackHandler(track)
	}
}

func (m *mockPeer) emitState(state connectionState) {
	if m.stateHandler != nil {
		m.stateHandler(state)
	}
}

type mockRemoteTrack struct {
	codec   string
	id      string
	stream  string
	packets chan []byte
	done    chan struct{}
}

func newMockRemoteTrack(codec, id, stream string) *mockRemoteTrack {
	return &mockRemoteTrack{
		codec:   codec,
		id:      id,
		stream:  stream,
		packets: make(chan []byte, 8),
		done:    make(chan struct{}),
	}
}

func (m *mockRemoteTrack) ReadOpusPacket() ([]byte, error) {
	select {
	case packet := <-m.packets:
		return packet, nil
	case <-m.done:
		return nil, io.EOF
	}
}

func (m *mockRemoteTrack) CodecMimeType() string { return m.codec }
func (m *mockRemoteTrack) ID() string            { return m.id }
func (m *mockRemoteTrack) StreamID() string      { return m.stream }

type mockOutgoingTrack struct {
	mu       sync.Mutex
	payloads [][]byte
	closed   bool
}

func (m *mockOutgoingTrack) WriteOpus(payload []byte, duration time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.payloads = append(m.payloads, append([]byte(nil), payload...))
	return nil
}

func (m *mockOutgoingTrack) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

type frameCollector struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (c *frameCollector) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return c.QueueFrame(frame, direction)
}

func (c *frameCollector) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.frames = append(c.frames, frame)
	return nil
}

func (c *frameCollector) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return c.QueueFrame(frame, direction)
}

func (c *frameCollector) Link(next processors.FrameProcessor)    {}
func (c *frameCollector) SetPrev(prev processors.FrameProcessor) {}
func (c *frameCollector) Start(ctx context.Context) error        { return nil }
func (c *frameCollector) Stop() error                            { return nil }
func (c *frameCollector) Name() string                           { return "collector" }

func (c *frameCollector) snapshot() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	copyFrames := make([]frames.Frame, len(c.frames))
	copy(copyFrames, c.frames)
	return copyFrames
}

func TestConnectAndRoomLifecycle(t *testing.T) {
	var roomCalls int
	var tokenCalls int

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer daily-key" {
			t.Fatalf("missing auth header")
		}

		switch r.URL.Path {
		case "/rooms":
			roomCalls++
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"name":"demo-room"}`))
		case "/meeting-tokens":
			tokenCalls++
			var body map[string]interface{}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode body: %v", err)
			}
			properties, _ := body["properties"].(map[string]interface{})
			if properties["room_name"] != "demo-room" {
				t.Fatalf("unexpected room_name payload: %#v", body)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"token":"meeting-token"}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	peer := newMockPeer()
	transport := NewDailyTransport(DailyConfig{
		APIKey:             "daily-key",
		APIBaseURL:         server.URL,
		RoomName:           "demo-room",
		CreateRoomIfAbsent: true,
		WebRTCConfig:       webrtc.Configuration{},
	})
	transport.peerFactory = func(cfg webrtc.Configuration) (peerConnection, error) { return peer, nil }

	if err := transport.Connect(context.Background()); err != nil {
		t.Fatalf("connect: %v", err)
	}

	if roomCalls != 1 || tokenCalls != 1 {
		t.Fatalf("expected room and token calls once, got room=%d token=%d", roomCalls, tokenCalls)
	}
	if transport.meetingToken != "meeting-token" {
		t.Fatalf("unexpected meeting token: %s", transport.meetingToken)
	}

	if err := transport.Disconnect(); err != nil {
		t.Fatalf("disconnect: %v", err)
	}
	if !peer.closed.Load() {
		t.Fatalf("expected peer to close")
	}
}

func TestInputProcessorEmitsAudioFrame(t *testing.T) {
	peer := newMockPeer()
	transport := NewDailyTransport(DailyConfig{
		APIKey:             "k",
		APIBaseURL:         "http://invalid",
		RoomName:           "room",
		CreateRoomIfAbsent: false,
	})
	transport.codec = &mockCodec{decode: []byte{0x10, 0x20, 0x30, 0x40}}
	transport.inputProc.codec = transport.codec
	transport.peerFactory = func(cfg webrtc.Configuration) (peerConnection, error) { return peer, nil }

	transport.joinRoomFn = func(ctx context.Context) error {
		transport.meetingToken = "token"
		return nil
	}

	collector := &frameCollector{}
	transport.inputProc.Link(collector)

	if err := transport.Connect(context.Background()); err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer transport.Disconnect()

	track := newMockRemoteTrack("audio/opus", "track-1", "participant-1")
	peer.emitTrack(track)
	track.packets <- []byte{0x01, 0x02}

	waitFor(t, 2*time.Second, func() bool {
		captured := collector.snapshot()
		for _, f := range captured {
			if af, ok := f.(*frames.AudioFrame); ok {
				return len(af.Data) == 4
			}
		}
		return false
	})

	close(track.done)
	if len(transport.Participants()) != 0 {
		waitFor(t, 2*time.Second, func() bool { return len(transport.Participants()) == 0 })
	}
}

func TestOutputProcessorSendsTTSAudioFrame(t *testing.T) {
	peer := newMockPeer()
	transport := NewDailyTransport(DailyConfig{
		APIKey:     "k",
		APIBaseURL: "http://invalid",
		RoomName:   "room",
	})
	transport.codec = &mockCodec{encode: []byte{0xaa, 0xbb}}
	transport.outputProc.codec = transport.codec
	transport.peerFactory = func(cfg webrtc.Configuration) (peerConnection, error) { return peer, nil }
	transport.joinRoomFn = func(ctx context.Context) error {
		transport.meetingToken = "token"
		return nil
	}

	if err := transport.Connect(context.Background()); err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer transport.Disconnect()

	tts := frames.NewTTSAudioFrame([]byte{0x01, 0x02, 0x03}, 16000, 1)
	tts.SetMetadata("codec", "linear16")

	if err := transport.outputProc.HandleFrame(context.Background(), tts, frames.Downstream); err != nil {
		t.Fatalf("handle frame: %v", err)
	}

	peer.outgoing.mu.Lock()
	defer peer.outgoing.mu.Unlock()
	if len(peer.outgoing.payloads) != 1 {
		t.Fatalf("expected one opus packet, got %d", len(peer.outgoing.payloads))
	}
	if len(peer.outgoing.payloads[0]) != 2 {
		t.Fatalf("expected encoded payload")
	}
}

func TestParticipantJoinLeaveAndReconnect(t *testing.T) {
	peer1 := newMockPeer()
	peer2 := newMockPeer()

	transport := NewDailyTransport(DailyConfig{
		APIKey:         "k",
		APIBaseURL:     "http://invalid",
		RoomName:       "room",
		ReconnectDelay: 5 * time.Millisecond,
	})
	transport.codec = &mockCodec{}
	transport.inputProc.codec = transport.codec
	transport.outputProc.codec = transport.codec

	var factoryCalls atomic.Int32
	transport.peerFactory = func(cfg webrtc.Configuration) (peerConnection, error) {
		if factoryCalls.Add(1) == 1 {
			return peer1, nil
		}
		return peer2, nil
	}
	transport.joinRoomFn = func(ctx context.Context) error {
		transport.meetingToken = "token"
		return nil
	}

	if err := transport.Connect(context.Background()); err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer transport.Disconnect()

	remote := newMockRemoteTrack("audio/opus", "track-1", "participant-x")
	peer1.emitTrack(remote)
	waitFor(t, time.Second, func() bool { return len(transport.Participants()) == 1 })

	close(remote.done)
	waitFor(t, time.Second, func() bool { return len(transport.Participants()) == 0 })

	peer1.emitState(connectionStateDisconnected)
	waitFor(t, 2*time.Second, func() bool {
		transport.peerMu.RLock()
		defer transport.peerMu.RUnlock()
		return transport.peer == peer2
	})
}

func TestConnectFailsOnAPIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusBadRequest)
	}))
	defer server.Close()

	transport := NewDailyTransport(DailyConfig{
		APIKey:             "daily-key",
		APIBaseURL:         server.URL,
		RoomName:           "demo-room",
		CreateRoomIfAbsent: true,
	})

	err := transport.Connect(context.Background())
	if err == nil {
		t.Fatalf("expected connect error")
	}
}

func waitFor(t *testing.T, timeout time.Duration, condition func() bool) {
	t.Helper()

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("condition not met within %v", timeout)
}
