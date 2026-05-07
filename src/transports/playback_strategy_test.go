package transports

import (
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

// mockAckSerializer implements FrameSerializer + PlaybackAckSerializer.
type mockAckSerializer struct{}

func (s *mockAckSerializer) Type() serializers.SerializerType { return serializers.SerializerTypeText }
func (s *mockAckSerializer) Setup(frames.Frame) error         { return nil }
func (s *mockAckSerializer) Serialize(frames.Frame) (interface{}, error) {
	return nil, nil
}
func (s *mockAckSerializer) Deserialize(interface{}) (frames.Frame, error) { return nil, nil }
func (s *mockAckSerializer) Cleanup() error                                { return nil }
func (s *mockAckSerializer) SerializePlaybackDoneAck(correlationID string) (interface{}, error) {
	return "ack-request:" + correlationID, nil
}

func newOutputWithSerializer(s serializers.FrameSerializer) *WebSocketOutputProcessor {
	t := NewWebSocketTransport(WebSocketConfig{Port: 0, Path: "/ws", Serializer: s})
	return t.outputProc
}

func newTransportWithSerializer(s serializers.FrameSerializer) *WebSocketTransport {
	return NewWebSocketTransport(WebSocketConfig{Port: 0, Path: "/ws", Serializer: s})
}

func TestResolvePlaybackStrategy_UserAckTakesPrecedence(t *testing.T) {
	p := newOutputWithSerializer(&mockAckSerializer{})
	p.RegisterPlaybackAckHandler()

	if got := p.resolvePlaybackStrategy(); got != stratUserAck {
		t.Fatalf("want stratUserAck, got %v", got)
	}

	p.UnregisterPlaybackAckHandler()
	if got := p.resolvePlaybackStrategy(); got != stratSerializerAck {
		t.Fatalf("after unregister, want stratSerializerAck, got %v", got)
	}
}

func TestResolvePlaybackStrategy_SerializerAck(t *testing.T) {
	p := newOutputWithSerializer(&mockAckSerializer{})
	if got := p.resolvePlaybackStrategy(); got != stratSerializerAck {
		t.Fatalf("want stratSerializerAck, got %v", got)
	}
}

func TestResolvePlaybackStrategy_LocalKind(t *testing.T) {
	tr := newTransportWithSerializer(&mockSerializer{})
	tr.SetPlaybackKind(PlaybackLocal)
	if got := tr.outputProc.resolvePlaybackStrategy(); got != stratLocal {
		t.Fatalf("want stratLocal, got %v", got)
	}
	tr.SetPlaybackKind(PlaybackNetworkBlind)
	if got := tr.outputProc.resolvePlaybackStrategy(); got != stratDrainPad {
		t.Fatalf("after revert, want stratDrainPad, got %v", got)
	}
}

func TestResolvePlaybackStrategy_DrainPadDefault(t *testing.T) {
	p := newOutputWithSerializer(&mockSerializer{})
	if got := p.resolvePlaybackStrategy(); got != stratDrainPad {
		t.Fatalf("want stratDrainPad, got %v", got)
	}
}

func TestPlaybackClassifier_DefaultIsNetworkBlind(t *testing.T) {
	tr := newTransportWithSerializer(&mockSerializer{})
	if got := tr.PlaybackKind(); got != PlaybackNetworkBlind {
		t.Fatalf("default PlaybackKind = %v, want PlaybackNetworkBlind", got)
	}
}

func TestTriggerPlaybackComplete_DoesNotBlock(t *testing.T) {
	p := newOutputWithSerializer(&mockSerializer{})
	defer p.Cleanup()

	// TriggerPlaybackComplete is fire-and-forget; it must never block
	// even if invoked many times in a row before any consumer drains.
	done := make(chan struct{})
	go func() {
		for i := 0; i < 100; i++ {
			p.TriggerPlaybackComplete()
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("TriggerPlaybackComplete blocked under repeated calls")
	}
}

func TestSetDrainPad(t *testing.T) {
	p := newOutputWithSerializer(&mockSerializer{})

	if got := p.drainPadNanos.Load(); got != int64(DefaultDrainPad) {
		t.Fatalf("default drain pad = %v, want %v", got, int64(DefaultDrainPad))
	}

	p.SetDrainPad(50_000_000) // 50 ms
	if got := p.drainPadNanos.Load(); got != 50_000_000 {
		t.Fatalf("after set, got %v", got)
	}

	p.SetDrainPad(-1)
	if got := p.drainPadNanos.Load(); got != 0 {
		t.Fatalf("negative should clamp to 0, got %v", got)
	}
}

func TestTransportForwardsPlaybackAckAPI(t *testing.T) {
	tr := newTransportWithSerializer(&mockSerializer{})
	defer tr.outputProc.Cleanup()

	tr.RegisterPlaybackAckHandler()
	if !tr.outputProc.userAckRegistered.Load() {
		t.Fatal("RegisterPlaybackAckHandler did not set userAckRegistered")
	}

	tr.UnregisterPlaybackAckHandler()
	if tr.outputProc.userAckRegistered.Load() {
		t.Fatal("UnregisterPlaybackAckHandler did not clear userAckRegistered")
	}

	// TriggerPlaybackComplete via transport facade should reach the ack
	// tracker without panicking or blocking.
	tr.TriggerPlaybackComplete()

	tr.SetDrainPad(123 * 1_000_000) // 123 ms
	if got := tr.outputProc.drainPadNanos.Load(); got != 123_000_000 {
		t.Fatalf("SetDrainPad forwarding failed: got %v", got)
	}
}
