package transports

import (
	"context"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

// Stale-audio gate tests previously lived here. They pinned to private
// fields on WebSocketOutputProcessor that have been removed in Phase 2:
// the gate now lives in the AudioGuard processor. See
// src/processors/audio_guard_test.go for the equivalent coverage and
// docs/adr/0001-audio-guard-as-dedicated-processor.md for context.

type mockSerializer struct{}

func (s *mockSerializer) Type() serializers.SerializerType {
	return serializers.SerializerTypeBinary
}

func (s *mockSerializer) Setup(frame frames.Frame) error {
	return nil
}

func (s *mockSerializer) Serialize(frame frames.Frame) (interface{}, error) {
	if _, ok := frame.(*frames.TTSAudioFrame); ok {
		return []byte("audio"), nil
	}
	return nil, nil
}

func (s *mockSerializer) Deserialize(data interface{}) (frames.Frame, error) {
	return nil, nil
}

func (s *mockSerializer) Cleanup() error {
	return nil
}

type frameCapture struct {
	frames []frames.Frame
}

func (f *frameCapture) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	f.frames = append(f.frames, frame)
	return nil
}

func (f *frameCapture) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}

func (f *frameCapture) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	f.frames = append(f.frames, frame)
	return nil
}

func (f *frameCapture) Link(next interface{}) {}

func (f *frameCapture) SetPrev(prev interface{}) {}

func (f *frameCapture) Start(ctx context.Context) error { return nil }

func (f *frameCapture) Stop() error { return nil }

func (f *frameCapture) Name() string { return "capture" }

// TestTransportPassesThroughTaggedAudio is a smoke test confirming the
// transport no longer drops tagged audio on its own — the gate has moved.
func TestTransportPassesThroughTaggedAudio(t *testing.T) {
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:       8080,
		Path:       "/ws",
		Serializer: &mockSerializer{},
	})

	processor := transport.outputProc
	ctx := context.Background()

	startFrame := frames.NewTTSStartedFrameWithContext("gen-1")
	if err := processor.HandleFrame(ctx, startFrame, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(TTSStartedFrame) error: %v", err)
	}

	audioFrame := frames.NewTTSAudioFrame([]byte("audio"), 24000, 1)
	audioFrame.SetMetadata("context_id", "gen-1")
	if err := processor.HandleFrame(ctx, audioFrame, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(TTSAudioFrame) error: %v", err)
	}
}
