package transports

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

type mockPlaybackAckSerializer struct{}

func (s *mockPlaybackAckSerializer) Type() serializers.SerializerType {
	return serializers.SerializerTypeText
}

func (s *mockPlaybackAckSerializer) Setup(frame frames.Frame) error {
	return nil
}

func (s *mockPlaybackAckSerializer) Serialize(frame frames.Frame) (interface{}, error) {
	if _, ok := frame.(*frames.TTSAudioFrame); ok {
		return "audio", nil
	}
	return nil, nil
}

func (s *mockPlaybackAckSerializer) Deserialize(data interface{}) (frames.Frame, error) {
	return nil, nil
}

func (s *mockPlaybackAckSerializer) Cleanup() error {
	return nil
}

func (s *mockPlaybackAckSerializer) SerializePlaybackDoneAck() (interface{}, error) {
	return "playback-done", nil
}

type queuedFrameCapture struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (c *queuedFrameCapture) record(frame frames.Frame) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.frames = append(c.frames, frame)
}

func (c *queuedFrameCapture) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	c.record(frame)
	return nil
}

func (c *queuedFrameCapture) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	c.record(frame)
	return nil
}

func (c *queuedFrameCapture) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	c.record(frame)
	return nil
}

func (c *queuedFrameCapture) Link(next processors.FrameProcessor) {}

func (c *queuedFrameCapture) SetPrev(prev processors.FrameProcessor) {}

func (c *queuedFrameCapture) Start(ctx context.Context) error { return nil }

func (c *queuedFrameCapture) Stop() error { return nil }

func (c *queuedFrameCapture) Name() string { return "queued-capture" }

func (c *queuedFrameCapture) count(name string) int {
	c.mu.Lock()
	defer c.mu.Unlock()

	count := 0
	for _, frame := range c.frames {
		if frame.Name() == name {
			count++
		}
	}
	return count
}

func (c *queuedFrameCapture) waitForFrame(name string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if c.count(name) > 0 {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return c.count(name) > 0
}

func TestPlaybackCompleteFrameDelaysBotStoppedSpeaking(t *testing.T) {
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:       8080,
		Path:       "/ws",
		Serializer: &mockPlaybackAckSerializer{},
	})
	processor := transport.outputProc
	defer processor.Cleanup()

	capture := &queuedFrameCapture{}
	processor.SetPrev(capture)

	ctx := context.Background()
	contextID := services.GenerateContextID()

	if err := processor.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(contextID), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(TTSStartedFrame) error: %v", err)
	}

	audioFrame := frames.NewTTSAudioFrame(make([]byte, 640), 16000, 1)
	audioFrame.SetMetadata("context_id", contextID)
	if err := processor.HandleFrame(ctx, audioFrame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(TTSAudioFrame) error: %v", err)
	}

	if !capture.waitForFrame("BotStartedSpeakingFrame", time.Second) {
		t.Fatal("timed out waiting for BotStartedSpeakingFrame")
	}

	if err := processor.HandleFrame(ctx, frames.NewLLMFullResponseEndFrame(), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(LLMFullResponseEndFrame) error: %v", err)
	}

	time.Sleep(500 * time.Millisecond)
	if got := capture.count("BotStoppedSpeakingFrame"); got != 0 {
		t.Fatalf("expected no BotStoppedSpeakingFrame before playback ack, got %d", got)
	}

	if err := processor.HandleFrame(ctx, frames.NewPlaybackCompleteFrame(), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(PlaybackCompleteFrame) error: %v", err)
	}

	if !capture.waitForFrame("BotStoppedSpeakingFrame", time.Second) {
		t.Fatal("timed out waiting for BotStoppedSpeakingFrame after playback ack")
	}
}

func TestTTSStartedFrameDrainsStalePlaybackAck(t *testing.T) {
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:       8080,
		Path:       "/ws",
		Serializer: &mockPlaybackAckSerializer{},
	})
	processor := transport.outputProc
	defer processor.Cleanup()

	capture := &queuedFrameCapture{}
	processor.SetPrev(capture)

	ctx := context.Background()

	if err := processor.HandleFrame(ctx, frames.NewPlaybackCompleteFrame(), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(stale PlaybackCompleteFrame) error: %v", err)
	}

	contextID := services.GenerateContextID()
	if err := processor.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(contextID), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(TTSStartedFrame) error: %v", err)
	}

	audioFrame := frames.NewTTSAudioFrame(make([]byte, 640), 16000, 1)
	audioFrame.SetMetadata("context_id", contextID)
	if err := processor.HandleFrame(ctx, audioFrame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(TTSAudioFrame) error: %v", err)
	}

	if !capture.waitForFrame("BotStartedSpeakingFrame", time.Second) {
		t.Fatal("timed out waiting for BotStartedSpeakingFrame")
	}

	if err := processor.HandleFrame(ctx, frames.NewLLMFullResponseEndFrame(), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(LLMFullResponseEndFrame) error: %v", err)
	}

	time.Sleep(500 * time.Millisecond)
	if got := capture.count("BotStoppedSpeakingFrame"); got != 0 {
		t.Fatalf("expected stale playback ack to be drained before new utterance, got %d stop frames", got)
	}
}
