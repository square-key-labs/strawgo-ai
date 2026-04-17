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
	"github.com/square-key-labs/strawgo-ai/src/turns"
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

func (s *mockPlaybackAckSerializer) SerializePlaybackDoneAck(correlationID string) (interface{}, error) {
	if correlationID == "" {
		correlationID = "playback-done"
	}
	return correlationID, nil
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
		Port:               8080,
		Path:               "/ws",
		Serializer:         &mockPlaybackAckSerializer{},
		PlaybackAckTimeout: 300 * time.Millisecond,
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

	playbackComplete := frames.NewPlaybackCompleteFrame()
	playbackComplete.SetMetadata("correlation_id", "playback-1")
	if err := processor.HandleFrame(ctx, playbackComplete, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(stale PlaybackCompleteFrame) error: %v", err)
	}

	time.Sleep(50 * time.Millisecond)
	if got := capture.count("BotStoppedSpeakingFrame"); got != 0 {
		t.Fatalf("expected stale playback completion to be ignored, got %d", got)
	}

	playbackComplete = frames.NewPlaybackCompleteFrame()
	playbackComplete.SetMetadata("correlation_id", "playback-2")
	if err := processor.HandleFrame(ctx, playbackComplete, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(PlaybackCompleteFrame) error: %v", err)
	}

	if !capture.waitForFrame("BotStoppedSpeakingFrame", time.Second) {
		t.Fatal("timed out waiting for BotStoppedSpeakingFrame after playback ack")
	}
}

func TestTTSStartedFrameDrainsStalePlaybackAck(t *testing.T) {
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:               8080,
		Path:               "/ws",
		Serializer:         &mockPlaybackAckSerializer{},
		PlaybackAckTimeout: 300 * time.Millisecond,
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

func TestInterruptionResetsPlaybackWaitState(t *testing.T) {
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:               8080,
		Path:               "/ws",
		Serializer:         &mockPlaybackAckSerializer{},
		PlaybackAckTimeout: 300 * time.Millisecond,
	})
	processor := transport.outputProc
	defer processor.Cleanup()

	capture := &queuedFrameCapture{}
	processor.SetPrev(capture)

	ctx := context.Background()

	start := frames.NewStartFrameWithConfig(true, turns.UserTurnStrategies{})
	if err := processor.HandleFrame(ctx, start, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(StartFrame) error: %v", err)
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
		t.Fatal("timed out waiting for initial BotStartedSpeakingFrame")
	}

	if err := processor.HandleFrame(ctx, frames.NewInterruptionFrame(), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(InterruptionFrame) error: %v", err)
	}

	time.Sleep(50 * time.Millisecond)

	clearedPlayback := frames.NewPlaybackCompleteFrame()
	clearedPlayback.SetMetadata("correlation_id", "playback-1")
	if err := processor.HandleFrame(ctx, clearedPlayback, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(cleared PlaybackCompleteFrame) error: %v", err)
	}

	time.Sleep(50 * time.Millisecond)
	if got := capture.count("BotStoppedSpeakingFrame"); got != 1 {
		t.Fatalf("expected exactly one BotStoppedSpeakingFrame from interruption, got %d", got)
	}

	nextContextID := services.GenerateContextID()
	if err := processor.HandleFrame(ctx, frames.NewTTSStartedFrameWithContext(nextContextID), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(next TTSStartedFrame) error: %v", err)
	}

	nextAudioFrame := frames.NewTTSAudioFrame(make([]byte, 640), 16000, 1)
	nextAudioFrame.SetMetadata("context_id", nextContextID)
	if err := processor.HandleFrame(ctx, nextAudioFrame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame(next TTSAudioFrame) error: %v", err)
	}

	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if capture.count("BotStartedSpeakingFrame") >= 2 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}

	t.Fatalf("expected second BotStartedSpeakingFrame after interruption reset, got %d", capture.count("BotStartedSpeakingFrame"))
}

func TestPlaybackAckTimeoutFallbackUsesConfig(t *testing.T) {
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:               8080,
		Path:               "/ws",
		Serializer:         &mockPlaybackAckSerializer{},
		PlaybackAckTimeout: 200 * time.Millisecond,
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

	if !capture.waitForFrame("BotStoppedSpeakingFrame", time.Second) {
		t.Fatal("timed out waiting for BotStoppedSpeakingFrame after fallback timeout")
	}
}
