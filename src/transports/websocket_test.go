package transports

import (
	"context"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/turns"
)

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

func TestContextIDTracking(t *testing.T) {
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:       8080,
		Path:       "/ws",
		Serializer: &mockSerializer{},
	})

	processor := transport.outputProc
	ctx := context.Background()

	contextID := services.GenerateContextID()

	startFrame := frames.NewTTSStartedFrameWithContext(contextID)
	if err := processor.HandleFrame(ctx, startFrame, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(TTSStartedFrame) error: %v", err)
	}

	processor.interruptionMu.Lock()
	expectedID := processor.expectedContextID
	currentID := processor.currentContextID
	processor.interruptionMu.Unlock()

	if expectedID != contextID {
		t.Errorf("Expected expectedContextID=%s, got %s", contextID, expectedID)
	}

	if currentID != "" {
		t.Errorf("Expected currentContextID to be empty before first audio, got %s", currentID)
	}

	audioFrame := frames.NewTTSAudioFrame([]byte("test audio"), 24000, 1)
	audioFrame.SetMetadata("context_id", contextID)

	if err := processor.HandleFrame(ctx, audioFrame, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(TTSAudioFrame) error: %v", err)
	}

	processor.interruptionMu.Lock()
	currentID = processor.currentContextID
	interrupted := processor.interrupted
	processor.interruptionMu.Unlock()

	if currentID != contextID {
		t.Errorf("Expected currentContextID=%s after first audio, got %s", contextID, currentID)
	}

	if interrupted {
		t.Error("Expected interrupted=false after accepting audio")
	}
}

func TestStaleAudioBlocking(t *testing.T) {
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:       8080,
		Path:       "/ws",
		Serializer: &mockSerializer{},
	})

	processor := transport.outputProc
	ctx := context.Background()

	startFrame := frames.NewStartFrameWithConfig(true, turns.UserTurnStrategies{})
	if err := processor.HandleFrame(ctx, startFrame, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(StartFrame) error: %v", err)
	}

	oldContextID := services.GenerateContextID()
	newContextID := services.GenerateContextID()

	startFrame1 := frames.NewTTSStartedFrameWithContext(oldContextID)
	if err := processor.HandleFrame(ctx, startFrame1, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(first TTSStartedFrame) error: %v", err)
	}

	audioFrame1 := frames.NewTTSAudioFrame([]byte("old audio"), 24000, 1)
	audioFrame1.SetMetadata("context_id", oldContextID)
	if err := processor.HandleFrame(ctx, audioFrame1, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(first audio) error: %v", err)
	}

	interruptFrame := frames.NewInterruptionFrame()
	if err := processor.HandleFrame(ctx, interruptFrame, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(InterruptionFrame) error: %v", err)
	}

	processor.interruptionMu.Lock()
	interrupted := processor.interrupted
	processor.interruptionMu.Unlock()

	if !interrupted {
		t.Error("Expected interrupted=true after InterruptionFrame")
	}

	startFrame2 := frames.NewTTSStartedFrameWithContext(newContextID)
	if err := processor.HandleFrame(ctx, startFrame2, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(new TTSStartedFrame) error: %v", err)
	}

	processor.interruptionMu.Lock()
	expectedID := processor.expectedContextID
	currentID := processor.currentContextID
	stillInterrupted := processor.interrupted
	processor.interruptionMu.Unlock()

	if expectedID != newContextID {
		t.Errorf("Expected expectedContextID=%s, got %s", newContextID, expectedID)
	}

	if currentID != "" {
		t.Errorf("Expected currentContextID to be reset, got %s", currentID)
	}

	if !stillInterrupted {
		t.Error("Expected interrupted=true until matching audio arrives")
	}

	staleAudio := frames.NewTTSAudioFrame([]byte("stale audio"), 24000, 1)
	staleAudio.SetMetadata("context_id", oldContextID)
	if err := processor.HandleFrame(ctx, staleAudio, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(stale audio) should not error: %v", err)
	}

	processor.interruptionMu.Lock()
	currentID = processor.currentContextID
	processor.interruptionMu.Unlock()

	if currentID == oldContextID {
		t.Error("Stale audio should not update currentContextID")
	}

	newAudio := frames.NewTTSAudioFrame([]byte("new audio"), 24000, 1)
	newAudio.SetMetadata("context_id", newContextID)
	if err := processor.HandleFrame(ctx, newAudio, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(new audio) error: %v", err)
	}

	processor.interruptionMu.Lock()
	currentID = processor.currentContextID
	interrupted = processor.interrupted
	processor.interruptionMu.Unlock()

	if currentID != newContextID {
		t.Errorf("Expected currentContextID=%s, got %s", newContextID, currentID)
	}

	if interrupted {
		t.Error("Expected interrupted=false after accepting new audio")
	}
}

func TestContextIDFilteringWithoutExpected(t *testing.T) {
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:       8080,
		Path:       "/ws",
		Serializer: &mockSerializer{},
	})

	processor := transport.outputProc
	ctx := context.Background()

	contextID1 := services.GenerateContextID()
	contextID2 := services.GenerateContextID()

	startFrame1 := frames.NewTTSStartedFrameWithContext(contextID1)
	if err := processor.HandleFrame(ctx, startFrame1, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(TTSStartedFrame) error: %v", err)
	}

	audioFrame1 := frames.NewTTSAudioFrame([]byte("audio 1"), 24000, 1)
	audioFrame1.SetMetadata("context_id", contextID1)
	if err := processor.HandleFrame(ctx, audioFrame1, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(audio 1) error: %v", err)
	}

	processor.interruptionMu.Lock()
	currentID := processor.currentContextID
	processor.interruptionMu.Unlock()

	if currentID != contextID1 {
		t.Errorf("Expected currentContextID=%s, got %s", contextID1, currentID)
	}

	startFrame2 := frames.NewTTSStartedFrameWithContext(contextID2)
	if err := processor.HandleFrame(ctx, startFrame2, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(new TTSStartedFrame) error: %v", err)
	}

	audioFrame2Old := frames.NewTTSAudioFrame([]byte("old audio"), 24000, 1)
	audioFrame2Old.SetMetadata("context_id", contextID1)
	if err := processor.HandleFrame(ctx, audioFrame2Old, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(old audio) should not error: %v", err)
	}

	processor.interruptionMu.Lock()
	currentID = processor.currentContextID
	processor.interruptionMu.Unlock()

	if currentID == contextID1 {
		t.Error("Old audio should not keep old context active")
	}

	audioFrame2 := frames.NewTTSAudioFrame([]byte("audio 2"), 24000, 1)
	audioFrame2.SetMetadata("context_id", contextID2)
	if err := processor.HandleFrame(ctx, audioFrame2, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(audio 2) error: %v", err)
	}

	processor.interruptionMu.Lock()
	currentID = processor.currentContextID
	processor.interruptionMu.Unlock()

	if currentID != contextID2 {
		t.Errorf("Expected currentContextID=%s, got %s", contextID2, currentID)
	}
}

func TestMultipleInterruptions(t *testing.T) {
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:       8080,
		Path:       "/ws",
		Serializer: &mockSerializer{},
	})

	processor := transport.outputProc
	ctx := context.Background()

	startFrame := frames.NewStartFrameWithConfig(true, turns.UserTurnStrategies{})
	if err := processor.HandleFrame(ctx, startFrame, frames.Downstream); err != nil {
		t.Errorf("HandleFrame(StartFrame) error: %v", err)
	}

	for iteration := 0; iteration < 5; iteration++ {
		contextID := services.GenerateContextID()

		startFrame := frames.NewTTSStartedFrameWithContext(contextID)
		if err := processor.HandleFrame(ctx, startFrame, frames.Downstream); err != nil {
			t.Errorf("iteration %d: HandleFrame(TTSStartedFrame) error: %v", iteration, err)
		}

		audioFrame := frames.NewTTSAudioFrame([]byte("audio"), 24000, 1)
		audioFrame.SetMetadata("context_id", contextID)
		if err := processor.HandleFrame(ctx, audioFrame, frames.Downstream); err != nil {
			t.Errorf("iteration %d: HandleFrame(audio) error: %v", iteration, err)
		}

		processor.interruptionMu.Lock()
		currentID := processor.currentContextID
		processor.interruptionMu.Unlock()

		if currentID != contextID {
			t.Errorf("iteration %d: Expected currentContextID=%s, got %s", iteration, contextID, currentID)
		}

		interruptFrame := frames.NewInterruptionFrame()
		if err := processor.HandleFrame(ctx, interruptFrame, frames.Downstream); err != nil {
			t.Errorf("iteration %d: HandleFrame(InterruptionFrame) error: %v", iteration, err)
		}

		processor.interruptionMu.Lock()
		interrupted := processor.interrupted
		processor.interruptionMu.Unlock()

		if !interrupted {
			t.Errorf("iteration %d: Expected interrupted=true", iteration)
		}
	}
}
