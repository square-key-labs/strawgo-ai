package serializers

import (
	"strings"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func TestTwilioSerializePlaybackDoneAck(t *testing.T) {
	serializer := NewTwilioFrameSerializer("stream-123", "call-456")

	data, err := serializer.SerializePlaybackDoneAck()
	if err != nil {
		t.Fatalf("SerializePlaybackDoneAck() error = %v", err)
	}

	msg, ok := data.(string)
	if !ok {
		t.Fatalf("SerializePlaybackDoneAck() type = %T, want string", data)
	}

	for _, want := range []string{`"event":"mark"`, `"streamSid":"stream-123"`, `"name":"playback-done"`} {
		if !strings.Contains(msg, want) {
			t.Fatalf("serialized ack %q does not contain %s", msg, want)
		}
	}
}

func TestTwilioDeserializeMarkReturnsPlaybackCompleteFrame(t *testing.T) {
	serializer := NewTwilioFrameSerializer("stream-123", "call-456")

	frame, err := serializer.Deserialize(`{"event":"mark","streamSid":"stream-123","mark":{"name":"playback-done"}}`)
	if err != nil {
		t.Fatalf("Deserialize(mark) error = %v", err)
	}

	if _, ok := frame.(*frames.PlaybackCompleteFrame); !ok {
		t.Fatalf("Deserialize(mark) frame = %T, want *frames.PlaybackCompleteFrame", frame)
	}
}

func TestAsteriskDeserializeQueueDrainedReturnsPlaybackCompleteFrame(t *testing.T) {
	serializer := NewAsteriskFrameSerializer(AsteriskSerializerConfig{})

	frame, err := serializer.Deserialize("QUEUE_DRAINED")
	if err != nil {
		t.Fatalf("Deserialize(QUEUE_DRAINED) error = %v", err)
	}

	if _, ok := frame.(*frames.PlaybackCompleteFrame); !ok {
		t.Fatalf("Deserialize(QUEUE_DRAINED) frame = %T, want *frames.PlaybackCompleteFrame", frame)
	}
}
