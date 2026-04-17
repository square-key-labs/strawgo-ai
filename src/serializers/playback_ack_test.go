package serializers

import (
	"strings"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func TestTwilioSerializePlaybackDoneAck(t *testing.T) {
	serializer := NewTwilioFrameSerializer("stream-123", "call-456")

	data, err := serializer.SerializePlaybackDoneAck("playback-123")
	if err != nil {
		t.Fatalf("SerializePlaybackDoneAck() error = %v", err)
	}

	msg, ok := data.(string)
	if !ok {
		t.Fatalf("SerializePlaybackDoneAck() type = %T, want string", data)
	}

	for _, want := range []string{`"event":"mark"`, `"streamSid":"stream-123"`, `"name":"playback-123"`} {
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

	if got := frame.Metadata()["correlation_id"]; got != "playback-done" {
		t.Fatalf("Deserialize(mark) correlation_id = %v, want playback-done", got)
	}
}

func TestAsteriskSerializePlaybackDoneAck(t *testing.T) {
	serializer := NewAsteriskFrameSerializer(AsteriskSerializerConfig{})

	data, err := serializer.SerializePlaybackDoneAck("playback-789")
	if err != nil {
		t.Fatalf("SerializePlaybackDoneAck() error = %v", err)
	}

	msg, ok := data.(string)
	if !ok {
		t.Fatalf("SerializePlaybackDoneAck() type = %T, want string", data)
	}

	if msg != "MARK_MEDIA correlation_id:playback-789" {
		t.Fatalf("SerializePlaybackDoneAck() = %q, want MARK_MEDIA correlation_id:playback-789", msg)
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

	if got := frame.Metadata()["correlation_id"]; got != "queue-drained" {
		t.Fatalf("Deserialize(QUEUE_DRAINED) correlation_id = %v, want queue-drained", got)
	}
}

func TestAsteriskDeserializeMediaMarkProcessedReturnsPlaybackCompleteFrame(t *testing.T) {
	serializer := NewAsteriskFrameSerializer(AsteriskSerializerConfig{})

	frame, err := serializer.Deserialize("MEDIA_MARK_PROCESSED correlation_id:playback-789")
	if err != nil {
		t.Fatalf("Deserialize(MEDIA_MARK_PROCESSED) error = %v", err)
	}

	if _, ok := frame.(*frames.PlaybackCompleteFrame); !ok {
		t.Fatalf("Deserialize(MEDIA_MARK_PROCESSED) frame = %T, want *frames.PlaybackCompleteFrame", frame)
	}

	if got := frame.Metadata()["correlation_id"]; got != "playback-789" {
		t.Fatalf("Deserialize(MEDIA_MARK_PROCESSED) correlation_id = %v, want playback-789", got)
	}
}
