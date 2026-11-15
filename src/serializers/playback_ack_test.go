package serializers

import (
	"strings"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func TestTwilioSerializePlaybackAck(t *testing.T) {
	serializer := NewTwilioFrameSerializer(TwilioConfig{StreamSID: "stream-123", CallSID: "call-456"})

	data, err := serializer.SerializePlaybackAck("playback-123")
	if err != nil {
		t.Fatalf("SerializePlaybackAck() error = %v", err)
	}

	if data.Type != TextMessage {
		t.Fatalf("SerializePlaybackAck() type = %v, want text", data.Type)
	}
	msg := string(data.Data)

	for _, want := range []string{`"event":"mark"`, `"streamSid":"stream-123"`, `"name":"playback-123"`} {
		if !strings.Contains(msg, want) {
			t.Fatalf("serialized ack %q does not contain %s", msg, want)
		}
	}
}

func TestTwilioDeserializeMarkReturnsPlaybackCompleteFrame(t *testing.T) {
	serializer := NewTwilioFrameSerializer(TwilioConfig{StreamSID: "stream-123", CallSID: "call-456"})

	frame, err := serializer.Deserialize(Text(`{"event":"mark","streamSid":"stream-123","mark":{"name":"playback-done"}}`))
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

func TestAsteriskSerializePlaybackAck(t *testing.T) {
	serializer := NewAsteriskFrameSerializer(AsteriskSerializerConfig{})

	data, err := serializer.SerializePlaybackAck("playback-789")
	if err != nil {
		t.Fatalf("SerializePlaybackAck() error = %v", err)
	}

	if data.Type != TextMessage {
		t.Fatalf("SerializePlaybackAck() type = %v, want text", data.Type)
	}
	msg := string(data.Data)

	if msg != "MARK_MEDIA correlation_id:playback-789" {
		t.Fatalf("SerializePlaybackAck() = %q, want MARK_MEDIA correlation_id:playback-789", msg)
	}
}

func TestAsteriskDeserializeQueueDrainedReturnsPlaybackCompleteFrame(t *testing.T) {
	serializer := NewAsteriskFrameSerializer(AsteriskSerializerConfig{})

	frame, err := serializer.Deserialize(Text("QUEUE_DRAINED"))
	if err != nil {
		t.Fatalf("Deserialize(QUEUE_DRAINED) error = %v", err)
	}

	if _, ok := frame.(*frames.PlaybackCompleteFrame); !ok {
		t.Fatalf("Deserialize(QUEUE_DRAINED) frame = %T, want *frames.PlaybackCompleteFrame", frame)
	}

	if got, ok := frame.Metadata()["correlation_id"]; ok {
		t.Fatalf("Deserialize(QUEUE_DRAINED) correlation_id = %v, want none", got)
	}
	if drained, _ := frame.Metadata()[frames.MetadataQueueDrained].(bool); !drained {
		t.Fatal("Deserialize(QUEUE_DRAINED) is not marked as a queue drain")
	}
}

func TestAsteriskDeserializeMediaMarkProcessedReturnsPlaybackCompleteFrame(t *testing.T) {
	serializer := NewAsteriskFrameSerializer(AsteriskSerializerConfig{})

	frame, err := serializer.Deserialize(Text("MEDIA_MARK_PROCESSED correlation_id:playback-789"))
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
