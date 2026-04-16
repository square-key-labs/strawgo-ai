package serializers

import (
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// SerializerType defines the serialization format type
type SerializerType string

const (
	SerializerTypeBinary SerializerType = "binary"
	SerializerTypeText   SerializerType = "text"
)

// FrameSerializer is the interface for serializing and deserializing frames
// to/from protocol-specific formats (e.g., Twilio, Asterisk, Telnyx)
type FrameSerializer interface {
	// Type returns the serialization type (binary or text)
	Type() SerializerType

	// Setup initializes the serializer with startup configuration
	Setup(frame frames.Frame) error

	// Serialize converts a frame to its serialized representation
	// Returns the serialized data (string or bytes) and any error
	Serialize(frame frames.Frame) (interface{}, error)

	// Deserialize converts serialized data back to a frame
	// Accepts either string or []byte depending on serializer type
	Deserialize(data interface{}) (frames.Frame, error)

	// Cleanup releases any resources held by the serializer
	Cleanup() error
}

// PlaybackAckSerializer is implemented by serializers that support client-side
// playback acknowledgement. When the server signals playback-done (e.g., a Twilio
// mark message), the client echoes it back, allowing the transport to emit
// BotStoppedSpeakingFrame at actual playback completion rather than on server send.
type PlaybackAckSerializer interface {
	// SerializePlaybackDoneAck returns the message to send to the client to
	// request a playback-done acknowledgement (e.g., a Twilio mark event).
	SerializePlaybackDoneAck(correlationID string) (interface{}, error)
}
