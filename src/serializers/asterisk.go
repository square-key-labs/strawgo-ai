package serializers

import (
	"encoding/json"
	"fmt"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// AsteriskFrameSerializer handles Asterisk WebSocket protocol
// Asterisk can use either binary mulaw frames or JSON control messages
type AsteriskFrameSerializer struct {
	channelID string
	useBinary bool // If true, send raw mulaw; if false, use JSON wrapper
}

// Asterisk JSON message structures
type asteriskMessage struct {
	Type      string                 `json:"type"`
	ChannelID string                 `json:"channel_id,omitempty"`
	Audio     string                 `json:"audio,omitempty"` // base64 encoded if JSON mode
	Data      map[string]interface{} `json:"data,omitempty"`
}

// NewAsteriskFrameSerializer creates a new Asterisk serializer
// useBinary: if true, sends raw mulaw bytes; if false, wraps in JSON
func NewAsteriskFrameSerializer(channelID string, useBinary bool) *AsteriskFrameSerializer {
	return &AsteriskFrameSerializer{
		channelID: channelID,
		useBinary: useBinary,
	}
}

// Type returns the serialization type
func (s *AsteriskFrameSerializer) Type() SerializerType {
	if s.useBinary {
		return SerializerTypeBinary
	}
	return SerializerTypeText
}

// Setup initializes the serializer
func (s *AsteriskFrameSerializer) Setup(frame frames.Frame) error {
	// Can extract channelID from StartFrame metadata if needed
	if frame != nil {
		if meta := frame.Metadata(); meta != nil {
			if channelID, ok := meta["channelID"].(string); ok {
				s.channelID = channelID
			}
		}
	}
	return nil
}

// Serialize converts a frame to Asterisk format
func (s *AsteriskFrameSerializer) Serialize(frame frames.Frame) (interface{}, error) {
	switch f := frame.(type) {
	case *frames.AudioFrame:
		if s.useBinary {
			// Send raw mulaw bytes
			return f.Data, nil
		} else {
			// Wrap in JSON
			msg := asteriskMessage{
				Type:      "audio",
				ChannelID: s.channelID,
				Audio:     string(f.Data), // Or base64 encode if needed
			}
			data, err := json.Marshal(msg)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal Asterisk message: %w", err)
			}
			return string(data), nil
		}

	case *frames.InterruptionFrame:
		// Send control message to interrupt
		msg := asteriskMessage{
			Type:      "interrupt",
			ChannelID: s.channelID,
		}
		data, err := json.Marshal(msg)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal Asterisk interrupt: %w", err)
		}
		return string(data), nil

	case *frames.EndFrame:
		// Send hangup message
		msg := asteriskMessage{
			Type:      "hangup",
			ChannelID: s.channelID,
		}
		data, err := json.Marshal(msg)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal Asterisk hangup: %w", err)
		}
		return string(data), nil

	default:
		// Ignore other frame types
		return nil, nil
	}
}

// Deserialize converts Asterisk data to frames
func (s *AsteriskFrameSerializer) Deserialize(data interface{}) (frames.Frame, error) {
	if s.useBinary {
		// Expecting raw mulaw bytes
		var audioData []byte
		switch v := data.(type) {
		case []byte:
			audioData = v
		case string:
			audioData = []byte(v)
		default:
			return nil, fmt.Errorf("expected []byte or string for binary mode, got %T", data)
		}

		// Create AudioFrame with mulaw data
		// Asterisk typically uses 8kHz mulaw
		audioFrame := frames.NewAudioFrame(audioData, 8000, 1)
		audioFrame.SetMetadata("codec", "mulaw")
		audioFrame.SetMetadata("channelID", s.channelID)
		return audioFrame, nil

	} else {
		// Expecting JSON
		jsonData, ok := data.(string)
		if !ok {
			if bytes, ok := data.([]byte); ok {
				jsonData = string(bytes)
			} else {
				return nil, fmt.Errorf("expected string or []byte for JSON mode, got %T", data)
			}
		}

		var msg asteriskMessage
		if err := json.Unmarshal([]byte(jsonData), &msg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal Asterisk message: %w", err)
		}

		switch msg.Type {
		case "start":
			if msg.ChannelID != "" {
				s.channelID = msg.ChannelID
			}
			startFrame := frames.NewStartFrame()
			startFrame.SetMetadata("channelID", s.channelID)
			return startFrame, nil

		case "audio":
			// Audio data in msg.Audio
			audioData := []byte(msg.Audio) // Or base64 decode if needed

			audioFrame := frames.NewAudioFrame(audioData, 8000, 1)
			audioFrame.SetMetadata("codec", "mulaw")
			audioFrame.SetMetadata("channelID", s.channelID)
			return audioFrame, nil

		case "hangup":
			endFrame := frames.NewEndFrame()
			endFrame.SetMetadata("channelID", s.channelID)
			return endFrame, nil

		default:
			// Unknown message type
			return nil, nil
		}
	}
}

// Cleanup releases any resources
func (s *AsteriskFrameSerializer) Cleanup() error {
	return nil
}

// GetChannelID returns the current channel ID
func (s *AsteriskFrameSerializer) GetChannelID() string {
	return s.channelID
}
