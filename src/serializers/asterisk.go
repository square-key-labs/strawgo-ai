package serializers

import (
	"encoding/json"
	"fmt"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// AsteriskFrameSerializer handles Asterisk WebSocket protocol
// Passes audio through without conversion when TTS/STT support the same codec
type AsteriskFrameSerializer struct {
	channelID  string
	useBinary  bool // If true, send raw codec bytes; if false, use JSON wrapper
	codec      string // e.g., "mulaw", "alaw", "ulaw", "PCMU", "PCMA"
	sampleRate int    // e.g., 8000
}

// Asterisk JSON message structures
type asteriskMessage struct {
	Type      string                 `json:"type"`
	ChannelID string                 `json:"channel_id,omitempty"`
	Audio     string                 `json:"audio,omitempty"` // base64 encoded if JSON mode
	Data      map[string]interface{} `json:"data,omitempty"`
}

// AsteriskSerializerConfig holds configuration for Asterisk serializer
type AsteriskSerializerConfig struct {
	ChannelID  string
	UseBinary  bool   // If true, sends raw codec bytes; if false, wraps in JSON
	Codec      string // Supported: "mulaw"/"ulaw"/"PCMU", "alaw"/"PCMA" (default: "alaw")
	SampleRate int    // Sample rate in Hz (default: 8000)
}

// NewAsteriskFrameSerializer creates a new Asterisk serializer with codec passthrough
// Passthrough strategy: audio is passed as-is to services that support the codec
func NewAsteriskFrameSerializer(config AsteriskSerializerConfig) *AsteriskFrameSerializer {
	codec := config.Codec
	if codec == "" {
		codec = "alaw" // Default to A-law (common in Europe/Asterisk)
	}

	sampleRate := config.SampleRate
	if sampleRate == 0 {
		sampleRate = 8000 // Telephony standard
	}

	return &AsteriskFrameSerializer{
		channelID:  config.ChannelID,
		useBinary:  config.UseBinary,
		codec:      normalizeAsteriskCodec(codec),
		sampleRate: sampleRate,
	}
}

// normalizeAsteriskCodec normalizes codec names for consistency
func normalizeAsteriskCodec(codec string) string {
	switch codec {
	case "ulaw", "PCMU":
		return "mulaw"
	case "PCMA":
		return "alaw"
	default:
		return codec
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
// Sends audio in native codec (passthrough from TTS)
func (s *AsteriskFrameSerializer) Serialize(frame frames.Frame) (interface{}, error) {
	switch f := frame.(type) {
	case *frames.AudioFrame:
		// Passthrough: send audio as-is in native codec
		// TTS should output in matching codec for best performance
		if s.useBinary {
			// Send raw codec bytes (mulaw/alaw/etc)
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
// Receives audio in native codec (passthrough to STT)
func (s *AsteriskFrameSerializer) Deserialize(data interface{}) (frames.Frame, error) {
	if s.useBinary {
		// Check for TEXT control messages (Asterisk sometimes sends these)
		if str, ok := data.(string); ok && len(str) > 0 {
			// Could be a control message, not audio
			if str[0] == '{' || str[0] == '<' {
				// Skip JSON/XML control messages
				return nil, nil
			}
		}

		// Expecting raw codec bytes (mulaw/alaw/etc)
		var audioData []byte
		switch v := data.(type) {
		case []byte:
			audioData = v
		case string:
			audioData = []byte(v)
		default:
			return nil, fmt.Errorf("expected []byte or string for binary mode, got %T", data)
		}

		// Passthrough: Create AudioFrame with native codec data
		// STT service (e.g., Deepgram) will handle decoding
		audioFrame := frames.NewAudioFrame(audioData, s.sampleRate, 1)
		audioFrame.SetMetadata("codec", s.codec)
		audioFrame.SetMetadata("channelID", s.channelID)
		audioFrame.SetMetadata("passthrough", true) // Indicate no conversion needed
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

			// Passthrough: preserve native codec
			audioFrame := frames.NewAudioFrame(audioData, s.sampleRate, 1)
			audioFrame.SetMetadata("codec", s.codec)
			audioFrame.SetMetadata("channelID", s.channelID)
			audioFrame.SetMetadata("passthrough", true)
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

// GetCodec returns the configured codec
func (s *AsteriskFrameSerializer) GetCodec() string {
	return s.codec
}

// GetSampleRate returns the configured sample rate
func (s *AsteriskFrameSerializer) GetSampleRate() int {
	return s.sampleRate
}
