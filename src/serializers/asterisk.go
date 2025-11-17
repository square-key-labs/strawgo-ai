package serializers

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// AsteriskFrameSerializer handles Asterisk WebSocket protocol
// Protocol: TEXT frames for control (MEDIA_START, HANGUP), BINARY frames for audio
// Codec is auto-detected from MEDIA_START message for true passthrough
type AsteriskFrameSerializer struct {
	channelID  string
	codec      string // Auto-detected from MEDIA_START, or fallback: "mulaw", "alaw", etc.
	sampleRate int    // Auto-detected from codec, or fallback: 8000
}

// Asterisk control message structure
// Format: MEDIA_START connection_id:xxx channel:xxx format:ulaw optimal_frame_size:160
type asteriskControlMessage struct {
	Type             string
	ConnectionID     string
	Channel          string
	Format           string // Codec: ulaw, alaw, slin, slin16, etc.
	OptimalFrameSize int
}

// AsteriskSerializerConfig holds configuration for Asterisk serializer
type AsteriskSerializerConfig struct {
	ChannelID  string
	Codec      string // Optional fallback codec if MEDIA_START not received: "mulaw"/"ulaw", "alaw" (default: "alaw")
	SampleRate int    // Optional fallback sample rate (default: 8000)
}

// NewAsteriskFrameSerializer creates a new Asterisk serializer with codec auto-detection
// Codec will be auto-detected from MEDIA_START message; config provides fallback values
func NewAsteriskFrameSerializer(config AsteriskSerializerConfig) *AsteriskFrameSerializer {
	codec := config.Codec
	if codec == "" {
		codec = "alaw" // Default fallback to A-law (common in Europe/Asterisk)
	}

	sampleRate := config.SampleRate
	if sampleRate == 0 {
		sampleRate = 8000 // Telephony standard
	}

	return &AsteriskFrameSerializer{
		channelID:  config.ChannelID,
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
	case "slin":
		return "linear16"
	case "slin16":
		return "linear16"
	default:
		return codec
	}
}

// parseControlMessage parses Asterisk plain text control messages
// Format: MEDIA_START connection_id:xxx channel:xxx format:ulaw optimal_frame_size:160
func parseControlMessage(text string) (*asteriskControlMessage, error) {
	parts := strings.Fields(text)
	if len(parts) == 0 {
		return nil, fmt.Errorf("empty control message")
	}

	msg := &asteriskControlMessage{
		Type: parts[0],
	}

	// Parse key:value pairs
	for i := 1; i < len(parts); i++ {
		kv := strings.SplitN(parts[i], ":", 2)
		if len(kv) != 2 {
			continue
		}
		key, value := kv[0], kv[1]

		switch key {
		case "connection_id":
			msg.ConnectionID = value
		case "channel":
			msg.Channel = value
		case "format":
			msg.Format = value
		case "optimal_frame_size":
			if size, err := strconv.Atoi(value); err == nil {
				msg.OptimalFrameSize = size
			}
		}
	}

	return msg, nil
}

// Type returns the serialization type
// Asterisk WebSocket always uses BINARY frames for audio
func (s *AsteriskFrameSerializer) Type() SerializerType {
	return SerializerTypeBinary
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
// Asterisk WebSocket protocol: BINARY frames for audio, TEXT frames for control messages
func (s *AsteriskFrameSerializer) Serialize(frame frames.Frame) (interface{}, error) {
	switch f := frame.(type) {
	case *frames.InterruptionFrame:
		// Send REPORT_QUEUE_DRAINED first to get feedback, then FLUSH_MEDIA to clear queue
		// Reference: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
		// REPORT_QUEUE_DRAINED: Tells Asterisk to send QUEUE_DRAINED event when queue is empty
		// FLUSH_MEDIA: Clears any queued but not yet sent audio chunks
		// Return as slice of commands to be sent in sequence
		commands := []string{"REPORT_QUEUE_DRAINED", "FLUSH_MEDIA"}
		fmt.Printf("[AsteriskSerializer] 🔴 Sending interruption commands: %v\n", commands)
		return commands, nil

	case *frames.AudioFrame:
		// Send raw codec bytes as BINARY frame
		// Asterisk expects raw audio data without any wrapper
		return f.Data, nil

	case *frames.TTSAudioFrame:
		// TTS audio also sent as raw binary
		return f.Data, nil

	default:
		// Ignore other frame types
		// Asterisk protocol doesn't expect other control messages from client
		return nil, nil
	}
}

// Deserialize converts Asterisk data to frames
// TEXT frames: Control messages (MEDIA_START, HANGUP, etc.)
// BINARY frames: Raw audio in native codec (passthrough to STT)
func (s *AsteriskFrameSerializer) Deserialize(data interface{}) (frames.Frame, error) {
	// Check if this is a TEXT control message
	if str, ok := data.(string); ok {
		fmt.Printf("[AsteriskSerializer] 📥 Received TEXT message: '%s'\n", str)
		// Parse plain text control message
		msg, err := parseControlMessage(str)
		if err != nil {
			return nil, fmt.Errorf("failed to parse control message: %w", err)
		}

		switch msg.Type {
		case "MEDIA_START":
			// Extract codec and channel from MEDIA_START message
			if msg.Format != "" {
				s.codec = normalizeAsteriskCodec(msg.Format)
			}
			if msg.Channel != "" {
				s.channelID = msg.Channel
			}
			// Update sample rate based on codec
			switch s.codec {
			case "mulaw", "alaw":
				s.sampleRate = 8000
			case "linear16":
				s.sampleRate = 16000
			}

			fmt.Printf("[AsteriskSerializer] ✅ MEDIA_START: codec=%s, channel=%s, rate=%d\n", s.codec, s.channelID, s.sampleRate)

			// DON'T create a new StartFrame - it would overwrite interruption settings from pipeline
			// MEDIA_START just updates our internal state for codec detection
			// Return nil to consume this control message without emitting a frame
			return nil, nil

		case "HANGUP":
			fmt.Printf("[AsteriskSerializer] 🔴 HANGUP received\n")
			endFrame := frames.NewEndFrame()
			endFrame.SetMetadata("channelID", s.channelID)
			return endFrame, nil

		case "MEDIA_XON":
			fmt.Printf("[AsteriskSerializer] ✅ MEDIA_XON: Resume sending (buffer below threshold)\n")
			// Flow control: Resume sending
			return nil, nil

		case "MEDIA_XOFF":
			fmt.Printf("[AsteriskSerializer] ⚠️  MEDIA_XOFF: Pause sending (buffer full ~900 frames)\n")
			// Flow control: Pause sending
			return nil, nil

		case "MEDIA_BUFFERING_COMPLETED":
			fmt.Printf("[AsteriskSerializer] ✅ MEDIA_BUFFERING_COMPLETED\n")
			return nil, nil

		case "QUEUE_DRAINED":
			fmt.Printf("[AsteriskSerializer] ✅ QUEUE_DRAINED: Audio queue has been flushed successfully\n")
			return nil, nil

		default:
			// Unknown control message, log and ignore
			fmt.Printf("[AsteriskSerializer] ⚠️  Unknown control message: %s\n", msg.Type)
			return nil, nil
		}
	}

	// This is a BINARY frame - raw audio data
	var audioData []byte
	switch v := data.(type) {
	case []byte:
		audioData = v
	case string:
		audioData = []byte(v)
	default:
		return nil, fmt.Errorf("expected []byte or string, got %T", data)
	}

	// Passthrough: Create AudioFrame with native codec data
	// STT service (e.g., Deepgram) will handle decoding
	audioFrame := frames.NewAudioFrame(audioData, s.sampleRate, 1)
	audioFrame.SetMetadata("codec", s.codec)
	audioFrame.SetMetadata("channelID", s.channelID)
	audioFrame.SetMetadata("passthrough", true) // Indicate no conversion needed
	return audioFrame, nil
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
