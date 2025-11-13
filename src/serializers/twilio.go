package serializers

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// TwilioFrameSerializer handles Twilio Media Streams WebSocket protocol
type TwilioFrameSerializer struct {
	streamSid string
	callSid   string
}

// Twilio message structures
type twilioMessage struct {
	Event     string                 `json:"event"`
	StreamSid string                 `json:"streamSid,omitempty"`
	Media     *twilioMedia           `json:"media,omitempty"`
	Start     *twilioStart           `json:"start,omitempty"`
	Mark      *twilioMark            `json:"mark,omitempty"`
	Stop      map[string]interface{} `json:"stop,omitempty"`
}

type twilioMedia struct {
	Track     string `json:"track"`
	Chunk     string `json:"chunk"`
	Timestamp string `json:"timestamp"`
	Payload   string `json:"payload"` // base64-encoded mulaw audio
}

type twilioStart struct {
	StreamSid        string                 `json:"streamSid"`
	CallSid          string                 `json:"callSid"`
	AccountSid       string                 `json:"accountSid"`
	Tracks           []string               `json:"tracks"`
	MediaFormat      map[string]interface{} `json:"mediaFormat"`
	CustomParameters map[string]string      `json:"customParameters,omitempty"`
}

type twilioMark struct {
	Name string `json:"name"`
}

// NewTwilioFrameSerializer creates a new Twilio serializer
func NewTwilioFrameSerializer(streamSid, callSid string) *TwilioFrameSerializer {
	return &TwilioFrameSerializer{
		streamSid: streamSid,
		callSid:   callSid,
	}
}

// Type returns the serialization type (Twilio uses JSON/text)
func (s *TwilioFrameSerializer) Type() SerializerType {
	return SerializerTypeText
}

// Setup initializes the serializer with startup configuration
func (s *TwilioFrameSerializer) Setup(frame frames.Frame) error {
	// Can extract streamSid/callSid from StartFrame metadata if needed
	return nil
}

// Serialize converts a frame to Twilio WebSocket JSON format
func (s *TwilioFrameSerializer) Serialize(frame frames.Frame) (interface{}, error) {
	switch f := frame.(type) {
	case *frames.AudioFrame:
		// Encode audio data (mulaw) to base64
		payload := base64.StdEncoding.EncodeToString(f.Data)

		msg := twilioMessage{
			Event:     "media",
			StreamSid: s.streamSid,
			Media: &twilioMedia{
				Payload: payload,
			},
		}

		data, err := json.Marshal(msg)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal Twilio media message: %w", err)
		}
		return string(data), nil

	case *frames.InterruptionFrame:
		// Send clear event to stop audio playback
		msg := twilioMessage{
			Event:     "clear",
			StreamSid: s.streamSid,
		}

		data, err := json.Marshal(msg)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal Twilio clear message: %w", err)
		}
		return string(data), nil

	case *frames.EndFrame:
		// Twilio doesn't have a specific end frame, return nil
		return nil, nil

	default:
		// Ignore other frame types
		return nil, nil
	}
}

// Deserialize converts Twilio WebSocket JSON data to frames
func (s *TwilioFrameSerializer) Deserialize(data interface{}) (frames.Frame, error) {
	jsonData, ok := data.(string)
	if !ok {
		// Try []byte
		if bytes, ok := data.([]byte); ok {
			jsonData = string(bytes)
		} else {
			return nil, fmt.Errorf("expected string or []byte, got %T", data)
		}
	}

	var msg twilioMessage
	if err := json.Unmarshal([]byte(jsonData), &msg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Twilio message: %w", err)
	}

	switch msg.Event {
	case "start":
		// Update streamSid and callSid from start message
		if msg.Start != nil {
			s.streamSid = msg.Start.StreamSid
			s.callSid = msg.Start.CallSid
		}

		// Create StartFrame with metadata
		startFrame := frames.NewStartFrame()
		startFrame.SetMetadata("streamSid", s.streamSid)
		startFrame.SetMetadata("callSid", s.callSid)
		if msg.Start != nil {
			startFrame.SetMetadata("accountSid", msg.Start.AccountSid)
		}
		return startFrame, nil

	case "media":
		if msg.Media == nil {
			return nil, fmt.Errorf("media event missing media data")
		}

		// Decode base64 mulaw audio
		audioData, err := base64.StdEncoding.DecodeString(msg.Media.Payload)
		if err != nil {
			return nil, fmt.Errorf("failed to decode audio payload: %w", err)
		}

		// Create AudioFrame with mulaw data
		// Twilio uses 8kHz mulaw
		audioFrame := frames.NewAudioFrame(audioData, 8000, 1)
		audioFrame.SetMetadata("codec", "mulaw")
		audioFrame.SetMetadata("streamSid", s.streamSid)
		return audioFrame, nil

	case "stop":
		// Call ended
		endFrame := frames.NewEndFrame()
		endFrame.SetMetadata("streamSid", s.streamSid)
		return endFrame, nil

	case "mark":
		// Mark events are used for synchronization, can be ignored or handled
		return nil, nil

	default:
		// Unknown event, ignore
		return nil, nil
	}
}

// Cleanup releases any resources (none for Twilio serializer)
func (s *TwilioFrameSerializer) Cleanup() error {
	return nil
}

// GetStreamSid returns the current stream SID
func (s *TwilioFrameSerializer) GetStreamSid() string {
	return s.streamSid
}

// GetCallSid returns the current call SID
func (s *TwilioFrameSerializer) GetCallSid() string {
	return s.callSid
}
