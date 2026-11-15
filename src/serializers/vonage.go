package serializers

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type VonageConfig struct {
	SampleRate int
}

type VonageFrameSerializer struct {
	rate  int
	fixed bool
	info  CallInfo
	rs    audio.Resampler
}

func NewVonageFrameSerializer(cfg VonageConfig) *VonageFrameSerializer {
	rate := cfg.SampleRate
	if rate <= 0 {
		rate = 16000
	}
	return &VonageFrameSerializer{
		rate:  rate,
		fixed: cfg.SampleRate > 0,
		info:  CallInfo{Provider: "vonage", Encoding: audio.CodecLinear16, SampleRate: rate},
	}
}

func (s *VonageFrameSerializer) WireSampleRates() (int, int) { return s.rate, s.rate }

func (s *VonageFrameSerializer) CallInfo() CallInfo { return s.info }

func (s *VonageFrameSerializer) Handshake(msg Message) (bool, error) {
	if msg.Type == BinaryMessage {
		return true, nil
	}
	var m map[string]any
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return false, fmt.Errorf("vonage handshake: %w", err)
	}
	if ev, _ := m["event"].(string); ev != "websocket:connected" {
		return false, nil
	}
	params := map[string]string{}
	for k, v := range m {
		switch k {
		case "event":
		case "content-type":
			ct, _ := v.(string)
			if i := strings.Index(ct, "rate="); i >= 0 && !s.fixed {
				if r, err := strconv.Atoi(strings.TrimSpace(ct[i+5:])); err == nil && r > 0 {
					s.rate = r
					s.info.SampleRate = r
				}
			}
		default:
			if str, ok := v.(string); ok {
				params[k] = str
			}
		}
	}
	if len(params) > 0 {
		s.info.CustomParameters = params
		s.info.CallID = firstNonEmpty(params["call_uuid"], params["uuid"], params["conversation_uuid"])
	}
	return true, nil
}

func (s *VonageFrameSerializer) SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error) {
	if sampleRate != s.rate {
		return append(dst, Binary(s.rs.Resample(make([]byte, 0, s.rs.MaxOutputBytes(len(pcm), sampleRate, s.rate)), pcm, sampleRate, s.rate))), nil
	}
	return append(dst, Binary(pcm)), nil
}

func (s *VonageFrameSerializer) Serialize(dst []Message, frame frames.Frame) ([]Message, error) {
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.rs.Reset()
		return append(dst, Text(`{"action":"clear"}`)), nil
	}
	if out, ok, err := serializeAudioFrame(s, &s.rs, dst, frame, s.rate); ok {
		return out, err
	}
	out, _, err := transportMessage(dst, frame, true)
	return out, err
}

func (s *VonageFrameSerializer) SerializePlaybackAck(correlationID string) (Message, error) {
	data, err := json.Marshal(map[string]any{"action": "notify", "payload": map[string]string{"name": correlationID}})
	return Message{Type: TextMessage, Data: data}, err
}

func (s *VonageFrameSerializer) Deserialize(msg Message) (frames.Frame, error) {
	if msg.Type == BinaryMessage {
		data := msg.Data
		if len(data)%2 != 0 {
			data = data[:len(data)-1]
		}
		return inputAudio(data, s.rate), nil
	}
	var m struct {
		Event string `json:"event"`
		Digit any    `json:"digit"`
		DTMF  *struct {
			Digit any `json:"digit"`
		} `json:"dtmf"`
		Payload *struct {
			Name string `json:"name"`
		} `json:"payload"`
	}
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return nil, fmt.Errorf("vonage: %w", err)
	}
	switch m.Event {
	case "websocket:dtmf":
		digit := m.Digit
		if digit == nil && m.DTMF != nil {
			digit = m.DTMF.Digit
		}
		if digit == nil {
			return nil, nil
		}
		return dtmfFrame(fmt.Sprint(digit)), nil
	case "websocket:notify":
		f := frames.NewPlaybackCompleteFrame()
		if m.Payload != nil {
			f.SetMetadata("correlation_id", m.Payload.Name)
		}
		return f, nil
	}
	return nil, nil
}
