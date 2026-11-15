package serializers

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type ExotelConfig struct {
	StreamSID  string
	CallSID    string
	SampleRate int
}

type ExotelFrameSerializer struct {
	cfg    ExotelConfig
	info   CallInfo
	rate   int
	prefix []byte
	rs     audio.Resampler
	rsBuf  []byte
}

type exotelMessage struct {
	Event     string `json:"event"`
	StreamSID string `json:"stream_sid"`
	Start     *struct {
		StreamSID        string            `json:"stream_sid"`
		CallSID          string            `json:"call_sid"`
		AccountSID       string            `json:"account_sid"`
		From             string            `json:"from"`
		To               string            `json:"to"`
		CustomParameters map[string]string `json:"custom_parameters"`
		MediaFormat      struct {
			SampleRate json.RawMessage `json:"sample_rate"`
		} `json:"media_format"`
	} `json:"start"`
	Media *struct {
		Payload string `json:"payload"`
	} `json:"media"`
	DTMF *struct {
		Digit string `json:"digit"`
	} `json:"dtmf"`
	Mark *struct {
		Name string `json:"name"`
	} `json:"mark"`
}

func NewExotelFrameSerializer(cfg ExotelConfig) *ExotelFrameSerializer {
	rate := cfg.SampleRate
	if rate <= 0 {
		rate = 8000
	}
	return &ExotelFrameSerializer{
		cfg:  cfg,
		rate: rate,
		info: CallInfo{Provider: "exotel", StreamID: cfg.StreamSID, CallID: cfg.CallSID, Encoding: audio.CodecLinear16, SampleRate: rate},
	}
}

func (s *ExotelFrameSerializer) WireSampleRates() (int, int) { return s.rate, s.rate }

func (s *ExotelFrameSerializer) CallInfo() CallInfo { return s.info }

func (s *ExotelFrameSerializer) Handshake(msg Message) (bool, error) {
	var m exotelMessage
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return false, fmt.Errorf("exotel handshake: %w", err)
	}
	if m.Event != "start" || m.Start == nil {
		return false, nil
	}
	s.cfg.StreamSID = firstNonEmpty(m.Start.StreamSID, m.StreamSID, s.cfg.StreamSID)
	s.cfg.CallSID = firstNonEmpty(m.Start.CallSID, s.cfg.CallSID)
	s.info.StreamID = s.cfg.StreamSID
	s.info.CallID = s.cfg.CallSID
	s.info.AccountID = m.Start.AccountSID
	s.info.From = m.Start.From
	s.info.To = m.Start.To
	s.info.CustomParameters = m.Start.CustomParameters
	if r, err := strconv.Atoi(strings.Trim(string(m.Start.MediaFormat.SampleRate), `"`)); err == nil && r > 0 && s.cfg.SampleRate == 0 {
		s.rate = r
		s.info.SampleRate = r
	}
	s.prefix = nil
	return true, nil
}

func (s *ExotelFrameSerializer) SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error) {
	if sampleRate != s.rate {
		s.rsBuf = s.rs.Resample(s.rsBuf[:0], pcm, sampleRate, s.rate)
		pcm = s.rsBuf
	}
	if s.prefix == nil {
		s.prefix = append(s.prefix, `{"event":"media","stream_sid":`...)
		s.prefix = appendJSONString(s.prefix, s.cfg.StreamSID)
		s.prefix = append(s.prefix, `,"media":{"payload":"`...)
	}
	return appendMediaMessage(dst, s.prefix, pcm), nil
}

func (s *ExotelFrameSerializer) Serialize(dst []Message, frame frames.Frame) ([]Message, error) {
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.rs.Reset()
		data, err := json.Marshal(map[string]string{"event": "clear", "stream_sid": s.cfg.StreamSID})
		if err != nil {
			return dst, err
		}
		return append(dst, Message{Type: TextMessage, Data: data}), nil
	}
	if out, ok, err := serializeAudioFrame(s, &s.rs, dst, frame, s.rate); ok {
		return out, err
	}
	out, _, err := transportMessage(dst, frame, true)
	return out, err
}

func (s *ExotelFrameSerializer) SerializePlaybackAck(correlationID string) (Message, error) {
	data, err := json.Marshal(map[string]any{"event": "mark", "stream_sid": s.cfg.StreamSID, "mark": map[string]string{"name": correlationID}})
	return Message{Type: TextMessage, Data: data}, err
}

func (s *ExotelFrameSerializer) Deserialize(msg Message) (frames.Frame, error) {
	var m exotelMessage
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return nil, fmt.Errorf("exotel: %w", err)
	}
	switch m.Event {
	case "media":
		if m.Media == nil || m.Media.Payload == "" {
			return nil, nil
		}
		raw, err := decodeBase64(m.Media.Payload)
		if err != nil {
			return nil, fmt.Errorf("exotel media payload: %w", err)
		}
		if len(raw)%2 != 0 {
			raw = raw[:len(raw)-1]
		}
		return inputAudio(raw, s.rate), nil
	case "dtmf":
		if m.DTMF == nil {
			return nil, nil
		}
		return dtmfFrame(m.DTMF.Digit), nil
	case "mark":
		f := frames.NewPlaybackCompleteFrame()
		if m.Mark != nil {
			f.SetMetadata("correlation_id", m.Mark.Name)
		}
		return f, nil
	case "stop":
		return frames.NewEndFrame(), nil
	}
	return nil, nil
}
