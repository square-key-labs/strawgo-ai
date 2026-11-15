package serializers

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

const (
	pbText          protowire.Number = 1
	pbAudio         protowire.Number = 2
	pbTranscription protowire.Number = 3
	pbMessage       protowire.Number = 4
	pbInterruption  protowire.Number = 5
)

type ProtobufFrameSerializer struct {
	inner []byte
}

func NewProtobufFrameSerializer() *ProtobufFrameSerializer {
	return &ProtobufFrameSerializer{}
}

func (s *ProtobufFrameSerializer) wrap(dst []Message, field protowire.Number) []Message {
	buf := make([]byte, 0, protowire.SizeTag(field)+protowire.SizeBytes(len(s.inner)))
	buf = protowire.AppendTag(buf, field, protowire.BytesType)
	buf = protowire.AppendBytes(buf, s.inner)
	return append(dst, Message{Type: BinaryMessage, Data: buf})
}

func (s *ProtobufFrameSerializer) header(id uint64, name string) {
	s.inner = s.inner[:0]
	if id != 0 {
		s.inner = protowire.AppendTag(s.inner, 1, protowire.VarintType)
		s.inner = protowire.AppendVarint(s.inner, id)
	}
	s.appendString(2, name)
}

func (s *ProtobufFrameSerializer) appendString(num protowire.Number, v string) {
	if v == "" {
		return
	}
	s.inner = protowire.AppendTag(s.inner, num, protowire.BytesType)
	s.inner = protowire.AppendString(s.inner, v)
}

func (s *ProtobufFrameSerializer) appendVarint(num protowire.Number, v uint64) {
	if v == 0 {
		return
	}
	s.inner = protowire.AppendTag(s.inner, num, protowire.VarintType)
	s.inner = protowire.AppendVarint(s.inner, v)
}

func (s *ProtobufFrameSerializer) SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error) {
	s.header(0, "OutputAudioRawFrame")
	if len(pcm) > 0 {
		s.inner = protowire.AppendTag(s.inner, 3, protowire.BytesType)
		s.inner = protowire.AppendBytes(s.inner, pcm)
	}
	s.appendVarint(4, uint64(sampleRate))
	s.appendVarint(5, 1)
	return s.wrap(dst, pbAudio), nil
}

func (s *ProtobufFrameSerializer) Serialize(dst []Message, frame frames.Frame) ([]Message, error) {
	switch f := frame.(type) {
	case *frames.TextFrame:
		s.header(f.ID(), f.Name())
		s.appendString(3, f.Text)
		return s.wrap(dst, pbText), nil
	case *frames.TranscriptionFrame:
		s.header(f.ID(), f.Name())
		s.appendString(3, f.Text)
		if uid, ok := f.Metadata()["user_id"].(string); ok {
			s.appendString(4, uid)
		}
		if !f.Timestamp.IsZero() {
			s.appendString(5, f.Timestamp.UTC().Format(time.RFC3339Nano))
		}
		return s.wrap(dst, pbTranscription), nil
	case *frames.InterruptionFrame:
		s.header(f.ID(), f.Name())
		return s.wrap(dst, pbInterruption), nil
	case *frames.OutputTransportMessageFrame, *frames.OutputTransportMessageUrgentFrame:
		var msg any
		if m, ok := f.(*frames.OutputTransportMessageFrame); ok {
			msg = m.Message
		} else {
			msg = f.(*frames.OutputTransportMessageUrgentFrame).Message
		}
		data, err := json.Marshal(msg)
		if err != nil {
			return dst, err
		}
		s.inner = s.inner[:0]
		s.inner = protowire.AppendTag(s.inner, 1, protowire.BytesType)
		s.inner = protowire.AppendBytes(s.inner, data)
		return s.wrap(dst, pbMessage), nil
	}
	if out, ok, err := serializeAudioFrame(s, nil, dst, frame, 0); ok {
		return out, err
	}
	return dst, nil
}

type pbFields struct {
	data      string
	name      string
	text      string
	userID    string
	timestamp string
	audio     []byte
	rate      uint64
	channels  uint64
}

func parsePBFields(b []byte) (pbFields, error) {
	var f pbFields
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return f, protowire.ParseError(n)
		}
		b = b[n:]
		switch {
		case typ == protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			if m < 0 {
				return f, protowire.ParseError(m)
			}
			b = b[m:]
			switch num {
			case 1:
				f.data = string(v)
			case 2:
				f.name = string(v)
			case 3:
				f.text = string(v)
				f.audio = v
			case 4:
				f.userID = string(v)
			case 5:
				f.timestamp = string(v)
			}
		case typ == protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			if m < 0 {
				return f, protowire.ParseError(m)
			}
			b = b[m:]
			switch num {
			case 4:
				f.rate = v
			case 5:
				f.channels = v
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			if m < 0 {
				return f, protowire.ParseError(m)
			}
			b = b[m:]
		}
	}
	return f, nil
}

func (s *ProtobufFrameSerializer) Deserialize(msg Message) (frames.Frame, error) {
	b := msg.Data
	num, typ, n := protowire.ConsumeTag(b)
	if n < 0 {
		return nil, fmt.Errorf("protobuf: %w", protowire.ParseError(n))
	}
	if typ != protowire.BytesType {
		return nil, errors.New("protobuf: unexpected wire type")
	}
	inner, m := protowire.ConsumeBytes(b[n:])
	if m < 0 {
		return nil, fmt.Errorf("protobuf: %w", protowire.ParseError(m))
	}
	f, err := parsePBFields(inner)
	if err != nil {
		return nil, fmt.Errorf("protobuf: %w", err)
	}
	switch num {
	case pbText:
		return frames.NewTextFrame(f.text), nil
	case pbAudio:
		channels := int(f.channels)
		if channels == 0 {
			channels = 1
		}
		data := append([]byte(nil), f.audio...)
		out := frames.NewAudioFrame(data, int(f.rate), channels)
		out.SetMetadata("codec", "linear16")
		return out, nil
	case pbTranscription:
		out := frames.NewTranscriptionFrame(f.text, true)
		if f.userID != "" {
			out.SetMetadata("user_id", f.userID)
		}
		if ts, err := time.Parse(time.RFC3339Nano, f.timestamp); err == nil {
			out.Timestamp = ts
		}
		return out, nil
	case pbMessage:
		var payload any
		if err := json.Unmarshal([]byte(f.data), &payload); err != nil {
			return nil, fmt.Errorf("protobuf message: %w", err)
		}
		return frames.NewInputTransportMessageFrame(payload), nil
	case pbInterruption:
		return frames.NewInterruptionFrame(), nil
	}
	return nil, nil
}
