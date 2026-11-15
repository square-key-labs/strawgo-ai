package serializers

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type RTVIClientSerializer struct {
	nextID uint64
}

func NewRTVIClientSerializer() *RTVIClientSerializer {
	return &RTVIClientSerializer{}
}

func (s *RTVIClientSerializer) SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error) {
	s.nextID++
	buf := make([]byte, 0, 128+base64.StdEncoding.EncodedLen(len(pcm)))
	buf = append(buf, `{"label":"rtvi-ai","type":"raw-audio","id":"`...)
	buf = strconv.AppendUint(buf, s.nextID, 10)
	buf = append(buf, `","data":{"base64Audio":"`...)
	buf = base64.StdEncoding.AppendEncode(buf, pcm)
	buf = append(buf, `","sampleRate":`...)
	buf = strconv.AppendInt(buf, int64(sampleRate), 10)
	buf = append(buf, `,"numChannels":1}}`...)
	return append(dst, Message{Type: TextMessage, Data: buf}), nil
}

func (s *RTVIClientSerializer) Serialize(dst []Message, frame frames.Frame) ([]Message, error) {
	var msg any
	switch f := frame.(type) {
	case *frames.OutputTransportMessageFrame:
		msg = f.Message
	case *frames.OutputTransportMessageUrgentFrame:
		msg = f.Message
	default:
		if out, ok, err := serializeAudioFrame(s, nil, dst, frame, 0); ok {
			return out, err
		}
		return dst, nil
	}
	if !isRTVIMessage(msg) {
		return dst, nil
	}
	out, _, err := transportMessage(dst, frame, false)
	return out, err
}

type rtviServerMessage struct {
	Label string         `json:"label"`
	Type  string         `json:"type"`
	Data  map[string]any `json:"data"`
}

func (s *RTVIClientSerializer) Deserialize(msg Message) (frames.Frame, error) {
	var m rtviServerMessage
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return nil, fmt.Errorf("rtvi client: %w", err)
	}
	if m.Label != rtviMessageLabel {
		return nil, nil
	}
	d := m.Data
	str := func(k string) string { v, _ := d[k].(string); return v }
	switch m.Type {
	case "bot-llm-started":
		return frames.NewLLMFullResponseStartFrame(), nil
	case "bot-llm-text":
		return frames.NewLLMTextFrame(str("text")), nil
	case "bot-llm-marker":
		var markers []string
		if list, ok := d["markers"].([]any); ok {
			for _, v := range list {
				if sv, ok := v.(string); ok {
					markers = append(markers, sv)
				}
			}
		}
		return frames.NewLLMMarkerResponseFrame(str("raw"), str("text"), str("kind"), markers), nil
	case "bot-llm-stopped":
		return frames.NewLLMFullResponseEndFrame(), nil
	case "bot-tts-text":
		return frames.NewTTSTextFrame(str("text"), frames.TextTypeSentence, ""), nil
	case "bot-started-speaking":
		return frames.NewBotStartedSpeakingFrame(), nil
	case "bot-stopped-speaking":
		return frames.NewBotStoppedSpeakingFrame(), nil
	case "bot-interrupted":
		return frames.NewInterruptionFrame(), nil
	case "llm-function-call-in-progress":
		args, _ := d["arguments"].(map[string]any)
		if args == nil {
			args = map[string]any{}
		}
		return frames.NewFunctionCallInProgressFrame(str("tool_call_id"), str("function_name"), args, false), nil
	case "llm-function-call-stopped":
		if c, _ := d["cancelled"].(bool); c {
			return frames.NewFunctionCallCancelFrame(str("tool_call_id"), str("function_name")), nil
		}
		f := frames.NewFunctionCallResultFrame(str("tool_call_id"), str("function_name"), d["result"], nil)
		f.Arguments = map[string]any{}
		return f, nil
	case "user-transcription":
		final := true
		if v, ok := d["final"].(bool); ok {
			final = v
		}
		ts, _ := time.Parse(time.RFC3339Nano, str("timestamp"))
		if !final {
			f := frames.NewInterimTranscriptionFrame(str("text"), str("user_id"))
			if !ts.IsZero() {
				f.Timestamp = ts
			}
			return f, nil
		}
		f := frames.NewTranscriptionFrame(str("text"), true)
		if uid := str("user_id"); uid != "" {
			f.SetMetadata("user_id", uid)
		}
		if !ts.IsZero() {
			f.Timestamp = ts
		}
		return f, nil
	}
	return nil, nil
}
