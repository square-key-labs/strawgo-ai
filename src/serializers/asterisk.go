package serializers

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type AsteriskSerializerConfig struct {
	ChannelID         string
	Codec             string
	SampleRate        int
	DisableAutoHangUp bool
}

type AsteriskFrameSerializer struct {
	cfg        AsteriskSerializerConfig
	channelID  string
	codec      string
	sampleRate int
	jsonMode   bool
	info       CallInfo
	wire       wireAudio
}

type asteriskControl struct {
	Type          string
	ConnectionID  string
	Channel       string
	Format        string
	FrameSize     int
	CorrelationID string
	Digit         string
	JSON          bool
}

func NewAsteriskFrameSerializer(cfg AsteriskSerializerConfig) *AsteriskFrameSerializer {
	s := &AsteriskFrameSerializer{cfg: cfg, channelID: cfg.ChannelID}
	s.setFormat(firstNonEmpty(cfg.Codec, "ulaw"))
	if cfg.SampleRate > 0 {
		s.sampleRate = cfg.SampleRate
	}
	s.info = CallInfo{Provider: "asterisk", CallID: cfg.ChannelID, Encoding: s.codec, SampleRate: s.sampleRate}
	s.syncWire()
	return s
}

func (s *AsteriskFrameSerializer) syncWire() {
	native := s.wire.native
	s.wire = newWireAudio(s.codec, s.codec, s.sampleRate)
	s.wire.native = native
}

func (s *AsteriskFrameSerializer) NativeAudioFormat() (string, int) { return s.codec, s.sampleRate }

func (s *AsteriskFrameSerializer) SetNativeAudioInput(enabled bool) { s.wire.native = enabled }

func (s *AsteriskFrameSerializer) SerializeNativeAudio(dst []Message, data []byte) ([]Message, error) {
	return append(dst, Binary(data)), nil
}

func (s *AsteriskFrameSerializer) setFormat(format string) {
	f := strings.ToLower(format)
	switch {
	case f == "slin":
		s.codec, s.sampleRate = audio.CodecLinear16, 8000
	case strings.HasPrefix(f, "slin"):
		s.codec = audio.CodecLinear16
		if k, err := strconv.Atoi(f[4:]); err == nil && k > 0 {
			if k == 44 {
				s.sampleRate = 44100
			} else {
				s.sampleRate = k * 1000
			}
		}
	default:
		s.codec = audio.NormalizeCodec(f)
		if s.codec == audio.CodecLinear16 && s.sampleRate == 0 {
			s.sampleRate = 16000
		} else if s.codec != audio.CodecLinear16 {
			s.sampleRate = 8000
		}
	}
}

func parseAsteriskControl(text string) (*asteriskControl, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, fmt.Errorf("empty control message")
	}
	msg := &asteriskControl{}
	if text[0] == '{' {
		var m map[string]any
		if err := json.Unmarshal([]byte(text), &m); err != nil {
			return nil, err
		}
		msg.JSON = true
		msg.Type, _ = m["event"].(string)
		msg.ConnectionID, _ = m["connection_id"].(string)
		msg.Channel, _ = m["channel"].(string)
		msg.Format, _ = m["format"].(string)
		msg.CorrelationID, _ = m["correlation_id"].(string)
		if d, ok := m["digit"]; ok {
			msg.Digit = fmt.Sprint(d)
		}
		if n, ok := m["optimal_frame_size"].(float64); ok {
			msg.FrameSize = int(n)
		}
		return msg, nil
	}
	parts := strings.Fields(text)
	msg.Type = parts[0]
	for _, part := range parts[1:] {
		key, value, ok := strings.Cut(part, ":")
		if !ok {
			continue
		}
		switch key {
		case "connection_id":
			msg.ConnectionID = value
		case "channel":
			msg.Channel = value
		case "format":
			msg.Format = value
		case "optimal_frame_size":
			msg.FrameSize, _ = strconv.Atoi(value)
		case "correlation_id":
			msg.CorrelationID = value
		case "digit":
			msg.Digit = value
		}
	}
	return msg, nil
}

func (s *AsteriskFrameSerializer) WireSampleRates() (int, int) { return s.sampleRate, s.sampleRate }

func (s *AsteriskFrameSerializer) CallInfo() CallInfo { return s.info }

func (s *AsteriskFrameSerializer) Handshake(msg Message) (bool, error) {
	if msg.Type == BinaryMessage {
		return true, nil
	}
	ctl, err := parseAsteriskControl(string(msg.Data))
	if err != nil {
		return false, fmt.Errorf("asterisk handshake: %w", err)
	}
	if ctl.Type != "MEDIA_START" {
		return true, nil
	}
	s.jsonMode = ctl.JSON
	if ctl.Format != "" && s.cfg.Codec == "" {
		s.setFormat(ctl.Format)
		if s.cfg.SampleRate > 0 {
			s.sampleRate = s.cfg.SampleRate
		}
	}
	s.channelID = firstNonEmpty(ctl.Channel, s.channelID)
	s.info = CallInfo{Provider: "asterisk", StreamID: ctl.ConnectionID, CallID: s.channelID, Encoding: s.codec, SampleRate: s.sampleRate}
	s.syncWire()
	return true, nil
}

func (s *AsteriskFrameSerializer) command(dst []Message, name string, args ...string) []Message {
	if s.jsonMode {
		m := map[string]string{"command": name}
		for i := 0; i+1 < len(args); i += 2 {
			m[args[i]] = args[i+1]
		}
		data, _ := json.Marshal(m)
		return append(dst, Message{Type: TextMessage, Data: data})
	}
	var b strings.Builder
	b.WriteString(name)
	for i := 0; i+1 < len(args); i += 2 {
		b.WriteString(" " + args[i] + ":" + args[i+1])
	}
	return append(dst, Text(b.String()))
}

func (s *AsteriskFrameSerializer) SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error) {
	enc, err := s.wire.encodeOwned(pcm, sampleRate)
	if err != nil {
		return dst, err
	}
	return append(dst, Binary(enc)), nil
}

func (s *AsteriskFrameSerializer) Serialize(dst []Message, frame frames.Frame) ([]Message, error) {
	switch frame.(type) {
	case *frames.InterruptionFrame:
		s.wire.reset()
		return s.command(dst, "FLUSH_MEDIA"), nil
	case *frames.EndFrame, *frames.CancelFrame:
		if s.cfg.DisableAutoHangUp {
			return dst, nil
		}
		return s.command(dst, "HANGUP"), nil
	}
	if out, ok, err := serializeAudioFrame(s, &s.wire.rs, dst, frame, s.sampleRate); ok {
		return out, err
	}
	out, _, err := transportMessage(dst, frame, true)
	return out, err
}

func (s *AsteriskFrameSerializer) SerializePlaybackAck(correlationID string) (Message, error) {
	if correlationID == "" {
		correlationID = "playback-done"
	}
	return s.command(nil, "MARK_MEDIA", "correlation_id", correlationID)[0], nil
}

func (s *AsteriskFrameSerializer) Deserialize(msg Message) (frames.Frame, error) {
	if msg.Type == BinaryMessage {
		return s.wire.decode(msg.Data)
	}
	ctl, err := parseAsteriskControl(string(msg.Data))
	if err != nil {
		return nil, fmt.Errorf("asterisk: %w", err)
	}
	switch ctl.Type {
	case "HANGUP":
		return frames.NewEndFrame(), nil
	case "DTMF_END":
		return dtmfFrame(ctl.Digit), nil
	case "MEDIA_XOFF":
		return NewMediaFlowFrame(true), nil
	case "MEDIA_XON":
		return NewMediaFlowFrame(false), nil
	case "MEDIA_MARK_PROCESSED":
		f := frames.NewPlaybackCompleteFrame()
		if ctl.CorrelationID != "" {
			f.SetMetadata("correlation_id", ctl.CorrelationID)
		}
		return f, nil
	case "QUEUE_DRAINED":
		f := frames.NewPlaybackCompleteFrame()
		f.SetMetadata(frames.MetadataQueueDrained, true)
		return f, nil
	}
	return nil, nil
}

func (s *AsteriskFrameSerializer) ChannelID() string { return s.channelID }

func (s *AsteriskFrameSerializer) Codec() string { return s.codec }

func (s *AsteriskFrameSerializer) SampleRate() int { return s.sampleRate }
