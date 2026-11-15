package serializers

import (
	"encoding/json"
	"fmt"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type JambonzConfig struct {
	AudioInSampleRate    int
	AudioOutSampleRate   int
	DisableDisconnectEnd bool
}

type JambonzFrameSerializer struct {
	cfg            JambonzConfig
	inRate         int
	outRate        int
	mixType        string
	callSID        string
	traceID        string
	meta           map[string]any
	preamble       bool
	warnedMix      bool
	disconnectSent atomic.Bool
	in             wireAudio
	out            wireAudio
}

func NewJambonzFrameSerializer(cfg JambonzConfig) *JambonzFrameSerializer {
	s := &JambonzFrameSerializer{cfg: cfg, inRate: cfg.AudioInSampleRate, outRate: cfg.AudioOutSampleRate}
	s.syncWire()
	return s
}

func (s *JambonzFrameSerializer) syncWire() {
	s.in = newWireAudio(audio.CodecLinear16, audio.CodecLinear16, s.inRate)
	s.out = newWireAudio(audio.CodecLinear16, audio.CodecLinear16, s.outRate)
}

func (s *JambonzFrameSerializer) WireSampleRates() (int, int) { return s.inRate, s.outRate }

func (s *JambonzFrameSerializer) MixType() string { return s.mixType }

func (s *JambonzFrameSerializer) CallSID() string { return s.callSID }

func (s *JambonzFrameSerializer) TraceID() string { return s.traceID }

func (s *JambonzFrameSerializer) CallMeta() map[string]any { return s.meta }

func (s *JambonzFrameSerializer) CallInfo() CallInfo {
	info := CallInfo{Provider: "jambonz", CallID: s.callSID, StreamID: s.traceID, Encoding: audio.CodecLinear16, SampleRate: s.inRate}
	info.From, _ = s.meta["from"].(string)
	info.To, _ = s.meta["to"].(string)
	return info
}

func (s *JambonzFrameSerializer) Handshake(msg Message) (bool, error) {
	if msg.Type == BinaryMessage {
		return true, nil
	}
	s.preamble = true
	var meta map[string]any
	if err := json.Unmarshal(msg.Data, &meta); err != nil {
		logger.Warn("[Jambonz] undecodable preamble: %v", err)
		return true, nil
	}
	s.meta = meta
	s.callSID, _ = meta["callSid"].(string)
	s.traceID, _ = meta["traceId"].(string)
	if mt, ok := meta["mixType"].(string); ok && mt != "" {
		s.mixType = mt
	}
	if r, ok := meta["sampleRate"].(float64); ok && r > 0 {
		if s.cfg.AudioInSampleRate == 0 {
			s.inRate = int(r)
		}
		if s.cfg.AudioOutSampleRate == 0 {
			s.outRate = int(r)
		}
	}
	s.syncWire()
	return true, nil
}

func (s *JambonzFrameSerializer) SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error) {
	if s.outRate == 0 {
		return dst, nil
	}
	enc, err := s.out.encodeOwned(pcm, sampleRate)
	if err != nil || len(enc) == 0 {
		return dst, err
	}
	return append(dst, Binary(enc)), nil
}

func (s *JambonzFrameSerializer) Serialize(dst []Message, frame frames.Frame) ([]Message, error) {
	switch frame.(type) {
	case *frames.EndFrame, *frames.CancelFrame:
		if s.cfg.DisableDisconnectEnd || !s.disconnectSent.CompareAndSwap(false, true) {
			return dst, nil
		}
		return append(dst, Text(`{"type":"disconnect"}`)), nil
	case *frames.InterruptionFrame:
		s.out.reset()
		return append(dst, Text(`{"type":"killAudio"}`)), nil
	}
	if s.outRate > 0 {
		if out, ok, err := serializeAudioFrame(s, &s.out.rs, dst, frame, s.outRate); ok {
			return out, err
		}
	}
	out, _, err := transportMessage(dst, frame, true)
	return out, err
}

func (s *JambonzFrameSerializer) Deserialize(msg Message) (frames.Frame, error) {
	if msg.Type == TextMessage {
		if !s.preamble {
			_, err := s.Handshake(msg)
			return nil, err
		}
		var ev map[string]any
		if err := json.Unmarshal(msg.Data, &ev); err != nil {
			return nil, nil
		}
		if ev["event"] == "dtmf" {
			return dtmfFrame(fmt.Sprint(ev["dtmf"])), nil
		}
		return nil, nil
	}
	if s.inRate == 0 {
		return nil, nil
	}
	if s.mixType != "" && s.mixType != "mono" {
		if !s.warnedMix {
			s.warnedMix = true
			logger.Warn("[Jambonz] dropping audio: mixType=%q is not usable, set mixType to \"mono\" on the listen verb", s.mixType)
		}
		return nil, nil
	}
	return s.in.decode(msg.Data)
}
