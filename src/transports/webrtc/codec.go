package webrtc

import (
	"strings"

	pwebrtc "github.com/pion/webrtc/v4"
	"github.com/square-key-labs/strawgo-ai/src/audio/g711"
)

type Encoder interface {
	Encode(dst, pcm []byte) ([]byte, error)
}

type Decoder interface {
	Decode(dst, payload []byte) ([]byte, error)
}

type Codec interface {
	Capability() pwebrtc.RTPCodecCapability
	PayloadType() pwebrtc.PayloadType
	SampleRate() int
	NewEncoder() (Encoder, error)
	NewDecoder() (Decoder, error)
}

type g711Codec struct {
	alaw bool
}

var (
	PCMU Codec = g711Codec{}
	PCMA Codec = g711Codec{alaw: true}
)

var defaultCodecs = []Codec{PCMU, PCMA}

func DefaultCodecs() []Codec {
	return append([]Codec(nil), defaultCodecs...)
}

func (c g711Codec) Capability() pwebrtc.RTPCodecCapability {
	mime := pwebrtc.MimeTypePCMU
	if c.alaw {
		mime = pwebrtc.MimeTypePCMA
	}
	return pwebrtc.RTPCodecCapability{MimeType: mime, ClockRate: 8000, Channels: 1}
}

func (c g711Codec) PayloadType() pwebrtc.PayloadType {
	if c.alaw {
		return 8
	}
	return 0
}

func (c g711Codec) SampleRate() int { return 8000 }

func (c g711Codec) NewEncoder() (Encoder, error) { return c, nil }

func (c g711Codec) NewDecoder() (Decoder, error) { return c, nil }

func (c g711Codec) Encode(dst, pcm []byte) ([]byte, error) {
	if c.alaw {
		return g711.PCM16ToAlaw(dst, pcm), nil
	}
	return g711.PCM16ToUlaw(dst, pcm), nil
}

func (c g711Codec) Decode(dst, payload []byte) ([]byte, error) {
	if c.alaw {
		return g711.AlawToPCM16(dst, payload), nil
	}
	return g711.UlawToPCM16(dst, payload), nil
}

func matchCodec(c Codec, name string, clockRate uint32) bool {
	capability := c.Capability()
	_, sub, _ := strings.Cut(capability.MimeType, "/")
	return strings.EqualFold(sub, name) && capability.ClockRate == clockRate
}

func codecForMime(codecs []Codec, capability pwebrtc.RTPCodecCapability) Codec {
	for _, c := range codecs {
		if strings.EqualFold(c.Capability().MimeType, capability.MimeType) && c.Capability().ClockRate == capability.ClockRate {
			return c
		}
	}
	return nil
}
