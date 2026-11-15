//go:build opus

package webrtc

import (
	"encoding/binary"

	pwebrtc "github.com/pion/webrtc/v4"
	"gopkg.in/hraban/opus.v2"
)

const (
	opusRate        = 48000
	opusMaxSamples  = opusRate * 120 / 1000
	opusMaxPacket   = 1500
	opusPayloadType = 111
)

type opusCodec struct{}

var Opus Codec = opusCodec{}

func init() {
	defaultCodecs = append([]Codec{Opus}, defaultCodecs...)
}

func (opusCodec) Capability() pwebrtc.RTPCodecCapability {
	return pwebrtc.RTPCodecCapability{
		MimeType:    pwebrtc.MimeTypeOpus,
		ClockRate:   opusRate,
		Channels:    2,
		SDPFmtpLine: "minptime=10;useinbandfec=1",
	}
}

func (opusCodec) PayloadType() pwebrtc.PayloadType { return opusPayloadType }

func (opusCodec) SampleRate() int { return opusRate }

func (opusCodec) NewEncoder() (Encoder, error) {
	enc, err := opus.NewEncoder(opusRate, 1, opus.AppVoIP)
	if err != nil {
		return nil, err
	}
	return &opusEncoder{enc: enc}, nil
}

func (opusCodec) NewDecoder() (Decoder, error) {
	dec, err := opus.NewDecoder(opusRate, 1)
	if err != nil {
		return nil, err
	}
	return &opusDecoder{dec: dec, pcm: make([]int16, opusMaxSamples)}, nil
}

type opusEncoder struct {
	enc *opus.Encoder
	pcm []int16
	out []byte
}

func (e *opusEncoder) Encode(dst, pcm []byte) ([]byte, error) {
	n := len(pcm) / 2
	if cap(e.pcm) < n {
		e.pcm = make([]int16, n)
	}
	e.pcm = e.pcm[:n]
	for i := range e.pcm {
		e.pcm[i] = int16(binary.LittleEndian.Uint16(pcm[2*i:]))
	}
	if e.out == nil {
		e.out = make([]byte, opusMaxPacket)
	}
	size, err := e.enc.Encode(e.pcm, e.out)
	if err != nil {
		return dst, err
	}
	return append(dst, e.out[:size]...), nil
}

type opusDecoder struct {
	dec *opus.Decoder
	pcm []int16
}

func (d *opusDecoder) Decode(dst, payload []byte) ([]byte, error) {
	n, err := d.dec.Decode(payload, d.pcm)
	if err != nil {
		return dst, err
	}
	for _, s := range d.pcm[:n] {
		dst = binary.LittleEndian.AppendUint16(dst, uint16(s))
	}
	return dst, nil
}
