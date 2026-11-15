package transports

import (
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/audio/g711"
	"github.com/square-key-labs/strawgo-ai/src/audio/pcm"
)

func FadeSamples(rate int, fade time.Duration) int {
	if rate <= 0 || fade <= 0 {
		return 0
	}
	return int(int64(rate) * int64(fade) / int64(time.Second))
}

func FadeOut(b []byte, codec string) {
	switch audio.NormalizeCodec(codec) {
	case audio.CodecMulaw:
		n := int32(len(b))
		for i := int32(0); i < n; i++ {
			b[i] = g711.UlawEncodeSample(int16(int32(g711.UlawDecodeSample(b[i])) * (n - 1 - i) / n))
		}
	case audio.CodecAlaw:
		n := int32(len(b))
		for i := int32(0); i < n; i++ {
			b[i] = g711.AlawEncodeSample(int16(int32(g711.AlawDecodeSample(b[i])) * (n - 1 - i) / n))
		}
	default:
		pcm.FadeOut(b[:len(b)&^1])
	}
}
