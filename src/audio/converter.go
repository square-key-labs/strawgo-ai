package audio

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/audio/g711"
	"github.com/square-key-labs/strawgo-ai/src/audio/pcm"
	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	CodecLinear16 = "linear16"
	CodecMulaw    = "mulaw"
	CodecAlaw     = "alaw"
)

type AudioConverterProcessor struct {
	*processors.BaseProcessor
	inputSampleRate  int
	inputCodec       string
	outputSampleRate int
	outputCodec      string

	inResampler  Resampler
	ttsResampler Resampler
	decoded      []byte
	resampled    []byte
}

type AudioConverterConfig struct {
	InputSampleRate  int
	InputCodec       string
	OutputSampleRate int
	OutputCodec      string
}

func NewAudioConverterProcessor(config AudioConverterConfig) *AudioConverterProcessor {
	ac := &AudioConverterProcessor{
		inputSampleRate:  config.InputSampleRate,
		inputCodec:       NormalizeCodec(config.InputCodec),
		outputSampleRate: config.OutputSampleRate,
		outputCodec:      NormalizeCodec(config.OutputCodec),
	}
	ac.BaseProcessor = processors.NewBaseProcessor("AudioConverter", ac)
	return ac
}

func (p *AudioConverterProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.AudioFrame:
		data, rate, err := p.convert(&p.inResampler, f.Data, f.SampleRate, f.Metadata())
		if err != nil {
			return p.PushError("audio conversion failed", err, false)
		}
		out := frames.NewAudioFrame(data, rate, f.Channels)
		copyMetadata(out, f, p.outputCodec)
		return p.PushFrame(out, direction)
	case *frames.TTSAudioFrame:
		data, rate, err := p.convert(&p.ttsResampler, f.Data, f.SampleRate, f.Metadata())
		if err != nil {
			return p.PushError("audio conversion failed", err, false)
		}
		out := frames.NewTTSAudioFrame(data, rate, f.Channels)
		out.ContextID = f.ContextID
		copyMetadata(out, f, p.outputCodec)
		return p.PushFrame(out, direction)
	case *frames.InterruptionFrame:
		p.ttsResampler.Reset()
	}
	return p.PushFrame(frame, direction)
}

func copyMetadata(dst, src frames.Frame, codec string) {
	for k, v := range src.Metadata() {
		dst.SetMetadata(k, v)
	}
	if orig, ok := src.Metadata()["codec"]; ok {
		dst.SetMetadata("original_codec", orig)
	}
	dst.SetMetadata("codec", codec)
}

func (p *AudioConverterProcessor) convert(rs *Resampler, data []byte, rate int, meta map[string]interface{}) ([]byte, int, error) {
	codec := p.inputCodec
	if c, ok := meta["codec"].(string); ok && c != "" {
		codec = NormalizeCodec(c)
	}
	if rate <= 0 {
		rate = p.inputSampleRate
	}
	outRate := p.outputSampleRate
	if outRate <= 0 {
		outRate = rate
	}
	if codec == p.outputCodec && rate == outRate {
		return data, rate, nil
	}
	pcm := data
	if codec != CodecLinear16 {
		var err error
		if p.decoded, err = DecodePCM16(p.decoded[:0], data, codec); err != nil {
			return nil, 0, err
		}
		pcm = p.decoded
	} else if len(pcm)%2 != 0 {
		return nil, 0, fmt.Errorf("invalid PCM data length: %d", len(pcm))
	}
	if rate != outRate {
		if p.outputCodec == CodecLinear16 {
			return rs.Resample(make([]byte, 0, rs.MaxOutputBytes(len(pcm), rate, outRate)), pcm, rate, outRate), outRate, nil
		}
		p.resampled = rs.Resample(p.resampled[:0], pcm, rate, outRate)
		pcm = p.resampled
	}
	out, err := EncodePCM16(make([]byte, 0, len(pcm)/2*BytesPerSample(p.outputCodec)), pcm, p.outputCodec)
	if err != nil {
		return nil, 0, err
	}
	return out, outRate, nil
}

func NormalizeCodec(codec string) string {
	switch strings.ToLower(codec) {
	case "mulaw", "ulaw", "pcmu", "g711_ulaw", "audio/x-mulaw", "mu-law", "audio/pcmu":
		return CodecMulaw
	case "alaw", "pcma", "g711_alaw", "audio/x-alaw", "a-law", "audio/pcma":
		return CodecAlaw
	case "", "linear16", "pcm", "l16", "slin", "pcm16", "s16le", "audio/l16":
		return CodecLinear16
	default:
		return codec
	}
}

func DecodePCM16(dst, src []byte, codec string) ([]byte, error) {
	switch NormalizeCodec(codec) {
	case CodecLinear16:
		if len(src)%2 != 0 {
			return nil, fmt.Errorf("invalid PCM data length: %d", len(src))
		}
		return append(dst, src...), nil
	case CodecMulaw:
		return g711.UlawToPCM16(dst, src), nil
	case CodecAlaw:
		return g711.AlawToPCM16(dst, src), nil
	}
	return nil, fmt.Errorf("unsupported codec: %s", codec)
}

func EncodePCM16(dst, pcm []byte, codec string) ([]byte, error) {
	if len(pcm)%2 != 0 {
		return nil, fmt.Errorf("invalid PCM data length: %d", len(pcm))
	}
	switch NormalizeCodec(codec) {
	case CodecLinear16:
		return append(dst, pcm...), nil
	case CodecMulaw:
		return g711.PCM16ToUlaw(dst, pcm), nil
	case CodecAlaw:
		return g711.PCM16ToAlaw(dst, pcm), nil
	}
	return nil, fmt.Errorf("unsupported codec: %s", codec)
}

func SilenceByte(codec string) byte {
	switch NormalizeCodec(codec) {
	case CodecMulaw:
		return 0xFF
	case CodecAlaw:
		return 0xD5
	}
	return 0
}

func BytesPerSample(codec string) int {
	switch NormalizeCodec(codec) {
	case CodecMulaw, CodecAlaw:
		return 1
	}
	return 2
}

type Resampler struct {
	r       *resample.Resampler
	pending bool
	tail    []int16
}

func (s *Resampler) Reset() {
	s.pending = false
	if s.r != nil {
		s.r.Reset()
	}
}

func (s *Resampler) Flush(dst []byte) []byte {
	if !s.pending || s.r == nil {
		return dst
	}
	s.pending = false
	s.tail = s.r.Flush(s.tail[:0])
	return pcm.FromInt16(dst, s.tail)
}

func (s *Resampler) Resample(dst, pcm []byte, inRate, outRate int) []byte {
	if inRate == outRate || inRate <= 0 || outRate <= 0 {
		return append(dst, pcm...)
	}
	if s.r == nil || s.r.InRate() != inRate || s.r.OutRate() != outRate {
		r, err := resample.New(inRate, outRate)
		if err != nil {
			return append(dst, pcm...)
		}
		s.r = r
		s.pending = false
	}
	s.pending = s.pending || len(pcm) > 0
	return s.r.ProcessBytes(dst, pcm)
}

func (s *Resampler) MaxOutputBytes(inBytes, inRate, outRate int) int {
	if inRate <= 0 || outRate <= 0 || inRate == outRate {
		return inBytes
	}
	return 2 * (int(int64(inBytes/2)*int64(outRate)/int64(inRate)) + 2)
}

func BytesToPCM(data []byte) ([]int16, error) {
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("invalid PCM data length: %d", len(data))
	}
	return pcm.ToInt16(nil, data), nil
}

func PCMToBytes(samples []int16) []byte {
	return pcm.FromInt16(nil, samples)
}

func Resample(input []int16, inputRate, outputRate int) []int16 {
	if inputRate == outputRate {
		return input
	}
	out, err := resample.Resample(input, inputRate, outputRate)
	if err != nil {
		return input
	}
	return out
}

func ClipAudio(pcm []int16, maxValue int16) []int16 {
	output := make([]int16, len(pcm))
	for i, val := range pcm {
		if val > maxValue {
			output[i] = maxValue
		} else if val < -maxValue {
			output[i] = -maxValue
		} else {
			output[i] = val
		}
	}
	return output
}

func NormalizeAudio(pcm []int16, targetRMS float64) []int16 {
	var sum float64
	for _, val := range pcm {
		sum += float64(val) * float64(val)
	}
	currentRMS := math.Sqrt(sum / float64(len(pcm)))

	if currentRMS == 0 {
		return pcm
	}

	gain := targetRMS / currentRMS

	output := make([]int16, len(pcm))
	for i, val := range pcm {
		scaled := float64(val) * gain
		if scaled > 32767 {
			output[i] = 32767
		} else if scaled < -32768 {
			output[i] = -32768
		} else {
			output[i] = int16(scaled)
		}
	}

	return output
}
