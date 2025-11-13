package audio

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

// AudioConverterProcessor converts audio between different formats and sample rates
type AudioConverterProcessor struct {
	*processors.BaseProcessor
	inputSampleRate  int
	inputCodec       string
	outputSampleRate int
	outputCodec      string
}

// AudioConverterConfig holds configuration for audio conversion
type AudioConverterConfig struct {
	InputSampleRate  int    // e.g., 8000, 16000, 24000
	InputCodec       string // Supported: "mulaw"/"ulaw"/"PCMU", "alaw"/"PCMA", "linear16"/"pcm"
	OutputSampleRate int    // e.g., 8000, 16000, 24000
	OutputCodec      string // Supported: "mulaw"/"ulaw"/"PCMU", "alaw"/"PCMA", "linear16"/"pcm"
}

// NewAudioConverterProcessor creates a new audio converter
func NewAudioConverterProcessor(config AudioConverterConfig) *AudioConverterProcessor {
	ac := &AudioConverterProcessor{
		inputSampleRate:  config.InputSampleRate,
		inputCodec:       config.InputCodec,
		outputSampleRate: config.OutputSampleRate,
		outputCodec:      config.OutputCodec,
	}
	ac.BaseProcessor = processors.NewBaseProcessor("AudioConverter", ac)
	return ac
}

func (p *AudioConverterProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Convert audio frames
	if audioFrame, ok := frame.(*frames.AudioFrame); ok {
		convertedData, err := p.convertAudio(audioFrame.Data, audioFrame.SampleRate)
		if err != nil {
			log.Printf("[AudioConverter] Error converting audio: %v", err)
			return p.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
		}

		// Create new frame with converted audio
		newFrame := frames.NewAudioFrame(convertedData, p.outputSampleRate, audioFrame.Channels)
		// Copy metadata
		for k, v := range audioFrame.Metadata() {
			newFrame.SetMetadata(k, v)
		}
		newFrame.SetMetadata("original_codec", p.inputCodec)
		newFrame.SetMetadata("codec", p.outputCodec)

		return p.PushFrame(newFrame, direction)
	}

	// Pass all other frames through
	return p.PushFrame(frame, direction)
}

func (p *AudioConverterProcessor) convertAudio(data []byte, inputRate int) ([]byte, error) {
	// Step 1: Decode to PCM int16
	var pcm []int16
	var err error

	// Normalize codec name
	inputCodec := normalizeCodecName(p.inputCodec)

	switch inputCodec {
	case "mulaw", "ulaw", "PCMU":
		pcm = MulawToPCM(data)
	case "alaw", "PCMA":
		pcm = AlawToPCM(data)
	case "linear16", "pcm":
		pcm, err = BytesToPCM(data)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unsupported input codec: %s", p.inputCodec)
	}

	// Step 2: Resample if needed
	if inputRate != p.outputSampleRate {
		pcm = Resample(pcm, inputRate, p.outputSampleRate)
	}

	// Step 3: Encode to output format
	outputCodec := normalizeCodecName(p.outputCodec)

	var output []byte
	switch outputCodec {
	case "linear16", "pcm":
		output = PCMToBytes(pcm)
	case "mulaw", "ulaw", "PCMU":
		output = PCMToMulaw(pcm)
	case "alaw", "PCMA":
		output = PCMToAlaw(pcm)
	default:
		return nil, fmt.Errorf("unsupported output codec: %s", p.outputCodec)
	}

	return output, nil
}

// normalizeCodecName converts codec name variations to a standard form
func normalizeCodecName(codec string) string {
	// Convert to lowercase for comparison
	switch codec {
	case "mulaw", "ulaw", "PCMU":
		return "mulaw"
	case "alaw", "PCMA":
		return "alaw"
	case "linear16", "pcm", "PCM":
		return "linear16"
	default:
		return codec
	}
}

// MulawToPCM converts mulaw audio to linear PCM int16
func MulawToPCM(mulaw []byte) []int16 {
	pcm := make([]int16, len(mulaw))
	for i, val := range mulaw {
		pcm[i] = mulawDecode(val)
	}
	return pcm
}

// PCMToMulaw converts linear PCM int16 to mulaw
func PCMToMulaw(pcm []int16) []byte {
	mulaw := make([]byte, len(pcm))
	for i, val := range pcm {
		mulaw[i] = mulawEncode(val)
	}
	return mulaw
}

// BytesToPCM converts byte array to int16 PCM (little-endian)
func BytesToPCM(data []byte) ([]int16, error) {
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("invalid PCM data length: %d", len(data))
	}
	pcm := make([]int16, len(data)/2)
	for i := 0; i < len(pcm); i++ {
		pcm[i] = int16(binary.LittleEndian.Uint16(data[i*2:]))
	}
	return pcm, nil
}

// PCMToBytes converts int16 PCM to byte array (little-endian)
func PCMToBytes(pcm []int16) []byte {
	data := make([]byte, len(pcm)*2)
	for i, val := range pcm {
		binary.LittleEndian.PutUint16(data[i*2:], uint16(val))
	}
	return data
}

// Resample performs simple linear interpolation resampling
// This is a basic implementation; for production, consider using a proper resampling library
func Resample(input []int16, inputRate, outputRate int) []int16 {
	if inputRate == outputRate {
		return input
	}

	ratio := float64(inputRate) / float64(outputRate)
	outputLen := int(float64(len(input)) / ratio)
	output := make([]int16, outputLen)

	for i := 0; i < outputLen; i++ {
		srcPos := float64(i) * ratio
		srcIdx := int(srcPos)
		frac := srcPos - float64(srcIdx)

		if srcIdx+1 < len(input) {
			// Linear interpolation
			sample1 := float64(input[srcIdx])
			sample2 := float64(input[srcIdx+1])
			output[i] = int16(sample1 + (sample2-sample1)*frac)
		} else if srcIdx < len(input) {
			output[i] = input[srcIdx]
		}
	}

	return output
}

// Mulaw encoding/decoding tables and functions
const (
	mulawBias = 0x84
	mulawClip = 32635
)

var mulawDecodeTable = [256]int16{
	-32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
	-23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
	-15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
	-11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
	-7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
	-5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
	-3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
	-2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
	-1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
	-1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
	-876, -844, -812, -780, -748, -716, -684, -652,
	-620, -588, -556, -524, -492, -460, -428, -396,
	-372, -356, -340, -324, -308, -292, -276, -260,
	-244, -228, -212, -196, -180, -164, -148, -132,
	-120, -112, -104, -96, -88, -80, -72, -64,
	-56, -48, -40, -32, -24, -16, -8, 0,
	32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
	23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
	15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
	11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
	7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
	5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
	3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
	2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
	1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
	1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
	876, 844, 812, 780, 748, 716, 684, 652,
	620, 588, 556, 524, 492, 460, 428, 396,
	372, 356, 340, 324, 308, 292, 276, 260,
	244, 228, 212, 196, 180, 164, 148, 132,
	120, 112, 104, 96, 88, 80, 72, 64,
	56, 48, 40, 32, 24, 16, 8, 0,
}

func mulawDecode(mulaw byte) int16 {
	return mulawDecodeTable[mulaw]
}

func mulawEncode(pcm int16) byte {
	// Get the sign bit
	sign := uint8(0)
	if pcm < 0 {
		sign = 0x80
		pcm = -pcm
	}

	// Clip the magnitude
	if pcm > mulawClip {
		pcm = mulawClip
	}

	// Add bias
	pcm += mulawBias

	// Find the position of the highest set bit
	var exponent uint8
	var mantissa uint8

	if pcm >= 0x1000 {
		exponent = 7
		mantissa = uint8((pcm >> 7) & 0x0F)
	} else if pcm >= 0x800 {
		exponent = 6
		mantissa = uint8((pcm >> 6) & 0x0F)
	} else if pcm >= 0x400 {
		exponent = 5
		mantissa = uint8((pcm >> 5) & 0x0F)
	} else if pcm >= 0x200 {
		exponent = 4
		mantissa = uint8((pcm >> 4) & 0x0F)
	} else if pcm >= 0x100 {
		exponent = 3
		mantissa = uint8((pcm >> 3) & 0x0F)
	} else if pcm >= 0x80 {
		exponent = 2
		mantissa = uint8((pcm >> 2) & 0x0F)
	} else if pcm >= 0x40 {
		exponent = 1
		mantissa = uint8((pcm >> 1) & 0x0F)
	} else {
		exponent = 0
		mantissa = uint8(pcm & 0x0F)
	}

	// Combine sign, exponent, and mantissa
	mulaw := sign | (exponent << 4) | mantissa

	// Invert all bits for mulaw
	return ^mulaw
}

// AlawToPCM converts A-law audio to linear PCM int16
func AlawToPCM(alaw []byte) []int16 {
	pcm := make([]int16, len(alaw))
	for i, val := range alaw {
		pcm[i] = alawDecode(val)
	}
	return pcm
}

// PCMToAlaw converts linear PCM int16 to A-law
func PCMToAlaw(pcm []int16) []byte {
	alaw := make([]byte, len(pcm))
	for i, val := range pcm {
		alaw[i] = alawEncode(val)
	}
	return alaw
}

// A-law encoding/decoding tables and functions
const (
	alawClip = 32767
)

var alawDecodeTable = [256]int16{
	-5504, -5248, -6016, -5760, -4480, -4224, -4992, -4736,
	-7552, -7296, -8064, -7808, -6528, -6272, -7040, -6784,
	-2752, -2624, -3008, -2880, -2240, -2112, -2496, -2368,
	-3776, -3648, -4032, -3904, -3264, -3136, -3520, -3392,
	-22016, -20992, -24064, -23040, -17920, -16896, -19968, -18944,
	-30208, -29184, -32256, -31232, -26112, -25088, -28160, -27136,
	-11008, -10496, -12032, -11520, -8960, -8448, -9984, -9472,
	-15104, -14592, -16128, -15616, -13056, -12544, -14080, -13568,
	-344, -328, -376, -360, -280, -264, -312, -296,
	-472, -456, -504, -488, -408, -392, -440, -424,
	-88, -72, -120, -104, -24, -8, -56, -40,
	-216, -200, -248, -232, -152, -136, -184, -168,
	-1376, -1312, -1504, -1440, -1120, -1056, -1248, -1184,
	-1888, -1824, -2016, -1952, -1632, -1568, -1760, -1696,
	-688, -656, -752, -720, -560, -528, -624, -592,
	-944, -912, -1008, -976, -816, -784, -880, -848,
	5504, 5248, 6016, 5760, 4480, 4224, 4992, 4736,
	7552, 7296, 8064, 7808, 6528, 6272, 7040, 6784,
	2752, 2624, 3008, 2880, 2240, 2112, 2496, 2368,
	3776, 3648, 4032, 3904, 3264, 3136, 3520, 3392,
	22016, 20992, 24064, 23040, 17920, 16896, 19968, 18944,
	30208, 29184, 32256, 31232, 26112, 25088, 28160, 27136,
	11008, 10496, 12032, 11520, 8960, 8448, 9984, 9472,
	15104, 14592, 16128, 15616, 13056, 12544, 14080, 13568,
	344, 328, 376, 360, 280, 264, 312, 296,
	472, 456, 504, 488, 408, 392, 440, 424,
	88, 72, 120, 104, 24, 8, 56, 40,
	216, 200, 248, 232, 152, 136, 184, 168,
	1376, 1312, 1504, 1440, 1120, 1056, 1248, 1184,
	1888, 1824, 2016, 1952, 1632, 1568, 1760, 1696,
	688, 656, 752, 720, 560, 528, 624, 592,
	944, 912, 1008, 976, 816, 784, 880, 848,
}

func alawDecode(alaw byte) int16 {
	return alawDecodeTable[alaw]
}

func alawEncode(pcm int16) byte {
	// Get sign bit
	sign := uint8(0)
	if pcm < 0 {
		sign = 0x80
		pcm = -pcm
	}

	// Clip the magnitude
	if pcm > alawClip {
		pcm = alawClip
	}

	// Find exponent and mantissa
	var exponent uint8
	var mantissa uint8

	if pcm >= 256 {
		exponent = 7
		for pcm >= 256 && exponent > 0 {
			pcm >>= 1
			exponent--
		}
		mantissa = uint8((pcm >> 4) & 0x0F)
	} else {
		exponent = 0
		mantissa = uint8((pcm >> 4) & 0x0F)
	}

	// Combine sign, exponent, and mantissa
	alaw := sign | (exponent << 4) | mantissa

	// XOR with 0x55 for A-law
	return alaw ^ 0x55
}

// ClipAudio clips audio samples to prevent overflow
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

// NormalizeAudio normalizes audio to a target RMS level
func NormalizeAudio(pcm []int16, targetRMS float64) []int16 {
	// Calculate current RMS
	var sum float64
	for _, val := range pcm {
		sum += float64(val) * float64(val)
	}
	currentRMS := math.Sqrt(sum / float64(len(pcm)))

	if currentRMS == 0 {
		return pcm
	}

	// Calculate gain
	gain := targetRMS / currentRMS

	// Apply gain
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
