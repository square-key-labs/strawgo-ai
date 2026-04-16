package daily

// Run with:
//   go test ./src/transports/daily/... -bench=. -benchmem
//   go test ./src/transports/daily/... -bench=BenchmarkEncode -benchmem -count=5
//
// Key question: are the two per-call allocs in EncodePCMToOpus worth pooling?
// Look at allocs/op and B/op in the output.
//
// With sync.Pool for the 4000-byte scratch buffer:
//   allocs/op stays 2 (samples slice + result copy)
//   B/op drops from ~5000 to ~1500 (result is actual Opus size, not 4000)

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/pion/webrtc/v3"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// pcmSine returns a linear16 PCM buffer of `samples` int16 samples at 440Hz.
func pcmSine(sampleRate, samples int) []byte {
	buf := make([]byte, samples*2)
	for i := 0; i < samples; i++ {
		s := int16(math.Sin(2*math.Pi*440*float64(i)/float64(sampleRate)) * 16000)
		binary.LittleEndian.PutUint16(buf[2*i:], uint16(s))
	}
	return buf
}

// BenchmarkEncodePCMToOpus_48kHz measures one 20ms frame (960 samples × 2 = 1920 bytes)
// at 48 kHz — the most common WebRTC sample rate.
// Check allocs/op: 2 heap allocs per call (samples slice + result copy).
// Check B/op: ~1500 bytes (pool eliminates the 4000-byte scratch allocation).
func BenchmarkEncodePCMToOpus_48kHz(b *testing.B) {
	codec := newPionAudioCodec().(*pionAudioCodec)
	pcm := pcmSine(48000, 48000/50) // 960 samples = 1920 bytes
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := codec.EncodePCMToOpus(pcm, 48000, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEncodePCMToOpus_24kHz is the same but at 24kHz (Cartesia / Deepgram default).
// frameBytes = (24000/50)*2 = 960 bytes = 480 samples.
func BenchmarkEncodePCMToOpus_24kHz(b *testing.B) {
	codec := newPionAudioCodec().(*pionAudioCodec)
	pcm := pcmSine(24000, 24000/50) // 480 samples = 960 bytes
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := codec.EncodePCMToOpus(pcm, 24000, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOutputProcessorHandleFrame measures the full egress path:
// HandleFrame → sendAudioFrame (chunking) → EncodePCMToOpus → WriteOpus.
// Input: one TTSAudioFrame containing exactly two 20ms frames at 24kHz.
// The mock track's WriteOpus is a slice append — negligible cost.
func BenchmarkOutputProcessorHandleFrame(b *testing.B) {
	peer := newMockPeer()
	transport := NewDailyTransport(DailyConfig{
		APIKey:     "k",
		APIBaseURL: "http://invalid",
		RoomName:   "room",
	})
	transport.peerFactory = func(cfg webrtc.Configuration) (peerConnection, error) { return peer, nil }
	transport.joinRoomFn = func(ctx context.Context) error {
		transport.meetingToken = "token"
		return nil
	}
	if err := transport.Connect(context.Background()); err != nil {
		b.Fatalf("connect: %v", err)
	}
	defer transport.Disconnect()

	const sampleRate = 24000
	pcm := pcmSine(sampleRate, (sampleRate/50)*2) // 2 × 20ms frames = 960 bytes
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tts := frames.NewTTSAudioFrame(pcm, sampleRate, 1)
		tts.SetMetadata("codec", "linear16")
		if err := transport.outputProc.HandleFrame(ctx, tts, frames.Downstream); err != nil {
			b.Fatal(err)
		}
	}
}
