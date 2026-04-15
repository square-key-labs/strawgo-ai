//go:build ignore

// generate_fixtures produces binary ground-truth fixtures from the Go ONNX code
// (silero.go + whisper_features.go) before any of that code is deleted.
// These fixtures are compared against the Rust sidecar to verify identical outputs.
//
// Run from the project root (required for ONNX dylib relative path):
//
//	go run ./cmd/generate_fixtures/
//
// The program always runs unconditionally; gate it via the GENERATE_FIXTURES env
// var if you want CI to skip it:
//
//	GENERATE_FIXTURES=1 go run ./cmd/generate_fixtures/
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
	"github.com/square-key-labs/strawgo-ai/src/models"
	ort "github.com/yalue/onnxruntime_go"
)

// Sample rates and durations for synthetic audio
const (
	sampleRate = 16000 // Hz — Silero and Whisper both require 16 kHz
	freq440Hz  = 440.0 // Hz — concert A
)

// sineWave generates numSamples of a pure sine wave at freqHz, normalized to [-1, 1]
func sineWave(numSamples int, freqHz, srHz float64) []float32 {
	out := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		out[i] = float32(math.Sin(2 * math.Pi * freqHz * float64(i) / srHz))
	}
	return out
}

// silence returns a slice of numSamples zero-valued float32 samples
func silence(numSamples int) []float32 {
	return make([]float32, numSamples)
}

// float32ToInt16LE converts normalized float32 audio to little-endian int16 PCM bytes.
// This is the format VoiceConfidence() expects.
func float32ToInt16LE(samples []float32) []byte {
	buf := make([]byte, len(samples)*2)
	for i, s := range samples {
		// Clamp to [-1, 1] before scaling
		if s > 1.0 {
			s = 1.0
		} else if s < -1.0 {
			s = -1.0
		}
		v := int16(s * 32767.0)
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
	}
	return buf
}

// float32SliceToBytes serialises a []float32 as raw IEEE 754 little-endian bytes.
func float32SliceToBytes(vals []float32) []byte {
	buf := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// writePCM writes raw int16 LE PCM bytes to path.
func writePCM(path string, pcm []byte) error {
	return os.WriteFile(path, pcm, 0644)
}

// writeMel writes a []float32 mel spectrogram as raw IEEE 754 LE bytes.
func writeMel(path string, mel []float32) error {
	return os.WriteFile(path, float32SliceToBytes(mel), 0644)
}

// writeVADChunks runs VoiceConfidence on successive 512-sample chunks of pcmBytes
// (at 16 kHz) and writes the resulting []float32 confidence values as a binary file.
// Silero VAD requires exactly 512 samples (1024 bytes) per call at 16 kHz.
// Leftover samples that don't fill a complete chunk are discarded.
//
// Returns the number of chunks written.
func writeVADChunks(path string, analyzer *vad.SileroVADAnalyzer, pcmBytes []byte) (int, error) {
	const chunkBytes = 512 * 2 // 512 int16 samples × 2 bytes each

	var confidences []float32
	for off := 0; off+chunkBytes <= len(pcmBytes); off += chunkBytes {
		chunk := pcmBytes[off : off+chunkBytes]
		c := analyzer.VoiceConfidence(chunk)
		confidences = append(confidences, c)
	}

	if len(confidences) == 0 {
		return 0, fmt.Errorf("no complete 512-sample chunks in %d byte buffer", len(pcmBytes))
	}

	return len(confidences), os.WriteFile(path, float32SliceToBytes(confidences), 0644)
}

// audioSample holds named synthetic audio in both float32 (for mel) and PCM (for VAD) form
type audioSample struct {
	name    string // human label, used in log output
	stem    string // file name stem (no extension)
	float32 []float32
	pcm     []byte
}

func main() {
	// Allow CI to skip fixture generation by checking env var
	if os.Getenv("GENERATE_FIXTURES") == "" {
		fmt.Fprintln(os.Stderr, "Set GENERATE_FIXTURES=1 to run fixture generation. Exiting.")
		os.Exit(0)
	}

	outDir := "testdata"
	if err := os.MkdirAll(outDir, 0755); err != nil {
		fatalf("creating testdata dir: %v", err)
	}

	// -------------------------------------------------------------------------
	// Synthetic audio samples
	// -------------------------------------------------------------------------
	samples100ms := 1600 // 100 ms × 16000 Hz
	samples500ms := 8000 // 500 ms × 16000 Hz

	silenceF32 := silence(samples100ms)
	sine100F32 := sineWave(samples100ms, freq440Hz, sampleRate)
	sine500F32 := sineWave(samples500ms, freq440Hz, sampleRate)

	audioSamples := []audioSample{
		{
			name:    "silence 100ms",
			stem:    "silence_100ms_16k",
			float32: silenceF32,
			pcm:     float32ToInt16LE(silenceF32),
		},
		{
			name:    "sine 440 Hz 100ms",
			stem:    "sine_440_100ms_16k",
			float32: sine100F32,
			pcm:     float32ToInt16LE(sine100F32),
		},
		{
			name:    "sine 440 Hz 500ms",
			stem:    "sine_440_500ms_16k",
			float32: sine500F32,
			pcm:     float32ToInt16LE(sine500F32),
		},
	}

	// -------------------------------------------------------------------------
	// Write raw PCM files
	// -------------------------------------------------------------------------
	fmt.Println("=== Writing PCM files ===")
	for _, s := range audioSamples {
		path := filepath.Join(outDir, s.stem+".pcm")
		if err := writePCM(path, s.pcm); err != nil {
			fatalf("writing PCM %s: %v", path, err)
		}
		fmt.Printf("  wrote %s (%d bytes, %d int16 samples)\n", path, len(s.pcm), len(s.pcm)/2)
	}

	// -------------------------------------------------------------------------
	// Mel spectrogram extraction (pure Go — no model required)
	// WhisperFeatureExtractor.Extract() returns []float32.
	// maxLengthSamples is set to the actual sample count so no artificial
	// padding or truncation is applied; the Rust side must use the same value.
	// -------------------------------------------------------------------------
	fmt.Println("\n=== Writing mel spectrogram fixtures ===")
	fe := turn.NewWhisperFeatureExtractor()
	melStems := []string{"mel_silence_100ms", "mel_sine_440_100ms", "mel_sine_440_500ms"}
	for i, s := range audioSamples {
		maxLen := len(s.float32)
		mel := fe.Extract(s.float32, maxLen)
		path := filepath.Join(outDir, melStems[i]+".bin")
		if err := writeMel(path, mel); err != nil {
			fatalf("writing mel %s: %v", path, err)
		}
		fmt.Printf("  wrote %s (%d float32 values, maxLengthSamples=%d)\n", path, len(mel), maxLen)
	}

	// -------------------------------------------------------------------------
	// Silero VAD confidence extraction
	// Requires:
	//   1. A compatible ONNX Runtime dylib. The onnxruntime_go v1.20.0 module
	//      uses ORT API version 22, which requires libonnxruntime ≥ 1.22.x.
	//      The project's lib/libonnxruntime.1.20.1.dylib is ORT 1.20.1 (API 20)
	//      and will NOT work.
	//
	//      We pre-initialise ORT here with the system dylib
	//      (/usr/local/lib/libonnxruntime.dylib → 1.22.0) so that silero.go's
	//      sync.Once sees an already-initialised environment and skips its own
	//      (incompatible) dylib selection.
	//
	//   2. Silero model at ~/.cache/strawgo/models/silero_vad.onnx
	//      (downloaded automatically on first run)
	//
	// The model runs on exactly 512-sample (1024-byte) chunks at 16 kHz.
	// Chunk counts per sample:
	//   100ms (1600 samples) → 3 chunks  (64 leftover samples discarded)
	//   500ms (8000 samples) → 15 chunks (320 leftover samples discarded)
	//
	// Each .f32 file is a raw sequence of IEEE 754 LE float32 values — one per
	// chunk — NOT a single scalar.
	// -------------------------------------------------------------------------
	fmt.Println("\n=== Writing Silero VAD fixtures ===")

	// Pre-initialise ORT with a compatible dylib before silero.go's sync.Once fires.
	// Priority order: ONNX_RUNTIME_LIB env var → system symlink → Homebrew.
	ortLibPath := os.Getenv("ONNX_RUNTIME_LIB")
	if ortLibPath == "" {
		candidates := []string{
			"/usr/local/lib/libonnxruntime.dylib",    // macOS Intel / most installs
			"/opt/homebrew/lib/libonnxruntime.dylib", // macOS Apple Silicon via Homebrew
		}
		for _, p := range candidates {
			if _, err := os.Stat(p); err == nil {
				ortLibPath = p
				break
			}
		}
	}
	if ortLibPath != "" {
		fmt.Printf("  Pre-initialising ORT with: %s\n", ortLibPath)
		ort.SetSharedLibraryPath(ortLibPath)
		if err := ort.InitializeEnvironment(); err != nil {
			fmt.Fprintf(os.Stderr, "  WARNING: ORT pre-init failed: %v\n", err)
			fmt.Fprintln(os.Stderr, "  Skipping VAD fixtures.")
			goto done
		}
		fmt.Printf("  ORT version: %s\n", ort.GetVersion())
	} else {
		fmt.Fprintln(os.Stderr, "  No compatible ONNX Runtime dylib found (set ONNX_RUNTIME_LIB to override).")
		fmt.Fprintln(os.Stderr, "  Skipping VAD fixtures.")
		goto done
	}

	{
		modelPath := filepath.Join(models.CacheDir(), models.SileroVADFile)
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			fmt.Printf("  Silero model not found at %s — attempting download…\n", modelPath)
		}

		var analyzer *vad.SileroVADAnalyzer
		var err error
		analyzer, err = vad.NewSileroVADAnalyzer(sampleRate, vad.DefaultVADParams())
		if err != nil {
			fmt.Fprintf(os.Stderr, "  WARNING: could not load Silero VAD model: %v\n", err)
			fmt.Fprintln(os.Stderr, "  Skipping VAD fixtures. Run again after downloading the model.")
		} else {
			defer analyzer.Cleanup()

			vadStems := []string{"vad_silence_100ms", "vad_sine_440_100ms", "vad_sine_440_500ms"}
			for i, s := range audioSamples {
				path := filepath.Join(outDir, vadStems[i]+".f32")
				n, err := writeVADChunks(path, analyzer, s.pcm)
				if err != nil {
					fmt.Fprintf(os.Stderr, "  ERROR writing %s: %v\n", path, err)
					continue
				}
				fmt.Printf("  wrote %s (%d chunks × 1 float32)\n", path, n)

				// Reset model state between samples so each sample is evaluated
				// independently (as a fresh utterance), matching Rust test behaviour.
				analyzer.Restart()
			}
		}
	}

done:
	fmt.Println("\nDone. Fixtures written to", outDir)
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "FATAL: "+format+"\n", args...)
	os.Exit(1)
}
