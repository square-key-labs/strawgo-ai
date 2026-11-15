//go:build integration

package audio_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
	"github.com/square-key-labs/strawgo-ai/src/worker"
)

func repoRoot() string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "..", "..")
}

func workerBinaryPath() string {
	return filepath.Join(repoRoot(), "onnx-worker", "target", "release", "onnx-worker")
}

func testdataPath(name string) string {
	return filepath.Join(repoRoot(), "testdata", name)
}

func readPCM(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(testdataPath(name))
	if err != nil {
		t.Fatalf("readPCM %s: %v", name, err)
	}
	return data
}

func readF32(t *testing.T, name string) []float32 {
	t.Helper()
	data, err := os.ReadFile(testdataPath(name))
	if err != nil {
		t.Fatalf("readF32 %s: %v", name, err)
	}
	if len(data)%4 != 0 {
		t.Fatalf("readF32 %s: size %d not multiple of 4", name, len(data))
	}
	vals := make([]float32, len(data)/4)
	for i := range vals {
		bits := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
		vals[i] = math.Float32frombits(bits)
	}
	return vals
}

func startSupervisor(t *testing.T) *worker.Supervisor {
	t.Helper()
	binary := workerBinaryPath()
	if _, err := os.Stat(binary); err != nil {
		t.Fatalf("onnx-worker binary not found at %s: %v\nBuild with: cargo build --release --manifest-path onnx-worker/Cargo.toml", binary, err)
	}
	sup, err := worker.Start(binary)
	if err != nil {
		t.Fatalf("worker.Start: %v", err)
	}
	return sup
}

func TestVADConfidenceParity(t *testing.T) {
	sup := startSupervisor(t)
	defer sup.Stop()

	cases := []struct {
		pcm  string
		refs string
	}{
		{"sine_440_100ms_16k.pcm", "vad_sine_440_100ms.f32"},
		{"silence_100ms_16k.pcm", "vad_silence_100ms.f32"},
		{"sine_440_500ms_16k.pcm", "vad_sine_440_500ms.f32"},
	}

	const sampleRate = 16000
	const tolerance float32 = 0.05 // Allow up to 5% absolute difference per frame

	for _, tc := range cases {
		t.Run(tc.pcm, func(t *testing.T) {
			pcmBytes := readPCM(t, tc.pcm)
			refConfs := readF32(t, tc.refs)

			client, err := vad.NewOnnxVADClient(sup.SocketPath())
			if err != nil {
				t.Fatalf("NewOnnxVADClient: %v", err)
			}
			defer client.Close()

			const frameBytes = 512 * 2 // 512 int16 = 1024 bytes
			frameIdx := 0
			var maxDiff float32

			for offset := 0; offset+frameBytes <= len(pcmBytes); offset += frameBytes {
				chunk := pcmBytes[offset : offset+frameBytes]
				conf, err := client.VoiceConfidence(chunk, sampleRate)
				if err != nil {
					t.Fatalf("VoiceConfidence frame %d: %v", frameIdx, err)
				}

				if frameIdx < len(refConfs) {
					diff := conf - refConfs[frameIdx]
					if diff < 0 {
						diff = -diff
					}
					if diff > maxDiff {
						maxDiff = diff
					}
					if diff > tolerance {
						t.Errorf("frame %d: conf=%.4f ref=%.4f diff=%.4f > tolerance=%.4f",
							frameIdx, conf, refConfs[frameIdx], diff, tolerance)
					}
				}
				frameIdx++
			}
			if frameIdx != len(refConfs) {
				t.Errorf("frame count mismatch: produced %d, reference has %d", frameIdx, len(refConfs))
			}
			t.Logf("%s: %d frames, max_diff=%.6f", tc.pcm, frameIdx, maxDiff)
		})
	}
}

func TestVADLatency(t *testing.T) {
	sup := startSupervisor(t)
	defer sup.Stop()

	client, err := vad.NewOnnxVADClient(sup.SocketPath())
	if err != nil {
		t.Fatalf("NewOnnxVADClient: %v", err)
	}
	defer client.Close()

	pcmBytes := readPCM(t, "sine_440_100ms_16k.pcm")
	const frameBytes = 512 * 2
	if len(pcmBytes) < frameBytes {
		t.Skip("PCM fixture too short for latency test")
	}
	chunk := pcmBytes[:frameBytes]

	const warmup = 3
	for i := 0; i < warmup; i++ {
		if _, err := client.VoiceConfidence(chunk, 16000); err != nil {
			t.Fatalf("warmup call %d: %v", i, err)
		}
	}

	const N = 100
	latencies := make([]time.Duration, N)
	for i := range latencies {
		start := time.Now()
		_, err := client.VoiceConfidence(chunk, 16000)
		latencies[i] = time.Since(start)
		if err != nil {
			t.Fatalf("VoiceConfidence call %d: %v", i, err)
		}
	}

	sortDurations(latencies)
	p50 := latencies[N*50/100]
	p95 := latencies[N*95/100]
	p99 := latencies[N*99/100]
	max := latencies[N-1]

	t.Logf("VAD latency over %d calls: p50=%v p95=%v p99=%v max=%v", N, p50, p95, p99, max)

	const target = 5 * time.Millisecond
	if p99 > target {
		t.Errorf("p99 latency %v exceeds target %v", p99, target)
	}
}

func sortDurations(d []time.Duration) {
	for i := 1; i < len(d); i++ {
		key := d[i]
		j := i - 1
		for j >= 0 && d[j] > key {
			d[j+1] = d[j]
			j--
		}
		d[j+1] = key
	}
}

func TestVADStateIsolation(t *testing.T) {
	sup := startSupervisor(t)
	defer sup.Stop()

	const sampleRate = 16000
	const frameBytes = 512 * 2

	sinePCM := readPCM(t, "sine_440_100ms_16k.pcm")
	silencePCM := readPCM(t, "silence_100ms_16k.pcm")

	if len(sinePCM) < frameBytes || len(silencePCM) < frameBytes {
		t.Skip("fixtures too short")
	}
	sineChunk := sinePCM[:frameBytes]
	silenceChunk := silencePCM[:frameBytes]

	const frames = 5

	refClient, err := vad.NewOnnxVADClient(sup.SocketPath())
	if err != nil {
		t.Fatalf("NewOnnxVADClient (reference): %v", err)
	}
	refConfs := make([]float32, frames)
	for j := 0; j < frames; j++ {
		c, err := refClient.VoiceConfidence(silenceChunk, sampleRate)
		if err != nil {
			t.Fatalf("reference silence frame %d: %v", j, err)
		}
		refConfs[j] = c
	}
	refClient.Close()

	const N = 5
	var wg sync.WaitGroup
	silenceConfs := make([]float32, frames)
	var silenceErr error

	wg.Add(1)
	go func() {
		defer wg.Done()
		c, err := vad.NewOnnxVADClient(sup.SocketPath())
		if err != nil {
			silenceErr = err
			return
		}
		defer c.Close()
		for j := 0; j < frames; j++ {
			conf, err := c.VoiceConfidence(silenceChunk, sampleRate)
			if err != nil {
				silenceErr = fmt.Errorf("silence client frame %d: %w", j, err)
				return
			}
			silenceConfs[j] = conf
		}
	}()

	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := vad.NewOnnxVADClient(sup.SocketPath())
			if err != nil {
				return // non-fatal for the sine goroutines
			}
			defer c.Close()
			for j := 0; j < frames; j++ {
				c.VoiceConfidence(sineChunk, sampleRate) //nolint:errcheck
			}
		}()
	}

	wg.Wait()

	if silenceErr != nil {
		t.Fatalf("silence client: %v", silenceErr)
	}

	const isolationTolerance float32 = 0.01
	for j := 0; j < frames; j++ {
		diff := silenceConfs[j] - refConfs[j]
		if diff < 0 {
			diff = -diff
		}
		if diff > isolationTolerance {
			t.Errorf("frame %d: concurrent silence conf=%.6f, isolated ref=%.6f, diff=%.6f > tolerance=%.6f",
				j, silenceConfs[j], refConfs[j], diff, isolationTolerance)
		}
	}
	t.Logf("State isolation verified over %d frames (max diff within %.4f)", frames, isolationTolerance)
}

func TestSmartTurnInference(t *testing.T) {
	sup := startSupervisor(t)
	defer sup.Stop()

	client, err := turn.NewOnnxTurnClient(sup.SocketPath())
	if err != nil {
		t.Fatalf("NewOnnxTurnClient: %v", err)
	}
	defer client.Close()

	pcmBytes := readPCM(t, "sine_440_500ms_16k.pcm")

	pred, err := client.Analyze(pcmBytes, 16000, 0)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}

	if pred.Probability < 0.0 || pred.Probability > 1.0 {
		t.Errorf("probability %.4f out of range [0, 1]", pred.Probability)
	}
	t.Logf("Smart Turn probability for sine_440_500ms: %.4f (> 0.5 = TurnComplete: %v)",
		pred.Probability, pred.Probability > 0.5)
}

func TestCrashRecovery(t *testing.T) {
	sup := startSupervisor(t)
	defer sup.Stop()

	client, err := vad.NewOnnxVADClient(sup.SocketPath())
	if err != nil {
		t.Fatalf("NewOnnxVADClient: %v", err)
	}
	defer client.Close()

	pcmBytes := readPCM(t, "sine_440_100ms_16k.pcm")
	chunk := pcmBytes[:512*2]

	if _, err := client.VoiceConfidence(chunk, 16000); err != nil {
		t.Fatalf("baseline call failed: %v", err)
	}

	sup.KillWorkerForTesting()

	disappearedDeadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(disappearedDeadline) {
		if _, err := os.Stat(sup.SocketPath()); err != nil {
			break // socket gone
		}
		time.Sleep(20 * time.Millisecond)
	}

	restartDeadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(restartDeadline) {
		if _, err := os.Stat(sup.SocketPath()); err == nil {
			break // socket back
		}
		time.Sleep(50 * time.Millisecond)
	}

	time.Sleep(100 * time.Millisecond)

	conf, err := client.VoiceConfidence(chunk, 16000)
	if err != nil {
		t.Fatalf("post-restart VoiceConfidence via auto-reconnect: %v", err)
	}
	t.Logf("Post-restart confidence (via auto-reconnect): %.4f", conf)
}
