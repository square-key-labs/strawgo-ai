// gen_burst_pcm — synthesize a realistic-ish 16 kHz int16 PCM that simulates
// a phone call: quiet noise preamble, periodic ~600 ms speech-like bursts,
// quiet between. Used by loadtest-pipeline to exercise the SNR gate.
//
// Build & run:
//
//	go run bench/gen_burst_pcm.go -out bench-burst-call-10s.pcm -dur 10
//
// The output frame layout matches Strawgo's 32 ms / 512-sample expectation:
// 10 s × 16 kHz = 160 000 samples = 320 000 bytes. Loops cleanly.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
)

func main() {
	out := flag.String("out", "bench-burst-call.pcm", "output PCM path")
	durSec := flag.Int("dur", 10, "duration seconds")
	noiseAmp := flag.Int("noise-amp", 600, "noise amplitude (int16, ~-35 dBFS)")
	burstAmp := flag.Int("burst-amp", 12000, "speech-burst amplitude (~-9 dBFS)")
	burstPeriodMs := flag.Int("burst-period-ms", 2500, "burst-on every N ms")
	burstWidthMs := flag.Int("burst-width-ms", 600, "each burst is N ms long")
	flag.Parse()

	const sr = 16000
	n := *durSec * sr
	r := rand.New(rand.NewSource(1))

	buf := make([]int16, n)
	periodSamples := *burstPeriodMs * sr / 1000
	widthSamples := *burstWidthMs * sr / 1000

	for i := 0; i < n; i++ {
		// Background noise (~-35 dBFS).
		noise := float64(*noiseAmp) * (2*r.Float64() - 1)

		// Burst on?
		phase := i % periodSamples
		v := noise
		if phase < widthSamples {
			// Voiced segment: 200 Hz F0 + light noise (formants approximated by
			// adding 800 Hz overtone). Envelope: cosine taper to avoid clicks.
			t := float64(phase) / float64(sr)
			env := 0.5 * (1 - math.Cos(2*math.Pi*float64(phase)/float64(widthSamples)))
			tone := math.Sin(2*math.Pi*200*t) + 0.3*math.Sin(2*math.Pi*800*t)
			v += float64(*burstAmp) * env * tone * 0.6
		}

		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		buf[i] = int16(v)
	}

	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, buf); err != nil {
		fmt.Fprintf(os.Stderr, "write: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %d samples (%d bytes, %.1f s @ 16 kHz) to %s\n",
		len(buf), len(buf)*2, float64(len(buf))/float64(sr), *out)
	fmt.Printf("burst pattern: every %d ms, width %d ms, noise amp %d, burst amp %d\n",
		*burstPeriodMs, *burstWidthMs, *noiseAmp, *burstAmp)
}
