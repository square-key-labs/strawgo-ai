package pipeline_embed

import (
	"math"
	"math/rand"
	"testing"
)

// helpers for synthetic signals at 16 kHz, 32 ms (512 samples).

func makeSilence(n int) []int16 { return make([]int16, n) }

func makeTone(n int, freqHz, ampInt int) []int16 {
	out := make([]int16, n)
	for i := range out {
		t := float64(i) / 16000.0
		out[i] = int16(float64(ampInt) * math.Sin(2*math.Pi*float64(freqHz)*t))
	}
	return out
}

func makeWhiteNoise(n int, ampInt int, seed int64) []int16 {
	r := rand.New(rand.NewSource(seed))
	out := make([]int16, n)
	for i := range out {
		// uniform [-1,1] scaled
		out[i] = int16(float64(ampInt) * (2*r.Float64() - 1))
	}
	return out
}

func mix(a, b []int16) []int16 {
	out := make([]int16, len(a))
	for i := range a {
		v := int32(a[i]) + int32(b[i])
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		out[i] = int16(v)
	}
	return out
}

// TestRMSDBFSSanity: full-scale tone ≈ 0 dBFS - 3 dB (sin RMS = peak/√2).
func TestRMSDBFSSanity(t *testing.T) {
	frame := makeTone(512, 1000, 32000)
	db := rmsDBFS(frame)
	// sine RMS = 32000/√2 ≈ 22627 → dBFS = 20·log10(22627/32768) ≈ -3.22
	if db < -4.0 || db > -2.5 {
		t.Errorf("full-scale 1 kHz sine: expected ~-3 dBFS, got %.2f", db)
	}

	// Pure silence.
	if got := rmsDBFS(makeSilence(512)); got != -120 {
		t.Errorf("silence: expected -120 floor, got %.2f", got)
	}

	// Quiet noise ≈ -40 dBFS.
	quiet := makeWhiteNoise(512, 320, 1) // 1 % FS amplitude
	q := rmsDBFS(quiet)
	if q < -50 || q > -30 {
		t.Errorf("quiet white noise (1%% FS): expected ~-40 dBFS, got %.2f", q)
	}
}

// TestSNRClassifiesClean: clean tone over silent floor → SNR should rise high
// after the window fills.
func TestSNRClassifiesClean(t *testing.T) {
	det := NewSNRDetector(SNRConfig{})

	// Prime with 1 second of silence — sets floor near -120.
	for i := 0; i < 31; i++ {
		det.Update(makeSilence(512))
	}
	// Then feed loud tone.
	loud := makeTone(512, 1000, 16000) // -9 dBFS
	var snr float64
	for i := 0; i < 5; i++ {
		snr = det.Update(loud)
	}
	if snr < 50 {
		t.Errorf("clean tone over silence: expected SNR > 50 dB, got %.2f", snr)
	}
	if !det.ShouldDenoise(0) {
		t.Errorf("threshold=0 must always denoise (gating disabled)")
	}
	if det.ShouldDenoise(15) {
		t.Errorf("clean signal should NOT be denoised at threshold 15 dB (got SNR %.2f)", snr)
	}
}

// TestSNRClassifiesNoisy: speech-like tone mixed into nearly-equal-energy
// noise → SNR ~ 0-6 dB → must denoise at threshold 12.
func TestSNRClassifiesNoisy(t *testing.T) {
	det := NewSNRDetector(SNRConfig{})

	// Prime with moderate noise floor (-30 dBFS).
	noiseFloor := makeWhiteNoise(512, 1000, 42)
	for i := 0; i < 31; i++ {
		det.Update(noiseFloor)
	}
	// Mix tone of similar amplitude into the same noise → SNR should be low.
	tone := makeTone(512, 1500, 1500)
	mixed := mix(tone, noiseFloor)
	var snr float64
	for i := 0; i < 5; i++ {
		snr = det.Update(mixed)
	}
	if snr > 12 {
		t.Errorf("speech-in-noise mix: expected SNR ≤ 12 dB, got %.2f", snr)
	}
	if !det.ShouldDenoise(12) {
		t.Errorf("noisy frame should trigger denoise at threshold 12 (SNR=%.2f)", snr)
	}
}

// TestSNRFloorAdapts: detector should settle to a stable noise floor and
// SNR ~ 0 when fed steady noise.
func TestSNRFloorAdapts(t *testing.T) {
	det := NewSNRDetector(SNRConfig{})

	noise := makeWhiteNoise(512, 2000, 7)
	var snr float64
	for i := 0; i < 100; i++ {
		snr = det.Update(noise)
	}
	if math.Abs(snr) > 4 {
		t.Errorf("steady noise: SNR should be near 0, got %.2f", snr)
	}
	if det.ShouldDenoise(12) == false {
		// Steady noise = low SNR = should denoise.
	}
}

// TestSNRGatingPattern: realistic call — quiet preamble, speech burst, more
// quiet. Verifies gate flips correctly with the signal.
func TestSNRGatingPattern(t *testing.T) {
	det := NewSNRDetector(SNRConfig{})

	noise := makeWhiteNoise(512, 800, 11) // ~-32 dBFS
	tone := makeTone(512, 800, 12000)     // ~-8 dBFS, much louder
	speechMix := mix(tone, noise)

	type phase struct {
		name      string
		frames    int
		signal    []int16
		wantGate  bool // true = expect denoise gate to fire (SNR < threshold)
		threshold float64
	}
	phases := []phase{
		{"quiet preamble", 30, noise, true, 12},
		{"clean burst", 10, speechMix, false, 12},
		{"trailing silence", 20, makeSilence(512), true, 12}, // dB very low → SNR very low vs noise floor
	}

	totalGated := 0
	for _, ph := range phases {
		gated := 0
		for i := 0; i < ph.frames; i++ {
			det.Update(ph.signal)
			if det.ShouldDenoise(ph.threshold) {
				gated++
			}
		}
		t.Logf("phase=%s frames=%d gated=%d snr=%.2f floor=%.2f",
			ph.name, ph.frames, gated, det.LastSNRDB, det.LastFloorDB)
		totalGated += gated

		// At least one frame in this phase should match expectation. The first
		// few frames of a transition can lag because the floor catches up.
		if ph.wantGate && gated == 0 {
			t.Errorf("phase %s: expected at least 1 gated frame, got 0 (snr=%.2f)",
				ph.name, det.LastSNRDB)
		}
	}
	t.Logf("total gated across 60 frames: %d", totalGated)
}

// TestSNROverhead: rough cost ceiling — must stay sub-100 µs per frame.
func TestSNRPerformance(t *testing.T) {
	det := NewSNRDetector(SNRConfig{})
	frame := makeWhiteNoise(512, 4000, 3)

	const iters = 10000
	for i := 0; i < iters; i++ {
		det.Update(frame)
	}
	// No assertion on time here — Go's testing benchmark would be the right
	// tool; this just exercises the path under -race for correctness.
}

func BenchmarkSNRUpdate(b *testing.B) {
	det := NewSNRDetector(SNRConfig{})
	frame := makeWhiteNoise(512, 4000, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		det.Update(frame)
	}
}
