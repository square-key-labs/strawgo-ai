package tone

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/dtmf"
	"github.com/square-key-labs/strawgo-ai/src/audio/g711"
)

func samples(rate int, d time.Duration) int {
	return int(d.Seconds() * float64(rate))
}

func sine(rate int, freq, amp float64, d time.Duration) []float64 {
	out := make([]float64, samples(rate, d))
	for i := range out {
		out[i] = amp * math.Sin(2*math.Pi*freq*float64(i)/float64(rate))
	}
	return out
}

func silence(rate int, d time.Duration) []float64 {
	return make([]float64, samples(rate, d))
}

func noise(rate int, amp float64, d time.Duration, seed uint64) []float64 {
	r := rand.New(rand.NewPCG(seed, seed+1))
	out := make([]float64, samples(rate, d))
	for i := range out {
		envelope := 0.5 + 0.5*math.Sin(2*math.Pi*4*float64(i)/float64(rate))
		out[i] = amp * envelope * (2*r.Float64() - 1)
	}
	return out
}

func vowel(rate int, f0 float64, d time.Duration) []float64 {
	formants := []float64{700, 1200, 2500}
	out := make([]float64, samples(rate, d))
	for h := 1; float64(h)*f0 < float64(rate)/2; h++ {
		f := float64(h) * f0
		gain := 0.0
		for _, fm := range formants {
			gain += 1 / (1 + math.Pow((f-fm)/120, 2))
		}
		for i := range out {
			out[i] += 3000 * gain * math.Sin(2*math.Pi*f*float64(i)/float64(rate))
		}
	}
	return out
}

func mix(a, b []float64) []float64 {
	out := append([]float64(nil), a...)
	for i := range out {
		if i < len(b) {
			out[i] += b[i]
		}
	}
	return out
}

func encode(parts ...[]float64) []byte {
	var out []byte
	for _, p := range parts {
		for _, v := range p {
			out = binary.LittleEndian.AppendUint16(out, uint16(int16(math.Max(math.Min(math.Round(v), 32767), -32768))))
		}
	}
	return out
}

func run(t *testing.T, rate int, pcm []byte, chunk int) []Event {
	t.Helper()
	d, err := NewDetector(DefaultConfig(rate))
	if err != nil {
		t.Fatal(err)
	}
	var events []Event
	for len(pcm) > 0 {
		n := min(chunk, len(pcm))
		events = append(events, d.Write(pcm[:n])...)
		pcm = pcm[n:]
	}
	return events
}

func TestDetectsBeepAtTelephonyAndWidebandRates(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		for _, freq := range []float64{440, 850, 1000, 1400} {
			pcm := encode(silence(rate, 300*time.Millisecond), sine(rate, freq, 8000, 600*time.Millisecond), silence(rate, 300*time.Millisecond))
			events := withoutConfirmed(run(t, rate, pcm, 2*samples(rate, 20*time.Millisecond)))
			if len(events) != 2 || events[0].Kind != Started || events[1].Kind != Ended {
				t.Fatalf("rate %d freq %.0f: events %+v", rate, freq, events)
			}
			if math.Abs(events[0].Frequency-freq) > 50 {
				t.Fatalf("rate %d freq %.0f: reported %.0f", rate, freq, events[0].Frequency)
			}
			if got := events[1].Duration; got < 500*time.Millisecond || got > 700*time.Millisecond {
				t.Fatalf("rate %d freq %.0f: duration %v", rate, freq, got)
			}
		}
	}
}

func TestDetectsBeepAcrossOddChunks(t *testing.T) {
	pcm := encode(silence(8000, 100*time.Millisecond), sine(8000, 1004, 6000, 500*time.Millisecond), silence(8000, 100*time.Millisecond))
	events := withoutConfirmed(run(t, 8000, pcm, 37))
	if len(events) != 2 || events[0].Kind != Started || events[1].Kind != Ended {
		t.Fatalf("events %+v", events)
	}
}

func TestDetectsBeepOverBackgroundNoise(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		beep := mix(sine(rate, 950, 8000, 500*time.Millisecond), noise(rate, 800, 500*time.Millisecond, 7))
		events := run(t, rate, encode(beep, silence(rate, 100*time.Millisecond)), 320)
		if len(events) == 0 || events[0].Kind != Started {
			t.Fatalf("rate %d: events %+v", rate, events)
		}
	}
}

func TestIgnoresShortTone(t *testing.T) {
	pcm := encode(sine(8000, 1000, 8000, 150*time.Millisecond), silence(8000, 200*time.Millisecond))
	if events := run(t, 8000, pcm, 320); len(events) != 0 {
		t.Fatalf("events %+v", events)
	}
}

func TestIgnoresQuietTone(t *testing.T) {
	pcm := encode(sine(8000, 1000, 100, 600*time.Millisecond))
	if events := run(t, 8000, pcm, 320); len(events) != 0 {
		t.Fatalf("events %+v", events)
	}
}

func TestIgnoresDTMFTones(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		for _, key := range []byte("0123456789*#ABCD") {
			pcm, err := dtmf.ToneWithOptions(key, rate, dtmf.Options{Tone: time.Second, Gap: 100 * time.Millisecond, Amplitude: 0.3})
			if err != nil {
				t.Fatal(err)
			}
			if events := run(t, rate, pcm, 320); len(events) != 0 {
				t.Fatalf("rate %d key %c: events %+v", rate, key, events)
			}
		}
	}
}

func TestIgnoresSustainedVowel(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		for _, f0 := range []float64{110, 140, 220} {
			pcm := encode(vowel(rate, f0, 1500*time.Millisecond))
			if events := run(t, rate, pcm, 320); len(events) != 0 {
				t.Fatalf("rate %d f0 %.0f: events %+v", rate, f0, events)
			}
		}
	}
}

func TestIgnoresSpeechLikeNoise(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		pcm := encode(noise(rate, 10000, 2*time.Second, 3))
		if events := run(t, rate, pcm, 320); len(events) != 0 {
			t.Fatalf("rate %d: events %+v", rate, events)
		}
	}
}

func TestIgnoresBusyTonesBelowBand(t *testing.T) {
	for _, freq := range []float64{400, 425} {
		pcm := encode(sine(8000, freq, 8000, time.Second))
		if events := run(t, 8000, pcm, 320); len(events) != 0 {
			t.Fatalf("freq %.0f: events %+v", freq, events)
		}
	}
}

func TestResetDropsPartialTone(t *testing.T) {
	d, err := NewDetector(DefaultConfig(8000))
	if err != nil {
		t.Fatal(err)
	}
	d.Write(encode(sine(8000, 1000, 8000, 200*time.Millisecond)))
	d.Reset()
	if events := d.Write(encode(sine(8000, 1000, 8000, 200*time.Millisecond))); len(events) != 0 {
		t.Fatalf("events %+v", events)
	}
}

func TestRejectsInvalidConfig(t *testing.T) {
	if _, err := NewDetector(Config{}); err == nil {
		t.Fatal("zero sample rate accepted")
	}
	if _, err := NewDetector(Config{SampleRate: 2000}); err == nil {
		t.Fatal("band above Nyquist accepted")
	}
}

func partials(rate int, d time.Duration, parts ...[2]float64) []float64 {
	out := make([]float64, samples(rate, d))
	for _, p := range parts {
		for i := range out {
			out[i] += p[1] * math.Sin(2*math.Pi*p[0]*float64(i)/float64(rate)+0.3)
		}
	}
	return out
}

func vibrato(rate int, freq, depth, speed, amp float64, d time.Duration) []float64 {
	out := make([]float64, samples(rate, d))
	for i := range out {
		t := float64(i) / float64(rate)
		out[i] = amp * math.Sin(2*math.Pi*freq*t+depth/speed*math.Sin(2*math.Pi*speed*t))
	}
	return out
}

func throughMuLaw(pcm []byte) []byte {
	out := make([]byte, len(pcm))
	for i := 0; i+1 < len(pcm); i += 2 {
		v := g711.UlawDecodeSample(g711.UlawEncodeSample(int16(binary.LittleEndian.Uint16(pcm[i:]))))
		binary.LittleEndian.PutUint16(out[i:], uint16(v))
	}
	return out
}

func withoutConfirmed(events []Event) []Event {
	var out []Event
	for _, e := range events {
		if e.Kind != Confirmed {
			out = append(out, e)
		}
	}
	return out
}

func confirmed(events []Event) bool {
	for _, e := range events {
		if e.Kind == Confirmed {
			return true
		}
	}
	return false
}

func started(events []Event) bool {
	for _, e := range events {
		if e.Kind == Started {
			return true
		}
	}
	return false
}

func TestDetectsBeepsUpTo2000Hz(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		for _, freq := range []float64{440, 625, 812.5, 1000, 1403, 1700, 2000} {
			pcm := encode(silence(rate, 200*time.Millisecond), sine(rate, freq, 9000, 600*time.Millisecond), silence(rate, 200*time.Millisecond))
			events := withoutConfirmed(run(t, rate, pcm, 320))
			if len(events) != 2 || events[0].Kind != Started || events[1].Kind != Ended {
				t.Fatalf("rate %d freq %.1f: events %+v", rate, freq, events)
			}
			if math.Abs(events[0].Frequency-freq) > 5 || events[0].Purity < 0.8 {
				t.Fatalf("rate %d freq %.1f: reported %+v", rate, freq, events[0])
			}
		}
	}
}

func TestDetectsBeepThroughMuLaw(t *testing.T) {
	for _, freq := range []float64{440, 1000, 1400, 2000} {
		pcm := throughMuLaw(encode(silence(8000, 200*time.Millisecond), sine(8000, freq, 3000, 600*time.Millisecond), silence(8000, 200*time.Millisecond)))
		events := withoutConfirmed(run(t, 8000, pcm, 160))
		if len(events) != 2 || events[0].Kind != Started || math.Abs(events[0].Frequency-freq) > 5 {
			t.Fatalf("freq %.0f: events %+v", freq, events)
		}
	}
}

func TestDetectsDistortedBeeps(t *testing.T) {
	cases := map[string][]float64{
		"second harmonic": partials(8000, 600*time.Millisecond, [2]float64{1000, 9000}, [2]float64{2000, 2700}),
		"third harmonic":  partials(8000, 600*time.Millisecond, [2]float64{1000, 9000}, [2]float64{3000, 4500}),
		"clipped":         sine(8000, 1000, 160000, 600*time.Millisecond),
	}
	for name, sig := range cases {
		if events := run(t, 8000, encode(sig, silence(8000, 100*time.Millisecond)), 320); !started(events) {
			t.Fatalf("%s: events %+v", name, events)
		}
	}
}

func TestTwoToneBeepIsOneRun(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		pcm := encode(silence(rate, 100*time.Millisecond), sine(rate, 950, 9000, 500*time.Millisecond), sine(rate, 1400, 9000, 500*time.Millisecond), silence(rate, 100*time.Millisecond))
		events := run(t, rate, pcm, 320)
		if len(events) != 3 || events[0].Kind != Started || events[1].Kind != Confirmed || events[2].Kind != Ended {
			t.Fatalf("rate %d: events %+v", rate, events)
		}
		if events[2].Duration < 950*time.Millisecond {
			t.Fatalf("rate %d: ended after %v, mid-beep", rate, events[2].Duration)
		}
	}
}

func TestIgnoresVoiceWithDominantHarmonic(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		cases := map[string][]float64{
			"240 Hz voice, strong second harmonic": partials(rate, 600*time.Millisecond, [2]float64{240, 2300}, [2]float64{480, 9800}, [2]float64{720, 2300}),
			"160 Hz voice, strong third harmonic":  partials(rate, 600*time.Millisecond, [2]float64{160, 2000}, [2]float64{320, 2000}, [2]float64{480, 9800}, [2]float64{640, 2000}),
			"480 Hz voice with harmonics":          partials(rate, 600*time.Millisecond, [2]float64{480, 9800}, [2]float64{960, 3100}, [2]float64{1440, 1540}),
		}
		for name, sig := range cases {
			for _, mu := range []bool{false, true} {
				pcm := encode(sig)
				if mu {
					pcm = throughMuLaw(pcm)
				}
				if events := run(t, rate, pcm, 320); len(events) != 0 {
					t.Fatalf("rate %d %s (mu-law %v): events %+v", rate, name, mu, events)
				}
			}
		}
	}
}

func TestIgnoresNoteWithVibrato(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		for _, freq := range []float64{660, 880, 1200} {
			if events := run(t, rate, encode(vibrato(rate, freq, 15, 5, 9800, time.Second)), 320); len(events) != 0 {
				t.Fatalf("rate %d freq %.0f: events %+v", rate, freq, events)
			}
		}
	}
}

func TestIgnoresDTMFWithTwist(t *testing.T) {
	rows := []float64{697, 770, 852, 941}
	cols := []float64{1209, 1336, 1477, 1633}
	for _, rate := range []int{8000, 16000} {
		for _, lo := range rows {
			for _, hi := range cols {
				for _, twist := range []float64{-8, 8} {
					g := math.Pow(10, twist/20)
					sig := partials(rate, 600*time.Millisecond, [2]float64{lo, 9800}, [2]float64{hi, 9800 * g})
					if events := run(t, rate, encode(sig), 320); len(events) != 0 {
						t.Fatalf("rate %d %v+%v twist %v dB: events %+v", rate, lo, hi, twist, events)
					}
				}
			}
		}
	}
}

func TestIgnoresBusyAndRingTones(t *testing.T) {
	var cadence450 [][]float64
	for i := 0; i < 4; i++ {
		cadence450 = append(cadence450, sine(8000, 450, 9800, 350*time.Millisecond), silence(8000, 350*time.Millisecond))
	}
	cases := map[string][]byte{
		"450 Hz busy cadence": encode(cadence450...),
		"400 Hz busy":         encode(sine(8000, 400, 9800, 750*time.Millisecond), silence(8000, 750*time.Millisecond)),
		"425 Hz busy":         encode(sine(8000, 425, 9800, time.Second)),
		"US busy 480+620":     encode(partials(8000, 500*time.Millisecond, [2]float64{480, 6500}, [2]float64{620, 6500})),
	}
	for name, pcm := range cases {
		for _, mu := range []bool{false, true} {
			if mu {
				pcm = throughMuLaw(pcm)
			}
			if events := run(t, 8000, pcm, 160); confirmed(events) {
				t.Fatalf("%s (mu-law %v): events %+v", name, mu, events)
			}
		}
	}
	for _, freq := range []float64{400, 425} {
		if events := run(t, 8000, encode(sine(8000, freq, 9800, time.Second)), 160); len(events) != 0 {
			t.Fatalf("%.0f Hz: events %+v", freq, events)
		}
	}
}

func TestIgnoresSpecialInformationTones(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		sequences := map[string][][]float64{
			"SIT 913.8/1370.6/1776.7": {sine(rate, 913.8, 9800, 274*time.Millisecond), sine(rate, 1370.6, 9800, 274*time.Millisecond), sine(rate, 1776.7, 9800, 380*time.Millisecond)},
			"SIT 950/1400/1800":       {sine(rate, 950, 9800, 330*time.Millisecond), sine(rate, 1400, 9800, 330*time.Millisecond), sine(rate, 1800, 9800, 330*time.Millisecond)},
		}
		for name, seq := range sequences {
			pcm := encode(append(seq, silence(rate, 200*time.Millisecond))...)
			if events := run(t, rate, pcm, 320); confirmed(events) {
				t.Fatalf("rate %d %s: events %+v", rate, name, events)
			}
		}
	}
}

func TestIgnoresFaxCallingTone(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		for _, freq := range []float64{1100, 1090, 1110} {
			pcm := encode(sine(rate, freq, 9800, 500*time.Millisecond), silence(rate, 3*time.Second), sine(rate, freq, 9800, 500*time.Millisecond))
			if events := run(t, rate, pcm, 320); len(events) != 0 {
				t.Fatalf("rate %d freq %.0f: events %+v", rate, freq, events)
			}
		}
	}
}

func sweepFrequencies() []float64 {
	var out []float64
	for f := 445.0; f <= 1995; f += 13.7 {
		out = append(out, f)
	}
	return out
}

type sweepSignal func(rate int, freq float64, d time.Duration) []float64

func TestFrequencySweep(t *testing.T) {
	const amp = 0.2 * 32767
	pureTone := func(rate int, f float64, d time.Duration) []float64 { return sine(rate, f, amp, d) }
	conditions := []struct {
		name   string
		rate   int
		mu     bool
		signal sweepSignal
		min300 int
		min400 int
		min450 int
	}{
		{"clean 16 kHz", 16000, false, pureTone, 111, 111, 111},
		{"mu-law", 8000, true, pureTone, 111, 111, 111},
		{"mu-law -30 dBFS", 8000, true, func(rate int, f float64, d time.Duration) []float64 { return sine(rate, f, 0.045*32767, d) }, 111, 111, 111},
		{"mu-law white noise 10 dB SNR", 8000, true, func(rate int, f float64, d time.Duration) []float64 {
			return mix(sine(rate, f, amp, d), gaussian(rate, amp/math.Sqrt2*math.Pow(10, -10.0/20), d, uint64(f)))
		}, 111, 111, 111},
		{"mu-law drift 25 Hz", 8000, true, func(rate int, f float64, d time.Duration) []float64 { return chirp(rate, f, 25/d.Seconds(), amp, d) }, 108, 108, 108},
		{"mu-law harmonics -18 dB", 8000, true, func(rate int, f float64, d time.Duration) []float64 {
			return partials(rate, d, [2]float64{f, amp}, [2]float64{2 * f, amp * 0.126}, [2]float64{3 * f, amp * 0.126})
		}, 111, 111, 111},
		{"mu-law clipped at 3x full scale", 8000, true, func(rate int, f float64, d time.Duration) []float64 { return sine(rate, f, 3*32767, d) }, 103, 103, 103},
	}
	freqs := sweepFrequencies()
	for _, c := range conditions {
		counts := map[time.Duration]int{}
		for _, dur := range []time.Duration{300 * time.Millisecond, 400 * time.Millisecond, 450 * time.Millisecond} {
			for i, f := range freqs {
				pcm := encode(silence(c.rate, time.Duration(200+i%20)*time.Millisecond), c.signal(c.rate, f, dur), silence(c.rate, 200*time.Millisecond))
				if c.mu {
					pcm = throughMuLaw(pcm)
				}
				for _, e := range run(t, c.rate, pcm, 2*c.rate/50) {
					if dur == 300*time.Millisecond && e.Kind == Started || dur > 300*time.Millisecond && e.Kind == Confirmed && e.Purity >= 0.85 {
						counts[dur]++
						break
					}
				}
			}
		}
		t.Logf("%-30s started at 300 ms %d/%d, confirmed with purity >= 0.85 at 400 ms %d/%d, at 450 ms %d/%d",
			c.name, counts[300*time.Millisecond], len(freqs), counts[400*time.Millisecond], len(freqs), counts[450*time.Millisecond], len(freqs))
		if counts[300*time.Millisecond] < c.min300 || counts[400*time.Millisecond] < c.min400 || counts[450*time.Millisecond] < c.min450 {
			t.Errorf("%s: below thresholds %d/%d/%d", c.name, c.min300, c.min400, c.min450)
		}
	}
}

func gaussian(rate int, amp float64, d time.Duration, seed uint64) []float64 {
	r := rand.New(rand.NewPCG(seed, 99))
	out := make([]float64, samples(rate, d))
	for i := range out {
		out[i] = amp * r.NormFloat64()
	}
	return out
}

func chirp(rate int, freq, slope, amp float64, d time.Duration) []float64 {
	out := make([]float64, samples(rate, d))
	for i := range out {
		t := float64(i) / float64(rate)
		out[i] = amp * math.Sin(2*math.Pi*(freq*t+0.5*slope*t*t))
	}
	return out
}
