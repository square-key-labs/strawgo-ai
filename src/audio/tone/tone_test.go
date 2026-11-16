package tone

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/dtmf"
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
			events := run(t, rate, pcm, 2*samples(rate, 20*time.Millisecond))
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
	events := run(t, 8000, pcm, 37)
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
