package resample

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

var ratios = [][2]int{
	{8000, 16000}, {16000, 8000},
	{8000, 24000}, {24000, 8000},
	{16000, 24000}, {24000, 16000},
	{16000, 48000}, {48000, 16000},
	{24000, 48000}, {48000, 24000},
	{44100, 48000}, {48000, 44100},
	{22050, 16000}, {16000, 22050},
}

func tone(n, rate int, freq, amp float64) []int16 {
	out := make([]int16, n)
	for i := range out {
		out[i] = int16(math.Round(amp * math.Sin(2*math.Pi*freq*float64(i)/float64(rate))))
	}
	return out
}

func goertzelAmp(x []int16, rate int, freq float64) float64 {
	w := 2 * math.Pi * freq / float64(rate)
	c := 2 * math.Cos(w)
	var s1, s2 float64
	for _, v := range x {
		s := float64(v) + c*s1 - s2
		s2, s1 = s1, s
	}
	re := s1 - s2*math.Cos(w)
	im := s2 * math.Sin(w)
	return 2 * math.Hypot(re, im) / float64(len(x))
}

func rms(x []int16) float64 {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s / float64(len(x)))
}

func TestNewRejectsInvalidRates(t *testing.T) {
	for _, r := range [][2]int{{0, 16000}, {16000, 0}, {-1, 8000}} {
		if _, err := New(r[0], r[1]); err == nil {
			t.Fatalf("New(%d,%d) succeeded", r[0], r[1])
		}
	}
	if _, err := Resample([]int16{1}, 0, 8000); err == nil {
		t.Fatal("Resample with zero rate succeeded")
	}
}

func TestSameRateIsPassthrough(t *testing.T) {
	r, err := New(16000, 16000)
	if err != nil {
		t.Fatal(err)
	}
	src := []int16{1, -2, 3, 32767, -32768}
	got := r.Process(nil, src)
	if fmt.Sprint(got) != fmt.Sprint(src) {
		t.Fatalf("got %v want %v", got, src)
	}
	if out := r.Flush(nil); len(out) != 0 {
		t.Fatalf("flush emitted %d samples", len(out))
	}
	b := []byte{1, 2, 3, 4, 5}
	if got := r.ProcessBytes(nil, b); len(got) != 4 {
		t.Fatalf("bytes passthrough length %d", len(got))
	}
	if r.InRate() != 16000 || r.OutRate() != 16000 {
		t.Fatal("rates not reported")
	}
}

func TestOneShotLengthRatio(t *testing.T) {
	for _, rr := range ratios {
		for _, n := range []int{0, 1, 159, 160, 4410, 16000} {
			src := tone(n, rr[0], 440, 8000)
			out, err := Resample(src, rr[0], rr[1])
			if err != nil {
				t.Fatal(err)
			}
			want := (n*rr[1] + rr[0] - 1) / rr[0]
			if len(out) != want {
				t.Fatalf("%v n=%d: len %d want %d", rr, n, len(out), want)
			}
		}
	}
}

func TestStreamingLengthTracksRatio(t *testing.T) {
	for _, rr := range ratios {
		r, err := New(rr[0], rr[1])
		if err != nil {
			t.Fatal(err)
		}
		chunk := rr[0] / 50
		var total int
		var dst []int16
		for i := 0; i < 100; i++ {
			dst = r.Process(dst[:0], tone(chunk, rr[0], 300, 1000))
			if len(dst) > r.MaxOutput(chunk)+r.MaxOutput(r.b.taps) {
				t.Fatalf("%v chunk produced %d samples", rr, len(dst))
			}
			total += len(dst)
		}
		total += len(r.Flush(nil))
		want := 100 * chunk * rr[1] / rr[0]
		if d := total - want; d < -2 || d > r.b.taps*rr[1]/rr[0]+2 {
			t.Fatalf("%v: total %d want about %d", rr, total, want)
		}
	}
}

func TestTonePreservesAmplitudeAndFrequency(t *testing.T) {
	for _, rr := range ratios {
		t.Run(fmt.Sprintf("%d_%d", rr[0], rr[1]), func(t *testing.T) {
			const freq, amp = 440.0, 10000.0
			src := tone(rr[0]/2, rr[0], freq, amp)
			out, err := Resample(src, rr[0], rr[1])
			if err != nil {
				t.Fatal(err)
			}
			edge := rr[1] / 20
			mid := out[edge : len(out)-edge]
			got := goertzelAmp(mid, rr[1], freq)
			if math.Abs(got-amp)/amp > 0.03 {
				t.Fatalf("tone amplitude %.0f want %.0f", got, amp)
			}
			if r := rms(mid); math.Abs(r-amp/math.Sqrt2)/(amp/math.Sqrt2) > 0.03 {
				t.Fatalf("rms %.0f want %.0f", r, amp/math.Sqrt2)
			}
			if off := goertzelAmp(mid, rr[1], freq*1.5); off > amp*0.02 {
				t.Fatalf("energy leaked to %.0f Hz: %.0f", freq*1.5, off)
			}
		})
	}
}

func TestDownsampleRejectsAboveNyquist(t *testing.T) {
	src := tone(48000/2, 48000, 6000, 10000)
	out, err := Resample(src, 48000, 8000)
	if err != nil {
		t.Fatal(err)
	}
	edge := 400
	if r := rms(out[edge : len(out)-edge]); r > 300 {
		t.Fatalf("6 kHz tone leaked into 8 kHz output with rms %.0f", r)
	}
}

func TestChunkedEqualsOneShot(t *testing.T) {
	for _, rr := range ratios {
		src := tone(rr[0]/4, rr[0], 523, 12000)
		whole, _ := New(rr[0], rr[1])
		ref := whole.Process(nil, src)
		ref = whole.Flush(ref)
		for _, chunk := range []int{1, 7, 160, 333} {
			s, _ := New(rr[0], rr[1])
			var got []int16
			for i := 0; i < len(src); i += chunk {
				got = s.Process(got, src[i:min(i+chunk, len(src))])
			}
			got = s.Flush(got)
			if len(got) != len(ref) {
				t.Fatalf("%v chunk %d: len %d want %d", rr, chunk, len(got), len(ref))
			}
			for i := range ref {
				if d := int(got[i]) - int(ref[i]); d < -1 || d > 1 {
					t.Fatalf("%v chunk %d: sample %d = %d want %d", rr, chunk, i, got[i], ref[i])
				}
			}
		}
	}
}

func TestProcessBytesMatchesProcess(t *testing.T) {
	src := tone(1600, 16000, 700, 9000)
	b := make([]byte, 2*len(src))
	for i, v := range src {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
	}
	a, _ := New(16000, 24000)
	c, _ := New(16000, 24000)
	want := a.Process(nil, src)
	got := c.ProcessBytes([]byte{9}, b)
	if got[0] != 9 || len(got) != 1+2*len(want) {
		t.Fatalf("bytes len %d want %d with prefix kept", len(got), 1+2*len(want))
	}
	for i, v := range want {
		if g := int16(binary.LittleEndian.Uint16(got[1+2*i:])); g != v {
			t.Fatalf("sample %d: %d want %d", i, g, v)
		}
	}
	rb, err := ResampleBytes(b, 16000, 8000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rb) != 2*800 {
		t.Fatalf("ResampleBytes len %d", len(rb))
	}
}

func TestZeroAllocsPerChunk(t *testing.T) {
	for _, rr := range ratios {
		r, _ := New(rr[0], rr[1])
		chunk := tone(rr[0]/50, rr[0], 440, 5000)
		dst := make([]int16, 0, 4*r.MaxOutput(len(chunk)))
		for i := 0; i < 5; i++ {
			dst = r.Process(dst[:0], chunk)
		}
		if a := testing.AllocsPerRun(100, func() { dst = r.Process(dst[:0], chunk) }); a != 0 {
			t.Fatalf("%v Process allocs %.1f per chunk", rr, a)
		}
		bchunk := make([]byte, 2*len(chunk))
		bdst := make([]byte, 0, 8*r.MaxOutput(len(chunk)))
		for i := 0; i < 5; i++ {
			bdst = r.ProcessBytes(bdst[:0], bchunk)
		}
		if a := testing.AllocsPerRun(100, func() { bdst = r.ProcessBytes(bdst[:0], bchunk) }); a != 0 {
			t.Fatalf("%v ProcessBytes allocs %.1f per chunk", rr, a)
		}
	}
}

func TestResetRestoresInitialState(t *testing.T) {
	src := tone(800, 16000, 440, 10000)
	r, _ := New(16000, 8000)
	first := r.Process(nil, src)
	r.Process(nil, tone(321, 16000, 1000, 20000))
	r.Reset()
	second := r.Process(nil, src)
	if fmt.Sprint(first) != fmt.Sprint(second) {
		t.Fatal("output after Reset differs from a fresh resampler")
	}
	r.Flush(nil)
	third := r.Process(nil, src)
	if fmt.Sprint(first) != fmt.Sprint(third) {
		t.Fatal("output after Flush differs from a fresh resampler")
	}
}

func TestClipsWithoutWrapping(t *testing.T) {
	src := make([]int16, 2000)
	for i := range src {
		if (i/20)%2 == 0 {
			src[i] = 32767
		} else {
			src[i] = -32768
		}
	}
	out, _ := Resample(src, 8000, 48000)
	for i := 1; i < len(out); i++ {
		if d := int(out[i]) - int(out[i-1]); d > 40000 || d < -40000 {
			t.Fatalf("wraparound at %d: %d -> %d", i, out[i-1], out[i])
		}
	}
}
