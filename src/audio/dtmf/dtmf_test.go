package dtmf

import (
	"encoding/binary"
	"math"
	"testing"
	"time"
)

var allFreqs = []float64{697, 770, 852, 941, 1209, 1336, 1477, 1633}

func goertzel(pcm []byte, rate int, freq float64) float64 {
	w := 2 * math.Pi * freq / float64(rate)
	c := 2 * math.Cos(w)
	var s1, s2 float64
	n := len(pcm) / 2
	for i := 0; i < n; i++ {
		x := float64(int16(binary.LittleEndian.Uint16(pcm[2*i:])))
		s := x + c*s1 - s2
		s2, s1 = s1, s
	}
	return (s1*s1 + s2*s2 - c*s1*s2) / float64(n*n)
}

func detect(pcm []byte, rate int) (lo, hi float64) {
	var bestLo, bestHi float64
	for _, f := range allFreqs {
		p := goertzel(pcm, rate, f)
		if f < 1000 {
			if p > bestLo {
				bestLo, lo = p, f
			}
		} else if p > bestHi {
			bestHi, hi = p, f
		}
	}
	return lo, hi
}

func TestEveryKeyHasItsTwoFrequencies(t *testing.T) {
	keys := "0123456789*#ABCD"
	for _, rate := range []int{8000, 16000, 24000} {
		for i := 0; i < len(keys); i++ {
			key := keys[i]
			wantLo, wantHi, err := Frequencies(key)
			if err != nil {
				t.Fatal(err)
			}
			pcm, err := Tone(key, rate)
			if err != nil {
				t.Fatal(err)
			}
			toneBytes := 2 * int(DefaultToneDuration.Seconds()*float64(rate))
			lo, hi := detect(pcm[:toneBytes], rate)
			if lo != wantLo || hi != wantHi {
				t.Errorf("rate %d key %c: detected %v/%v want %v/%v", rate, key, lo, hi, wantLo, wantHi)
			}
			pLo, pHi := goertzel(pcm[:toneBytes], rate, wantLo), goertzel(pcm[:toneBytes], rate, wantHi)
			for _, f := range allFreqs {
				if f == wantLo || f == wantHi {
					continue
				}
				if p := goertzel(pcm[:toneBytes], rate, f); p*20 > min(pLo, pHi) {
					t.Errorf("rate %d key %c: %v Hz too strong", rate, key, f)
				}
			}
		}
	}
}

func TestKnownKeyFrequencies(t *testing.T) {
	cases := map[byte][2]float64{'1': {697, 1209}, '5': {770, 1336}, '9': {852, 1477}, '0': {941, 1336}, '*': {941, 1209}, '#': {941, 1477}, 'D': {941, 1633}, 'a': {697, 1633}}
	for k, want := range cases {
		lo, hi, err := Frequencies(k)
		if err != nil || lo != want[0] || hi != want[1] {
			t.Errorf("%c: %v %v %v", k, lo, hi, err)
		}
	}
	for _, k := range []byte{'E', 'x', ' ', 0} {
		if Valid(k) {
			t.Errorf("%q reported valid", k)
		}
		if _, err := Tone(k, 8000); err == nil {
			t.Errorf("%q tone succeeded", k)
		}
	}
}

func TestToneLayoutAndAmplitude(t *testing.T) {
	const rate = 8000
	pcm, err := Tone('5', rate)
	if err != nil {
		t.Fatal(err)
	}
	toneN := int(DefaultToneDuration.Seconds() * rate)
	gapN := int(DefaultGapDuration.Seconds() * rate)
	if len(pcm) != 2*(toneN+gapN) {
		t.Fatalf("len %d want %d", len(pcm), 2*(toneN+gapN))
	}
	for i := toneN; i < toneN+gapN; i++ {
		if pcm[2*i] != 0 || pcm[2*i+1] != 0 {
			t.Fatalf("gap not silent at sample %d", i)
		}
	}
	if s := int16(binary.LittleEndian.Uint16(pcm)); s != 0 {
		t.Fatalf("tone does not fade in: first sample %d", s)
	}
	limit := int(math.Round(2*DefaultAmplitude*32767)) + 1
	peak := 0
	for i := 0; i < toneN; i++ {
		v := int(int16(binary.LittleEndian.Uint16(pcm[2*i:])))
		if v < 0 {
			v = -v
		}
		peak = max(peak, v)
	}
	if peak > limit || peak < limit/2 {
		t.Fatalf("peak %d outside (%d, %d]", peak, limit/2, limit)
	}
}

func TestLowercaseSharesCachedTone(t *testing.T) {
	a, _ := Tone('b', 8000)
	b, _ := Tone('B', 8000)
	if len(a) == 0 || &a[0] != &b[0] {
		t.Fatal("lowercase and uppercase keys should return the same cached tone")
	}
}

func TestSequenceAndAppend(t *testing.T) {
	one, _ := Tone('1', 16000)
	seq, err := Sequence("123", 16000)
	if err != nil {
		t.Fatal(err)
	}
	if len(seq) != 3*len(one) {
		t.Fatalf("sequence len %d want %d", len(seq), 3*len(one))
	}
	lo, hi := detect(seq[len(one):2*len(one)], 16000)
	if lo != 697 || hi != 1336 {
		t.Fatalf("second tone detected as %v/%v", lo, hi)
	}
	if _, err := Sequence("1x", 16000); err == nil {
		t.Fatal("invalid key in sequence accepted")
	}
	got, err := AppendTone([]byte{1, 2}, '1', 16000)
	if err != nil || len(got) != 2+len(one) || got[0] != 1 {
		t.Fatalf("AppendTone len %d err %v", len(got), err)
	}
}

func TestGenerateOptionsAndErrors(t *testing.T) {
	opts := Options{Tone: 100 * time.Millisecond, Gap: 0, Amplitude: 0.5}
	pcm, err := Generate(nil, '7', 8000, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(pcm) != 2*800 {
		t.Fatalf("len %d", len(pcm))
	}
	if lo, hi := detect(pcm, 8000); lo != 852 || hi != 1209 {
		t.Fatalf("detected %v/%v", lo, hi)
	}
	if _, err := Generate(nil, '7', 0, opts); err == nil {
		t.Fatal("zero sample rate accepted")
	}
	low, err := Generate(nil, 'A', 2000, opts)
	if err != nil {
		t.Fatal(err)
	}
	if p := goertzel(low, 2000, 697); p == 0 {
		t.Fatal("low tone missing when high tone is above nyquist")
	}
}
