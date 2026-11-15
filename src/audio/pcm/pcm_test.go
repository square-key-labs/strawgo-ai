package pcm

import (
	"encoding/binary"
	"math"
	"testing"
)

func le(s ...int16) []byte {
	b := make([]byte, 2*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
	}
	return b
}

func samples(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return out
}

func eq(t *testing.T, got []byte, want ...int16) {
	t.Helper()
	g := samples(got)
	if len(g) != len(want) {
		t.Fatalf("got %v want %v", g, want)
	}
	for i := range want {
		if g[i] != want[i] {
			t.Fatalf("got %v want %v", g, want)
		}
	}
}

func TestInt16Conversions(t *testing.T) {
	src := []int16{0, 1, -1, 32767, -32768}
	b := FromInt16([]byte{9, 9}, src)
	if len(b) != 2+2*len(src) || b[0] != 9 {
		t.Fatal("FromInt16 did not append")
	}
	back := ToInt16([]int16{5}, b[2:])
	if back[0] != 5 || len(back) != 1+len(src) {
		t.Fatal("ToInt16 did not append")
	}
	for i, v := range src {
		if back[i+1] != v {
			t.Fatalf("round trip %d: %d", v, back[i+1])
		}
	}
	if got := ToInt16(nil, []byte{1, 2, 3}); len(got) != 1 {
		t.Fatalf("odd byte length gave %d samples", len(got))
	}
}

func TestAppendSilenceAndGrow(t *testing.T) {
	dst := make([]byte, 2, 10)
	dst[0], dst[1] = 7, 7
	full := dst[:10]
	for i := 2; i < 10; i++ {
		full[i] = 0xFF
	}
	out := AppendSilence(dst, 6)
	if len(out) != 8 || out[0] != 7 {
		t.Fatalf("len %d", len(out))
	}
	for _, v := range out[2:] {
		if v != 0 {
			t.Fatal("reused capacity not cleared")
		}
	}
	g := Grow(nil, 4)
	if len(g) != 4 {
		t.Fatal("Grow length")
	}
}

func TestMixSumsAndClips(t *testing.T) {
	eq(t, Mix(nil, le(100, 30000, -30000, 5), le(-50, 10000, -10000)), 50, 32767, -32768, 5)
}

func TestMixInPlaceGain(t *testing.T) {
	dst := le(1000, 1000, 32000)
	MixInPlace(dst, le(1000, -2000, 32000, 99), 0.5)
	eq(t, dst, 1500, 0, 32767)
}

func TestInterleavePadsShorterChannel(t *testing.T) {
	eq(t, Interleave(nil, le(1, 2, 3), le(-1)), 1, -1, 2, 0, 3, 0)
}

func TestToMono(t *testing.T) {
	eq(t, ToMono(nil, le(100, 200, -32768, -32768, 7), 2), 150, -32768)
	eq(t, ToMono(nil, le(3, 6, 9), 3), 6)
	eq(t, ToMono(nil, le(4, 5), 1), 4, 5)
}

func TestScale(t *testing.T) {
	eq(t, Scale(nil, le(100, -100, 20000), 2), 200, -200, 32767)
	eq(t, Scale(nil, le(3, -3), 0.5), 2, -2)
}

func TestLevels(t *testing.T) {
	if Peak(le(5, -32768, 100)) != 32768 {
		t.Fatal("peak of -32768")
	}
	if Peak(nil) != 0 || RMS(nil) != 0 {
		t.Fatal("empty buffer levels")
	}
	square := le(16384, -16384, 16384, -16384)
	if r := RMS(square); math.Abs(r-0.5) > 1e-9 {
		t.Fatalf("rms %v", r)
	}
	if d := DBFS(square); math.Abs(d-(-6.0206)) > 0.001 {
		t.Fatalf("dbfs %v", d)
	}
	if DBFS(le(0, 0)) != MinDBFS || ToDBFS(1e-12) != MinDBFS {
		t.Fatal("silence should be MinDBFS")
	}
	if v := Volume(square); math.Abs(v-(60-6.0206)/60) > 0.001 {
		t.Fatalf("volume %v", v)
	}
	if Volume(le(0)) != 0 {
		t.Fatal("silent volume")
	}
}

func TestSilenceAndHelpers(t *testing.T) {
	if !IsSilence(le(SpeakingThreshold, -SpeakingThreshold)) || IsSilence(le(SpeakingThreshold+1)) {
		t.Fatal("IsSilence threshold")
	}
	if !IsSilenceThreshold(le(500), 500) || IsSilenceThreshold(le(-501), 500) {
		t.Fatal("IsSilenceThreshold")
	}
	if NormalizeValue(5, 0, 10) != 0.5 || NormalizeValue(-1, 0, 10) != 0 || NormalizeValue(11, 0, 10) != 1 || NormalizeValue(1, 2, 2) != 0 {
		t.Fatal("NormalizeValue")
	}
	if ExpSmoothing(10, 0, 0.25) != 2.5 {
		t.Fatal("ExpSmoothing")
	}
}

func TestHotHelpersDoNotAllocateWithCapacity(t *testing.T) {
	a, b := make([]byte, 640), make([]byte, 640)
	dst := make([]byte, 0, 2048)
	if n := testing.AllocsPerRun(50, func() {
		dst = Mix(dst[:0], a, b)
		dst = Interleave(dst[:0], a, b)
		dst = Scale(dst[:0], a, 0.5)
	}); n != 0 {
		t.Fatalf("allocs %.1f", n)
	}
}

func TestFadeOutRampsLinearlyToZero(t *testing.T) {
	b := le(1000, 1000, 1000, 1000, -1000)
	if n := testing.AllocsPerRun(10, func() { FadeOut(b[:0]) }); n != 0 {
		t.Fatalf("allocs %.1f", n)
	}
	FadeOut(b)
	want := le(800, 600, 400, 200, 0)
	for i := range want {
		if b[i] != want[i] {
			t.Fatalf("FadeOut = %v, want %v", b, want)
		}
	}
}
