package g711

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestUlawReferenceDecode(t *testing.T) {
	cases := map[byte]int16{
		0x00: -32124, 0x80: 32124, 0xFF: 0, 0x7F: 0,
		0x7E: -8, 0xFE: 8, 0x70: -120, 0xF0: 120,
		0x0F: -16764, 0x8F: 16764, 0x5A: -556, 0xDA: 556,
	}
	for code, want := range cases {
		if got := UlawDecodeSample(code); got != want {
			t.Errorf("ulaw decode 0x%02X = %d want %d", code, got, want)
		}
	}
}

func TestAlawReferenceDecode(t *testing.T) {
	cases := map[byte]int16{
		0xD5: 8, 0x55: -8, 0xAA: 32256, 0x2A: -32256,
		0x80: 5504, 0x00: -5504, 0xD4: 24, 0x54: -24,
		0xC5: 264, 0x45: -264,
	}
	for code, want := range cases {
		if got := AlawDecodeSample(code); got != want {
			t.Errorf("alaw decode 0x%02X = %d want %d", code, got, want)
		}
	}
}

func TestReferenceEncode(t *testing.T) {
	u := map[int16]byte{0: 0xFF, 32767: 0x80, -32768: 0x00, 32124: 0x80, -32124: 0x00, 8: 0xFE, 120: 0xF0}
	for s, want := range u {
		if got := UlawEncodeSample(s); got != want {
			t.Errorf("ulaw encode %d = 0x%02X want 0x%02X", s, got, want)
		}
	}
	a := map[int16]byte{0: 0xD5, -8: 0x55, 32767: 0xAA, -32768: 0x2A, 8: 0xD5, 264: 0xC5}
	for s, want := range a {
		if got := AlawEncodeSample(s); got != want {
			t.Errorf("alaw encode %d = 0x%02X want 0x%02X", s, got, want)
		}
	}
}

func TestExhaustiveCodeRoundTrip(t *testing.T) {
	for c := 0; c < 256; c++ {
		code := byte(c)
		if got := UlawEncodeSample(UlawDecodeSample(code)); got != code && code != 0x7F {
			t.Errorf("ulaw 0x%02X -> %d -> 0x%02X", code, UlawDecodeSample(code), got)
		}
		if got := AlawEncodeSample(AlawDecodeSample(code)); got != code {
			t.Errorf("alaw 0x%02X -> %d -> 0x%02X", code, AlawDecodeSample(code), got)
		}
	}
	if UlawEncodeSample(UlawDecodeSample(0x7F)) != 0xFF {
		t.Error("ulaw negative zero should re-encode as positive zero")
	}
}

func TestExhaustiveLinearQuantisationError(t *testing.T) {
	for i := math.MinInt16; i <= math.MaxInt16; i++ {
		s := int16(i)
		bound := int(math.Abs(float64(s)))/16 + 16
		if e := int(UlawDecodeSample(UlawEncodeSample(s))) - i; e > bound || e < -bound {
			t.Fatalf("ulaw error for %d is %d (bound %d)", i, e, bound)
		}
		if e := int(AlawDecodeSample(AlawEncodeSample(s))) - i; e > bound || e < -bound {
			t.Fatalf("alaw error for %d is %d (bound %d)", i, e, bound)
		}
	}
}

func TestDecodeIsMonotonic(t *testing.T) {
	prevU, prevA := int16(math.MinInt16), int16(math.MinInt16)
	for i := math.MinInt16; i <= math.MaxInt16; i += 7 {
		u := UlawDecodeSample(UlawEncodeSample(int16(i)))
		a := AlawDecodeSample(AlawEncodeSample(int16(i)))
		if u < prevU || a < prevA {
			t.Fatalf("non-monotonic at %d", i)
		}
		prevU, prevA = u, a
	}
}

func TestBulkHelpersMatchSampleFunctions(t *testing.T) {
	src := []int16{0, 1, -1, 100, -100, 5000, -5000, 32767, -32768}
	pcmBytes := make([]byte, 2*len(src))
	for i, s := range src {
		binary.LittleEndian.PutUint16(pcmBytes[2*i:], uint16(s))
	}
	u := EncodeUlaw([]byte{0xAB}, src)
	a := EncodeAlaw([]byte{0xAB}, src)
	ub := PCM16ToUlaw(nil, pcmBytes)
	ab := PCM16ToAlaw(nil, pcmBytes)
	if u[0] != 0xAB || a[0] != 0xAB || len(u) != len(src)+1 || len(a) != len(src)+1 {
		t.Fatal("encode did not append to dst")
	}
	for i, s := range src {
		if u[i+1] != UlawEncodeSample(s) || ub[i] != u[i+1] {
			t.Fatalf("ulaw mismatch at %d", i)
		}
		if a[i+1] != AlawEncodeSample(s) || ab[i] != a[i+1] {
			t.Fatalf("alaw mismatch at %d", i)
		}
	}
	du := DecodeUlaw([]int16{7}, u[1:])
	da := DecodeAlaw([]int16{7}, a[1:])
	upcm := UlawToPCM16([]byte{1, 2}, u[1:])
	apcm := AlawToPCM16(nil, a[1:])
	if du[0] != 7 || da[0] != 7 || len(du) != len(src)+1 || len(upcm) != 2+2*len(src) {
		t.Fatal("decode did not append to dst")
	}
	for i := range src {
		if du[i+1] != UlawDecodeSample(u[i+1]) || int16(binary.LittleEndian.Uint16(upcm[2+2*i:])) != du[i+1] {
			t.Fatalf("ulaw decode mismatch at %d", i)
		}
		if da[i+1] != AlawDecodeSample(a[i+1]) || int16(binary.LittleEndian.Uint16(apcm[2*i:])) != da[i+1] {
			t.Fatalf("alaw decode mismatch at %d", i)
		}
	}
	if got := PCM16ToUlaw(nil, []byte{1, 2, 3}); len(got) != 1 {
		t.Fatalf("odd byte input produced %d codes", len(got))
	}
}

func TestBulkHelpersDoNotAllocateWithCapacity(t *testing.T) {
	src := make([]byte, 320)
	dst := make([]byte, 0, 640)
	if n := testing.AllocsPerRun(100, func() { dst = UlawToPCM16(dst[:0], src[:160]) }); n != 0 {
		t.Fatalf("UlawToPCM16 allocs %.1f", n)
	}
	if n := testing.AllocsPerRun(100, func() { dst = PCM16ToUlaw(dst[:0], src) }); n != 0 {
		t.Fatalf("PCM16ToUlaw allocs %.1f", n)
	}
}
