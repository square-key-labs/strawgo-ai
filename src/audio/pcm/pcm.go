package pcm

import (
	"encoding/binary"
	"math"
)

const (
	SpeakingThreshold = 20
	MinDBFS           = -96.0
)

func sample(b []byte, i int) int32 {
	return int32(int16(binary.LittleEndian.Uint16(b[2*i:])))
}

func put(b []byte, i int, v int32) {
	if v > math.MaxInt16 {
		v = math.MaxInt16
	} else if v < math.MinInt16 {
		v = math.MinInt16
	}
	binary.LittleEndian.PutUint16(b[2*i:], uint16(int16(v)))
}

func Grow(s []byte, n int) []byte {
	l := len(s)
	if cap(s)-l < n {
		ns := make([]byte, l, l+n)
		copy(ns, s)
		s = ns
	}
	return s[:l+n]
}

func AppendSilence(dst []byte, n int) []byte {
	l := len(dst)
	dst = Grow(dst, n)
	clear(dst[l:])
	return dst
}

func ToInt16(dst []int16, src []byte) []int16 {
	n := len(src) / 2
	l := len(dst)
	if cap(dst)-l < n {
		nd := make([]int16, l, l+n)
		copy(nd, dst)
		dst = nd
	}
	dst = dst[:l+n]
	out := dst[l:]
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(src[2*i:]))
	}
	return dst
}

func FromInt16(dst []byte, src []int16) []byte {
	l := len(dst)
	dst = Grow(dst, 2*len(src))
	out := dst[l:]
	for i, s := range src {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(s))
	}
	return dst
}

func Mix(dst, a, b []byte) []byte {
	na, nb := len(a)/2, len(b)/2
	n := max(na, nb)
	l := len(dst)
	dst = Grow(dst, 2*n)
	out := dst[l:]
	for i := 0; i < n; i++ {
		var v int32
		if i < na {
			v += sample(a, i)
		}
		if i < nb {
			v += sample(b, i)
		}
		put(out, i, v)
	}
	return dst
}

func MixInPlace(dst, src []byte, gain float64) {
	n := min(len(dst), len(src)) / 2
	g := int32(gain * 65536)
	for i := 0; i < n; i++ {
		v := sample(dst, i) + int32((int64(sample(src, i))*int64(g))>>16)
		put(dst, i, v)
	}
}

func Interleave(dst, left, right []byte) []byte {
	nl, nr := len(left)/2, len(right)/2
	n := max(nl, nr)
	l := len(dst)
	dst = Grow(dst, 4*n)
	out := dst[l:]
	for i := 0; i < n; i++ {
		var lv, rv uint16
		if i < nl {
			lv = binary.LittleEndian.Uint16(left[2*i:])
		}
		if i < nr {
			rv = binary.LittleEndian.Uint16(right[2*i:])
		}
		binary.LittleEndian.PutUint16(out[4*i:], lv)
		binary.LittleEndian.PutUint16(out[4*i+2:], rv)
	}
	return dst
}

func ToMono(dst, src []byte, channels int) []byte {
	if channels <= 1 {
		return append(dst, src...)
	}
	frames := len(src) / (2 * channels)
	l := len(dst)
	dst = Grow(dst, 2*frames)
	out := dst[l:]
	for i := 0; i < frames; i++ {
		var s int32
		for c := 0; c < channels; c++ {
			s += sample(src, i*channels+c)
		}
		put(out, i, s/int32(channels))
	}
	return dst
}

func Scale(dst, src []byte, factor float64) []byte {
	n := len(src) / 2
	l := len(dst)
	dst = Grow(dst, 2*n)
	out := dst[l:]
	for i := 0; i < n; i++ {
		put(out, i, int32(math.Round(float64(sample(src, i))*factor)))
	}
	return dst
}

func Peak(b []byte) int {
	n := len(b) / 2
	var m int32
	for i := 0; i < n; i++ {
		v := sample(b, i)
		if v < 0 {
			v = -v
		}
		if v > m {
			m = v
		}
	}
	return int(m)
}

func RMS(b []byte) float64 {
	n := len(b) / 2
	if n == 0 {
		return 0
	}
	var sum int64
	for i := 0; i < n; i++ {
		v := int64(sample(b, i))
		sum += v * v
	}
	return math.Sqrt(float64(sum)/float64(n)) / 32768
}

func DBFS(b []byte) float64 {
	return ToDBFS(RMS(b))
}

func ToDBFS(rms float64) float64 {
	if rms <= 0 {
		return MinDBFS
	}
	return max(20*math.Log10(rms), MinDBFS)
}

func Volume(b []byte) float64 {
	return NormalizeValue(DBFS(b), -60, 0)
}

func NormalizeValue(v, lo, hi float64) float64 {
	if hi <= lo {
		return 0
	}
	return min(max((v-lo)/(hi-lo), 0), 1)
}

func IsSilence(b []byte) bool {
	return Peak(b) <= SpeakingThreshold
}

func IsSilenceThreshold(b []byte, threshold int) bool {
	return Peak(b) <= threshold
}

func ExpSmoothing(value, prev, factor float64) float64 {
	return prev + factor*(value-prev)
}

func FadeOut(b []byte) {
	n := int32(len(b) / 2)
	for i := int32(0); i < n; i++ {
		put(b, int(i), sample(b, int(i))*(n-1-i)/n)
	}
}
