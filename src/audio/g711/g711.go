package g711

import "encoding/binary"

var (
	ulawDecode [256]int16
	alawDecode [256]int16
	ulawEncode [1 << 14]byte
	alawEncode [1 << 13]byte
)

var (
	ulawSegEnd = [8]int{0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF}
	alawSegEnd = [8]int{0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF}
)

func init() {
	for i := 0; i < 256; i++ {
		ulawDecode[i] = ulawToLinear(byte(i))
		alawDecode[i] = alawToLinear(byte(i))
	}
	for i := range ulawEncode {
		ulawEncode[i] = linearToUlaw(int16(uint16(i << 2)))
	}
	for i := range alawEncode {
		alawEncode[i] = linearToAlaw(int16(uint16(i << 3)))
	}
}

func segment(v int, ends *[8]int) int {
	for i, e := range ends {
		if v <= e {
			return i
		}
	}
	return 8
}

func linearToUlaw(pcm int16) byte {
	v := int(pcm) >> 2
	mask := 0xFF
	if v < 0 {
		v = -v
		mask = 0x7F
	}
	if v > 8159 {
		v = 8159
	}
	v += 0x84 >> 2
	seg := segment(v, &ulawSegEnd)
	if seg >= 8 {
		return byte(0x7F ^ mask)
	}
	u := (seg << 4) | ((v >> (seg + 1)) & 0xF)
	return byte(u ^ mask)
}

func ulawToLinear(u byte) int16 {
	u = ^u
	t := (int(u&0x0F) << 3) + 0x84
	t <<= (u & 0x70) >> 4
	if u&0x80 != 0 {
		return int16(0x84 - t)
	}
	return int16(t - 0x84)
}

func linearToAlaw(pcm int16) byte {
	v := int(pcm) >> 3
	mask := 0xD5
	if v < 0 {
		mask = 0x55
		v = -v - 1
	}
	seg := segment(v, &alawSegEnd)
	if seg >= 8 {
		return byte(0x7F ^ mask)
	}
	a := seg << 4
	if seg < 2 {
		a |= (v >> 1) & 0xF
	} else {
		a |= (v >> seg) & 0xF
	}
	return byte(a ^ mask)
}

func alawToLinear(a byte) int16 {
	a ^= 0x55
	t := int(a&0x0F) << 4
	seg := int(a&0x70) >> 4
	switch seg {
	case 0:
		t += 8
	case 1:
		t += 0x108
	default:
		t += 0x108
		t <<= seg - 1
	}
	if a&0x80 != 0 {
		return int16(t)
	}
	return int16(-t)
}

func UlawDecodeSample(u byte) int16 { return ulawDecode[u] }
func AlawDecodeSample(a byte) int16 { return alawDecode[a] }
func UlawEncodeSample(s int16) byte { return ulawEncode[uint16(s)>>2] }
func AlawEncodeSample(s int16) byte { return alawEncode[uint16(s)>>3] }

func DecodeUlaw(dst []int16, src []byte) []int16 {
	dst = growI16(dst, len(src))
	out := dst[len(dst)-len(src):]
	for i, b := range src {
		out[i] = ulawDecode[b]
	}
	return dst
}

func DecodeAlaw(dst []int16, src []byte) []int16 {
	dst = growI16(dst, len(src))
	out := dst[len(dst)-len(src):]
	for i, b := range src {
		out[i] = alawDecode[b]
	}
	return dst
}

func EncodeUlaw(dst []byte, src []int16) []byte {
	dst = growBytes(dst, len(src))
	out := dst[len(dst)-len(src):]
	for i, s := range src {
		out[i] = ulawEncode[uint16(s)>>2]
	}
	return dst
}

func EncodeAlaw(dst []byte, src []int16) []byte {
	dst = growBytes(dst, len(src))
	out := dst[len(dst)-len(src):]
	for i, s := range src {
		out[i] = alawEncode[uint16(s)>>3]
	}
	return dst
}

func UlawToPCM16(dst, src []byte) []byte {
	dst = growBytes(dst, 2*len(src))
	out := dst[len(dst)-2*len(src):]
	for i, b := range src {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(ulawDecode[b]))
	}
	return dst
}

func AlawToPCM16(dst, src []byte) []byte {
	dst = growBytes(dst, 2*len(src))
	out := dst[len(dst)-2*len(src):]
	for i, b := range src {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(alawDecode[b]))
	}
	return dst
}

func PCM16ToUlaw(dst, src []byte) []byte {
	n := len(src) / 2
	dst = growBytes(dst, n)
	out := dst[len(dst)-n:]
	for i := range out {
		out[i] = ulawEncode[binary.LittleEndian.Uint16(src[2*i:])>>2]
	}
	return dst
}

func PCM16ToAlaw(dst, src []byte) []byte {
	n := len(src) / 2
	dst = growBytes(dst, n)
	out := dst[len(dst)-n:]
	for i := range out {
		out[i] = alawEncode[binary.LittleEndian.Uint16(src[2*i:])>>3]
	}
	return dst
}

func growI16(s []int16, n int) []int16 {
	l := len(s)
	if cap(s)-l < n {
		ns := make([]int16, l, l+n)
		copy(ns, s)
		s = ns
	}
	return s[:l+n]
}

func growBytes(s []byte, n int) []byte {
	l := len(s)
	if cap(s)-l < n {
		ns := make([]byte, l, l+n)
		copy(ns, s)
		s = ns
	}
	return s[:l+n]
}
