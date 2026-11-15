package resample

import (
	"encoding/binary"
	"fmt"
	"math"
)

type Resampler struct {
	inRate, outRate int
	b               *bank
	stepInt         int
	stepRem         int
	phase           int
	base            int
	buf             []float32
	scratch         []int16
	phaseScale      float64
}

func New(inRate, outRate int) (*Resampler, error) {
	if inRate <= 0 || outRate <= 0 {
		return nil, fmt.Errorf("resample: invalid rates %d -> %d", inRate, outRate)
	}
	r := &Resampler{inRate: inRate, outRate: outRate}
	if inRate == outRate {
		return r, nil
	}
	r.b = getBank(inRate, outRate)
	r.stepInt = r.b.m / r.b.l
	r.stepRem = r.b.m % r.b.l
	r.phaseScale = float64(r.b.phases) / float64(r.b.l)
	r.Reset()
	return r, nil
}

func (r *Resampler) InRate() int  { return r.inRate }
func (r *Resampler) OutRate() int { return r.outRate }

func (r *Resampler) Reset() {
	if r.b == nil {
		return
	}
	h := r.b.taps - 1
	if cap(r.buf) < h {
		r.buf = make([]float32, h, h+1024)
	}
	r.buf = r.buf[:h]
	clear(r.buf)
	r.base = h
	r.phase = 0
}

func (r *Resampler) MaxOutput(inSamples int) int {
	if r.b == nil {
		return inSamples
	}
	return int((int64(inSamples)*int64(r.b.l))/int64(r.b.m)) + 2
}

func (r *Resampler) Process(dst, src []int16) []int16 {
	if r.b == nil {
		return append(dst, src...)
	}
	r.buf = growF32(r.buf, len(src))
	tail := r.buf[len(r.buf)-len(src):]
	for i, s := range src {
		tail[i] = float32(s)
	}
	return r.run(dst)
}

func (r *Resampler) ProcessBytes(dst, src []byte) []byte {
	n := len(src) / 2
	if r.b == nil {
		return append(dst, src[:n*2]...)
	}
	r.buf = growF32(r.buf, n)
	tail := r.buf[len(r.buf)-n:]
	for i := range tail {
		tail[i] = float32(int16(binary.LittleEndian.Uint16(src[2*i:])))
	}
	r.scratch = r.run(r.scratch[:0])
	return appendBytes(dst, r.scratch)
}

func (r *Resampler) Flush(dst []int16) []int16 {
	if r.b == nil {
		return dst
	}
	pad := r.b.taps
	r.buf = growF32(r.buf, pad)
	clear(r.buf[len(r.buf)-pad:])
	dst = r.run(dst)
	r.Reset()
	return dst
}

func (r *Resampler) run(dst []int16) []int16 {
	b := r.b
	taps := b.taps
	coefs := b.coefs
	buf := r.buf
	base, phase := r.base, r.phase
	l := b.l
	if need := len(dst) + r.MaxOutput(len(buf)-base); cap(dst) < need {
		nd := make([]int16, len(dst), need)
		copy(nd, dst)
		dst = nd
	}
	for base < len(buf) {
		win := buf[base-taps+1 : base+1]
		var acc float32
		if b.exact {
			acc = dot(coefs[phase*taps:(phase+1)*taps], win)
		} else {
			pos := float64(phase) * r.phaseScale
			i := int(pos)
			frac := float32(pos - float64(i))
			a0 := dot(coefs[i*taps:(i+1)*taps], win)
			a1 := dot(coefs[(i+1)*taps:(i+2)*taps], win)
			acc = a0 + frac*(a1-a0)
		}
		dst = append(dst, clamp16(acc))
		phase += r.stepRem
		base += r.stepInt
		if phase >= l {
			phase -= l
			base++
		}
	}
	keep := taps - 1
	drop := len(buf) - keep
	copy(buf, buf[drop:])
	r.buf = buf[:keep]
	r.base = base - drop
	r.phase = phase
	return dst
}

func dot(c, x []float32) float32 {
	x = x[:len(c)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(c); i += 4 {
		s0 += c[i] * x[i]
		s1 += c[i+1] * x[i+1]
		s2 += c[i+2] * x[i+2]
		s3 += c[i+3] * x[i+3]
	}
	for ; i < len(c); i++ {
		s0 += c[i] * x[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func clamp16(v float32) int16 {
	if v >= 32767 {
		return 32767
	}
	if v <= -32768 {
		return -32768
	}
	return int16(math.Round(float64(v)))
}

func growF32(s []float32, n int) []float32 {
	l := len(s)
	if cap(s)-l < n {
		ns := make([]float32, l, (l+n)*2)
		copy(ns, s)
		s = ns
	}
	return s[:l+n]
}

func appendBytes(dst []byte, src []int16) []byte {
	l := len(dst)
	need := l + 2*len(src)
	if cap(dst) < need {
		nd := make([]byte, l, need)
		copy(nd, dst)
		dst = nd
	}
	dst = dst[:need]
	out := dst[l:]
	for i, s := range src {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(s))
	}
	return dst
}

func Resample(src []int16, inRate, outRate int) ([]int16, error) {
	r, err := New(inRate, outRate)
	if err != nil {
		return nil, err
	}
	if r.b == nil {
		return append([]int16(nil), src...), nil
	}
	want := int((int64(len(src))*int64(outRate) + int64(inRate) - 1) / int64(inRate))
	d := r.b.delayOutput
	out := make([]int16, 0, r.MaxOutput(len(src)+2*r.b.taps)+d)
	out = r.Process(out, src)
	out = r.Flush(out)
	for len(out) < d+want {
		out = append(out, 0)
	}
	return out[d : d+want], nil
}

func ResampleBytes(src []byte, inRate, outRate int) ([]byte, error) {
	in := make([]int16, len(src)/2)
	for i := range in {
		in[i] = int16(binary.LittleEndian.Uint16(src[2*i:]))
	}
	out, err := Resample(in, inRate, outRate)
	if err != nil {
		return nil, err
	}
	return appendBytes(make([]byte, 0, 2*len(out)), out), nil
}
