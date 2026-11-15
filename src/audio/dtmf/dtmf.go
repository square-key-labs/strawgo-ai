package dtmf

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"time"
)

const (
	DefaultToneDuration = 300 * time.Millisecond
	DefaultGapDuration  = 200 * time.Millisecond
	DefaultAmplitude    = 0.2
	fadeDuration        = 5 * time.Millisecond
)

var lowFreqs = [4]float64{697, 770, 852, 941}
var highFreqs = [4]float64{1209, 1336, 1477, 1633}

const layout = "123A456B789C*0#D"

func Frequencies(key byte) (low, high float64, err error) {
	for i := 0; i < len(layout); i++ {
		if layout[i] == key || (key >= 'a' && key <= 'd' && layout[i] == key-32) {
			return lowFreqs[i/4], highFreqs[i%4], nil
		}
	}
	return 0, 0, fmt.Errorf("dtmf: invalid key %q", key)
}

func Valid(key byte) bool {
	_, _, err := Frequencies(key)
	return err == nil
}

type Options struct {
	Tone      time.Duration
	Gap       time.Duration
	Amplitude float64
}

func DefaultOptions() Options {
	return Options{Tone: DefaultToneDuration, Gap: DefaultGapDuration, Amplitude: DefaultAmplitude}
}

type cacheKey struct {
	key  byte
	rate int
	opts Options
}

var cache sync.Map

func Tone(key byte, sampleRate int) ([]byte, error) {
	return ToneWithOptions(key, sampleRate, DefaultOptions())
}

func ToneWithOptions(key byte, sampleRate int, opts Options) ([]byte, error) {
	if key >= 'a' && key <= 'd' {
		key -= 32
	}
	ck := cacheKey{key: key, rate: sampleRate, opts: opts}
	if v, ok := cache.Load(ck); ok {
		return v.([]byte), nil
	}
	pcm, err := Generate(nil, key, sampleRate, opts)
	if err != nil {
		return nil, err
	}
	v, _ := cache.LoadOrStore(ck, pcm)
	return v.([]byte), nil
}

func AppendTone(dst []byte, key byte, sampleRate int) ([]byte, error) {
	t, err := Tone(key, sampleRate)
	if err != nil {
		return dst, err
	}
	return append(dst, t...), nil
}

func Sequence(keys string, sampleRate int) ([]byte, error) {
	var out []byte
	for i := 0; i < len(keys); i++ {
		var err error
		if out, err = AppendTone(out, keys[i], sampleRate); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func Generate(dst []byte, key byte, sampleRate int, opts Options) ([]byte, error) {
	if sampleRate <= 0 {
		return dst, fmt.Errorf("dtmf: invalid sample rate %d", sampleRate)
	}
	lo, hi, err := Frequencies(key)
	if err != nil {
		return dst, err
	}
	toneN := int(opts.Tone.Seconds() * float64(sampleRate))
	gapN := int(opts.Gap.Seconds() * float64(sampleRate))
	fadeN := min(int(fadeDuration.Seconds()*float64(sampleRate)), toneN/2)
	nyq := float64(sampleRate) / 2
	l := len(dst)
	dst = append(dst, make([]byte, 2*(toneN+gapN))...)
	out := dst[l:]
	amp := opts.Amplitude * 32767
	wl, wh := 2*math.Pi*lo/float64(sampleRate), 2*math.Pi*hi/float64(sampleRate)
	ah := amp
	if hi >= nyq {
		ah = 0
	}
	al := amp
	if lo >= nyq {
		al = 0
	}
	for i := 0; i < toneN; i++ {
		v := al*math.Sin(wl*float64(i)) + ah*math.Sin(wh*float64(i))
		if fadeN > 0 {
			if i < fadeN {
				v *= 0.5 - 0.5*math.Cos(math.Pi*float64(i)/float64(fadeN))
			} else if j := toneN - 1 - i; j < fadeN {
				v *= 0.5 - 0.5*math.Cos(math.Pi*float64(j)/float64(fadeN))
			}
		}
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(math.Round(max(min(v, 32767), -32768)))))
	}
	return dst, nil
}
