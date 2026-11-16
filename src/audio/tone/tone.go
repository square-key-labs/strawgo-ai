package tone

import (
	"encoding/binary"
	"errors"
	"math"
	"time"
)

const (
	DefaultMinFrequency = 440.0
	DefaultMaxFrequency = 1400.0
	DefaultMinDuration  = 250 * time.Millisecond
	DefaultMinPurity    = 0.7
	DefaultMinLevelDBFS = -40.0
	DefaultBlock        = 20 * time.Millisecond
)

type Config struct {
	SampleRate   int
	MinFrequency float64
	MaxFrequency float64
	MinDuration  time.Duration
	MinPurity    float64
	MinLevelDBFS float64
	Block        time.Duration
}

func DefaultConfig(sampleRate int) Config {
	return Config{
		SampleRate:   sampleRate,
		MinFrequency: DefaultMinFrequency,
		MaxFrequency: DefaultMaxFrequency,
		MinDuration:  DefaultMinDuration,
		MinPurity:    DefaultMinPurity,
		MinLevelDBFS: DefaultMinLevelDBFS,
		Block:        DefaultBlock,
	}
}

type EventKind int

const (
	Started EventKind = iota + 1
	Ended
)

func (k EventKind) String() string {
	switch k {
	case Started:
		return "started"
	case Ended:
		return "ended"
	}
	return "none"
}

type Event struct {
	Kind      EventKind
	Frequency float64
	Duration  time.Duration
}

type Detector struct {
	cfg       Config
	n         int
	step      float64
	freqs     []float64
	coeffs    []float64
	minEnergy float64
	block     []float64
	carry     []byte

	runBlocks int
	runFreq   float64
	reported  bool
}

func NewDetector(cfg Config) (*Detector, error) {
	def := DefaultConfig(cfg.SampleRate)
	if cfg.MinFrequency <= 0 {
		cfg.MinFrequency = def.MinFrequency
	}
	if cfg.MaxFrequency <= 0 {
		cfg.MaxFrequency = def.MaxFrequency
	}
	if cfg.MinDuration <= 0 {
		cfg.MinDuration = def.MinDuration
	}
	if cfg.MinPurity <= 0 {
		cfg.MinPurity = def.MinPurity
	}
	if cfg.MinLevelDBFS == 0 {
		cfg.MinLevelDBFS = def.MinLevelDBFS
	}
	if cfg.Block <= 0 {
		cfg.Block = def.Block
	}
	if cfg.SampleRate <= 0 {
		return nil, errors.New("tone: sample rate must be positive")
	}
	if cfg.MinFrequency >= cfg.MaxFrequency || cfg.MaxFrequency >= float64(cfg.SampleRate)/2 {
		return nil, errors.New("tone: frequency range must be ascending and below the Nyquist frequency")
	}
	n := int(cfg.Block.Seconds() * float64(cfg.SampleRate))
	if n < 16 {
		return nil, errors.New("tone: block too short for the sample rate")
	}
	d := &Detector{cfg: cfg, n: n, step: float64(cfg.SampleRate) / float64(2*n)}
	for f := cfg.MinFrequency - d.step; f <= cfg.MaxFrequency+d.step+1e-9; f += d.step {
		d.freqs = append(d.freqs, f)
		d.coeffs = append(d.coeffs, Coefficient(f, cfg.SampleRate))
	}
	rms := 32768 * math.Pow(10, cfg.MinLevelDBFS/20)
	d.minEnergy = rms * rms * float64(n)
	d.block = make([]float64, n)
	return d, nil
}

func (d *Detector) Config() Config {
	return d.cfg
}

func (d *Detector) Reset() {
	d.carry = d.carry[:0]
	d.runBlocks = 0
	d.runFreq = 0
	d.reported = false
}

func (d *Detector) Write(pcm []byte) []Event {
	var events []Event
	blockBytes := 2 * d.n
	if len(d.carry) > 0 {
		need := blockBytes - len(d.carry)
		if len(pcm) < need {
			d.carry = append(d.carry, pcm...)
			return nil
		}
		d.carry = append(d.carry, pcm[:need]...)
		pcm = pcm[need:]
		events = d.process(d.carry, events)
		d.carry = d.carry[:0]
	}
	for len(pcm) >= blockBytes {
		events = d.process(pcm[:blockBytes], events)
		pcm = pcm[blockBytes:]
	}
	d.carry = append(d.carry, pcm...)
	return events
}

func (d *Detector) process(raw []byte, events []Event) []Event {
	freq, ok := d.analyse(raw)
	if ok && d.runBlocks > 0 && math.Abs(freq-d.runFreq) > 2*d.step {
		events = d.endRun(events)
	}
	if !ok {
		return d.endRun(events)
	}
	if d.runBlocks == 0 {
		d.runFreq = freq
	}
	d.runBlocks++
	if !d.reported && d.runDuration() >= d.cfg.MinDuration {
		d.reported = true
		events = append(events, Event{Kind: Started, Frequency: d.runFreq, Duration: d.runDuration()})
	}
	return events
}

func (d *Detector) endRun(events []Event) []Event {
	if d.reported {
		events = append(events, Event{Kind: Ended, Frequency: d.runFreq, Duration: d.runDuration()})
	}
	d.runBlocks = 0
	d.runFreq = 0
	d.reported = false
	return events
}

func (d *Detector) runDuration() time.Duration {
	return time.Duration(d.runBlocks) * d.cfg.Block
}

func (d *Detector) analyse(raw []byte) (float64, bool) {
	var mean float64
	for i := range d.block {
		v := float64(int16(binary.LittleEndian.Uint16(raw[2*i:])))
		d.block[i] = v
		mean += v
	}
	mean /= float64(d.n)
	var energy float64
	for i, v := range d.block {
		v -= mean
		d.block[i] = v
		energy += v * v
	}
	if energy < d.minEnergy {
		return 0, false
	}
	best, bestFreq := 0.0, 0.0
	for k, c := range d.coeffs {
		p := Goertzel(d.block, c)
		if p > best {
			best, bestFreq = p, d.freqs[k]
		}
	}
	if bestFreq < d.cfg.MinFrequency || bestFreq > d.cfg.MaxFrequency {
		return 0, false
	}
	purity := 2 * best / (float64(d.n) * energy)
	return bestFreq, purity >= d.cfg.MinPurity
}

func Goertzel(samples []float64, coeff float64) float64 {
	var s1, s2 float64
	for _, x := range samples {
		s := x + coeff*s1 - s2
		s2, s1 = s1, s
	}
	return s1*s1 + s2*s2 - coeff*s1*s2
}

func Coefficient(frequency float64, sampleRate int) float64 {
	return 2 * math.Cos(2*math.Pi*frequency/float64(sampleRate))
}
