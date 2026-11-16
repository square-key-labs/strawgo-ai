package tone

import (
	"encoding/binary"
	"errors"
	"math"
	"time"
)

const (
	DefaultMinFrequency = 440.0
	DefaultMaxFrequency = 2000.0
	DefaultMinDuration  = 400 * time.Millisecond
	DefaultMinPurity    = 0.7
	DefaultMinLevelDBFS = -40.0
	DefaultBlock        = 20 * time.Millisecond
	DefaultMaxDeviation = 10.0
	DefaultRejectWidth  = 40.0
	FaxCallingTone      = 1100.0
	sidebandRatio       = 0.01
	bandMargin          = 5.0
)

var dtmfRows = [4]float64{697, 770, 852, 941}
var dtmfCols = [4]float64{1209, 1336, 1477, 1633}

type Config struct {
	SampleRate        int
	MinFrequency      float64
	MaxFrequency      float64
	MinDuration       time.Duration
	MinPurity         float64
	MinLevelDBFS      float64
	Block             time.Duration
	MaxDeviation      float64
	RejectFrequencies []float64
	RejectWidth       float64
}

func DefaultConfig(sampleRate int) Config {
	return Config{
		SampleRate:        sampleRate,
		MinFrequency:      DefaultMinFrequency,
		MaxFrequency:      DefaultMaxFrequency,
		MinDuration:       DefaultMinDuration,
		MinPurity:         DefaultMinPurity,
		MinLevelDBFS:      DefaultMinLevelDBFS,
		Block:             DefaultBlock,
		MaxDeviation:      DefaultMaxDeviation,
		RejectFrequencies: []float64{FaxCallingTone},
		RejectWidth:       DefaultRejectWidth,
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
	Purity    float64
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
	window    []float64
	windowed  []float64
	carry     []byte

	runBlocks int
	segBlocks int
	segSum    float64
	segPurity float64
	segSide   sidebands
	reported  bool
	freq      float64
	purity    float64
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
	if cfg.MaxDeviation <= 0 {
		cfg.MaxDeviation = def.MaxDeviation
	}
	if cfg.RejectFrequencies == nil {
		cfg.RejectFrequencies = def.RejectFrequencies
	}
	if cfg.RejectWidth <= 0 {
		cfg.RejectWidth = def.RejectWidth
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
	for f := cfg.MinFrequency - 2*d.step; f <= cfg.MaxFrequency+2*d.step+1e-9; f += d.step {
		d.freqs = append(d.freqs, f)
		d.coeffs = append(d.coeffs, Coefficient(f, cfg.SampleRate))
	}
	rms := 32768 * math.Pow(10, cfg.MinLevelDBFS/20)
	d.minEnergy = rms * rms * float64(n)
	d.block = make([]float64, n)
	d.window = make([]float64, n)
	d.windowed = make([]float64, n)
	for i := range d.window {
		d.window[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n-1))
	}
	return d, nil
}

func (d *Detector) Config() Config {
	return d.cfg
}

func (d *Detector) Reset() {
	d.carry = d.carry[:0]
	d.endRun(nil)
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

type sidebands struct {
	peak, sub, dtmf, h2, h3 float64
}

func (s *sidebands) add(o sidebands) {
	s.peak += o.peak
	s.sub += o.sub
	s.dtmf += o.dtmf
	s.h2 += o.h2
	s.h3 += o.h3
}

func (s sidebands) clean() bool {
	limit := sidebandRatio * s.peak
	return s.sub < limit && s.dtmf < limit && (s.h2 < limit || s.h3 < limit)
}

func (d *Detector) process(raw []byte, events []Event) []Event {
	freq, purity, side, ok := d.analyse(raw)
	if !ok {
		return d.endRun(events)
	}
	if d.segBlocks > 0 && math.Abs(freq-d.segSum/float64(d.segBlocks)) > d.cfg.MaxDeviation {
		d.segBlocks, d.segSum, d.segPurity, d.segSide = 0, 0, 0, sidebands{}
	}
	if d.segBlocks == 0 || purity < d.segPurity {
		d.segPurity = purity
	}
	d.segBlocks++
	d.segSum += freq
	d.segSide.add(side)
	d.runBlocks++
	if !d.reported && time.Duration(d.segBlocks)*d.cfg.Block >= d.cfg.MinDuration && d.segSide.clean() {
		d.reported = true
		d.freq, d.purity = d.segSum/float64(d.segBlocks), d.segPurity
		events = append(events, Event{Kind: Started, Frequency: d.freq, Purity: d.purity, Duration: d.runDuration()})
	}
	return events
}

func (d *Detector) endRun(events []Event) []Event {
	if d.reported {
		events = append(events, Event{Kind: Ended, Frequency: d.freq, Purity: d.purity, Duration: d.runDuration()})
	}
	d.runBlocks, d.segBlocks, d.segSum, d.segPurity, d.segSide = 0, 0, 0, 0, sidebands{}
	d.reported = false
	return events
}

func (d *Detector) runDuration() time.Duration {
	return time.Duration(d.runBlocks) * d.cfg.Block
}

func (d *Detector) analyse(raw []byte) (float64, float64, sidebands, bool) {
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
		return 0, 0, sidebands{}, false
	}
	best, k := 0.0, 0
	for i, c := range d.coeffs {
		if p := Goertzel(d.block, c); p > best {
			best, k = p, i
		}
	}
	purity := 2 * best / (float64(d.n) * energy)
	if purity < d.cfg.MinPurity || k == 0 || k == len(d.coeffs)-1 {
		return 0, 0, sidebands{}, false
	}
	freq := d.refine(k, best)
	if freq < d.cfg.MinFrequency-bandMargin || freq > d.cfg.MaxFrequency+bandMargin {
		return 0, 0, sidebands{}, false
	}
	for _, r := range d.cfg.RejectFrequencies {
		if math.Abs(freq-r) <= d.cfg.RejectWidth {
			return 0, 0, sidebands{}, false
		}
	}
	for i, v := range d.block {
		d.windowed[i] = v * d.window[i]
	}
	side := sidebands{
		peak: d.power(freq),
		sub:  max(d.power(freq/2), d.power(freq/3), d.power(2*freq/3)),
		dtmf: d.dtmfPartner(freq),
		h2:   d.power(2 * freq),
		h3:   d.power(3 * freq),
	}
	return freq, purity, side, true
}

func (d *Detector) refine(k int, peak float64) float64 {
	lo := math.Sqrt(Goertzel(d.block, d.coeffs[k-1]))
	hi := math.Sqrt(Goertzel(d.block, d.coeffs[k+1]))
	mid := math.Sqrt(peak)
	den := lo - 2*mid + hi
	if den == 0 {
		return d.freqs[k]
	}
	offset := 0.5 * (lo - hi) / den
	return d.freqs[k] + max(min(offset, 1), -1)*d.step
}

func (d *Detector) power(freq float64) float64 {
	if freq <= 0 || freq >= float64(d.cfg.SampleRate)/2 {
		return 0
	}
	return Goertzel(d.windowed, Coefficient(freq, d.cfg.SampleRate))
}

func (d *Detector) dtmfPartner(freq float64) float64 {
	near := func(set [4]float64) bool {
		for _, f := range set {
			if math.Abs(freq-f) <= 2*d.step {
				return true
			}
		}
		return false
	}
	var other [4]float64
	switch {
	case near(dtmfRows):
		other = dtmfCols
	case near(dtmfCols):
		other = dtmfRows
	default:
		return 0
	}
	strongest := 0.0
	for _, f := range other {
		strongest = max(strongest, d.power(f))
	}
	return strongest
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
