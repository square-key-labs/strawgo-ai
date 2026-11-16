package tone

import (
	"encoding/binary"
	"errors"
	"math"
	"time"
)

const (
	DefaultMinFrequency    = 440.0
	DefaultMaxFrequency    = 2000.0
	DefaultMinDuration     = 250 * time.Millisecond
	DefaultConfirmDuration = 400 * time.Millisecond
	DefaultMinPurity       = 0.7
	DefaultMinLevelDBFS    = -40.0
	DefaultBlock           = 20 * time.Millisecond
	DefaultHop             = 5 * time.Millisecond
	DefaultMaxDeviation    = 4.0
	DefaultMaxDrift        = 30.0
	DefaultMaxJump         = 40.0
	DefaultRejectWidth     = 15.0
	FaxCallingTone         = 1100.0
	subharmonicRatio       = 0.01
	secondHarmonicRatio    = 0.04
	thirdHarmonicRatio     = 0.01
	bandMargin             = 5.0
	coverage               = 0.8
)

var dtmfRows = [4]float64{697, 770, 852, 941}
var dtmfCols = [4]float64{1209, 1336, 1477, 1633}

type Config struct {
	SampleRate        int
	MinFrequency      float64
	MaxFrequency      float64
	MinDuration       time.Duration
	ConfirmDuration   time.Duration
	MinPurity         float64
	MinLevelDBFS      float64
	Block             time.Duration
	Hop               time.Duration
	MaxDeviation      float64
	MaxDrift          float64
	MaxJump           float64
	RejectFrequencies []float64
	RejectWidth       float64
}

func DefaultConfig(sampleRate int) Config {
	return Config{
		SampleRate:        sampleRate,
		MinFrequency:      DefaultMinFrequency,
		MaxFrequency:      DefaultMaxFrequency,
		MinDuration:       DefaultMinDuration,
		ConfirmDuration:   DefaultConfirmDuration,
		MinPurity:         DefaultMinPurity,
		MinLevelDBFS:      DefaultMinLevelDBFS,
		Block:             DefaultBlock,
		Hop:               DefaultHop,
		MaxDeviation:      DefaultMaxDeviation,
		MaxDrift:          DefaultMaxDrift,
		MaxJump:           DefaultMaxJump,
		RejectFrequencies: []float64{FaxCallingTone},
		RejectWidth:       DefaultRejectWidth,
	}
}

type EventKind int

const (
	Started EventKind = iota + 1
	Confirmed
	Ended
)

func (k EventKind) String() string {
	switch k {
	case Started:
		return "started"
	case Confirmed:
		return "confirmed"
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
	return s.sub < subharmonicRatio*s.peak && s.dtmf < subharmonicRatio*s.peak &&
		(s.h2 < secondHarmonicRatio*s.peak || s.h3 < thirdHarmonicRatio*s.peak)
}

type sample struct {
	at     int64
	freq   float64
	purity float64
	side   sidebands
}

type Detector struct {
	cfg       Config
	n         int
	hop       int
	gap       int
	step      float64
	freqs     []float64
	coeffs    []float64
	minEnergy float64
	block     []float64
	window    []float64
	windowed  []float64
	pending   []byte
	pos       int64

	seg       []sample
	runStart  int64
	runEnd    int64
	running   bool
	started   bool
	confirmed bool
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
	if cfg.ConfirmDuration <= 0 {
		cfg.ConfirmDuration = def.ConfirmDuration
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
	if cfg.Hop <= 0 || cfg.Hop > cfg.Block {
		cfg.Hop = min(def.Hop, cfg.Block)
	}
	if cfg.MaxDeviation <= 0 {
		cfg.MaxDeviation = def.MaxDeviation
	}
	if cfg.MaxDrift <= 0 {
		cfg.MaxDrift = def.MaxDrift
	}
	if cfg.MaxJump <= 0 {
		cfg.MaxJump = def.MaxJump
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
	n := int(cfg.Block.Seconds() * float64(cfg.SampleRate))
	if n < 16 {
		return nil, errors.New("tone: block too short for the sample rate")
	}
	step := float64(cfg.SampleRate) / float64(2*n)
	if cfg.MinFrequency >= cfg.MaxFrequency || cfg.MaxFrequency+2*step >= float64(cfg.SampleRate)/2 {
		return nil, errors.New("tone: frequency range must be ascending and below the Nyquist frequency")
	}
	d := &Detector{
		cfg:  cfg,
		n:    n,
		hop:  max(int(cfg.Hop.Seconds()*float64(cfg.SampleRate)), 1),
		step: step,
	}
	d.gap = n
	for f := cfg.MinFrequency - 2*step; f <= cfg.MaxFrequency+2*step+1e-9; f += step {
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
	d.pending = d.pending[:0]
	d.pos = 0
	d.reset()
}

func (d *Detector) reset() {
	d.seg = d.seg[:0]
	d.running, d.started, d.confirmed = false, false, false
}

func (d *Detector) Write(pcm []byte) []Event {
	var events []Event
	d.pending = append(d.pending, pcm...)
	off := 0
	for len(d.pending)-off >= 2*d.n {
		events = d.process(d.pending[off:off+2*d.n], events)
		off += 2 * d.hop
		d.pos += int64(d.hop)
	}
	if off > 0 {
		d.pending = d.pending[:copy(d.pending, d.pending[off:])]
	}
	return events
}

func (d *Detector) duration(samples int64) time.Duration {
	return time.Duration(samples) * time.Second / time.Duration(d.cfg.SampleRate)
}

func (d *Detector) samples(t time.Duration) int64 {
	return int64(t.Seconds() * float64(d.cfg.SampleRate))
}

func (d *Detector) process(raw []byte, events []Event) []Event {
	freq, purity, side, ok := d.analyse(raw)
	if !ok {
		if d.running && d.pos-d.runEnd+int64(d.n) > int64(d.gap) {
			events = d.endRun(events)
		}
		return events
	}
	if !d.running {
		d.running, d.runStart = true, d.pos
	}
	if len(d.seg) > 0 && math.Abs(freq-d.seg[len(d.seg)-1].freq) > d.cfg.MaxJump {
		d.seg = d.seg[:0]
	}
	d.seg = append(d.seg, sample{at: d.pos, freq: freq, purity: purity, side: side})
	if keep := d.pos - d.samples(max(d.cfg.MinDuration, d.cfg.ConfirmDuration)) - int64(2*d.hop); d.seg[0].at < keep {
		i := 0
		for d.seg[i].at < keep {
			i++
		}
		d.seg = d.seg[:copy(d.seg, d.seg[i:])]
	}
	d.runEnd = d.pos + int64(d.n)
	if !d.started {
		if f, p, ok := d.steady(d.cfg.MinDuration); ok {
			d.started, d.freq, d.purity = true, f, p
			events = append(events, Event{Kind: Started, Frequency: f, Purity: p, Duration: d.duration(d.runEnd - d.runStart)})
		}
	}
	if d.started && !d.confirmed && d.cfg.ConfirmDuration > d.cfg.MinDuration {
		if f, p, ok := d.steady(d.cfg.ConfirmDuration); ok {
			d.confirmed, d.freq, d.purity = true, f, p
			events = append(events, Event{Kind: Confirmed, Frequency: f, Purity: p, Duration: d.duration(d.runEnd - d.runStart)})
		}
	}
	return events
}

func (d *Detector) endRun(events []Event) []Event {
	if d.started {
		events = append(events, Event{Kind: Ended, Frequency: d.freq, Purity: d.purity, Duration: d.duration(d.runEnd - d.runStart)})
	}
	d.reset()
	return events
}

func (d *Detector) steady(span time.Duration) (float64, float64, bool) {
	need := d.samples(span)
	last := d.seg[len(d.seg)-1].at + int64(d.n)
	from := last - need
	if d.seg[0].at > from+int64(d.hop) {
		return 0, 0, false
	}
	var window []sample
	for i := len(d.seg) - 1; i >= 0 && d.seg[i].at >= from; i-- {
		window = d.seg[i:]
	}
	expected := float64(need-int64(d.n))/float64(d.hop) + 1
	if float64(len(window)) < coverage*expected || len(window) < 3 {
		return 0, 0, false
	}
	base := window[0].freq
	var n, st, sf, stt, stf, sff, purity float64
	var side sidebands
	for _, s := range window {
		t := float64(s.at-window[0].at) / float64(d.cfg.SampleRate)
		f := s.freq - base
		n++
		st += t
		sf += f
		stt += t * t
		stf += t * f
		sff += f * f
		purity += s.purity
		side.add(s.side)
	}
	den := n*stt - st*st
	if den <= 0 {
		return 0, 0, false
	}
	slope := (n*stf - st*sf) / den
	intercept := (sf - slope*st) / n
	sse := sff - intercept*sf - slope*stf
	if math.Sqrt(max(sse, 0)/n) > d.cfg.MaxDeviation {
		return 0, 0, false
	}
	spanSecs := float64(window[len(window)-1].at-window[0].at) / float64(d.cfg.SampleRate)
	if math.Abs(slope*spanSecs) > d.cfg.MaxDrift || !side.clean() {
		return 0, 0, false
	}
	return base + sf/n, purity / n, true
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
	scale := 2 / (float64(d.n) * energy)
	best, k := 0.0, 0
	for i, c := range d.coeffs {
		if p := Goertzel(d.block, c); p > best {
			best, k = p, i
		}
	}
	if best*scale < 0.75*d.cfg.MinPurity || k == 0 || k == len(d.coeffs)-1 {
		return 0, 0, sidebands{}, false
	}
	freq := d.refine(k, best)
	purity := Goertzel(d.block, Coefficient(freq, d.cfg.SampleRate)) * scale
	if purity < d.cfg.MinPurity {
		return 0, 0, sidebands{}, false
	}
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
	return freq, min(purity, 1), side, true
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
		if math.Abs(f-2*freq) <= 2*d.step || math.Abs(f-3*freq) <= 2*d.step {
			continue
		}
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
