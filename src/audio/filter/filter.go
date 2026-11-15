package filter

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type Filter interface {
	Start(sampleRate int) error
	Stop() error
	ProcessFrame(frame frames.Frame) error
	Filter(dst, audio []byte) []byte
}

type Func func(dst, audio []byte) []byte

type FuncFilter struct {
	fn      Func
	mu      sync.Mutex
	enabled bool
}

func NewFuncFilter(fn Func) *FuncFilter {
	return &FuncFilter{fn: fn, enabled: true}
}

func (f *FuncFilter) Start(int) error { return nil }
func (f *FuncFilter) Stop() error     { return nil }

func (f *FuncFilter) ProcessFrame(frame frames.Frame) error {
	if e, ok := frame.(*frames.FilterEnableFrame); ok {
		f.mu.Lock()
		f.enabled = e.Enable
		f.mu.Unlock()
	}
	return nil
}

func (f *FuncFilter) Filter(dst, audio []byte) []byte {
	f.mu.Lock()
	enabled := f.enabled
	f.mu.Unlock()
	if !enabled || f.fn == nil {
		return append(dst, audio...)
	}
	return f.fn(dst, audio)
}

type Chain struct {
	filters []Filter
	a, b    []byte
}

func NewChain(filters ...Filter) *Chain {
	return &Chain{filters: filters}
}

func (c *Chain) Start(sampleRate int) error {
	for _, f := range c.filters {
		if err := f.Start(sampleRate); err != nil {
			return err
		}
	}
	return nil
}

func (c *Chain) Stop() error {
	var first error
	for _, f := range c.filters {
		if err := f.Stop(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func (c *Chain) ProcessFrame(frame frames.Frame) error {
	for _, f := range c.filters {
		if err := f.ProcessFrame(frame); err != nil {
			return err
		}
	}
	return nil
}

func (c *Chain) Filter(dst, audio []byte) []byte {
	if len(c.filters) == 0 {
		return append(dst, audio...)
	}
	cur := audio
	last := len(c.filters) - 1
	for i, f := range c.filters[:last] {
		if i%2 == 0 {
			c.a = f.Filter(c.a[:0], cur)
			cur = c.a
		} else {
			c.b = f.Filter(c.b[:0], cur)
			cur = c.b
		}
	}
	return c.filters[last].Filter(dst, cur)
}

type NoiseGateParams struct {
	ThresholdDB float64
	AttackMs    float64
	HoldMs      float64
	ReleaseMs   float64
	FloorDB     float64
}

func DefaultNoiseGateParams() NoiseGateParams {
	return NoiseGateParams{ThresholdDB: -45, AttackMs: 2, HoldMs: 150, ReleaseMs: 80, FloorDB: -80}
}

type NoiseGate struct {
	mu         sync.Mutex
	params     NoiseGateParams
	enabled    bool
	sampleRate int

	threshold   float64
	floor       float64
	envAttack   float64
	envRelease  float64
	gainAttack  float64
	gainRelease float64
	holdSamples int

	env  float64
	gain float64
	hold int
}

func NewNoiseGate(params NoiseGateParams) *NoiseGate {
	return &NoiseGate{params: params, enabled: true, gain: 1}
}

func (g *NoiseGate) Start(sampleRate int) error {
	if sampleRate <= 0 {
		return fmt.Errorf("noise gate: invalid sample rate %d", sampleRate)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.sampleRate = sampleRate
	g.recompute()
	g.env, g.gain, g.hold = 0, 1, 0
	return nil
}

func (g *NoiseGate) Stop() error { return nil }

func (g *NoiseGate) Params() NoiseGateParams {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.params
}

func (g *NoiseGate) SetParams(p NoiseGateParams) {
	g.mu.Lock()
	g.params = p
	g.recompute()
	g.mu.Unlock()
}

func (g *NoiseGate) SetEnabled(enabled bool) {
	g.mu.Lock()
	g.enabled = enabled
	g.mu.Unlock()
}

func (g *NoiseGate) ProcessFrame(frame frames.Frame) error {
	switch f := frame.(type) {
	case *frames.FilterEnableFrame:
		g.SetEnabled(f.Enable)
	case *frames.FilterUpdateSettingsFrame:
		g.mu.Lock()
		defer g.mu.Unlock()
		for k, v := range f.Settings {
			fv, ok := toFloat(v)
			if !ok {
				return fmt.Errorf("noise gate: setting %q: not a number", k)
			}
			switch k {
			case "threshold_db":
				g.params.ThresholdDB = fv
			case "attack_ms":
				g.params.AttackMs = fv
			case "hold_ms":
				g.params.HoldMs = fv
			case "release_ms":
				g.params.ReleaseMs = fv
			case "floor_db":
				g.params.FloorDB = fv
			default:
				return fmt.Errorf("noise gate: unknown setting %q", k)
			}
		}
		g.recompute()
	}
	return nil
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	}
	return 0, false
}

func coef(ms float64, rate int) float64 {
	if ms <= 0 || rate <= 0 {
		return 1
	}
	return 1 - math.Exp(-1/(ms*0.001*float64(rate)))
}

func (g *NoiseGate) recompute() {
	p := g.params
	g.threshold = math.Pow(10, p.ThresholdDB/20) * 32768
	g.floor = math.Pow(10, p.FloorDB/20)
	g.envAttack = coef(0.5, g.sampleRate)
	g.envRelease = coef(20, g.sampleRate)
	g.gainAttack = coef(p.AttackMs, g.sampleRate)
	g.gainRelease = coef(p.ReleaseMs, g.sampleRate)
	g.holdSamples = int(p.HoldMs * 0.001 * float64(g.sampleRate))
}

func (g *NoiseGate) Filter(dst, audio []byte) []byte {
	n := len(audio) / 2
	l := len(dst)
	if cap(dst)-l < 2*n {
		nd := make([]byte, l, l+2*n)
		copy(nd, dst)
		dst = nd
	}
	dst = dst[:l+2*n]
	out := dst[l:]
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.enabled || g.sampleRate == 0 {
		copy(out, audio[:2*n])
		return dst
	}
	env, gain, hold := g.env, g.gain, g.hold
	for i := 0; i < n; i++ {
		s := float64(int16(binary.LittleEndian.Uint16(audio[2*i:])))
		a := math.Abs(s)
		if a > env {
			env += g.envAttack * (a - env)
		} else {
			env += g.envRelease * (a - env)
		}
		target := g.floor
		if env >= g.threshold {
			target = 1
			hold = g.holdSamples
		} else if hold > 0 {
			hold--
			target = 1
		}
		if target > gain {
			gain += g.gainAttack * (target - gain)
		} else {
			gain += g.gainRelease * (target - gain)
		}
		v := math.Round(s * gain)
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(max(min(v, 32767), -32768))))
	}
	g.env, g.gain, g.hold = env, gain, hold
	return dst
}
