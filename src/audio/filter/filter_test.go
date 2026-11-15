package filter

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func sine(n, rate int, freq, amp float64) []byte {
	b := make([]byte, 2*n)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(int16(math.Round(amp*math.Sin(2*math.Pi*freq*float64(i)/float64(rate))))))
	}
	return b
}

func peak(b []byte) float64 {
	var m float64
	for i := 0; i+1 < len(b); i += 2 {
		m = math.Max(m, math.Abs(float64(int16(binary.LittleEndian.Uint16(b[i:])))))
	}
	return m
}

func runGate(g *NoiseGate, in []byte, chunk int) []byte {
	var out []byte
	for i := 0; i < len(in); i += chunk {
		out = g.Filter(out, in[i:min(i+chunk, len(in))])
	}
	return out
}

func TestNoiseGateAttenuatesQuietNoise(t *testing.T) {
	g := NewNoiseGate(DefaultNoiseGateParams())
	if err := g.Start(16000); err != nil {
		t.Fatal(err)
	}
	out := runGate(g, sine(16000, 16000, 300, 60), 320)
	if p := peak(out[len(out)-3200:]); p > 1 {
		t.Fatalf("noise below threshold not gated, peak %.0f", p)
	}
}

func TestNoiseGatePassesSpeechLevelSignal(t *testing.T) {
	g := NewNoiseGate(DefaultNoiseGateParams())
	g.Start(16000)
	in := sine(8000, 16000, 300, 10000)
	out := runGate(g, in, 320)
	tail := out[len(out)-3200:]
	ref := in[len(in)-3200:]
	for i := 0; i < len(tail); i += 2 {
		a := int16(binary.LittleEndian.Uint16(tail[i:]))
		b := int16(binary.LittleEndian.Uint16(ref[i:]))
		if d := int(a) - int(b); d > 2 || d < -2 {
			t.Fatalf("loud signal altered at %d: %d vs %d", i/2, a, b)
		}
	}
}

func TestNoiseGateHoldsThenReleases(t *testing.T) {
	g := NewNoiseGate(NoiseGateParams{ThresholdDB: -30, AttackMs: 1, HoldMs: 100, ReleaseMs: 10, FloorDB: -80})
	g.Start(16000)
	runGate(g, sine(3200, 16000, 300, 10000), 320)
	quiet := sine(16000, 16000, 300, 300)
	out := runGate(g, quiet, 320)
	if p := peak(out[2*800 : 2*1200]); p < 250 {
		t.Fatalf("gate closed during hold, peak %.0f", p)
	}
	if p := peak(out[2*6400:]); p > 1 {
		t.Fatalf("gate did not release after hold, peak %.0f", p)
	}
}

func TestNoiseGateEnableDisableAndPassthroughBeforeStart(t *testing.T) {
	quiet := sine(8000, 16000, 300, 60)
	unstarted := NewNoiseGate(DefaultNoiseGateParams())
	if p := peak(runGate(unstarted, quiet, 320)); p < 59 {
		t.Fatal("gate filtered before Start")
	}
	g := NewNoiseGate(DefaultNoiseGateParams())
	g.Start(16000)
	if err := g.ProcessFrame(frames.NewFilterEnableFrame(false)); err != nil {
		t.Fatal(err)
	}
	if p := peak(runGate(g, quiet, 320)); p < 59 {
		t.Fatal("disabled gate still filtering")
	}
	g.ProcessFrame(frames.NewFilterEnableFrame(true))
	out := runGate(g, quiet, 320)
	if p := peak(out[len(out)-3200:]); p > 1 {
		t.Fatal("re-enabled gate not filtering")
	}
}

func TestNoiseGateSettingsFrame(t *testing.T) {
	g := NewNoiseGate(DefaultNoiseGateParams())
	g.Start(16000)
	err := g.ProcessFrame(frames.NewFilterUpdateSettingsFrame(map[string]any{"threshold_db": -70, "attack_ms": float32(1), "hold_ms": int64(10), "release_ms": 5.0, "floor_db": -60}))
	if err != nil {
		t.Fatal(err)
	}
	p := g.Params()
	if p.ThresholdDB != -70 || p.AttackMs != 1 || p.HoldMs != 10 || p.ReleaseMs != 5 || p.FloorDB != -60 {
		t.Fatalf("params %+v", p)
	}
	out := runGate(g, sine(8000, 16000, 300, 60), 320)
	if pk := peak(out[len(out)-3200:]); pk < 55 {
		t.Fatalf("lowered threshold should pass the signal, peak %.0f", pk)
	}
	if err := g.ProcessFrame(frames.NewFilterUpdateSettingsFrame(map[string]any{"bogus": 1})); err == nil {
		t.Fatal("unknown setting accepted")
	}
	if err := g.ProcessFrame(frames.NewFilterUpdateSettingsFrame(map[string]any{"floor_db": "x"})); err == nil {
		t.Fatal("non-number accepted")
	}
	g.SetParams(DefaultNoiseGateParams())
	if g.Params() != DefaultNoiseGateParams() {
		t.Fatal("SetParams not applied")
	}
	if err := g.Start(0); err == nil {
		t.Fatal("zero rate accepted")
	}
}

func TestNoiseGateAppendsAndDoesNotAllocate(t *testing.T) {
	g := NewNoiseGate(DefaultNoiseGateParams())
	g.Start(16000)
	in := sine(320, 16000, 300, 1000)
	out := g.Filter([]byte{1, 2}, in)
	if len(out) != 2+len(in) || out[0] != 1 {
		t.Fatal("Filter did not append")
	}
	dst := make([]byte, 0, len(in))
	if n := testing.AllocsPerRun(50, func() { dst = g.Filter(dst[:0], in) }); n != 0 {
		t.Fatalf("allocs %.1f", n)
	}
}

func TestFuncFilterEnableToggle(t *testing.T) {
	f := NewFuncFilter(func(dst, audio []byte) []byte {
		for _, b := range audio {
			dst = append(dst, b+1)
		}
		return dst
	})
	if got := f.Filter(nil, []byte{1, 2}); got[0] != 2 || got[1] != 3 {
		t.Fatalf("got %v", got)
	}
	f.ProcessFrame(frames.NewFilterEnableFrame(false))
	if got := f.Filter(nil, []byte{1, 2}); got[0] != 1 {
		t.Fatalf("disabled filter applied: %v", got)
	}
	if got := NewFuncFilter(nil).Filter([]byte{9}, []byte{1}); len(got) != 2 || got[1] != 1 {
		t.Fatalf("nil func got %v", got)
	}
}

type recFilter struct {
	add      byte
	started  int
	stopErr  error
	startErr error
	frames   int
}

func (r *recFilter) Start(int) error { r.started++; return r.startErr }
func (r *recFilter) Stop() error     { return r.stopErr }
func (r *recFilter) ProcessFrame(frames.Frame) error {
	r.frames++
	return nil
}
func (r *recFilter) Filter(dst, audio []byte) []byte {
	for _, b := range audio {
		dst = append(dst, b*2+r.add)
	}
	return dst
}

func TestChainAppliesInOrder(t *testing.T) {
	a, b, c := &recFilter{add: 1}, &recFilter{add: 0}, &recFilter{add: 3}
	ch := NewChain(a, b, c)
	if err := ch.Start(16000); err != nil {
		t.Fatal(err)
	}
	got := ch.Filter([]byte{0xAA}, []byte{1, 2})
	if len(got) != 3 || got[0] != 0xAA || got[1] != ((1*2+1)*2)*2+3 || got[2] != ((2*2+1)*2)*2+3 {
		t.Fatalf("got %v", got)
	}
	again := ch.Filter(nil, []byte{1, 2})
	if again[0] != got[1] || again[1] != got[2] {
		t.Fatal("chain not stable across calls")
	}
	ch.ProcessFrame(frames.NewFilterEnableFrame(true))
	if a.started != 1 || c.started != 1 || a.frames != 1 || c.frames != 1 {
		t.Fatal("chain did not fan out Start/ProcessFrame")
	}
	if got := NewChain().Filter(nil, []byte{5}); got[0] != 5 {
		t.Fatal("empty chain not passthrough")
	}
}

func TestChainErrors(t *testing.T) {
	boom := errors.New("boom")
	if err := NewChain(&recFilter{}, &recFilter{startErr: boom}).Start(8000); !errors.Is(err, boom) {
		t.Fatalf("start err %v", err)
	}
	other := errors.New("other")
	if err := NewChain(&recFilter{stopErr: boom}, &recFilter{stopErr: other}).Stop(); !errors.Is(err, boom) {
		t.Fatalf("stop err %v", err)
	}
}
