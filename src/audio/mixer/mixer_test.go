package mixer

import (
	"encoding/binary"
	"path/filepath"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/audio/wav"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func constWav(t *testing.T, value int16, n, rate int) string {
	t.Helper()
	b := make([]byte, 2*n)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(value))
	}
	path := filepath.Join(t.TempDir(), "s.wav")
	if err := wav.WriteFile(path, b, rate, 1); err != nil {
		t.Fatal(err)
	}
	return path
}

func rampWav(t *testing.T, n, rate int) string {
	t.Helper()
	b := make([]byte, 2*n)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(int16(i+1)))
	}
	path := filepath.Join(t.TempDir(), "r.wav")
	if err := wav.WriteFile(path, b, rate, 1); err != nil {
		t.Fatal(err)
	}
	return path
}

func vals(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return out
}

func silence(n int) []byte { return make([]byte, 2*n) }

func TestMixAddsSoundAtVolumeDuringSilence(t *testing.T) {
	m := NewSoundfileMixer(map[string]string{"bg": constWav(t, 1000, 100, 16000)}, "bg", WithVolume(0.5))
	if err := m.Start(16000); err != nil {
		t.Fatal(err)
	}
	for _, v := range vals(m.Mix(nil, silence(40))) {
		if v != 500 {
			t.Fatalf("mixed sample %d want 500", v)
		}
	}
}

func TestMixAddsOnTopOfSpeechAndClips(t *testing.T) {
	m := NewSoundfileMixer(map[string]string{"bg": constWav(t, 20000, 100, 8000)}, "bg", WithVolume(1))
	if err := m.Start(8000); err != nil {
		t.Fatal(err)
	}
	in := make([]byte, 4)
	binary.LittleEndian.PutUint16(in, uint16(int16(100)))
	binary.LittleEndian.PutUint16(in[2:], uint16(int16(20000)))
	got := vals(m.Mix([]byte{0xEE}, in)[1:])
	if got[0] != 20100 || got[1] != 32767 {
		t.Fatalf("got %v", got)
	}
}

func TestLoopWrapsAround(t *testing.T) {
	m := NewSoundfileMixer(map[string]string{"r": rampWav(t, 5, 8000)}, "r", WithVolume(1))
	if err := m.Start(8000); err != nil {
		t.Fatal(err)
	}
	got := vals(m.Mix(nil, silence(12)))
	want := []int16{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
}

func TestNoLoopStopsAtEnd(t *testing.T) {
	m := NewSoundfileMixer(map[string]string{"r": rampWav(t, 6, 8000)}, "r", WithVolume(1), WithLoop(false))
	if err := m.Start(8000); err != nil {
		t.Fatal(err)
	}
	first := vals(m.Mix(nil, silence(4)))
	if first[3] != 4 {
		t.Fatalf("first chunk %v", first)
	}
	for _, v := range vals(m.Mix(nil, silence(4))) {
		if v != 0 {
			t.Fatal("sound mixed past its end with loop disabled")
		}
	}
}

func TestEnableDisableAndSettingsFrames(t *testing.T) {
	m := NewSoundfileMixer(map[string]string{"a": constWav(t, 1000, 50, 8000), "b": constWav(t, 3000, 50, 8000)}, "a", WithVolume(1))
	if err := m.Start(8000); err != nil {
		t.Fatal(err)
	}
	if err := m.ProcessFrame(frames.NewMixerEnableFrame(false)); err != nil {
		t.Fatal(err)
	}
	if v := vals(m.Mix(nil, silence(2))); v[0] != 0 {
		t.Fatal("disabled mixer still mixing")
	}
	m.ProcessFrame(frames.NewMixerEnableFrame(true))
	if v := vals(m.Mix(nil, silence(2))); v[0] != 1000 {
		t.Fatalf("re-enabled mixer gave %d", v[0])
	}
	if err := m.ProcessFrame(frames.NewMixerUpdateSettingsFrame(map[string]any{"sound": "b", "volume": 0.5})); err != nil {
		t.Fatal(err)
	}
	if v := vals(m.Mix(nil, silence(2))); v[0] != 1500 {
		t.Fatalf("after settings update got %d", v[0])
	}
	m.ProcessFrame(frames.NewMixerUpdateSettingsFrame(map[string]any{"volume": 0}))
	if v := vals(m.Mix(nil, silence(2))); v[0] != 0 {
		t.Fatal("volume 0 still audible")
	}
	for _, bad := range []map[string]any{{"sound": "nope"}, {"sound": 1}, {"volume": "x"}, {"loop": "yes"}} {
		if err := m.ProcessFrame(frames.NewMixerUpdateSettingsFrame(bad)); err == nil {
			t.Errorf("%v accepted", bad)
		}
	}
	if err := m.ProcessFrame(frames.NewMixerUpdateSettingsFrame(map[string]any{"loop": false})); err != nil {
		t.Fatal(err)
	}
}

func TestWithMixingFalseStartsSilent(t *testing.T) {
	m := NewSoundfileMixer(map[string]string{"a": constWav(t, 1000, 50, 8000)}, "a", WithMixing(false))
	m.Start(8000)
	if v := vals(m.Mix(nil, silence(3))); v[0] != 0 {
		t.Fatal("mixing despite WithMixing(false)")
	}
}

func TestStartResamplesSoundToOutputRate(t *testing.T) {
	m := NewSoundfileMixer(map[string]string{"a": constWav(t, 8000, 800, 8000)}, "a", WithVolume(1), WithLoop(false))
	if err := m.Start(16000); err != nil {
		t.Fatal(err)
	}
	if v := vals(m.Mix(nil, silence(1590))); v[1000] < 7800 || v[1000] > 8200 {
		t.Fatalf("resampled sound value %d", v[1000])
	}
	if v := vals(m.Mix(nil, silence(20))); v[0] != 0 {
		t.Fatal("resampled sound longer than expected")
	}
}

func TestStartErrors(t *testing.T) {
	if err := NewSoundfileMixer(nil, "").Start(0); err == nil {
		t.Fatal("zero rate accepted")
	}
	missing := filepath.Join(t.TempDir(), "missing.wav")
	if err := NewSoundfileMixer(map[string]string{"x": missing}, "x").Start(8000); err == nil {
		t.Fatal("all-missing sounds accepted")
	}
	m := NewSoundfileMixer(map[string]string{"x": missing, "ok": constWav(t, 10, 10, 8000)}, "x")
	if err := m.Start(8000); err != nil {
		t.Fatalf("partial load should succeed: %v", err)
	}
	if v := vals(m.Mix(nil, silence(2))); v[0] != 0 {
		t.Fatal("missing current sound produced audio")
	}
}

func TestMixDoesNotAllocateWithCapacity(t *testing.T) {
	m := NewSoundfileMixer(map[string]string{"a": constWav(t, 100, 1000, 16000)}, "a")
	m.Start(16000)
	in := silence(320)
	dst := make([]byte, 0, 640)
	if n := testing.AllocsPerRun(50, func() { dst = m.Mix(dst[:0], in) }); n != 0 {
		t.Fatalf("allocs %.1f", n)
	}
}
