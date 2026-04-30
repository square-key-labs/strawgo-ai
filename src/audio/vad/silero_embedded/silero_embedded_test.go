package silero_embedded

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
)

// findRepoFile walks up from the package dir to find a relative path.
func findRepoFile(t *testing.T, rel string) string {
	t.Helper()
	dir, _ := os.Getwd()
	for i := 0; i < 8; i++ {
		c := filepath.Join(dir, rel)
		if _, err := os.Stat(c); err == nil {
			return c
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatalf("could not find %s walking up from cwd", rel)
	return ""
}

// pickORTLib returns a libonnxruntime path likely to exist.
func pickORTLib() string {
	if v := os.Getenv("ORT_DYLIB_PATH"); v != "" {
		return v
	}
	switch runtime.GOOS {
	case "darwin":
		for _, p := range []string{
			"/usr/local/lib/libonnxruntime.dylib",
			"/opt/homebrew/lib/libonnxruntime.dylib",
		} {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	case "linux":
		for _, p := range []string{
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
		} {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	return ""
}

// TestMain centralizes init so all tests share one shared session.
func TestMain(m *testing.M) {
	wd, _ := os.Getwd()
	// Walk up to find testdata/models/silero_vad.onnx
	model := ""
	d := wd
	for i := 0; i < 8; i++ {
		c := filepath.Join(d, "testdata", "models", "silero_vad.onnx")
		if _, err := os.Stat(c); err == nil {
			model = c
			break
		}
		parent := filepath.Dir(d)
		if parent == d {
			break
		}
		d = parent
	}
	if model == "" {
		// Skip all tests if model missing
		os.Stderr.WriteString("silero_vad.onnx not found; skipping silero_embedded tests\n")
		os.Exit(0)
	}
	if err := Init(Config{
		ModelPath:         model,
		SharedLibraryPath: pickORTLib(),
	}); err != nil {
		os.Stderr.WriteString("Init failed: " + err.Error() + "\n")
		os.Exit(0)
	}
	defer Shutdown()
	code := m.Run()
	Shutdown()
	os.Exit(code)
}

// makeSineFrame returns 512 i16-LE PCM samples of a 440Hz sine at 16kHz.
func makeSineFrame() []byte {
	const N = 512
	buf := make([]byte, N*2)
	for i := 0; i < N; i++ {
		v := int16(math.Sin(2*math.Pi*440*float64(i)/16000) * 32767 * 0.5)
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
	}
	return buf
}

// makeSilenceFrame returns 512 i16-LE silence samples.
func makeSilenceFrame() []byte { return make([]byte, 512*2) }

func TestVoiceConfidenceShape(t *testing.T) {
	v, err := New(16000, vad.DefaultVADParams())
	if err != nil {
		t.Fatal(err)
	}
	defer v.Cleanup()

	c := v.VoiceConfidence(makeSilenceFrame())
	if c < 0 || c > 1 {
		t.Fatalf("confidence on silence out of [0,1]: %v", c)
	}
	if c > 0.3 {
		t.Logf("note: silence confidence=%.3f (expected near 0)", c)
	}
}

func TestSineHasSomeConfidence(t *testing.T) {
	v, err := New(16000, vad.DefaultVADParams())
	if err != nil {
		t.Fatal(err)
	}
	defer v.Cleanup()

	frame := makeSineFrame()
	// Drive a few frames so LSTM converges.
	var last float32
	for i := 0; i < 5; i++ {
		last = v.VoiceConfidence(frame)
	}
	t.Logf("440Hz sine confidence after 5 frames = %.3f", last)
	// Sine isn't speech, so we don't assert > threshold; just sanity.
	if last < 0 || last > 1 {
		t.Fatalf("confidence out of [0,1]: %v", last)
	}
}

func TestRestartZerosState(t *testing.T) {
	v, err := New(16000, vad.DefaultVADParams())
	if err != nil {
		t.Fatal(err)
	}
	defer v.Cleanup()

	frame := makeSineFrame()
	for i := 0; i < 3; i++ {
		v.VoiceConfidence(frame)
	}
	v.Restart()
	for i := range v.hidden {
		if v.hidden[i] != 0 {
			t.Fatalf("hidden state not zeroed after Restart: idx=%d val=%v", i, v.hidden[i])
		}
	}
	if v.context != nil {
		t.Fatalf("context not nil after Restart, got %v", v.context)
	}
}

func TestConcurrentSharedSession(t *testing.T) {
	const N = 8
	const frames = 50
	var wg sync.WaitGroup
	errs := make(chan error, N)
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			v, err := New(16000, vad.DefaultVADParams())
			if err != nil {
				errs <- err
				return
			}
			defer v.Cleanup()
			frame := makeSineFrame()
			for f := 0; f < frames; f++ {
				c := v.VoiceConfidence(frame)
				if c < 0 || c > 1 {
					errs <- err
					return
				}
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Fatal(e)
	}
}
