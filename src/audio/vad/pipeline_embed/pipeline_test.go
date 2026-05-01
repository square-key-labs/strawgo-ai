package pipeline_embed

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func findAncestorFile(rel string) string {
	cwd, _ := os.Getwd()
	for dir := cwd; dir != "/"; dir = filepath.Dir(dir) {
		c := filepath.Join(dir, rel)
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

func libPath() string {
	if v := os.Getenv("ORT_DYLIB_PATH"); v != "" {
		return v
	}
	if runtime.GOOS == "darwin" {
		if c := findAncestorFile("lib/libonnxruntime.dylib"); c != "" {
			return c
		}
	}
	if runtime.GOOS == "linux" {
		if c := findAncestorFile("lib/libonnxruntime.so"); c != "" {
			return c
		}
	}
	return ""
}

// TestMain initializes the shared pipeline once for all tests in this package.
// initOnce only fires the first call so tests must NOT call Shutdown.
func TestMain(m *testing.M) {
	vad := findAncestorFile("testdata/models/silero_vad.onnx")
	denoise := findAncestorFile("testdata/models/gtcrn_simple.onnx")
	smartTurn := findAncestorFile("testdata/models/smart-turn-v3.1-cpu.onnx")
	lib := libPath()
	if vad == "" || denoise == "" || smartTurn == "" || lib == "" {
		// Models or lib missing — skip all tests by exiting cleanly.
		os.Exit(0)
	}
	if err := Init(Config{
		VADModelPath:       vad,
		DenoiserModelPath:  denoise,
		SmartTurnModelPath: smartTurn,
		SharedLibraryPath:  lib,
	}); err != nil {
		_, _ = os.Stderr.WriteString("pipeline_embed test: Init failed: " + err.Error() + "\n")
		os.Exit(1)
	}
	code := m.Run()
	Shutdown()
	os.Exit(code)
}

// makeFrame16k returns 1024 bytes of int16-LE PCM containing a 440 Hz sine.
func makeFrame16k(amp float64, phase *float64) []byte {
	const N = 512
	buf := make([]byte, N*2)
	step := 2 * math.Pi * 440.0 / 16000.0
	for i := 0; i < N; i++ {
		v := int16(math.Sin(*phase) * amp * 32767)
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
		*phase += step
	}
	return buf
}

func makeSilenceFrame() []byte {
	return make([]byte, 512*2)
}

func TestPipelineSmoke(t *testing.T) {
	p, err := NewPipelineAnalyzer()
	if err != nil {
		t.Fatalf("NewPipelineAnalyzer: %v", err)
	}
	defer p.Cleanup()

	var phase float64
	speechProbs := []float32{}
	for i := 0; i < 30; i++ {
		frame := makeFrame16k(0.5, &phase)
		prob, _, _, err := p.ProcessFrame(frame)
		if err != nil {
			t.Fatalf("ProcessFrame[%d]: %v", i, err)
		}
		if prob < 0 || prob > 1 {
			t.Fatalf("VAD prob out of range: %f", prob)
		}
		speechProbs = append(speechProbs, prob)
	}
	t.Logf("speech VAD probs (sine 440 Hz @ amp 0.5): first=%.3f mid=%.3f last=%.3f",
		speechProbs[0], speechProbs[15], speechProbs[29])

	silenceRan := 0
	for i := 0; i < 30; i++ {
		_, _, ran, err := p.ProcessFrame(makeSilenceFrame())
		if err != nil {
			t.Fatalf("silence frame[%d]: %v", i, err)
		}
		if ran {
			silenceRan++
		}
	}
	t.Logf("smart-turn fired %d/30 times on silence after sine speech", silenceRan)
	// Whether it fires depends on Silero's sine response; don't fail the test
	// on that, but exercise the code path.
}

func TestPipelineSmartTurnDirect(t *testing.T) {
	st, err := NewSmartTurn()
	if err != nil {
		t.Fatalf("NewSmartTurn: %v", err)
	}
	defer st.Cleanup()

	// Append 2 seconds of speech-like sine.
	const N = 16000 * 2
	samples := make([]int16, N)
	for i := 0; i < N; i++ {
		samples[i] = int16(math.Sin(2*math.Pi*440*float64(i)/16000) * 0.3 * 32767)
	}
	st.AppendAudio(samples)

	prob, err := st.PredictEnd()
	if err != nil {
		t.Fatalf("PredictEnd: %v", err)
	}
	if prob < 0 || prob > 1 {
		t.Errorf("smart-turn prob out of range: %f", prob)
	}
	t.Logf("smart-turn prob on 2s sine: %.3f", prob)
}

func TestPipelineMultipleStreams(t *testing.T) {
	const N = 5
	pipes := make([]*PipelineAnalyzer, N)
	for i := 0; i < N; i++ {
		p, err := NewPipelineAnalyzer()
		if err != nil {
			t.Fatalf("NewPipelineAnalyzer[%d]: %v", i, err)
		}
		pipes[i] = p
		defer p.Cleanup()
	}
	var phase float64
	for f := 0; f < 10; f++ {
		frame := makeFrame16k(0.5, &phase)
		for i, p := range pipes {
			if _, _, _, err := p.ProcessFrame(frame); err != nil {
				t.Fatalf("stream %d frame %d: %v", i, f, err)
			}
		}
	}
}

func TestForceSmartTurnFire(t *testing.T) {
	// Force the high→low VAD edge by manually constructing pipe state.
	p, err := NewPipelineAnalyzer()
	if err != nil {
		t.Fatalf("NewPipelineAnalyzer: %v", err)
	}
	defer p.Cleanup()

	// Append 2s of speech via direct smart-turn path so PredictEnd has audio.
	const N = 16000 * 2
	samples := make([]int16, N)
	for i := 0; i < N; i++ {
		samples[i] = int16(math.Sin(2*math.Pi*440*float64(i)/16000) * 0.3 * 32767)
	}
	p.smartTurn.AppendAudio(samples)

	// Manually drive: first set prevVADProb high, then call ProcessFrame on
	// silence and observe smart-turn fires.
	p.prevVADProb = 0.9
	frame := makeSilenceFrame()
	_, _, ran, err := p.ProcessFrame(frame)
	if err != nil {
		t.Fatalf("ProcessFrame: %v", err)
	}
	if !ran {
		t.Errorf("expected smart-turn to fire on forced high→low edge")
	}
	if p.LastSmartTurnNS == 0 {
		t.Errorf("LastSmartTurnNS not recorded")
	}
	t.Logf("smart-turn ran=%v lastSmartTurnNS=%d (=%.2f ms)",
		ran, p.LastSmartTurnNS, float64(p.LastSmartTurnNS)/1e6)
}
