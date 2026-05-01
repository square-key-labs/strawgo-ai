// bench-denoise-cost — microbenchmark for raw ORT inference cost across
// candidate denoiser ONNX files. Loads each model with a shared session,
// builds synthetic input tensors of the model's expected shape, and times
// .Run() across N concurrent goroutines for a fixed wall window. Reports
// per-call median + p99 + total throughput.
//
// This is intentionally narrow: it does NOT do STFT/iSTFT, does NOT drive
// real audio, does NOT assess output quality. It answers exactly one
// question: "how expensive is the ONNX graph itself, called as a streaming
// load, on this CPU?"
//
// Models supported (auto-detected by filename):
//
//   - gtcrn_simple.onnx          — 4-in / 4-out, 16 kHz, single STFT frame
//   - nsnet2-20ms-baseline.onnx  — 1-in / 1-out, 16 kHz, 1 frame at a time
//   - rnnoise.onnx               — 1-in / 2-out, 48 kHz, 1 s window per call
//
// Add more by extending modelSpecs below.
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

type modelSpec struct {
	name        string
	tag         string // matched against filename
	inputs      []string
	outputs     []string
	shapes      [][]int64
	dtype       string
	frameMs     float64 // logical frame duration this call covers
	description string
}

var specs = []modelSpec{
	{
		name:    "gtcrn",
		tag:     "gtcrn",
		inputs:  []string{"mix", "conv_cache", "tra_cache", "inter_cache"},
		outputs: []string{"enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"},
		shapes: [][]int64{
			{1, 257, 1, 2},        // mix (single STFT frame, re/im)
			{2, 1, 16, 16, 33},    // conv_cache
			{2, 3, 1, 1, 16},      // tra_cache
			{2, 1, 33, 16},        // inter_cache
		},
		dtype:       "f32",
		frameMs:     16.0, // one STFT frame at 16 kHz / 256 hop
		description: "GTCRN streaming, 16 kHz, 4 caches",
	},
	{
		name:    "nsnet2",
		tag:     "nsnet2",
		inputs:  []string{"input"},
		outputs: []string{"output"},
		shapes: [][]int64{
			{1, 1, 161}, // [batch, frames, freq_bins], one 20 ms frame
		},
		dtype:       "f32",
		frameMs:     20.0, // 20 ms STFT hop at 16 kHz
		description: "NSNet2 baseline, 16 kHz, 20 ms, no state cache",
	},
	{
		name:    "rnnoise",
		tag:     "rnnoise",
		inputs:  []string{"main_input:0"},
		outputs: []string{"denoise_output/Sigmoid:0", "vad_output/Sigmoid:0"},
		shapes: [][]int64{
			{1, 100, 42}, // [batch, frames=100×10ms=1s window, features=42]
		},
		dtype:       "f32",
		frameMs:     1000.0, // each call is one 1-second window
		description: "RNNoise (ailia export), 48 kHz, 1 s windows, no state",
	},
	// DFN3 split into 3 sub-graphs. Each gets benched separately; sum
	// p50s gives the per-frame chain cost. Streaming S=1.
	{
		name:    "dfn3_enc",
		tag:     "dfn3_enc",
		inputs:  []string{"feat_erb", "feat_spec"},
		outputs: []string{"e0", "e1", "e2", "e3", "emb", "c0", "lsnr"},
		shapes: [][]int64{
			{1, 1, 1, 32},  // feat_erb
			{1, 2, 1, 96},  // feat_spec
		},
		dtype:       "f32",
		frameMs:     10.0,
		description: "DFN3 encoder, 48 kHz, 10 ms frames, S=1",
	},
	{
		name:    "dfn3_erb_dec",
		tag:     "dfn3_erb_dec",
		inputs:  []string{"emb", "e3", "e2", "e1", "e0"},
		outputs: []string{"m"},
		shapes: [][]int64{
			{1, 1, 512},     // emb
			{1, 64, 1, 8},   // e3
			{1, 64, 1, 8},   // e2
			{1, 64, 1, 16},  // e1
			{1, 64, 1, 32},  // e0
		},
		dtype:       "f32",
		frameMs:     10.0,
		description: "DFN3 ERB decoder (gain mask), 48 kHz, 10 ms, S=1",
	},
	{
		name:    "dfn3_df_dec",
		tag:     "dfn3_df_dec",
		inputs:  []string{"emb", "c0"},
		outputs: []string{"coefs"},
		shapes: [][]int64{
			{1, 1, 512},    // emb
			{1, 64, 1, 96}, // c0
		},
		dtype:       "f32",
		frameMs:     10.0,
		description: "DFN3 DF decoder (filter coeffs), 48 kHz, 10 ms, S=1",
	},
}

func detectSpec(modelPath string) (modelSpec, bool) {
	base := strings.ToLower(filepath.Base(modelPath))
	for _, s := range specs {
		if strings.Contains(base, s.tag) {
			return s, true
		}
	}
	return modelSpec{}, false
}

func makeRandFloats(shape []int64) []float32 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = rand.Float32()*2 - 1
	}
	return out
}

type runner struct {
	mu      sync.Mutex
	lats    []time.Duration
	calls   int64
	errs    int64
	session *ort.DynamicAdvancedSession
	spec    modelSpec
}

func (r *runner) callOnce() error {
	// Build input tensors fresh per call. ORT accepts re-use but for a fair
	// "real workload" cost we allocate as the production path would.
	ins := make([]ort.Value, len(r.spec.inputs))
	for i, sh := range r.spec.shapes {
		data := makeRandFloats(sh)
		t, err := ort.NewTensor(ort.NewShape(sh...), data)
		if err != nil {
			return err
		}
		ins[i] = t
	}
	defer func() {
		for _, v := range ins {
			if v != nil {
				_ = v.Destroy()
			}
		}
	}()

	outs := make([]ort.Value, len(r.spec.outputs))
	defer func() {
		for _, v := range outs {
			if v != nil {
				_ = v.Destroy()
			}
		}
	}()

	t0 := time.Now()
	if err := r.session.Run(ins, outs); err != nil {
		atomic.AddInt64(&r.errs, 1)
		return err
	}
	lat := time.Since(t0)

	r.mu.Lock()
	r.lats = append(r.lats, lat)
	r.calls++
	r.mu.Unlock()
	return nil
}

func (r *runner) loop(stop <-chan struct{}) {
	for {
		select {
		case <-stop:
			return
		default:
			_ = r.callOnce()
		}
	}
}

func pct(s []time.Duration, p float64) time.Duration {
	if len(s) == 0 {
		return 0
	}
	idx := int(float64(len(s)-1) * p / 100)
	return s[idx]
}

func main() {
	model := flag.String("model", "", "ONNX model path")
	lib := flag.String("lib", "", "libonnxruntime.{so,dylib}")
	dur := flag.Int("dur", 5, "seconds")
	n := flag.Int("n", 1, "concurrent goroutines")
	flag.Parse()

	if *model == "" || *lib == "" {
		fmt.Fprintln(os.Stderr, "usage: bench-denoise-cost -model X.onnx -lib libonnxruntime.so [-n N] [-dur S]")
		os.Exit(2)
	}

	spec, ok := detectSpec(*model)
	if !ok {
		fmt.Fprintf(os.Stderr, "unknown model (filename must contain %v)\n", []string{"gtcrn", "nsnet2", "rnnoise"})
		os.Exit(2)
	}

	ort.SetSharedLibraryPath(*lib)
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Fprintf(os.Stderr, "init: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = ort.DestroyEnvironment() }()

	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Fprintf(os.Stderr, "session opts: %v\n", err)
		os.Exit(1)
	}
	defer opts.Destroy()
	_ = opts.SetIntraOpNumThreads(1)
	_ = opts.SetInterOpNumThreads(1)

	sess, err := ort.NewDynamicAdvancedSession(*model, spec.inputs, spec.outputs, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "session new: %v\n", err)
		os.Exit(1)
	}
	defer sess.Destroy()

	runners := make([]*runner, *n)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < *n; i++ {
		runners[i] = &runner{session: sess, spec: spec}
		wg.Add(1)
		go func(r *runner) {
			defer wg.Done()
			r.loop(stop)
		}(runners[i])
	}

	t0 := time.Now()
	time.Sleep(time.Duration(*dur) * time.Second)
	close(stop)
	wg.Wait()
	wall := time.Since(t0)

	// Aggregate.
	var allLats []time.Duration
	var totCalls, totErr int64
	for _, r := range runners {
		allLats = append(allLats, r.lats...)
		totCalls += r.calls
		totErr += r.errs
	}
	sort.Slice(allLats, func(i, j int) bool { return allLats[i] < allLats[j] })

	fmt.Printf("model      : %s (%s)\n", spec.name, spec.description)
	fmt.Printf("CPU        : %s/%d cores\n", runtime.GOOS+"/"+runtime.GOARCH, runtime.NumCPU())
	fmt.Printf("workload   : N=%d concurrent, %.1fs wall\n", *n, wall.Seconds())
	fmt.Printf("calls      : total=%d errs=%d (%.0f calls/s)\n",
		totCalls, totErr, float64(totCalls)/wall.Seconds())
	fmt.Printf("per-call   : p50=%s p95=%s p99=%s\n",
		fmtDur(pct(allLats, 50)), fmtDur(pct(allLats, 95)), fmtDur(pct(allLats, 99)))
	if spec.frameMs > 0 {
		// frames-equivalent throughput: each call covers `frameMs` of audio
		audioMs := float64(totCalls) * spec.frameMs
		realtimeFactor := audioMs / 1000.0 / wall.Seconds()
		perAgentRT := realtimeFactor / float64(*n)
		fmt.Printf("audio cov. : %.1fs of audio processed, %.1f× realtime aggregate, %.2f× per-agent\n",
			audioMs/1000.0, realtimeFactor, perAgentRT)
	}
}

func fmtDur(d time.Duration) string {
	switch {
	case d < time.Microsecond:
		return fmt.Sprintf("%dns", d.Nanoseconds())
	case d < time.Millisecond:
		return fmt.Sprintf("%dµs", d.Microseconds())
	default:
		return fmt.Sprintf("%.2fms", float64(d.Microseconds())/1000)
	}
}
