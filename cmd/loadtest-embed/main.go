// Package main is a load tester for the in-process (cgo) Silero VAD.
//
// It mirrors cmd/loadtest's protocol and stdout schema (so bench/compare_vad.py
// parses both equally), but bypasses the Unix-socket round-trip. Each "worker"
// owns its own SileroVAD instance — all of which share one ONNX Runtime
// session. Latency measured here is the end-to-end cost a real Strawgo agent
// would observe: ticker fire → ORT inference complete.
//
// Usage:
//
//	go run ./cmd/loadtest-embed [flags]
//	  -model    Path to silero_vad.onnx (default: testdata/models/silero_vad.onnx)
//	  -lib      Path to libonnxruntime.{so,dylib} (auto-detected if empty)
//	  -pcm      PCM fixture (int16 LE 16kHz). Auto-detected from testdata/ if empty.
//	  -dur      Seconds of measurement per concurrency level (default: 20)
//	  -levels   Comma-separated concurrency levels (default: 1,2,5,10,20,50)
//	  -n        Single concurrency level (overrides -levels if set; convenience flag)
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
	"github.com/square-key-labs/strawgo-ai/src/audio/vad/silero_embedded"
)

const (
	sampleRate16kHz = 16000
	samplesPerFrame = 512
	bytesPerSample  = 2
	frameBytes      = samplesPerFrame * bytesPerSample
	frameInterval   = 32 * time.Millisecond
)

func sortDurations(d []time.Duration) {
	sort.Slice(d, func(i, j int) bool { return d[i] < d[j] })
}

func pct(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	return sorted[int(float64(len(sorted)-1)*p/100)]
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

// processRSSKB returns the current process's resident set size in KB.
func processRSSKB() int64 {
	if runtime.GOOS == "linux" {
		data, err := os.ReadFile("/proc/self/status")
		if err != nil {
			return 0
		}
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "VmRSS:") {
				if f := strings.Fields(line); len(f) >= 2 {
					v, _ := strconv.ParseInt(f[1], 10, 64)
					return v
				}
			}
		}
		return 0
	}
	// Darwin: use ps.
	pid := os.Getpid()
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return 0
	}
	v, _ := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	return v
}

// pickLib auto-selects libonnxruntime path.
func pickLib() string {
	if v := os.Getenv("ORT_DYLIB_PATH"); v != "" {
		return v
	}
	candidates := []string{}
	switch runtime.GOOS {
	case "darwin":
		candidates = []string{
			"./lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
			"/opt/homebrew/lib/libonnxruntime.dylib",
		}
	case "linux":
		candidates = []string{
			"./lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
		}
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			abs, _ := filepath.Abs(p)
			return abs
		}
	}
	return ""
}

// findFile walks up to project root to find a relative file path.
func findFile(rel string) string {
	cwd, _ := os.Getwd()
	for dir := cwd; dir != "/"; dir = filepath.Dir(dir) {
		c := filepath.Join(dir, rel)
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// ── per-call worker ─────────────────────────────────────────────────────────

type worker struct {
	pcmFrames [][]byte
	startCh   <-chan struct{}
	stop      chan struct{}
	readyCh   chan struct{}
	wg        sync.WaitGroup

	mu        sync.Mutex
	latencies []time.Duration

	scheduled int64
	succeeded int64
	errors    int64

	vad *silero_embedded.SileroVAD
}

func newWorker(pcmFrames [][]byte, startCh <-chan struct{}) *worker {
	return &worker{
		pcmFrames: pcmFrames,
		startCh:   startCh,
		stop:      make(chan struct{}),
		readyCh:   make(chan struct{}),
	}
}

func (w *worker) run() {
	defer w.wg.Done()

	v, err := silero_embedded.New(sampleRate16kHz, vad.DefaultVADParams())
	if err != nil {
		atomic.AddInt64(&w.errors, 1)
		close(w.readyCh)
		return
	}
	w.vad = v
	defer v.Cleanup()

	close(w.readyCh)

	select {
	case <-w.stop:
		return
	case <-w.startCh:
	}

	ticker := time.NewTicker(frameInterval)
	defer ticker.Stop()

	frameIdx := 0
	for {
		select {
		case <-w.stop:
			return

		case tickTime := <-ticker.C:
			atomic.AddInt64(&w.scheduled, 1)

			frame := w.pcmFrames[frameIdx%len(w.pcmFrames)]
			frameIdx++

			_ = v.VoiceConfidence(frame)
			lat := time.Since(tickTime)

			atomic.AddInt64(&w.succeeded, 1)
			w.mu.Lock()
			w.latencies = append(w.latencies, lat)
			w.mu.Unlock()
		}
	}
}

func (w *worker) stopAndCollect() (scheduled, succeeded, errors int64, lats []time.Duration) {
	close(w.stop)
	w.wg.Wait()
	return atomic.LoadInt64(&w.scheduled),
		atomic.LoadInt64(&w.succeeded),
		atomic.LoadInt64(&w.errors),
		w.latencies
}

// ── RSS sampler ──────────────────────────────────────────────────────────────

type rssSampler struct {
	interval time.Duration
	stop     chan struct{}
	wg       sync.WaitGroup
	mu       sync.Mutex
	readings []float64
}

func newRSSSampler(interval time.Duration) *rssSampler {
	return &rssSampler{interval: interval, stop: make(chan struct{})}
}

func (s *rssSampler) start() {
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		t := time.NewTicker(s.interval)
		defer t.Stop()
		for {
			select {
			case <-s.stop:
				return
			case <-t.C:
				if kb := processRSSKB(); kb > 0 {
					s.mu.Lock()
					s.readings = append(s.readings, float64(kb)/1024.0)
					s.mu.Unlock()
				}
			}
		}
	}()
}

func (s *rssSampler) stopAndPeak() float64 {
	close(s.stop)
	s.wg.Wait()
	var peak float64
	for _, r := range s.readings {
		if r > peak {
			peak = r
		}
	}
	return peak
}

// ── main ─────────────────────────────────────────────────────────────────────

func main() {
	model := flag.String("model", "", "Path to silero_vad.onnx (auto-detected from testdata/ if empty)")
	lib := flag.String("lib", "", "Path to libonnxruntime.{so,dylib}; auto-detected if empty")
	pcmPath := flag.String("pcm", "", "PCM fixture (int16 LE 16kHz). Auto-detected from testdata/ if empty.")
	durSecs := flag.Int("dur", 20, "Seconds of measurement per concurrency level")
	levelsStr := flag.String("levels", "1,2,5,10,20,50", "Comma-separated concurrency levels")
	nFlag := flag.Int("n", 0, "Single concurrency level; overrides -levels if > 0")
	flag.Parse()

	if *model == "" {
		*model = findFile(filepath.Join("testdata", "models", "silero_vad.onnx"))
	}
	if *model == "" {
		fmt.Fprintln(os.Stderr, "cannot find silero_vad.onnx — pass -model explicitly")
		os.Exit(1)
	}
	if *lib == "" {
		*lib = pickLib()
	}
	if *pcmPath == "" {
		*pcmPath = findFile(filepath.Join("testdata", "sine_440_500ms_16k.pcm"))
	}
	if *pcmPath == "" {
		fmt.Fprintln(os.Stderr, "cannot find testdata/sine_440_500ms_16k.pcm — pass -pcm explicitly")
		os.Exit(1)
	}

	raw, err := os.ReadFile(*pcmPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read PCM: %v\n", err)
		os.Exit(1)
	}
	var pcmFrames [][]byte
	for off := 0; off+frameBytes <= len(raw); off += frameBytes {
		f := make([]byte, frameBytes)
		copy(f, raw[off:])
		pcmFrames = append(pcmFrames, f)
	}
	if len(pcmFrames) == 0 {
		fmt.Fprintln(os.Stderr, "PCM file too short (need >=1024 bytes)")
		os.Exit(1)
	}

	if err := silero_embedded.Init(silero_embedded.Config{
		ModelPath:         *model,
		SharedLibraryPath: *lib,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "Init: %v\n", err)
		os.Exit(1)
	}
	defer silero_embedded.Shutdown()

	fmt.Printf("Model: %s\nLib  : %s\nPCM  : %s  (%d frames × 32ms = %.1fs looped)\n",
		*model, *lib, *pcmPath, len(pcmFrames), float64(len(pcmFrames))*0.032)

	var levels []int
	if *nFlag > 0 {
		levels = []int{*nFlag}
	} else {
		for _, s := range strings.Split(*levelsStr, ",") {
			if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil && n > 0 {
				levels = append(levels, n)
			}
		}
		sort.Ints(levels)
	}

	dur := time.Duration(*durSecs) * time.Second

	// Header matches cmd/loadtest exactly so bench/compare_vad.py parses cleanly.
	fmt.Printf("\n%-6s  %-9s %-9s %-9s  %-8s %-8s %-8s  %-13s %-13s\n",
		"N", "p50", "p95", "p99",
		"sched", "ok", "err",
		"rss_base_MB", "rss_peak_MB")
	fmt.Println(strings.Repeat("─", 90))

	for _, n := range levels {
		startCh := make(chan struct{})
		workers := make([]*worker, n)
		for i := range workers {
			workers[i] = newWorker(pcmFrames, startCh)
			workers[i].wg.Add(1)
			go workers[i].run()
		}
		for _, w := range workers {
			<-w.readyCh
		}

		baselineRSS := float64(processRSSKB()) / 1024.0
		rss := newRSSSampler(300 * time.Millisecond)
		rss.start()

		close(startCh)
		time.Sleep(dur)

		var (
			totalSched, totalOK, totalErr int64
			allLats                       []time.Duration
		)
		for _, w := range workers {
			s, ok, e, lats := w.stopAndCollect()
			totalSched += s
			totalOK += ok
			totalErr += e
			allLats = append(allLats, lats...)
		}
		peakRSS := rss.stopAndPeak()
		sortDurations(allLats)

		fmt.Printf("%-6d  %-9s %-9s %-9s  %-8d %-8d %-8d  %-13.1f %-13.1f\n",
			n,
			fmtDur(pct(allLats, 50)),
			fmtDur(pct(allLats, 95)),
			fmtDur(pct(allLats, 99)),
			totalSched, totalOK, totalErr,
			baselineRSS, peakRSS,
		)
	}

	fmt.Println(strings.Repeat("─", 90))
	fmt.Println("\nDone.")
}
