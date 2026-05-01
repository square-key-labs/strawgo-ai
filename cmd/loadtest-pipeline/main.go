// Package main is a load tester for the in-process 3-model voice pipeline:
// GTCRN denoise → Silero VAD → smart-turn. Mirrors cmd/loadtest-embed flags
// and stdout schema so bench/compare_vad.py can parse it.
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

	"github.com/square-key-labs/strawgo-ai/src/audio/vad/pipeline_embed"
)

const (
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
	pid := os.Getpid()
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return 0
	}
	v, _ := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	return v
}

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

// findHomeFile expands ~/X for paths like ~/smart-turn-v3.1-cpu.onnx.
func findHomeFile(name string) string {
	if home, err := os.UserHomeDir(); err == nil {
		c := filepath.Join(home, name)
		if _, err := os.Stat(c); err == nil {
			return c
		}
		c = filepath.Join(home, ".cache/strawgo/models", name)
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

	denoiseNS    []int64
	vadNS        []int64
	smartNS      []int64
	snrNS        []int64
	smartCalls   int64
	denoiseCalls int64

	scheduled int64
	succeeded int64
	errors    int64

	snrThresholdDB float64

	pipeline *pipeline_embed.PipelineAnalyzer
}

func newWorker(pcmFrames [][]byte, startCh <-chan struct{}, snrThresholdDB float64) *worker {
	return &worker{
		pcmFrames:      pcmFrames,
		startCh:        startCh,
		stop:           make(chan struct{}),
		readyCh:        make(chan struct{}),
		snrThresholdDB: snrThresholdDB,
	}
}

func (w *worker) run() {
	defer w.wg.Done()

	p, err := pipeline_embed.NewPipelineAnalyzer()
	if err != nil {
		atomic.AddInt64(&w.errors, 1)
		fmt.Fprintf(os.Stderr, "worker: NewPipelineAnalyzer: %v\n", err)
		close(w.readyCh)
		return
	}
	if w.snrThresholdDB > 0 {
		p.EnableSNRGating(w.snrThresholdDB, pipeline_embed.SNRConfig{})
	}
	w.pipeline = p
	defer p.Cleanup()

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

			_, _, smartRan, err := p.ProcessFrame(frame)
			lat := time.Since(tickTime)

			if err != nil {
				atomic.AddInt64(&w.errors, 1)
				continue
			}
			atomic.AddInt64(&w.succeeded, 1)
			w.mu.Lock()
			w.latencies = append(w.latencies, lat)
			if p.LastDenoiseRan {
				w.denoiseNS = append(w.denoiseNS, p.LastDenoiseNS)
				w.denoiseCalls++
			}
			w.vadNS = append(w.vadNS, p.LastVADNS)
			if p.LastSNRNS > 0 {
				w.snrNS = append(w.snrNS, p.LastSNRNS)
			}
			if smartRan {
				w.smartNS = append(w.smartNS, p.LastSmartTurnNS)
				w.smartCalls++
			}
			w.mu.Unlock()
		}
	}
}

func (w *worker) stopAndCollect() (scheduled, succeeded, errors int64, lats []time.Duration,
	denoise, vad, smart, snr []int64, smartCalls, denoiseCalls int64) {
	close(w.stop)
	w.wg.Wait()
	return atomic.LoadInt64(&w.scheduled),
		atomic.LoadInt64(&w.succeeded),
		atomic.LoadInt64(&w.errors),
		w.latencies, w.denoiseNS, w.vadNS, w.smartNS, w.snrNS, w.smartCalls, w.denoiseCalls
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

func medianNS(xs []int64) int64 {
	if len(xs) == 0 {
		return 0
	}
	cp := make([]int64, len(xs))
	copy(cp, xs)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return cp[len(cp)/2]
}

// ── main ─────────────────────────────────────────────────────────────────────

func main() {
	vadModel := flag.String("vad-model", "", "Path to silero_vad.onnx (auto-detected from testdata/ if empty)")
	denoiseModel := flag.String("dfn-model", "", "Path to denoiser ONNX (gtcrn_simple.onnx). Auto-detected if empty.")
	smartTurnModel := flag.String("smart-turn-model", "", "Path to smart-turn ONNX. Auto-detected if empty.")
	lib := flag.String("lib", "", "Path to libonnxruntime.{so,dylib}; auto-detected if empty")
	pcmPath := flag.String("pcm", "", "PCM fixture (int16 LE 16kHz). Auto-detected from testdata/ if empty.")
	durSecs := flag.Int("dur", 20, "Seconds of measurement per concurrency level")
	levelsStr := flag.String("levels", "1,2,5,10,20,50", "Comma-separated concurrency levels")
	nFlag := flag.Int("n", 0, "Single concurrency level; overrides -levels if > 0")
	snrThresholdDB := flag.Float64("snr-threshold-db", 0, "Gate denoise on SNR < threshold (dB). 0 = always denoise (legacy). Typical: 12-15.")
	denoiserKind := flag.String("denoiser-kind", "gtcrn", "Per-stream denoiser: 'gtcrn' (default, 16 kHz streaming) or 'nsnet2' (Microsoft baseline, ~2× faster).")
	flag.Parse()

	if *vadModel == "" {
		*vadModel = findFile(filepath.Join("testdata", "models", "silero_vad.onnx"))
	}
	if *vadModel == "" {
		fmt.Fprintln(os.Stderr, "cannot find silero_vad.onnx — pass -vad-model explicitly")
		os.Exit(1)
	}
	if *denoiseModel == "" {
		// Auto-detect based on -denoiser-kind so swapping is one-flag.
		switch *denoiserKind {
		case "nsnet2":
			*denoiseModel = findFile(filepath.Join("testdata", "models", "nsnet2-20ms.onnx"))
		default:
			*denoiseModel = findFile(filepath.Join("testdata", "models", "gtcrn_simple.onnx"))
		}
	}
	if *denoiseModel == "" {
		fmt.Fprintf(os.Stderr, "cannot find denoiser model for kind=%s — pass -dfn-model explicitly\n", *denoiserKind)
		os.Exit(1)
	}
	if *smartTurnModel == "" {
		*smartTurnModel = findFile(filepath.Join("testdata", "models", "smart-turn-v3.1-cpu.onnx"))
	}
	if *smartTurnModel == "" {
		*smartTurnModel = findHomeFile("smart-turn-v3.1-cpu.onnx")
	}
	if *smartTurnModel == "" {
		fmt.Fprintln(os.Stderr, "cannot find smart-turn-v3.1-cpu.onnx — pass -smart-turn-model explicitly")
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

	if err := pipeline_embed.Init(pipeline_embed.Config{
		VADModelPath:       *vadModel,
		DenoiserModelPath:  *denoiseModel,
		SmartTurnModelPath: *smartTurnModel,
		DenoiserKind:       *denoiserKind,
		SharedLibraryPath:  *lib,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "Init: %v\n", err)
		os.Exit(1)
	}
	defer pipeline_embed.Shutdown()

	fmt.Printf("VAD       : %s\nDenoiser  : %s\nSmartTurn : %s\nLib       : %s\nPCM       : %s  (%d frames × 32ms = %.1fs looped)\n",
		*vadModel, *denoiseModel, *smartTurnModel, *lib, *pcmPath,
		len(pcmFrames), float64(len(pcmFrames))*0.032)

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

	fmt.Printf("\n%-6s  %-9s %-9s %-9s  %-8s %-8s %-8s  %-13s %-13s\n",
		"N", "p50", "p95", "p99",
		"sched", "ok", "err",
		"rss_base_MB", "rss_peak_MB")
	fmt.Println(strings.Repeat("─", 90))

	for _, n := range levels {
		startCh := make(chan struct{})
		workers := make([]*worker, n)
		for i := range workers {
			workers[i] = newWorker(pcmFrames, startCh, *snrThresholdDB)
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
			totalSched, totalOK, totalErr     int64
			allLats                           []time.Duration
			allDen, allVad, allSmart, allSNR  []int64
			totalSmartCalls, totalDenoiseCalls int64
		)
		for _, w := range workers {
			s, ok, e, lats, den, vd, sm, sn, sc, dc := w.stopAndCollect()
			totalSched += s
			totalOK += ok
			totalErr += e
			allLats = append(allLats, lats...)
			allDen = append(allDen, den...)
			allVad = append(allVad, vd...)
			allSmart = append(allSmart, sm...)
			allSNR = append(allSNR, sn...)
			totalSmartCalls += sc
			totalDenoiseCalls += dc
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
		denoisePct := 0.0
		if totalOK > 0 {
			denoisePct = 100.0 * float64(totalDenoiseCalls) / float64(totalOK)
		}
		fmt.Printf("        per-frame medians: denoise=%dµs vad=%dµs smartturn=%dµs snr=%dµs (denoise %d/%d=%.1f%%, st %d)\n",
			medianNS(allDen)/1000, medianNS(allVad)/1000, medianNS(allSmart)/1000, medianNS(allSNR)/1000,
			totalDenoiseCalls, totalOK, denoisePct, totalSmartCalls)
	}

	fmt.Println(strings.Repeat("─", 90))
	fmt.Println("\nDone.")
}
