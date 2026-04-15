// Package main is a load tester for the onnx-worker Unix socket.
//
// It simulates N concurrent voice calls, each sending Silero VAD frames at
// real-time pace (31.25 Hz for 16kHz audio), and measures:
//   - Round-trip latency from ticker fire → response received (p50/p95/p99)
//   - onnx-worker process RSS delta (peak − baseline after all sessions ready)
//   - Sustained throughput: scheduled / succeeded / dropped frame counts
//
// Measurement design:
//   - All N workers connect and warm up (1 s ORT init) before measurement starts.
//   - A shared start barrier releases all workers simultaneously → no bias from
//     staggered startup bleeding into the measurement window.
//   - RSS baseline is captured after all sessions are loaded but before any
//     frames are sent, so delta = per-call session cost only.
//   - stopAndCollect() waits for run() to exit (WaitGroup) before reading any
//     counters → no race between stop signal and final writes.
//
// Usage:
//
//	go run ./cmd/loadtest [flags]
//	  -socket   Unix socket path (default: /tmp/onnx-worker-demo.sock)
//	  -pcm      PCM fixture (int16 LE 16kHz). Auto-detected from testdata/ if empty.
//	  -dur      Seconds of measurement per concurrency level (default: 20)
//	  -levels   Comma-separated concurrency levels (default: 1,2,5,10,20,50)
//	  -pid      onnx-worker PID for RSS sampling. Required for accurate RSS;
//	            auto-detected via pgrep as a fallback (unreliable if multiple
//	            onnx-worker processes exist).
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"net"
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
)

// ── protocol constants ──────────────────────────────────────────────────────

const (
	msgTypeVAD      byte = 0x01
	sampleRate16kHz      = 16000
	samplesPerFrame      = 512 // Silero expects exactly 512 samples at 16kHz
	bytesPerSample       = 2   // int16
	frameBytes           = samplesPerFrame * bytesPerSample // 1024 bytes
	frameInterval        = 32 * time.Millisecond            // 512/16000 = 32 ms
	ortWarmupDelay       = 1200 * time.Millisecond          // conservative ORT init budget
	rpcDeadline          = 5 * time.Second
)

// ── helpers ─────────────────────────────────────────────────────────────────

// buildRequest encodes one VAD frame.
// Wire format (matches onnx_client.go):
//
//	[u8 msg_type=0x01][u32 payload_len LE][u32 sample_rate LE][i16 PCM bytes...]
func buildRequest(pcmFrame []byte) []byte {
	payloadLen := 4 + len(pcmFrame)
	buf := make([]byte, 1+4+4+len(pcmFrame))
	buf[0] = msgTypeVAD
	binary.LittleEndian.PutUint32(buf[1:5], uint32(payloadLen))
	binary.LittleEndian.PutUint32(buf[5:9], sampleRate16kHz)
	copy(buf[9:], pcmFrame)
	return buf
}

func writeFull(conn net.Conn, buf []byte) error {
	for len(buf) > 0 {
		n, err := conn.Write(buf)
		if err != nil {
			return err
		}
		buf = buf[n:]
	}
	return nil
}

func readFull(conn net.Conn, buf []byte) error {
	for total := 0; total < len(buf); {
		n, err := conn.Read(buf[total:])
		total += n
		if err != nil {
			return err
		}
	}
	return nil
}

func readF32(conn net.Conn) (float32, error) {
	var b [4]byte
	if err := readFull(conn, b[:]); err != nil {
		return 0, err
	}
	return math.Float32frombits(binary.LittleEndian.Uint32(b[:])), nil
}

func processRSSKB(pid int) int64 {
	if pid <= 0 {
		return 0
	}
	switch runtime.GOOS {
	case "darwin":
		out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(pid)).Output()
		if err != nil {
			return 0
		}
		v, _ := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
		return v
	default:
		data, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
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
}

func sortDurations(d []time.Duration) {
	for i := 1; i < len(d); i++ {
		k, j := d[i], i-1
		for j >= 0 && d[j] > k {
			d[j+1] = d[j]
			j--
		}
		d[j+1] = k
	}
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

// ── per-call worker ─────────────────────────────────────────────────────────

type worker struct {
	sockPath  string
	pcmFrames [][]byte

	// startCh is shared across all workers in a level; closed by main to
	// release all workers simultaneously after all sessions are ready.
	startCh <-chan struct{}

	// stop is closed by stopAndCollect to signal the run goroutine to exit.
	stop chan struct{}

	// readyCh is closed by run() once the ORT session is initialised and the
	// worker is waiting on startCh. Main waits on all readyChs before releasing.
	readyCh chan struct{}

	// wg tracks run()'s lifetime; stopAndCollect() calls wg.Wait() to ensure
	// run() has fully exited before reading any counters or latencies.
	wg sync.WaitGroup

	// mu protects latencies; only written by run(), read by stopAndCollect()
	// after wg.Wait() (no live contention at read time, but Lock is kept for
	// correctness during the measurement window).
	mu        sync.Mutex
	latencies []time.Duration

	// atomics — written by run(), read by stopAndCollect() after wg.Wait().
	scheduled int64 // ticker fires (attempted frames)
	succeeded int64 // frames with successful response
	errors    int64 // write/read failures + dial failures
}

func newWorker(sockPath string, pcmFrames [][]byte, startCh <-chan struct{}) *worker {
	return &worker{
		sockPath:  sockPath,
		pcmFrames: pcmFrames,
		startCh:   startCh,
		stop:      make(chan struct{}),
		readyCh:   make(chan struct{}),
	}
}

// connectAndWarm dials the socket and waits for ORT session initialisation.
// Used for both the initial connection and reconnects (reconnect also needs to
// wait — the server creates a new SileroSession on every accepted connection).
func (w *worker) connectAndWarm() (net.Conn, error) {
	conn, err := net.Dial("unix", w.sockPath)
	if err != nil {
		return nil, err
	}
	time.Sleep(ortWarmupDelay)
	return conn, nil
}

func (w *worker) run() {
	defer w.wg.Done()

	conn, err := w.connectAndWarm()
	if err != nil {
		atomic.AddInt64(&w.errors, 1)
		close(w.readyCh) // still signal ready so main isn't blocked
		return
	}
	// Closure captures the conn variable itself (not its value at defer time),
	// so the current connection — including any reconnected one — is closed on exit.
	defer func() { conn.Close() }()

	// Signal to main that this session is initialised and ready.
	close(w.readyCh)

	// Wait for all other workers to also be ready before starting measurement.
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
			// Count every ticker fire as a scheduled attempt regardless of outcome.
			atomic.AddInt64(&w.scheduled, 1)

			frame := w.pcmFrames[frameIdx%len(w.pcmFrames)]
			frameIdx++
			req := buildRequest(frame)

			conn.SetDeadline(time.Now().Add(rpcDeadline))

			if err := writeFull(conn, req); err != nil {
				atomic.AddInt64(&w.errors, 1)
				conn.Close()
				conn, err = w.connectAndWarm()
				if err != nil {
					atomic.AddInt64(&w.errors, 1)
					return
				}
				continue
			}

			_, err := readF32(conn)
			// Latency is measured from ticker fire → response received.
			// This captures: ticker wakeup jitter + IPC write + ORT inference +
			// IPC read. It is the end-to-end cost a real caller would observe.
			lat := time.Since(tickTime)
			if err != nil {
				atomic.AddInt64(&w.errors, 1)
				conn.Close()
				conn, err = w.connectAndWarm()
				if err != nil {
					atomic.AddInt64(&w.errors, 1)
					return
				}
				continue
			}

			atomic.AddInt64(&w.succeeded, 1)
			w.mu.Lock()
			w.latencies = append(w.latencies, lat)
			w.mu.Unlock()
		}
	}
}

// stopAndCollect signals run() to stop, waits for it to fully exit, then
// returns counters and latency slice. No data races: all writes have ceased
// by the time wg.Wait() returns.
func (w *worker) stopAndCollect() (scheduled, succeeded, errors int64, lats []time.Duration) {
	close(w.stop)
	w.wg.Wait()
	// Safe to read without lock — run() has exited.
	return atomic.LoadInt64(&w.scheduled),
		atomic.LoadInt64(&w.succeeded),
		atomic.LoadInt64(&w.errors),
		w.latencies
}

// ── RSS sampler ──────────────────────────────────────────────────────────────

type rssSampler struct {
	pid      int
	interval time.Duration
	stop     chan struct{}
	wg       sync.WaitGroup
	mu       sync.Mutex
	readings []float64
}

func newRSSSampler(pid int, interval time.Duration) *rssSampler {
	return &rssSampler{pid: pid, interval: interval, stop: make(chan struct{})}
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
				if kb := processRSSKB(s.pid); kb > 0 {
					s.mu.Lock()
					s.readings = append(s.readings, float64(kb)/1024.0)
					s.mu.Unlock()
				}
			}
		}
	}()
}

// stopAndPeak stops the sampler, waits for it to exit, then returns peak RSS.
func (s *rssSampler) stopAndPeak() float64 {
	close(s.stop)
	s.wg.Wait() // wait for goroutine exit before reading slice
	var peak float64
	for _, r := range s.readings { // no lock needed — goroutine has exited
		if r > peak {
			peak = r
		}
	}
	return peak
}

// ── main ─────────────────────────────────────────────────────────────────────

func main() {
	sockPath  := flag.String("socket", "/tmp/onnx-worker-demo.sock", "onnx-worker Unix socket path")
	pcmPath   := flag.String("pcm", "", "PCM fixture (int16 LE 16kHz). Auto-detected from testdata/ if empty.")
	durSecs   := flag.Int("dur", 20, "Seconds of measurement per concurrency level (excludes warmup)")
	levelsStr := flag.String("levels", "1,2,5,10,20,50", "Comma-separated concurrency levels")
	workerPID := flag.Int("pid", 0, "onnx-worker PID for RSS. Pass explicitly for reliable numbers.")
	flag.Parse()

	// ── resolve PCM path ──
	if *pcmPath == "" {
		cwd, _ := os.Getwd()
		for dir := cwd; dir != "/"; dir = filepath.Dir(dir) {
			c := filepath.Join(dir, "testdata", "sine_440_500ms_16k.pcm")
			if _, err := os.Stat(c); err == nil {
				*pcmPath = c
				break
			}
		}
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
		frame := make([]byte, frameBytes)
		copy(frame, raw[off:])
		pcmFrames = append(pcmFrames, frame)
	}
	if len(pcmFrames) == 0 {
		fmt.Fprintln(os.Stderr, "PCM file too short (need ≥1024 bytes)")
		os.Exit(1)
	}
	fmt.Printf("PCM: %s  (%d frames × 32ms = %.1fs looped)\n",
		*pcmPath, len(pcmFrames), float64(len(pcmFrames))*0.032)

	// ── PID resolution ──
	pid := *workerPID
	if pid == 0 {
		out, _ := exec.Command("pgrep", "-f", "onnx-worker").Output()
		lines := strings.Fields(strings.TrimSpace(string(out)))
		if len(lines) == 1 {
			pid, _ = strconv.Atoi(lines[0])
		} else if len(lines) > 1 {
			fmt.Fprintf(os.Stderr,
				"WARNING: pgrep found %d onnx-worker processes — pass -pid for accurate RSS\n",
				len(lines))
		}
	}
	if pid > 0 {
		fmt.Printf("onnx-worker PID: %d  idle RSS: %.1f MB\n", pid, float64(processRSSKB(pid))/1024)
	} else {
		fmt.Println("onnx-worker PID: not found — RSS columns will be 0")
	}

	// ── verify socket reachable ──
	if probe, err := net.Dial("unix", *sockPath); err != nil {
		fmt.Fprintf(os.Stderr, "cannot connect to %s: %v\n", *sockPath, err)
		os.Exit(1)
	} else {
		probe.Close()
	}
	fmt.Printf("Socket: %s — reachable ✓\n\n", *sockPath)

	var levels []int
	for _, s := range strings.Split(*levelsStr, ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil && n > 0 {
			levels = append(levels, n)
		}
	}
	sort.Ints(levels)

	dur := time.Duration(*durSecs) * time.Second

	fmt.Printf("%-6s  %-9s %-9s %-9s  %-8s %-8s %-8s  %-13s %-13s\n",
		"N", "p50", "p95", "p99",
		"sched", "ok", "err",
		"rss_base_MB", "rss_peak_MB")
	fmt.Println(strings.Repeat("─", 90))

	for _, n := range levels {
		startCh := make(chan struct{})

		workers := make([]*worker, n)
		for i := range workers {
			workers[i] = newWorker(*sockPath, pcmFrames, startCh)
			workers[i].wg.Add(1)
			go workers[i].run()
		}

		// Wait for every worker to finish ORT init before starting measurement.
		// This eliminates measurement-window bias from staggered warmup.
		for _, w := range workers {
			<-w.readyCh
		}

		// Capture baseline RSS after all sessions are loaded, before any frames.
		baselineRSS := float64(processRSSKB(pid)) / 1024.0

		// Start RSS sampler.
		rss := newRSSSampler(pid, 300*time.Millisecond)
		rss.start()

		// Release all workers simultaneously.
		close(startCh)

		time.Sleep(dur)

		// Collect results — stopAndCollect() blocks until run() exits cleanly.
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
