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

const (
	msgTypeVAD      byte = 0x01
	sampleRate16kHz      = 16000
	samplesPerFrame      = 512                              // Silero expects exactly 512 samples at 16kHz
	bytesPerSample       = 2                                // int16
	frameBytes           = samplesPerFrame * bytesPerSample // 1024 bytes
	frameInterval        = 32 * time.Millisecond            // 512/16000 = 32 ms
	ortWarmupDelay       = 1200 * time.Millisecond          // conservative ORT init budget
	rpcDeadline          = 5 * time.Second
)

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

type worker struct {
	sockPath  string
	pcmFrames [][]byte

	startCh <-chan struct{}

	stop chan struct{}

	readyCh chan struct{}

	wg sync.WaitGroup

	mu        sync.Mutex
	latencies []time.Duration

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
	defer func() { conn.Close() }()

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

func (w *worker) stopAndCollect() (scheduled, succeeded, errors int64, lats []time.Duration) {
	close(w.stop)
	w.wg.Wait()
	return atomic.LoadInt64(&w.scheduled),
		atomic.LoadInt64(&w.succeeded),
		atomic.LoadInt64(&w.errors),
		w.latencies
}

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

func main() {
	sockPath := flag.String("socket", "/tmp/onnx-worker-demo.sock", "onnx-worker Unix socket path")
	pcmPath := flag.String("pcm", "", "PCM fixture (int16 LE 16kHz). Auto-detected from testdata/ if empty.")
	durSecs := flag.Int("dur", 20, "Seconds of measurement per concurrency level (excludes warmup)")
	levelsStr := flag.String("levels", "1,2,5,10,20,50", "Comma-separated concurrency levels")
	workerPID := flag.Int("pid", 0, "onnx-worker PID for RSS. Pass explicitly for reliable numbers.")
	flag.Parse()

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

		for _, w := range workers {
			<-w.readyCh
		}

		baselineRSS := float64(processRSSKB(pid)) / 1024.0

		rss := newRSSSampler(pid, 300*time.Millisecond)
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
