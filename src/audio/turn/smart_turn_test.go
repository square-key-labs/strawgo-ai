package turn

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func speechPCM(samples int, v int16) []byte {
	out := make([]byte, 0, 2*samples)
	for i := 0; i < samples; i++ {
		out = binary.LittleEndian.AppendUint16(out, uint16(v))
	}
	return out
}

type smartTurnServer struct {
	*httptest.Server
	calls   atomic.Int32
	status  atomic.Int32
	block   atomic.Bool
	release chan struct{}
	lastReq atomic.Pointer[capturedRequest]
}

type capturedRequest struct {
	header http.Header
	body   []byte
}

func newSmartTurnServer(t *testing.T) *smartTurnServer {
	t.Helper()
	s := &smartTurnServer{release: make(chan struct{})}
	s.status.Store(http.StatusOK)
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.calls.Add(1)
		body, _ := io.ReadAll(r.Body)
		s.lastReq.Store(&capturedRequest{header: r.Header.Clone(), body: body})
		if s.block.Load() {
			select {
			case <-s.release:
			case <-r.Context().Done():
			}
			return
		}
		if code := int(s.status.Load()); code != http.StatusOK {
			http.Error(w, "worker down", code)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"prediction":1,"probability":0.87,"metrics":{"inference_time":0.012,"total_time":0.02}}`)
	}))
	t.Cleanup(func() {
		close(s.release)
		s.Close()
	})
	return s
}

func TestHTTPSmartTurnPostsRawNPYWithHeaderAuth(t *testing.T) {
	srv := newSmartTurnServer(t)
	a := NewHTTPSmartTurn(HTTPSmartTurnConfig{URL: srv.URL, APIKey: "secret", Headers: map[string]string{"X-Tenant": "acme"}})
	a.AppendAudio(speechPCM(1600, 16384), true)

	state, metrics, err := a.AnalyzeEndOfTurnContext(context.Background())
	if err != nil || state != TurnComplete {
		t.Fatalf("state=%v err=%v, want COMPLETE", state, err)
	}
	if metrics == nil || !metrics.IsComplete || metrics.Probability != 0.87 || math.Abs(metrics.InferenceTimeMs-12) > 1e-6 {
		t.Fatalf("metrics = %+v", metrics)
	}
	req := srv.lastReq.Load()
	if ct := req.header.Get("Content-Type"); ct != "application/octet-stream" {
		t.Fatalf("Content-Type = %q, want raw bytes", ct)
	}
	if req.header.Get("Authorization") != "Key secret" || req.header.Get("X-Tenant") != "acme" {
		t.Fatalf("auth headers = %v", req.header)
	}
	body := req.body
	if string(body[:6]) != "\x93NUMPY" || body[6] != 1 || body[7] != 0 {
		t.Fatalf("body is not an NPY v1 array: % x", body[:8])
	}
	headerLen := int(binary.LittleEndian.Uint16(body[8:10]))
	if (10+headerLen)%64 != 0 {
		t.Fatalf("NPY header not 64-byte aligned: %d", 10+headerLen)
	}
	dict := string(body[10 : 10+headerLen])
	if !strings.Contains(dict, "'descr': '<f4'") || !strings.Contains(dict, "'shape': (1600,)") || !strings.HasSuffix(dict, "\n") {
		t.Fatalf("NPY header = %q", dict)
	}
	data := body[10+headerLen:]
	if len(data) != 4*1600 {
		t.Fatalf("NPY payload = %d bytes, want %d", len(data), 4*1600)
	}
	if v := math.Float32frombits(binary.LittleEndian.Uint32(data)); v != 0.5 {
		t.Fatalf("first sample = %v, want 0.5", v)
	}
	if a.SpeechTriggered() {
		t.Fatal("turn state not cleared after COMPLETE")
	}
}

func TestFalSmartTurnUsesRawEndpointAndKeyHeader(t *testing.T) {
	if !strings.HasSuffix(DefaultFalSmartTurnURL, "/fal-ai/smart-turn/raw") {
		t.Fatalf("fal URL = %q", DefaultFalSmartTurnURL)
	}
	srv := newSmartTurnServer(t)
	a := NewFalSmartTurn(FalSmartTurnConfig{URL: srv.URL + "/?no_key_here", APIKey: "fal-key"})
	a.AppendAudio(speechPCM(800, 100), true)
	if state, _, err := a.AnalyzeEndOfTurn(); err != nil || state != TurnComplete {
		t.Fatalf("state=%v err=%v", state, err)
	}
	req := srv.lastReq.Load()
	if req.header.Get("Authorization") != "Key fal-key" || req.header.Get("Content-Type") != "application/octet-stream" {
		t.Fatalf("fal request headers = %v", req.header)
	}
}

func TestHTTPSmartTurnLegacyJSONFormat(t *testing.T) {
	srv := newSmartTurnServer(t)
	a := NewHTTPSmartTurn(HTTPSmartTurnConfig{URL: srv.URL, Format: SmartTurnFormatJSON})
	a.AppendAudio(speechPCM(800, 100), true)
	if state, _, err := a.AnalyzeEndOfTurn(); err != nil || state != TurnComplete {
		t.Fatalf("state=%v err=%v", state, err)
	}
	req := srv.lastReq.Load()
	if req.header.Get("Content-Type") != "application/json" || !strings.Contains(string(req.body), `"audio_base64"`) {
		t.Fatalf("JSON request = %v %.60s", req.header, req.body)
	}
}

func TestHTTPSmartTurnCancelledByCaller(t *testing.T) {
	srv := newSmartTurnServer(t)
	srv.block.Store(true)
	a := NewHTTPSmartTurn(HTTPSmartTurnConfig{URL: srv.URL, Breaker: BreakerConfig{FailureThreshold: 1}})
	a.AppendAudio(speechPCM(800, 100), true)

	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(50*time.Millisecond, cancel)
	start := time.Now()
	state, _, err := a.AnalyzeEndOfTurnContext(ctx)
	if !errors.Is(err, context.Canceled) || state != TurnIncomplete {
		t.Fatalf("state=%v err=%v, want INCOMPLETE and context.Canceled", state, err)
	}
	if d := time.Since(start); d > time.Second {
		t.Fatalf("cancellation took %v", d)
	}
	if a.BreakerOpen() {
		t.Fatal("a caller cancellation tripped the circuit breaker")
	}
	if !a.SpeechTriggered() {
		t.Fatal("a cancelled inference ended the turn")
	}
}

func TestHTTPSmartTurnTimeoutBoundedByStopSecsIsComplete(t *testing.T) {
	srv := newSmartTurnServer(t)
	srv.block.Store(true)
	a := NewHTTPSmartTurn(HTTPSmartTurnConfig{URL: srv.URL, Params: &SmartTurnParams{StopSecs: 0.15, MaxDurationSecs: 8}})
	a.AppendAudio(speechPCM(800, 100), true)

	start := time.Now()
	state, metrics, err := a.AnalyzeEndOfTurn()
	d := time.Since(start)
	if err != nil || state != TurnComplete || metrics != nil {
		t.Fatalf("state=%v metrics=%v err=%v, want COMPLETE on timeout", state, metrics, err)
	}
	if d < 100*time.Millisecond || d > time.Second {
		t.Fatalf("returned after %v, want about stop_secs (150ms)", d)
	}
	if a.SpeechTriggered() {
		t.Fatal("turn state not cleared after the timeout")
	}
}

func TestSmartTurnCircuitBreakerBacksOffAndRecovers(t *testing.T) {
	srv := newSmartTurnServer(t)
	srv.status.Store(http.StatusInternalServerError)
	a := NewHTTPSmartTurn(HTTPSmartTurnConfig{URL: srv.URL, Breaker: BreakerConfig{FailureThreshold: 2, Cooldown: 150 * time.Millisecond, MaxCooldown: 300 * time.Millisecond}})
	analyze := func() (EndOfTurnState, error) {
		a.AppendAudio(speechPCM(800, 100), true)
		state, _, err := a.AnalyzeEndOfTurn()
		return state, err
	}

	for i := 0; i < 2; i++ {
		if state, err := analyze(); err == nil || state != TurnIncomplete {
			t.Fatalf("failure %d: state=%v err=%v", i, state, err)
		}
	}
	if !a.BreakerOpen() || srv.calls.Load() != 2 {
		t.Fatalf("breaker open=%v after %d calls", a.BreakerOpen(), srv.calls.Load())
	}
	for i := 0; i < 5; i++ {
		if state, err := analyze(); err != nil || state != TurnComplete {
			t.Fatalf("open breaker: state=%v err=%v, want silence-based COMPLETE", state, err)
		}
	}
	if n := srv.calls.Load(); n != 2 {
		t.Fatalf("dead worker called %d times while the breaker was open", n)
	}

	time.Sleep(170 * time.Millisecond)
	if state, err := analyze(); err == nil || state != TurnIncomplete {
		t.Fatalf("failed probe: state=%v err=%v", state, err)
	}
	if n := srv.calls.Load(); n != 3 {
		t.Fatalf("calls after probe = %d, want 3", n)
	}
	time.Sleep(170 * time.Millisecond)
	if state, err := analyze(); err != nil || state != TurnComplete || srv.calls.Load() != 3 {
		t.Fatalf("cooldown did not double after a failed probe: state=%v err=%v calls=%d", state, err, srv.calls.Load())
	}

	srv.status.Store(http.StatusOK)
	time.Sleep(170 * time.Millisecond)
	a.AppendAudio(speechPCM(800, 100), true)
	state, metrics, err := a.AnalyzeEndOfTurn()
	if err != nil || state != TurnComplete || metrics == nil || srv.calls.Load() != 4 {
		t.Fatalf("recovery probe: state=%v metrics=%v err=%v calls=%d", state, metrics, err, srv.calls.Load())
	}
	if a.BreakerOpen() {
		t.Fatal("breaker still open after a successful probe")
	}
}

func hangingTurnWorker(t *testing.T) string {
	t.Helper()
	f, err := os.CreateTemp("", "hang-turn-*.sock")
	if err != nil {
		t.Fatal(err)
	}
	path := f.Name()
	f.Close()
	os.Remove(path)
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	t.Cleanup(func() {
		close(done)
		ln.Close()
		os.Remove(path)
	})
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				go io.Copy(io.Discard, conn)
				<-done
			}()
		}
	}()
	return path
}

func TestOnnxSmartTurnCancelledByCaller(t *testing.T) {
	st, err := NewOnnxSmartTurn(OnnxSmartTurnConfig{SockPath: hangingTurnWorker(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	st.AppendAudio(speechPCM(1600, 100), true)

	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(50*time.Millisecond, cancel)
	start := time.Now()
	state, _, err := st.AnalyzeEndOfTurnContext(ctx)
	if !errors.Is(err, context.Canceled) || state != TurnIncomplete {
		t.Fatalf("state=%v err=%v, want INCOMPLETE and context.Canceled", state, err)
	}
	if d := time.Since(start); d > time.Second {
		t.Fatalf("cancellation took %v", d)
	}
}

func TestOnnxSmartTurnHungWorkerBoundedByStopSecs(t *testing.T) {
	st, err := NewOnnxSmartTurn(OnnxSmartTurnConfig{SockPath: hangingTurnWorker(t), Params: &SmartTurnParams{StopSecs: 0.2, MaxDurationSecs: 8}})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	st.AppendAudio(speechPCM(1600, 100), true)

	start := time.Now()
	state, _, err := st.AnalyzeEndOfTurn()
	d := time.Since(start)
	if err != nil || state != TurnComplete {
		t.Fatalf("state=%v err=%v, want COMPLETE on timeout", state, err)
	}
	if d < 150*time.Millisecond || d > 1500*time.Millisecond {
		t.Fatalf("returned after %v, want about stop_secs (200ms)", d)
	}
}

func TestOnnxSmartTurnDeadWorkerTripsBreaker(t *testing.T) {
	f, err := os.CreateTemp("", "dead-turn-*.sock")
	if err != nil {
		t.Fatal(err)
	}
	path := f.Name()
	f.Close()
	os.Remove(path)
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	st, err := NewOnnxSmartTurn(OnnxSmartTurnConfig{SockPath: path, Breaker: BreakerConfig{FailureThreshold: 1, Cooldown: time.Minute}})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ln.Close()
	os.Remove(path)
	if conn := st.client.current(); conn != nil {
		st.client.drop(conn)
	}

	st.AppendAudio(speechPCM(1600, 100), true)
	start := time.Now()
	if state, _, err := st.AnalyzeEndOfTurn(); err == nil || state != TurnIncomplete {
		t.Fatalf("dead worker: state=%v err=%v", state, err)
	}
	if d := time.Since(start); d > 2*time.Second {
		t.Fatalf("dead worker blocked for %v", d)
	}
	start = time.Now()
	if state, _, err := st.AnalyzeEndOfTurn(); err != nil || state != TurnComplete {
		t.Fatalf("open breaker: state=%v err=%v", state, err)
	}
	if d := time.Since(start); d > 20*time.Millisecond {
		t.Fatalf("open breaker still waited %v", d)
	}
}

func TestOnnxTurnClientAnalyzeContextDeadline(t *testing.T) {
	client, err := NewOnnxTurnClient(hangingTurnWorker(t))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := client.AnalyzeContext(ctx, make([]byte, 320), 16000, 0); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
	if d := time.Since(start); d > time.Second {
		t.Fatalf("deadline ignored for %v", d)
	}
}
