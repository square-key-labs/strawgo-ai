package httpretry

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func get(url string) RequestFunc {
	return func(ctx context.Context) (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	}
}

func stalledThenFast(t *testing.T, stalls int32) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) <= stalls {
			w.(http.Flusher).Flush()
			select {
			case <-r.Context().Done():
			case <-time.After(5 * time.Second):
			}
			return
		}
		_, _ = io.WriteString(w, "hello")
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func TestRetriesWhenNoFirstByte(t *testing.T) {
	srv, hits := stalledThenFast(t, 2)
	policy := Policy{MaxAttempts: 3, FirstByteTimeout: 80 * time.Millisecond, Backoff: time.Millisecond}
	resp, err := Do(context.Background(), srv.Client(), policy, get(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil || string(raw) != "hello" {
		t.Fatalf("body = %q err = %v", raw, err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if hits.Load() != 3 {
		t.Fatalf("attempts = %d", hits.Load())
	}
}

func TestAttemptsAreBounded(t *testing.T) {
	srv, hits := stalledThenFast(t, 100)
	policy := Policy{MaxAttempts: 2, FirstByteTimeout: 50 * time.Millisecond, Backoff: time.Millisecond}
	_, err := Do(context.Background(), srv.Client(), policy, get(srv.URL))
	if !errors.Is(err, ErrFirstByteTimeout) {
		t.Fatalf("err = %v", err)
	}
	if hits.Load() != 2 {
		t.Fatalf("attempts = %d", hits.Load())
	}
}

func TestDisabledPolicyMakesOneAttempt(t *testing.T) {
	var dials atomic.Int32
	client := &http.Client{Transport: &http.Transport{DialContext: func(context.Context, string, string) (net.Conn, error) {
		dials.Add(1)
		return nil, &net.OpError{Op: "dial", Err: errors.New("refused")}
	}}}
	if _, err := Do(context.Background(), client, Policy{}, get("http://example.invalid")); err == nil {
		t.Fatal("expected an error")
	}
	if dials.Load() != 1 {
		t.Fatalf("dials = %d", dials.Load())
	}
}

func TestRetriesDialErrorsWithBackoff(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "ok")
	}))
	defer srv.Close()
	var dials atomic.Int32
	client := &http.Client{Transport: &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
		if dials.Add(1) == 1 {
			return nil, &net.OpError{Op: "dial", Err: errors.New("refused")}
		}
		return (&net.Dialer{}).DialContext(ctx, network, srv.Listener.Addr().String())
	}}}
	started := time.Now()
	resp, err := Do(context.Background(), client, Policy{MaxAttempts: 2, Backoff: 40 * time.Millisecond}, get(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if dials.Load() != 2 || time.Since(started) < 40*time.Millisecond {
		t.Fatalf("dials = %d elapsed = %s", dials.Load(), time.Since(started))
	}
}

func TestDoesNotRetryHTTPStatusOrAfterFirstByte(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if r.URL.Path == "/status" {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, "busy")
			return
		}
		_, _ = io.WriteString(w, "x")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()
	policy := Policy{MaxAttempts: 3, FirstByteTimeout: 60 * time.Millisecond, Backoff: time.Millisecond}

	resp, err := Do(context.Background(), srv.Client(), policy, get(srv.URL+"/status"))
	if err != nil || resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("resp = %v err = %v", resp, err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if string(raw) != "busy" || hits.Load() != 1 {
		t.Fatalf("body = %q attempts = %d", raw, hits.Load())
	}

	hits.Store(0)
	resp, err = Do(context.Background(), srv.Client(), policy, get(srv.URL+"/stream"))
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(150 * time.Millisecond)
	buf := make([]byte, 8)
	n, err := resp.Body.Read(buf)
	if n != 1 || buf[0] != 'x' || err != nil {
		t.Fatalf("first read n=%d err=%v", n, err)
	}
	resp.Body.Close()
	if hits.Load() != 1 {
		t.Fatalf("stream was retried after its first byte: attempts = %d", hits.Load())
	}
}

func TestHonoursContext(t *testing.T) {
	srv, hits := stalledThenFast(t, 100)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Millisecond)
	defer cancel()
	policy := Policy{MaxAttempts: 50, FirstByteTimeout: 5 * time.Second, Backoff: time.Second}
	started := time.Now()
	_, err := Do(ctx, srv.Client(), policy, get(srv.URL))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v", err)
	}
	if time.Since(started) > time.Second || hits.Load() != 1 {
		t.Fatalf("elapsed = %s attempts = %d", time.Since(started), hits.Load())
	}

	ctx2, cancel2 := context.WithCancel(context.Background())
	hits.Store(0)
	go func() {
		time.Sleep(80 * time.Millisecond)
		cancel2()
	}()
	_, err = Do(ctx2, srv.Client(), Policy{MaxAttempts: 50, FirstByteTimeout: 30 * time.Millisecond, Backoff: 10 * time.Second}, get(srv.URL))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if hits.Load() != 1 {
		t.Fatalf("attempts during backoff = %d", hits.Load())
	}
}
