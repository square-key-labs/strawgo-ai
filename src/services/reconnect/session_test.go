package reconnect

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

const guard = 3 * time.Second

type wsServer struct {
	*httptest.Server
	conns atomic.Int32

	mu   sync.Mutex
	msgs []string
	recv chan string
}

func newWSServer(t *testing.T, handle func(n int32, c *websocket.Conn, s *wsServer)) *wsServer {
	t.Helper()
	s := &wsServer{recv: make(chan string, 4096)}
	up := websocket.Upgrader{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := up.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer c.Close()
		n := s.conns.Add(1)
		handle(n, c, s)
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *wsServer) url() string {
	return "ws" + strings.TrimPrefix(s.URL, "http")
}

func (s *wsServer) endpoint() func() (string, http.Header) {
	return func() (string, http.Header) { return s.url(), nil }
}

func (s *wsServer) record(m string) {
	s.mu.Lock()
	s.msgs = append(s.msgs, m)
	s.mu.Unlock()
	s.recv <- m
}

func readAll(_ int32, c *websocket.Conn, s *wsServer) {
	for {
		_, data, err := c.ReadMessage()
		if err != nil {
			return
		}
		s.record(string(data))
	}
}

func (s *wsServer) expect(t *testing.T, want ...string) {
	t.Helper()
	for i, w := range want {
		select {
		case got := <-s.recv:
			if got != w {
				t.Fatalf("message %d: got %q want %q", i, got, w)
			}
		case <-time.After(guard):
			t.Fatalf("timed out waiting for message %d (%q)", i, w)
		}
	}
}

func waitClosed(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(guard):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func fastBackoff(attempts int) Backoff {
	return Backoff{Initial: 2 * time.Millisecond, Max: 8 * time.Millisecond, MaxAttempts: attempts}
}

func TestMessagesDeliveredInOrder(t *testing.T) {
	srv := newWSServer(t, readAll)
	s := New(Config{Endpoint: srv.endpoint(), Backoff: fastBackoff(3), QueueSize: 1024})
	s.Start(context.Background())
	defer s.Close(false)

	want := make([]string, 300)
	for i := range want {
		want[i] = strconv.Itoa(i)
		if !s.Send(websocket.TextMessage, []byte(want[i])) {
			t.Fatalf("send %d rejected", i)
		}
	}
	srv.expect(t, want...)
}

func TestConcurrentSendersUseSingleWriter(t *testing.T) {
	srv := newWSServer(t, readAll)
	s := New(Config{Endpoint: srv.endpoint(), Backoff: fastBackoff(3), QueueSize: 4096})
	s.Start(context.Background())
	defer s.Close(false)

	const senders, per = 8, 100
	var wg sync.WaitGroup
	for g := 0; g < senders; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < per; i++ {
				s.Send(websocket.TextMessage, []byte(fmt.Sprintf("%d:%d", g, i)))
			}
		}(g)
	}
	wg.Wait()

	next := make([]int, senders)
	for n := 0; n < senders*per; n++ {
		select {
		case m := <-srv.recv:
			var g, i int
			if _, err := fmt.Sscanf(m, "%d:%d", &g, &i); err != nil {
				t.Fatalf("bad message %q", m)
			}
			if i != next[g] {
				t.Fatalf("sender %d out of order: got %d want %d", g, i, next[g])
			}
			next[g]++
		case <-time.After(guard):
			t.Fatalf("timed out after %d messages", n)
		}
	}
}

func TestDroppableEvictsOldestAndNeverBlocks(t *testing.T) {
	srv := newWSServer(t, readAll)
	s := New(Config{Endpoint: srv.endpoint(), Backoff: fastBackoff(3), QueueSize: 4})

	done := make(chan struct{})
	go func() {
		defer close(done)
		s.Send(websocket.TextMessage, []byte("ctl"))
		for i := 1; i <= 5; i++ {
			if !s.SendDroppable(websocket.BinaryMessage, []byte("a"+strconv.Itoa(i))) {
				t.Errorf("droppable %d rejected", i)
			}
		}
	}()
	waitClosed(t, done, "droppable sends")

	if got := s.Dropped(); got != 2 {
		t.Fatalf("dropped = %d, want 2", got)
	}

	s.Start(context.Background())
	defer s.Close(false)
	srv.expect(t, "ctl", "a3", "a4", "a5")
}

func TestQueueFullOfReliableMessagesRejects(t *testing.T) {
	s := New(Config{Endpoint: func() (string, http.Header) { return "ws://127.0.0.1:1", nil }, QueueSize: 2})
	if !s.Send(websocket.TextMessage, []byte("1")) || !s.Send(websocket.TextMessage, []byte("2")) {
		t.Fatal("sends within capacity rejected")
	}
	if s.Send(websocket.TextMessage, []byte("3")) {
		t.Fatal("send beyond capacity accepted")
	}
	if s.SendDroppable(websocket.BinaryMessage, []byte("x")) {
		t.Fatal("droppable accepted with no droppable to evict")
	}
	if err := s.SendJSON(map[string]int{"a": 1}); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("SendJSON err = %v, want ErrQueueFull", err)
	}
	if s.Dropped() != 3 {
		t.Fatalf("dropped = %d, want 3", s.Dropped())
	}
	s.Clear()
	if !s.Send(websocket.TextMessage, []byte("4")) {
		t.Fatal("send after Clear rejected")
	}
}

func TestReconnectsAfterServerClose(t *testing.T) {
	srv := newWSServer(t, func(n int32, c *websocket.Conn, s *wsServer) {
		_, data, err := c.ReadMessage()
		if err != nil {
			return
		}
		s.record(fmt.Sprintf("%d:%s", n, data))
		if n == 1 {
			return
		}
		readAll(n, c, s)
	})
	disconnected := make(chan error, 4)
	opened := make(chan struct{}, 4)
	s := New(Config{
		Endpoint:     srv.endpoint(),
		Backoff:      fastBackoff(5),
		OnOpen:       func(Writer) error { opened <- struct{}{}; return nil },
		OnDisconnect: func(err error) { disconnected <- err },
	})
	s.Start(context.Background())
	defer s.Close(false)

	s.Send(websocket.TextMessage, []byte("first"))
	srv.expect(t, "1:first")
	select {
	case <-disconnected:
	case <-time.After(guard):
		t.Fatal("OnDisconnect not called")
	}
	<-opened
	select {
	case <-opened:
	case <-time.After(guard):
		t.Fatal("did not reconnect")
	}
	s.Send(websocket.TextMessage, []byte("second"))
	srv.expect(t, "2:second")
	if !s.Connected() {
		t.Fatal("not connected after reconnect")
	}
}

func TestOnOpenWritesHandshakeFirst(t *testing.T) {
	srv := newWSServer(t, readAll)
	s := New(Config{
		Endpoint: srv.endpoint(),
		Backoff:  fastBackoff(3),
		OnOpen: func(w Writer) error {
			return w.WriteMessage(websocket.TextMessage, []byte("hello"))
		},
	})
	s.Send(websocket.TextMessage, []byte("queued-before-start"))
	s.Start(context.Background())
	defer s.Close(false)
	srv.expect(t, "hello", "queued-before-start")
}

func TestAcceptThenCloseLoopEndsFatal(t *testing.T) {
	srv := newWSServer(t, func(int32, *websocket.Conn, *wsServer) {})
	var mu sync.Mutex
	var fatals, warnings int
	s := New(Config{
		Endpoint: srv.endpoint(),
		Backoff:  fastBackoff(4),
		OnError: func(err error, fatal bool) {
			mu.Lock()
			defer mu.Unlock()
			if fatal {
				fatals++
			} else {
				warnings++
			}
		},
	})
	s.Start(context.Background())
	waitClosed(t, s.Done(), "session to give up")
	s.Close(false)

	mu.Lock()
	defer mu.Unlock()
	if fatals != 1 {
		t.Fatalf("fatal errors = %d, want 1", fatals)
	}
	if got := srv.conns.Load(); got != 4 {
		t.Fatalf("connections = %d, want 4", got)
	}
}

func TestDialFailureEndsFatalAfterMaxAttempts(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		http.Error(w, "nope", http.StatusUnauthorized)
	}))
	defer srv.Close()
	fatal := make(chan error, 1)
	s := New(Config{
		Endpoint: func() (string, http.Header) { return "ws" + strings.TrimPrefix(srv.URL, "http"), nil },
		Backoff:  fastBackoff(3),
		OnError: func(err error, f bool) {
			if f {
				fatal <- err
			}
		},
	})
	s.Start(context.Background())
	waitClosed(t, s.Done(), "session to give up")
	select {
	case err := <-fatal:
		if !strings.Contains(err.Error(), "401") {
			t.Fatalf("fatal error lacks status: %v", err)
		}
	default:
		t.Fatal("no fatal error reported")
	}
	if attempts.Load() != 3 {
		t.Fatalf("attempts = %d, want 3", attempts.Load())
	}
}

func TestCloseDuringBackoffReturnsPromptly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "down", http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	failed := make(chan struct{}, 1)
	s := New(Config{
		Endpoint: func() (string, http.Header) { return "ws" + strings.TrimPrefix(srv.URL, "http"), nil },
		Backoff:  Backoff{Initial: time.Minute, Max: time.Minute, MaxAttempts: 10},
		OnError: func(error, bool) {
			select {
			case failed <- struct{}{}:
			default:
			}
		},
	})
	s.Start(context.Background())
	waitClosed(t, failed, "first failure")

	began := time.Now()
	s.Close(true)
	if d := time.Since(began); d > 500*time.Millisecond {
		t.Fatalf("Close took %v during backoff", d)
	}
	waitClosed(t, s.Done(), "done")
}

func TestContextCancelStopsSession(t *testing.T) {
	srv := newWSServer(t, readAll)
	ctx, cancel := context.WithCancel(context.Background())
	s := New(Config{Endpoint: srv.endpoint(), Backoff: fastBackoff(3)})
	s.Start(ctx)
	s.Send(websocket.TextMessage, []byte("x"))
	srv.expect(t, "x")
	cancel()
	waitClosed(t, s.Done(), "done after ctx cancel")
}

func TestPingSent(t *testing.T) {
	pings := make(chan struct{}, 16)
	srv := newWSServer(t, func(n int32, c *websocket.Conn, s *wsServer) {
		c.SetPingHandler(func(data string) error {
			select {
			case pings <- struct{}{}:
			default:
			}
			return c.WriteControl(websocket.PongMessage, []byte(data), time.Now().Add(time.Second))
		})
		readAll(n, c, s)
	})
	s := New(Config{Endpoint: srv.endpoint(), Backoff: fastBackoff(3), PingInterval: 15 * time.Millisecond})
	s.Start(context.Background())
	defer s.Close(false)
	for i := 0; i < 3; i++ {
		select {
		case <-pings:
		case <-time.After(guard):
			t.Fatalf("ping %d not received", i)
		}
	}
	if !s.Connected() || srv.conns.Load() != 1 {
		t.Fatalf("connection churned with pongs answered: conns=%d", srv.conns.Load())
	}
}

func TestKeepaliveSentWhenIdle(t *testing.T) {
	srv := newWSServer(t, readAll)
	s := New(Config{
		Endpoint:      srv.endpoint(),
		Backoff:       fastBackoff(3),
		KeepaliveIdle: 30 * time.Millisecond,
		Keepalive:     func() (Message, bool) { return Message{Type: websocket.TextMessage, Data: []byte("ka")}, true },
	})
	s.Start(context.Background())
	defer s.Close(false)
	srv.expect(t, "ka", "ka")
}

func TestGracefulCloseFlushesAndSendsCloseMessage(t *testing.T) {
	srv := newWSServer(t, readAll)
	s := New(Config{
		Endpoint:     srv.endpoint(),
		Backoff:      fastBackoff(3),
		CloseMessage: &Message{Type: websocket.TextMessage, Data: []byte("bye")},
		CloseTimeout: 100 * time.Millisecond,
	})
	s.Start(context.Background())
	s.Send(websocket.TextMessage, []byte("warm"))
	srv.expect(t, "warm")
	s.Send(websocket.TextMessage, []byte("last"))
	s.Close(true)
	srv.expect(t, "last", "bye")
	waitClosed(t, s.Done(), "done")
}

func TestExplicitReconnectOpensNewConnection(t *testing.T) {
	srv := newWSServer(t, readAll)
	s := New(Config{Endpoint: srv.endpoint(), Backoff: fastBackoff(3)})
	s.Start(context.Background())
	defer s.Close(false)
	s.Send(websocket.TextMessage, []byte("a"))
	srv.expect(t, "a")
	s.Reconnect()
	deadline := time.Now().Add(guard)
	for srv.conns.Load() < 2 {
		if time.Now().After(deadline) {
			t.Fatal("no second connection")
		}
		time.Sleep(time.Millisecond)
	}
	s.Send(websocket.TextMessage, []byte("b"))
	srv.expect(t, "b")
}

func TestOnMessagePanicIsRecovered(t *testing.T) {
	srv := newWSServer(t, func(n int32, c *websocket.Conn, s *wsServer) {
		_ = c.WriteMessage(websocket.TextMessage, []byte("boom"))
		_ = c.WriteMessage(websocket.TextMessage, []byte("ok"))
		readAll(n, c, s)
	})
	got := make(chan string, 4)
	errs := make(chan error, 4)
	s := New(Config{
		Endpoint: srv.endpoint(),
		Backoff:  fastBackoff(3),
		OnMessage: func(_ int, data []byte) {
			if string(data) == "boom" {
				panic("handler exploded")
			}
			got <- string(data)
		},
		OnError: func(err error, _ bool) { errs <- err },
	})
	s.Start(context.Background())
	defer s.Close(false)
	select {
	case m := <-got:
		if m != "ok" {
			t.Fatalf("got %q", m)
		}
	case <-time.After(guard):
		t.Fatal("message after panic not delivered")
	}
	select {
	case err := <-errs:
		if !strings.Contains(err.Error(), "panic") {
			t.Fatalf("unexpected error %v", err)
		}
	case <-time.After(guard):
		t.Fatal("panic not reported")
	}
	if srv.conns.Load() != 1 {
		t.Fatal("panic caused a reconnect")
	}
}

func TestBackoffDelayBounds(t *testing.T) {
	b := Backoff{Initial: 100 * time.Millisecond, Max: time.Second, MaxAttempts: 5}
	cases := []struct {
		attempt int
		base    time.Duration
	}{{1, 100 * time.Millisecond}, {2, 200 * time.Millisecond}, {3, 400 * time.Millisecond}, {4, 800 * time.Millisecond}, {5, time.Second}, {50, time.Second}}
	for _, c := range cases {
		for i := 0; i < 50; i++ {
			d := b.Delay(c.attempt)
			if d < c.base/2 || d > c.base {
				t.Fatalf("attempt %d: delay %v outside [%v,%v]", c.attempt, d, c.base/2, c.base)
			}
		}
	}
}

func TestNoGoroutineGrowthAfterClose(t *testing.T) {
	srv := newWSServer(t, readAll)
	cycle := func() {
		s := New(Config{Endpoint: srv.endpoint(), Backoff: fastBackoff(3), PingInterval: 10 * time.Millisecond})
		s.Start(context.Background())
		s.Send(websocket.TextMessage, []byte("x"))
		<-srv.recv
		s.Close(true)
		<-s.Done()
	}
	cycle()
	runtime.GC()
	base := runtime.NumGoroutine()
	for i := 0; i < 10; i++ {
		cycle()
	}
	deadline := time.Now().Add(guard)
	for {
		n := runtime.NumGoroutine()
		if n <= base+2 {
			return
		}
		if time.Now().After(deadline) {
			buf := make([]byte, 1<<16)
			t.Fatalf("goroutines grew from %d to %d\n%s", base, n, buf[:runtime.Stack(buf, true)])
		}
		time.Sleep(5 * time.Millisecond)
	}
}
