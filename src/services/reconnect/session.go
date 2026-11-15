package reconnect

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

var (
	ErrQueueFull = errors.New("websocket send queue full")
	errStopped   = errors.New("websocket session stopped")
	errReconnect = errors.New("websocket reconnect requested")
)

type Backoff struct {
	Initial     time.Duration
	Max         time.Duration
	MaxAttempts int
}

func (b Backoff) withDefaults() Backoff {
	if b.Initial <= 0 {
		b.Initial = 250 * time.Millisecond
	}
	if b.Max <= 0 {
		b.Max = 10 * time.Second
	}
	if b.MaxAttempts == 0 {
		b.MaxAttempts = 10
	}
	return b
}

func (b Backoff) Delay(attempt int) time.Duration {
	b = b.withDefaults()
	d := b.Initial
	for i := 1; i < attempt && d < b.Max; i++ {
		d *= 2
	}
	if d > b.Max {
		d = b.Max
	}
	half := d / 2
	return half + time.Duration(rand.Int64N(int64(half)+1))
}

type Message struct {
	Type int
	Data []byte
}

type Writer interface {
	WriteMessage(messageType int, data []byte) error
}

type Config struct {
	Name          string
	Endpoint      func() (string, http.Header)
	Dialer        *websocket.Dialer
	OnOpen        func(w Writer) error
	OnMessage     func(messageType int, data []byte)
	OnDisconnect  func(err error)
	OnError       func(err error, fatal bool)
	CloseMessage  *Message
	DrainOnClose  bool
	KeepaliveIdle time.Duration
	Keepalive     func() (Message, bool)
	PingInterval  time.Duration
	WriteTimeout  time.Duration
	CloseTimeout  time.Duration
	QueueSize     int
	Backoff       Backoff
}

type queued struct {
	msg       Message
	droppable bool
}

type Session struct {
	cfg Config
	log *logger.Logger

	mu    sync.Mutex
	queue []queued
	head  int
	n     int

	wake        chan struct{}
	stop        chan struct{}
	reconnectCh chan struct{}
	done        chan struct{}

	lifeMu   sync.Mutex
	started  bool
	cancel   context.CancelFunc
	stopOnce sync.Once
	graceful atomic.Bool

	connected atomic.Bool
	dropped   atomic.Uint64
}

func New(cfg Config) *Session {
	if cfg.WriteTimeout <= 0 {
		cfg.WriteTimeout = 5 * time.Second
	}
	if cfg.CloseTimeout <= 0 {
		cfg.CloseTimeout = 2 * time.Second
	}
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 512
	}
	if cfg.Name == "" {
		cfg.Name = "websocket"
	}
	cfg.Backoff = cfg.Backoff.withDefaults()
	return &Session{
		cfg:         cfg,
		log:         logger.WithPrefix(cfg.Name),
		queue:       make([]queued, cfg.QueueSize),
		wake:        make(chan struct{}, 1),
		stop:        make(chan struct{}),
		reconnectCh: make(chan struct{}, 1),
		done:        make(chan struct{}),
	}
}

func (s *Session) Start(ctx context.Context) {
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()
	if s.started {
		return
	}
	s.started = true
	runCtx, cancel := context.WithCancel(ctx)
	s.cancel = cancel
	go s.run(runCtx)
}

func (s *Session) Close(graceful bool) {
	s.lifeMu.Lock()
	started, cancel := s.started, s.cancel
	s.lifeMu.Unlock()
	if !started {
		return
	}
	s.stopOnce.Do(func() {
		s.graceful.Store(graceful)
		close(s.stop)
	})
	if !graceful || !s.connected.Load() {
		cancel()
	}
	timer := time.NewTimer(s.cfg.CloseTimeout + s.cfg.WriteTimeout)
	select {
	case <-s.done:
	case <-timer.C:
		cancel()
		<-s.done
	}
	timer.Stop()
	cancel()
}

func (s *Session) Done() <-chan struct{} {
	return s.done
}

func (s *Session) Connected() bool {
	return s.connected.Load()
}

func (s *Session) Dropped() uint64 {
	return s.dropped.Load()
}

func (s *Session) Reconnect() {
	select {
	case s.reconnectCh <- struct{}{}:
	default:
	}
}

func (s *Session) Send(messageType int, data []byte) bool {
	return s.enqueue(queued{msg: Message{Type: messageType, Data: data}})
}

func (s *Session) SendDroppable(messageType int, data []byte) bool {
	return s.enqueue(queued{msg: Message{Type: messageType, Data: data}, droppable: true})
}

func (s *Session) SendJSON(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if !s.Send(websocket.TextMessage, data) {
		return ErrQueueFull
	}
	return nil
}

func (s *Session) Clear() {
	s.mu.Lock()
	for i := 0; i < s.n; i++ {
		s.queue[(s.head+i)%len(s.queue)] = queued{}
	}
	s.head, s.n = 0, 0
	s.mu.Unlock()
}

func (s *Session) enqueue(m queued) bool {
	s.mu.Lock()
	ok := s.n < len(s.queue) || s.evictOldestDroppableLocked()
	if ok {
		s.queue[(s.head+s.n)%len(s.queue)] = m
		s.n++
	}
	s.mu.Unlock()
	if !ok {
		if s.dropped.Add(1)%100 == 1 {
			s.log.Warn("send queue full, dropped %d messages so far", s.dropped.Load())
		}
		return false
	}
	select {
	case s.wake <- struct{}{}:
	default:
	}
	return true
}

func (s *Session) evictOldestDroppableLocked() bool {
	size := len(s.queue)
	for i := 0; i < s.n; i++ {
		if !s.queue[(s.head+i)%size].droppable {
			continue
		}
		if i == 0 {
			s.queue[s.head] = queued{}
			s.head = (s.head + 1) % size
		} else {
			for j := i; j < s.n-1; j++ {
				s.queue[(s.head+j)%size] = s.queue[(s.head+j+1)%size]
			}
			s.queue[(s.head+s.n-1)%size] = queued{}
		}
		s.n--
		s.dropped.Add(1)
		return true
	}
	return false
}

func (s *Session) pop() (Message, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.n == 0 {
		return Message{}, false
	}
	m := s.queue[s.head].msg
	s.queue[s.head] = queued{}
	s.head = (s.head + 1) % len(s.queue)
	s.n--
	return m, true
}

func (s *Session) stopping() bool {
	select {
	case <-s.stop:
		return true
	default:
		return false
	}
}

func (s *Session) report(err error, fatal bool) {
	if fatal {
		s.log.Error("%v", err)
	} else {
		s.log.Warn("%v", err)
	}
	if s.cfg.OnError != nil {
		s.cfg.OnError(err, fatal)
	}
}

func (s *Session) run(ctx context.Context) {
	defer close(s.done)
	failures := 0
	for {
		if s.stopping() || ctx.Err() != nil {
			return
		}
		conn, err := s.dial(ctx)
		if err == nil {
			err = s.open(conn)
		}
		if err != nil {
			if s.stopping() || ctx.Err() != nil {
				return
			}
			failures++
			if s.cfg.Backoff.MaxAttempts > 0 && failures >= s.cfg.Backoff.MaxAttempts {
				s.report(fmt.Errorf("connection failed after %d attempts: %w", failures, err), true)
				return
			}
			s.report(fmt.Errorf("connection attempt %d failed: %w", failures, err), false)
			if !s.sleep(ctx, s.cfg.Backoff.Delay(failures)) {
				return
			}
			continue
		}
		s.connected.Store(true)
		connectedAt := time.Now()
		err = s.serve(ctx, conn)
		s.connected.Store(false)
		stable := time.Since(connectedAt) >= minStableConnection
		if stable {
			failures = 0
		}
		if errors.Is(err, errStopped) || s.stopping() || ctx.Err() != nil {
			return
		}
		if errors.Is(err, errReconnect) {
			continue
		}
		s.log.Warn("connection lost: %v", err)
		if s.cfg.OnDisconnect != nil {
			s.cfg.OnDisconnect(err)
		}
		failures++
		if !stable && s.cfg.Backoff.MaxAttempts > 0 && failures >= s.cfg.Backoff.MaxAttempts {
			s.report(fmt.Errorf("connection dropped %d times in a row: %w", failures, err), true)
			return
		}
		if !s.sleep(ctx, s.cfg.Backoff.Delay(failures)) {
			return
		}
	}
}

const minStableConnection = 5 * time.Second

func (s *Session) sleep(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	case <-s.stop:
		return false
	}
}

func (s *Session) dial(ctx context.Context) (*websocket.Conn, error) {
	if s.cfg.Endpoint == nil {
		return nil, errors.New("no endpoint configured")
	}
	url, header := s.cfg.Endpoint()
	d := s.cfg.Dialer
	if d == nil {
		d = &websocket.Dialer{Proxy: http.ProxyFromEnvironment, HandshakeTimeout: 10 * time.Second}
	}
	conn, resp, err := d.DialContext(ctx, url, header)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		if resp != nil {
			return nil, fmt.Errorf("dial: %w (HTTP %d)", err, resp.StatusCode)
		}
		return nil, fmt.Errorf("dial: %w", err)
	}
	return conn, nil
}

func (s *Session) open(conn *websocket.Conn) error {
	if s.cfg.OnOpen == nil {
		return nil
	}
	if err := s.cfg.OnOpen(connWriter{conn: conn, timeout: s.cfg.WriteTimeout}); err != nil {
		_ = conn.Close()
		return fmt.Errorf("open: %w", err)
	}
	return nil
}

type connWriter struct {
	conn    *websocket.Conn
	timeout time.Duration
}

func (w connWriter) WriteMessage(messageType int, data []byte) error {
	_ = w.conn.SetWriteDeadline(time.Now().Add(w.timeout))
	return w.conn.WriteMessage(messageType, data)
}

func (s *Session) readTimeout() time.Duration {
	if s.cfg.PingInterval <= 0 {
		return 0
	}
	return 2*s.cfg.PingInterval + s.cfg.WriteTimeout
}

func (s *Session) tickInterval() time.Duration {
	interval := s.cfg.PingInterval
	if s.cfg.KeepaliveIdle > 0 && s.cfg.Keepalive != nil {
		half := s.cfg.KeepaliveIdle / 2
		if interval <= 0 || half < interval {
			interval = half
		}
	}
	return interval
}

func (s *Session) serve(ctx context.Context, conn *websocket.Conn) error {
	readDone := make(chan error, 1)
	if rt := s.readTimeout(); rt > 0 {
		_ = conn.SetReadDeadline(time.Now().Add(rt))
		conn.SetPongHandler(func(string) error {
			return conn.SetReadDeadline(time.Now().Add(rt))
		})
	}
	go s.readLoop(conn, readDone)

	var tick <-chan time.Time
	if interval := s.tickInterval(); interval > 0 {
		t := time.NewTicker(interval)
		defer t.Stop()
		tick = t.C
	}

	w := connWriter{conn: conn, timeout: s.cfg.WriteTimeout}
	lastWrite := time.Now()
	lastPing := lastWrite
	fail := func(err error) error {
		_ = conn.Close()
		<-readDone
		return err
	}

	for {
		if err := s.flush(w, &lastWrite); err != nil {
			return fail(fmt.Errorf("write: %w", err))
		}
		select {
		case <-ctx.Done():
			return fail(ctx.Err())
		case err := <-readDone:
			_ = conn.Close()
			return fmt.Errorf("read: %w", err)
		case <-s.wake:
		case <-s.reconnectCh:
			return fail(errReconnect)
		case <-s.stop:
			if !s.graceful.Load() {
				return fail(errStopped)
			}
			s.shutdown(conn, w, readDone, &lastWrite)
			return errStopped
		case now := <-tick:
			if s.cfg.PingInterval > 0 && now.Sub(lastPing) >= s.cfg.PingInterval {
				if err := conn.WriteControl(websocket.PingMessage, nil, now.Add(s.cfg.WriteTimeout)); err != nil {
					return fail(fmt.Errorf("ping: %w", err))
				}
				lastPing = now
			}
			if s.cfg.KeepaliveIdle > 0 && s.cfg.Keepalive != nil && now.Sub(lastWrite) >= s.cfg.KeepaliveIdle {
				if m, ok := s.cfg.Keepalive(); ok {
					if err := w.WriteMessage(m.Type, m.Data); err != nil {
						return fail(fmt.Errorf("keepalive: %w", err))
					}
					lastWrite = now
				}
			}
		}
	}
}

func (s *Session) flush(w Writer, lastWrite *time.Time) error {
	for {
		m, ok := s.pop()
		if !ok {
			return nil
		}
		if err := w.WriteMessage(m.Type, m.Data); err != nil {
			return err
		}
		*lastWrite = time.Now()
	}
}

func (s *Session) shutdown(conn *websocket.Conn, w Writer, readDone chan error, lastWrite *time.Time) {
	readFinished := false
	_ = s.flush(w, lastWrite)
	timer := time.NewTimer(s.cfg.CloseTimeout)
	defer timer.Stop()
	wait := s.cfg.DrainOnClose
	if cm := s.cfg.CloseMessage; cm != nil {
		wait = w.WriteMessage(cm.Type, cm.Data) == nil
	}
	if wait {
		select {
		case <-readDone:
			readFinished = true
		case <-timer.C:
		}
	}
	if !readFinished {
		_ = conn.WriteControl(websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""),
			time.Now().Add(s.cfg.WriteTimeout))
		grace := time.NewTimer(250 * time.Millisecond)
		select {
		case <-readDone:
			readFinished = true
		case <-grace.C:
		}
		grace.Stop()
	}
	_ = conn.Close()
	if !readFinished {
		<-readDone
	}
}

func (s *Session) readLoop(conn *websocket.Conn, done chan<- error) {
	rt := s.readTimeout()
	for {
		mt, data, err := conn.ReadMessage()
		if err != nil {
			done <- err
			return
		}
		if rt > 0 {
			_ = conn.SetReadDeadline(time.Now().Add(rt))
		}
		if s.cfg.OnMessage != nil {
			s.dispatch(mt, data)
		}
	}
}

func (s *Session) dispatch(mt int, data []byte) {
	defer func() {
		if r := recover(); r != nil {
			s.report(fmt.Errorf("message handler panic: %v", r), false)
		}
	}()
	s.cfg.OnMessage(mt, data)
}
