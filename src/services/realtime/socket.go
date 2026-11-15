package realtime

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

var ErrNotConnected = errors.New("realtime: websocket not connected")

type Backoff struct {
	Initial     time.Duration
	Max         time.Duration
	MaxAttempts int
}

func (b Backoff) withDefaults() Backoff {
	if b.Initial <= 0 {
		b.Initial = 500 * time.Millisecond
	}
	if b.Max <= 0 {
		b.Max = 15 * time.Second
	}
	if b.MaxAttempts == 0 {
		b.MaxAttempts = 6
	}
	return b
}

func (b Backoff) delay(attempt int) time.Duration {
	d := b.Initial
	for i := 1; i < attempt && d < b.Max; i++ {
		d *= 2
	}
	if d > b.Max {
		d = b.Max
	}
	return d
}

type SocketConfig struct {
	Name         string
	Dial         func(ctx context.Context) (*websocket.Conn, error)
	OnOpen       func(ctx context.Context, w Writer) error
	ReadyOnOpen  bool
	OnMessage    func(data []byte)
	OnDisconnect func(err error)
	OnFatal      func(err error)
	Backoff      Backoff
	WriteTimeout time.Duration
}

type Writer interface {
	WriteJSON(v any) error
}

type Socket struct {
	cfg SocketConfig

	writeMu sync.Mutex
	conn    *websocket.Conn
	ready   bool
	buf     []byte

	forced atomic.Bool

	lifeMu sync.Mutex
	cancel context.CancelFunc
	done   chan struct{}
}

func NewSocket(cfg SocketConfig) *Socket {
	cfg.Backoff = cfg.Backoff.withDefaults()
	if cfg.WriteTimeout <= 0 {
		cfg.WriteTimeout = 5 * time.Second
	}
	return &Socket{cfg: cfg}
}

func (s *Socket) Start(parent context.Context) {
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()
	if s.cancel != nil {
		return
	}
	ctx, cancel := context.WithCancel(parent)
	s.cancel = cancel
	s.done = make(chan struct{})
	go s.run(ctx, s.done)
}

func (s *Socket) Stop() {
	s.lifeMu.Lock()
	cancel, done := s.cancel, s.done
	s.cancel, s.done = nil, nil
	s.lifeMu.Unlock()
	if cancel == nil {
		return
	}
	s.writeMu.Lock()
	if s.conn != nil {
		_ = s.conn.WriteControl(websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""),
			time.Now().Add(250*time.Millisecond))
	}
	s.writeMu.Unlock()
	cancel()
	<-done
}

func (s *Socket) Running() bool {
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()
	return s.cancel != nil
}

func (s *Socket) Ready() bool {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn != nil && s.ready
}

func (s *Socket) MarkReady() {
	s.writeMu.Lock()
	if s.conn != nil {
		s.ready = true
	}
	s.writeMu.Unlock()
}

func (s *Socket) Reconnect() {
	s.writeMu.Lock()
	conn := s.conn
	s.writeMu.Unlock()
	if conn == nil {
		return
	}
	s.forced.Store(true)
	_ = conn.Close()
}

func (s *Socket) WriteJSON(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return s.WriteMessage(data)
}

func (s *Socket) WriteMessage(data []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.writeLocked(data, true)
}

func (s *Socket) WriteBase64(prefix string, payload []byte, suffix string) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if s.conn == nil || !s.ready {
		return ErrNotConnected
	}
	buf := append(s.buf[:0], prefix...)
	buf = base64.StdEncoding.AppendEncode(buf, payload)
	buf = append(buf, suffix...)
	s.buf = buf
	return s.writeLocked(buf, true)
}

func (s *Socket) writeLocked(data []byte, needReady bool) error {
	if s.conn == nil || (needReady && !s.ready) {
		return ErrNotConnected
	}
	_ = s.conn.SetWriteDeadline(time.Now().Add(s.cfg.WriteTimeout))
	if err := s.conn.WriteMessage(websocket.TextMessage, data); err != nil {
		_ = s.conn.Close()
		return err
	}
	return nil
}

type openWriter struct{ s *Socket }

func (w openWriter) WriteJSON(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	w.s.writeMu.Lock()
	defer w.s.writeMu.Unlock()
	return w.s.writeLocked(data, false)
}

func (s *Socket) run(ctx context.Context, done chan struct{}) {
	defer close(done)
	attempt := 0
	for {
		err := s.session(ctx, &attempt)
		if ctx.Err() != nil {
			return
		}
		if s.forced.Swap(false) {
			continue
		}
		attempt++
		if s.cfg.Backoff.MaxAttempts > 0 && attempt > s.cfg.Backoff.MaxAttempts {
			if s.cfg.OnFatal != nil {
				s.cfg.OnFatal(err)
			}
			return
		}
		if s.cfg.OnDisconnect != nil {
			s.cfg.OnDisconnect(err)
		}
		delay := s.cfg.Backoff.delay(attempt)
		logger.Warn("[%s] websocket disconnected (%v), reconnecting in %v (attempt %d)", s.cfg.Name, err, delay, attempt)
		t := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			t.Stop()
			return
		case <-t.C:
		}
	}
}

func (s *Socket) session(ctx context.Context, attempt *int) error {
	conn, err := s.cfg.Dial(ctx)
	if err != nil {
		return err
	}
	stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
	defer func() {
		stop()
		s.writeMu.Lock()
		if s.conn == conn {
			s.conn = nil
			s.ready = false
		}
		s.writeMu.Unlock()
		_ = conn.Close()
	}()

	s.writeMu.Lock()
	s.conn = conn
	s.ready = false
	s.writeMu.Unlock()

	if s.cfg.OnOpen != nil {
		if err := s.cfg.OnOpen(ctx, openWriter{s}); err != nil {
			return err
		}
	}
	if s.cfg.ReadyOnOpen {
		s.MarkReady()
	}
	*attempt = 0

	var buf bytes.Buffer
	for {
		_, r, err := conn.NextReader()
		if err != nil {
			return err
		}
		buf.Reset()
		if _, err := buf.ReadFrom(r); err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		if s.cfg.OnMessage != nil {
			s.cfg.OnMessage(buf.Bytes())
		}
	}
}
