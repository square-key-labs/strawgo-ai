package transports

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

type SessionHandler func(ctx context.Context, s *Session) error

type ServerConfig struct {
	Addr          string
	Path          string
	NewSerializer func(r *http.Request) serializers.FrameSerializer
	Handler       SessionHandler
	Params        TransportParams
	CheckOrigin   func(r *http.Request) bool
}

type Server struct {
	cfg      ServerConfig
	params   TransportParams
	upgrader websocket.Upgrader
	log      *logger.Logger

	mu       sync.Mutex
	sessions map[*Session]struct{}
	baseCtx  context.Context
	cancel   context.CancelFunc
	wg       sync.WaitGroup
	draining atomic.Bool
}

func NewServer(cfg ServerConfig) (*Server, error) {
	if cfg.NewSerializer == nil {
		return nil, errors.New("transports: ServerConfig.NewSerializer is required")
	}
	if cfg.Handler == nil {
		return nil, errors.New("transports: ServerConfig.Handler is required")
	}
	if cfg.Path == "" {
		cfg.Path = "/ws"
	}
	ctx, cancel := context.WithCancel(context.Background())
	s := &Server{
		cfg:      cfg,
		params:   cfg.Params.withDefaults(),
		log:      logger.WithPrefix("WebSocketServer"),
		sessions: make(map[*Session]struct{}),
		baseCtx:  ctx,
		cancel:   cancel,
	}
	s.upgrader.CheckOrigin = cfg.CheckOrigin
	if s.upgrader.CheckOrigin == nil {
		s.upgrader.CheckOrigin = func(*http.Request) bool { return true }
	}
	return s, nil
}

func (s *Server) ListenAndServe(ctx context.Context) error {
	ln, err := net.Listen("tcp", s.cfg.Addr)
	if err != nil {
		return fmt.Errorf("websocket server listen: %w", err)
	}
	return s.Serve(ctx, ln)
}

func (s *Server) Serve(ctx context.Context, ln net.Listener) error {
	mux := http.NewServeMux()
	mux.Handle(s.cfg.Path, s)
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: s.params.HandshakeTimeout}
	stop := context.AfterFunc(ctx, func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), s.params.CloseTimeout)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
		s.Shutdown()
	})
	defer stop()
	s.log.Info("listening on %s%s", ln.Addr(), s.cfg.Path)
	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("websocket server: %w", err)
	}
	s.Shutdown()
	bound, cancel := context.WithTimeout(context.Background(), s.params.HangupTimeout+s.params.CloseTimeout)
	defer cancel()
	if !s.waitSessions(bound) {
		s.log.Warn("%d session handlers still running after shutdown", s.ActiveSessions())
	}
	return nil
}

func (s *Server) ActiveSessions() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.sessions)
}

func (s *Server) waitSessions(ctx context.Context) bool {
	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		return true
	case <-ctx.Done():
		return false
	}
}

func (s *Server) ShutdownContext(ctx context.Context) error {
	s.mu.Lock()
	s.draining.Store(true)
	s.mu.Unlock()
	if s.waitSessions(ctx) {
		s.cancel()
		return nil
	}
	s.Shutdown()
	bound, cancel := context.WithTimeout(context.Background(), s.params.CloseTimeout)
	defer cancel()
	s.waitSessions(bound)
	return ctx.Err()
}

func (s *Server) Shutdown() {
	s.mu.Lock()
	s.cancel()
	sessions := make([]*Session, 0, len(s.sessions))
	for sess := range s.sessions {
		sessions = append(sessions, sess)
	}
	s.mu.Unlock()
	for _, sess := range sessions {
		_ = sess.Close()
	}
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	if s.draining.Load() || s.baseCtx.Err() != nil {
		s.mu.Unlock()
		http.Error(w, "server shutting down", http.StatusServiceUnavailable)
		return
	}
	s.wg.Add(1)
	s.mu.Unlock()
	defer s.wg.Done()
	ws, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		s.log.Warn("upgrade: %v", err)
		return
	}

	conn := newWSConn(ws, s.params)
	ser := s.cfg.NewSerializer(r)
	if ser == nil {
		s.log.Warn("no serializer for %s", r.URL.Path)
		_ = conn.Close()
		return
	}
	info, err := handshake(conn, ser, s.params.HandshakeTimeout)
	if err != nil {
		s.log.Warn("handshake: %v", err)
		_ = conn.Close()
		return
	}
	sess := newSession(conn, ser, s.params, info, r)
	s.mu.Lock()
	s.sessions[sess] = struct{}{}
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.sessions, sess)
		s.mu.Unlock()
	}()

	sess.log.Info("connected (provider=%s stream=%s call=%s)", info.Provider, info.StreamID, info.CallID)
	sess.start()
	if err := s.runHandler(sess); err != nil {
		sess.log.Warn("handler: %v", err)
	}
	_ = sess.Close()
	sess.Wait()
	sess.log.Info("closed")
}

func (s *Server) runHandler(sess *Session) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("session handler panic: %v", r)
		}
	}()
	return s.cfg.Handler(s.baseCtx, sess)
}

func handshake(conn *wsConn, ser serializers.FrameSerializer, timeout time.Duration) (serializers.CallInfo, error) {
	hs, ok := ser.(serializers.Handshaker)
	if !ok {
		return serializers.CallInfo{}, nil
	}
	var deadline time.Time
	if timeout > 0 {
		deadline = time.Now().Add(timeout)
	}
	for {
		msg, err := conn.read(deadline)
		if err != nil {
			return serializers.CallInfo{}, err
		}
		done, err := hs.Handshake(msg)
		if err != nil {
			return serializers.CallInfo{}, err
		}
		if done {
			return hs.CallInfo(), nil
		}
	}
}
