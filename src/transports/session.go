package transports

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

var sessionCounter atomic.Uint64

type Session struct {
	id         string
	conn       wireConn
	serializer serializers.FrameSerializer
	params     TransportParams
	info       serializers.CallInfo
	request    *http.Request
	log        *logger.Logger

	input  *InputProcessor
	output *OutputProcessor

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	closing      atomic.Bool
	disconnected atomic.Bool
	disconnectMu sync.Mutex
	disconnectOK bool

	timerMu sync.Mutex
	timer   *time.Timer

	closeOnce sync.Once
	closeErr  error
	writeErr  atomic.Bool
}

func newSession(conn wireConn, ser serializers.FrameSerializer, params TransportParams, info serializers.CallInfo, req *http.Request) *Session {
	params = params.withDefaults()
	if params.NewAudioInAnalyzer != nil {
		params.AudioInAnalyzer = params.NewAudioInAnalyzer()
	}
	if params.NewOutputMixer != nil {
		params.OutputMixer = params.NewOutputMixer()
	}
	if params.NewInputFilter != nil {
		params.InputFilter = params.NewInputFilter()
	}
	id := fmt.Sprintf("ws-%d", sessionCounter.Add(1))
	ctx, cancel := context.WithCancel(context.Background())
	s := &Session{
		id:         id,
		conn:       conn,
		serializer: ser,
		params:     params,
		info:       info,
		request:    req,
		log:        logger.WithPrefix("WebSocketSession " + id),
		ctx:        ctx,
		cancel:     cancel,
	}
	if params.AudioInNative {
		if ns, ok := ser.(serializers.NativeAudioSerializer); ok {
			ns.SetNativeAudioInput(true)
		}
	}
	s.input = newInputProcessor(s)
	s.output = newOutputProcessor(s)
	return s
}

func (s *Session) reply(msg any) {
	data, err := json.Marshal(msg)
	if err != nil {
		s.log.Warn("reply: %v", err)
		return
	}
	_ = s.write(serializers.Message{Type: serializers.TextMessage, Data: data})
}

func (s *Session) start() {
	s.wg.Add(1)
	go s.readLoop()
	if s.params.PingInterval > 0 {
		s.wg.Add(1)
		go s.keepalive()
	}
}

func (s *Session) ID() string { return s.id }

func (s *Session) Input() *InputProcessor { return s.input }

func (s *Session) Output() *OutputProcessor { return s.output }

func (s *Session) CallInfo() serializers.CallInfo { return s.info }

func (s *Session) Request() *http.Request { return s.request }

func (s *Session) Serializer() serializers.FrameSerializer { return s.serializer }

func (s *Session) Done() <-chan struct{} { return s.ctx.Done() }

func (s *Session) Close() error {
	s.closeOnce.Do(func() {
		s.closing.Store(true)
		s.stopTimer()
		s.cancel()
		s.input.stopFilter()
		s.closeErr = s.conn.Close()
	})
	return s.closeErr
}

func (s *Session) Wait() {
	s.wg.Wait()
}

func (s *Session) write(msg serializers.Message) error {
	err := s.conn.WriteMessage(msg)
	if err != nil && !s.closing.Load() && s.writeErr.CompareAndSwap(false, true) {
		s.log.Warn("write failed: %v", err)
	}
	return err
}

func (s *Session) keepalive() {
	defer s.wg.Done()
	ticker := time.NewTicker(s.params.PingInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			if err := s.conn.Ping(); err != nil && !s.closing.Load() {
				s.log.Debug("ping failed: %v", err)
			}
		}
	}
}

func (s *Session) readLoop() {
	defer s.wg.Done()
	for {
		msg, err := s.conn.ReadMessage()
		if err != nil {
			if !s.closing.Load() && !errors.Is(err, errConnClosed) &&
				websocket.IsUnexpectedCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway, websocket.CloseNoStatusReceived, websocket.CloseAbnormalClosure) {
				s.log.Warn("read error: %v", err)
			}
			s.markDisconnected()
			return
		}
		frame, err := s.serializer.Deserialize(msg)
		if err != nil {
			s.log.Debug("deserialize: %v", err)
			continue
		}
		switch f := frame.(type) {
		case nil:
			continue
		case *serializers.MediaFlowFrame:
			s.output.setFlow(f.Paused)
			continue
		case *frames.OutputTransportMessageUrgentFrame:
			s.reply(f.Message)
			continue
		}
		if !s.input.waitStarted(s.ctx) {
			return
		}
		switch f := frame.(type) {
		case *frames.EndFrame, *frames.CancelFrame:
			s.markDisconnected()
		case *frames.PlaybackCompleteFrame:
			id, _ := f.Metadata()["correlation_id"].(string)
			drained, _ := f.Metadata()[frames.MetadataQueueDrained].(bool)
			s.output.ack(id, drained)
		default:
			s.input.deliver(frame)
		}
	}
}

func (s *Session) markDisconnected() {
	if s.disconnected.CompareAndSwap(false, true) {
		s.maybeHandleDisconnect()
	}
}

func (s *Session) maybeHandleDisconnect() {
	if !s.disconnected.Load() || !s.input.started() {
		return
	}
	s.disconnectMu.Lock()
	if s.disconnectOK {
		s.disconnectMu.Unlock()
		return
	}
	s.disconnectOK = true
	s.disconnectMu.Unlock()
	if s.closing.Load() || s.input.ending.Load() {
		return
	}
	s.log.Info("client disconnected")
	s.stopTimer()
	_ = s.input.PushFrame(frames.NewClientDisconnectedFrame(), frames.Downstream)
	if cb := s.params.OnClientDisconnected; cb != nil {
		cb(s)
		return
	}
	_ = s.input.PushFrame(frames.NewCancelTaskFrame("client disconnected"), frames.Upstream)
}

func (s *Session) startTimer() {
	if s.params.SessionTimeout <= 0 {
		return
	}
	s.timerMu.Lock()
	defer s.timerMu.Unlock()
	if s.timer != nil || s.closing.Load() {
		return
	}
	s.timer = time.AfterFunc(s.params.SessionTimeout, s.onTimeout)
}

func (s *Session) stopTimer() {
	s.timerMu.Lock()
	defer s.timerMu.Unlock()
	if s.timer != nil {
		s.timer.Stop()
	}
}

func (s *Session) onTimeout() {
	if s.closing.Load() || s.input.ending.Load() {
		return
	}
	s.log.Info("session timeout after %v", s.params.SessionTimeout)
	if cb := s.params.OnSessionTimeout; cb != nil {
		cb(s)
		return
	}
	_ = s.input.PushFrame(frames.NewEndTaskFrame("session timeout"), frames.Upstream)
}
