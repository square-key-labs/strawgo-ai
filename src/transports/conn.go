package transports

import (
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

var errConnClosed = errors.New("connection closed")

type wireConn interface {
	ReadMessage() (serializers.Message, error)
	WriteMessage(msg serializers.Message) error
	Ping() error
	Close() error
}

type wsConn struct {
	ws           *websocket.Conn
	writeTimeout time.Duration
	closeTimeout time.Duration
	readTimeout  time.Duration
	mu           sync.Mutex
	closed       atomic.Bool
}

func newWSConn(ws *websocket.Conn, params TransportParams) *wsConn {
	c := &wsConn{
		ws:           ws,
		writeTimeout: params.WriteTimeout,
		closeTimeout: params.CloseTimeout,
	}
	if params.PingInterval > 0 {
		c.readTimeout = 2*params.PingInterval + params.WriteTimeout
		ws.SetPongHandler(func(string) error {
			return ws.SetReadDeadline(time.Now().Add(c.readTimeout))
		})
	}
	return c
}

func (c *wsConn) ReadMessage() (serializers.Message, error) {
	var deadline time.Time
	if c.readTimeout > 0 {
		deadline = time.Now().Add(c.readTimeout)
	}
	return c.read(deadline)
}

func (c *wsConn) read(deadline time.Time) (serializers.Message, error) {
	_ = c.ws.SetReadDeadline(deadline)
	typ, data, err := c.ws.ReadMessage()
	if err != nil {
		return serializers.Message{}, err
	}
	if typ == websocket.BinaryMessage {
		return serializers.Message{Type: serializers.BinaryMessage, Data: data}, nil
	}
	return serializers.Message{Type: serializers.TextMessage, Data: data}, nil
}

func (c *wsConn) WriteMessage(msg serializers.Message) error {
	if c.closed.Load() {
		return errConnClosed
	}
	typ := websocket.TextMessage
	if msg.Type == serializers.BinaryMessage {
		typ = websocket.BinaryMessage
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.writeTimeout > 0 {
		_ = c.ws.SetWriteDeadline(time.Now().Add(c.writeTimeout))
	}
	return c.ws.WriteMessage(typ, msg.Data)
}

func (c *wsConn) Ping() error {
	if c.closed.Load() {
		return errConnClosed
	}
	return c.ws.WriteControl(websocket.PingMessage, nil, time.Now().Add(c.writeTimeout))
}

func (c *wsConn) Close() error {
	if !c.closed.CompareAndSwap(false, true) {
		return nil
	}
	_ = c.ws.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""), time.Now().Add(c.closeTimeout))
	return c.ws.Close()
}
