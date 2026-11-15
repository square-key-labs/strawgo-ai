package turn

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"net"
	"sync"
	"time"
)

type TurnPrediction struct {
	Probability float32
}

const DefaultOnnxTurnTimeout = 5 * time.Second

type OnnxTurnClient struct {
	sockPath string
	sem      chan struct{}
	frame    []byte

	mu     sync.Mutex
	conn   net.Conn
	closed bool
}

func NewOnnxTurnClient(sockPath string) (*OnnxTurnClient, error) {
	conn, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: sockPath, Net: "unix"})
	if err != nil {
		return nil, fmt.Errorf("onnx_turn: dial %s: %w", sockPath, err)
	}
	return &OnnxTurnClient{sockPath: sockPath, conn: conn, sem: make(chan struct{}, 1)}, nil
}

func (c *OnnxTurnClient) Analyze(audio []byte, sampleRate int, speechStartMs int) (TurnPrediction, error) {
	ctx, cancel := context.WithTimeout(context.Background(), DefaultOnnxTurnTimeout)
	defer cancel()
	return c.AnalyzeContext(ctx, audio, sampleRate, speechStartMs)
}

func (c *OnnxTurnClient) AnalyzeContext(ctx context.Context, audio []byte, sampleRate int, speechStartMs int) (TurnPrediction, error) {
	select {
	case c.sem <- struct{}{}:
	case <-ctx.Done():
		return TurnPrediction{}, ctx.Err()
	}
	defer func() { <-c.sem }()

	n := 13 + len(audio)
	if cap(c.frame) < n {
		c.frame = make([]byte, n)
	}
	frame := c.frame[:n]
	frame[0] = 0x02
	binary.LittleEndian.PutUint32(frame[1:5], uint32(8+len(audio)))
	binary.LittleEndian.PutUint32(frame[5:9], uint32(sampleRate))
	binary.LittleEndian.PutUint32(frame[9:13], uint32(speechStartMs))
	copy(frame[13:], audio)

	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(DefaultOnnxTurnTimeout)
	}

	var lastErr error
	const maxRetries = 3
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return TurnPrediction{}, err
		}
		conn := c.current()
		if conn == nil {
			if attempt > 0 {
				select {
				case <-time.After(50 * time.Millisecond):
				case <-ctx.Done():
					return TurnPrediction{}, ctx.Err()
				}
			}
			var d net.Dialer
			dialed, err := d.DialContext(ctx, "unix", c.sockPath)
			if err != nil {
				lastErr = fmt.Errorf("onnx_turn: dial attempt %d: %w", attempt, err)
				continue
			}
			if !c.adopt(dialed) {
				return TurnPrediction{}, net.ErrClosed
			}
			conn = dialed
		}
		pred, err := c.roundTrip(ctx, conn, frame, deadline)
		if err == nil {
			return pred, nil
		}
		c.drop(conn)
		if cerr := ctx.Err(); cerr != nil {
			return TurnPrediction{}, cerr
		}
		lastErr = fmt.Errorf("onnx_turn: attempt %d: %w", attempt, err)
	}
	return TurnPrediction{}, lastErr
}

func (c *OnnxTurnClient) roundTrip(ctx context.Context, conn net.Conn, frame []byte, deadline time.Time) (TurnPrediction, error) {
	if err := conn.SetDeadline(deadline); err != nil {
		return TurnPrediction{}, err
	}
	stop := context.AfterFunc(ctx, func() { _ = conn.SetDeadline(time.Unix(1, 0)) })
	defer stop()
	if err := turnWriteFull(conn, frame); err != nil {
		return TurnPrediction{}, err
	}
	var buf [4]byte
	if err := turnReadFull(conn, buf[:]); err != nil {
		return TurnPrediction{}, err
	}
	return TurnPrediction{Probability: math.Float32frombits(binary.LittleEndian.Uint32(buf[:]))}, nil
}

func (c *OnnxTurnClient) current() net.Conn {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.conn
}

func (c *OnnxTurnClient) adopt(conn net.Conn) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		_ = conn.Close()
		return false
	}
	c.conn = conn
	return true
}

func (c *OnnxTurnClient) drop(conn net.Conn) {
	_ = conn.Close()
	c.mu.Lock()
	if c.conn == conn {
		c.conn = nil
	}
	c.mu.Unlock()
}

func (c *OnnxTurnClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	if c.conn != nil {
		err := c.conn.Close()
		c.conn = nil
		return err
	}
	return nil
}

func turnWriteFull(conn net.Conn, buf []byte) error {
	for len(buf) > 0 {
		n, err := conn.Write(buf)
		buf = buf[n:]
		if err != nil {
			return err
		}
	}
	return nil
}

func turnReadFull(conn net.Conn, buf []byte) error {
	total := 0
	for total < len(buf) {
		n, err := conn.Read(buf[total:])
		total += n
		if err != nil {
			return err
		}
	}
	return nil
}
