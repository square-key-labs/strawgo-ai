package turn

import (
	"encoding/binary"
	"fmt"
	"math"
	"net"
	"sync"
	"time"
)

// TurnPrediction holds the result of a Smart Turn inference.
type TurnPrediction struct {
	Probability float32 // 0.0–1.0, >0.5 means turn complete
}

// OnnxTurnClient sends audio to the onnx-worker for Smart Turn inference.
type OnnxTurnClient struct {
	sockPath string
	conn     net.Conn
	mu       sync.Mutex
}

// NewOnnxTurnClient dials the onnx-worker Unix socket.
func NewOnnxTurnClient(sockPath string) (*OnnxTurnClient, error) {
	conn, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: sockPath, Net: "unix"})
	if err != nil {
		return nil, fmt.Errorf("onnx_turn: dial %s: %w", sockPath, err)
	}
	return &OnnxTurnClient{
		sockPath: sockPath,
		conn:     conn,
	}, nil
}

// Analyze sends audio and metadata to the worker and returns a TurnPrediction.
// Returns TurnPrediction with Probability from the worker.
// Caller interprets Probability > 0.5 as TurnComplete.
// On error returns TurnPrediction{Probability: 0.0} (treated as TurnIncomplete by caller).
func (c *OnnxTurnClient) Analyze(audio []byte, sampleRate int, speechStartMs int) (TurnPrediction, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	payloadLen := uint32(4 + 4 + len(audio))

	buildFrame := func() []byte {
		frame := make([]byte, 1+4+4+4+len(audio))
		frame[0] = 0x02
		binary.LittleEndian.PutUint32(frame[1:5], payloadLen)
		binary.LittleEndian.PutUint32(frame[5:9], uint32(sampleRate))
		binary.LittleEndian.PutUint32(frame[9:13], uint32(speechStartMs))
		copy(frame[13:], audio)
		return frame
	}

	var lastErr error
	const maxRetries = 3
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if c.conn == nil {
			if attempt > 0 {
				time.Sleep(50 * time.Millisecond)
			}
			conn, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: c.sockPath, Net: "unix"})
			if err != nil {
				lastErr = fmt.Errorf("onnx_turn: dial attempt %d: %w", attempt, err)
				continue
			}
			c.conn = conn
		}

		frame := buildFrame()
		if err := c.conn.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
			c.conn.Close()
			c.conn = nil
			lastErr = err
			continue
		}
		if err := turnWriteFull(c.conn, frame); err != nil {
			c.conn.Close()
			c.conn = nil
			lastErr = fmt.Errorf("onnx_turn: write attempt %d: %w", attempt, err)
			continue
		}

		var buf [4]byte
		if err := turnReadFull(c.conn, buf[:]); err != nil {
			c.conn.Close()
			c.conn = nil
			lastErr = fmt.Errorf("onnx_turn: read attempt %d: %w", attempt, err)
			continue
		}

		bits := binary.LittleEndian.Uint32(buf[:])
		prob := math.Float32frombits(bits)
		return TurnPrediction{Probability: prob}, nil
	}

	return TurnPrediction{Probability: 0.0}, lastErr
}

// Close closes the connection.
func (c *OnnxTurnClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn != nil {
		err := c.conn.Close()
		c.conn = nil
		return err
	}
	return nil
}

// turnWriteFull writes all bytes to conn, handling partial writes on stream sockets.
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

// turnReadFull reads exactly len(buf) bytes from conn.
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
