package vad

import (
	"encoding/binary"
	"fmt"
	"math"
	"net"
	"sync"
	"time"
)

// OnnxVADClient sends audio frames to the Rust onnx-worker over a Unix socket
// and receives voice confidence values back.
// Each client maintains a persistent connection — the Rust side creates a new
// SileroSession (with independent hidden state) per connection, so each
// OnnxVADClient instance has independent VAD state.
type OnnxVADClient struct {
	sockPath string
	conn     net.Conn
	mu       sync.Mutex
}

// NewOnnxVADClient dials the onnx-worker Unix socket and returns a client.
// sockPath is the path to the Unix socket (e.g. /tmp/onnx-worker-1234.sock).
func NewOnnxVADClient(sockPath string) (*OnnxVADClient, error) {
	conn, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: sockPath, Net: "unix"})
	if err != nil {
		return nil, fmt.Errorf("onnx_vad: dial %s: %w", sockPath, err)
	}
	return &OnnxVADClient{
		sockPath: sockPath,
		conn:     conn,
	}, nil
}

// VoiceConfidence sends audio bytes (int16 LE PCM) to the worker and returns
// the confidence score [0.0, 1.0].
// On transient error it retries up to 3 times with 50ms backoff, reconnecting
// each time. If all retries fail it returns 0.0 and the last error.
func (c *OnnxVADClient) VoiceConfidence(audio []byte, sampleRate int) (float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Wire layout: [u8 msg_type=0x01][u32 payload_len LE][u32 sample_rate LE][i16 PCM bytes...]
	payloadLen := uint32(4 + len(audio))

	buildFrame := func() []byte {
		frame := make([]byte, 1+4+4+len(audio))
		frame[0] = 0x01
		binary.LittleEndian.PutUint32(frame[1:5], payloadLen)
		binary.LittleEndian.PutUint32(frame[5:9], uint32(sampleRate))
		copy(frame[9:], audio)
		return frame
	}

	var lastErr error
	const maxRetries = 3
	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Reconnect if the connection is nil (initial state after a failed reconnect
		// on a prior attempt, or a nil conn passed in).
		if c.conn == nil {
			if attempt > 0 {
				time.Sleep(50 * time.Millisecond)
			}
			conn, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: c.sockPath, Net: "unix"})
			if err != nil {
				lastErr = fmt.Errorf("onnx_vad: dial attempt %d: %w", attempt, err)
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
		if err := writeFull(c.conn, frame); err != nil {
			c.conn.Close()
			c.conn = nil
			lastErr = fmt.Errorf("onnx_vad: write attempt %d: %w", attempt, err)
			continue
		}

		var buf [4]byte
		if err := readFull(c.conn, buf[:]); err != nil {
			c.conn.Close()
			c.conn = nil
			lastErr = fmt.Errorf("onnx_vad: read attempt %d: %w", attempt, err)
			continue
		}

		bits := binary.LittleEndian.Uint32(buf[:])
		return math.Float32frombits(bits), nil
	}

	return 0.0, lastErr
}

// Close closes the underlying Unix socket connection.
func (c *OnnxVADClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn != nil {
		err := c.conn.Close()
		c.conn = nil
		return err
	}
	return nil
}

// writeFull writes all bytes to conn, handling partial writes on stream sockets.
func writeFull(conn net.Conn, buf []byte) error {
	for len(buf) > 0 {
		n, err := conn.Write(buf)
		buf = buf[n:]
		if err != nil {
			return err
		}
	}
	return nil
}

// readFull reads exactly len(buf) bytes from conn.
func readFull(conn net.Conn, buf []byte) error {
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
