package vad

import (
	"encoding/binary"
	"fmt"
	"math"
	"net"
	"sync"
	"time"
)

type OnnxVADClient struct {
	sockPath string
	conn     net.Conn
	mu       sync.Mutex
	frame    []byte
}

func NewOnnxVADClient(sockPath string) (*OnnxVADClient, error) {
	conn, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: sockPath, Net: "unix"})
	if err != nil {
		return nil, fmt.Errorf("onnx_vad: dial %s: %w", sockPath, err)
	}
	return &OnnxVADClient{sockPath: sockPath, conn: conn}, nil
}

func (c *OnnxVADClient) VoiceConfidence(audio []byte, sampleRate int) (float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	n := 9 + len(audio)
	if cap(c.frame) < n {
		c.frame = make([]byte, n)
	}
	frame := c.frame[:n]
	frame[0] = 0x01
	binary.LittleEndian.PutUint32(frame[1:5], uint32(4+len(audio)))
	binary.LittleEndian.PutUint32(frame[5:9], uint32(sampleRate))
	copy(frame[9:], audio)

	var lastErr error
	const maxRetries = 3
	for attempt := 0; attempt <= maxRetries; attempt++ {
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

		if err := c.conn.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
			c.drop()
			lastErr = err
			continue
		}
		if err := writeFull(c.conn, frame); err != nil {
			c.drop()
			lastErr = fmt.Errorf("onnx_vad: write attempt %d: %w", attempt, err)
			continue
		}

		var buf [4]byte
		if err := readFull(c.conn, buf[:]); err != nil {
			c.drop()
			lastErr = fmt.Errorf("onnx_vad: read attempt %d: %w", attempt, err)
			continue
		}
		return math.Float32frombits(binary.LittleEndian.Uint32(buf[:])), nil
	}

	return 0.0, lastErr
}

func (c *OnnxVADClient) drop() {
	c.conn.Close()
	c.conn = nil
}

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
