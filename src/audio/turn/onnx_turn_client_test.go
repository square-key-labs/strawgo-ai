package turn

import (
	"encoding/binary"
	"math"
	"net"
	"os"
	"testing"
)

// startMockTurnServer starts a Unix socket server that reads one request frame
// and responds with the provided f32 value. Returns the socket path and a
// channel that closes when the server goroutine exits.
func startMockTurnServer(t *testing.T, response float32) (sockPath string, done <-chan struct{}) {
	t.Helper()

	f, err := os.CreateTemp("", "mock-turn-*.sock")
	if err != nil {
		t.Fatalf("create temp file: %v", err)
	}
	sockPath = f.Name()
	f.Close()
	os.Remove(sockPath)

	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() {
		ln.Close()
		os.Remove(sockPath)
	})

	ch := make(chan struct{})
	go func() {
		defer close(ch)
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()

		// Drain the incoming frame.
		// Header: 1 byte msg_type + 4 bytes payload_len LE.
		var hdr [5]byte
		if err := turnReadFull(conn, hdr[:]); err != nil {
			return
		}
		payloadLen := binary.LittleEndian.Uint32(hdr[1:5])
		payload := make([]byte, payloadLen)
		if err := turnReadFull(conn, payload); err != nil {
			return
		}

		// Respond with f32 LE.
		var resp [4]byte
		binary.LittleEndian.PutUint32(resp[:], math.Float32bits(response))
		conn.Write(resp[:])
	}()

	return sockPath, ch
}

func TestOnnxTurnClient_Analyze(t *testing.T) {
	const want float32 = 0.42

	sockPath, done := startMockTurnServer(t, want)

	client, err := NewOnnxTurnClient(sockPath)
	if err != nil {
		t.Fatalf("NewOnnxTurnClient: %v", err)
	}
	defer client.Close()

	// 32 samples of silence (int16 LE zeros).
	audio := make([]byte, 64)

	pred, err := client.Analyze(audio, 16000, 500)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}

	if pred.Probability != want {
		t.Errorf("Probability = %v, want %v", pred.Probability, want)
	}

	<-done
}

func TestTurnPrediction_Threshold(t *testing.T) {
	// Confirm callers can interpret the threshold correctly.
	complete := TurnPrediction{Probability: 0.8}
	incomplete := TurnPrediction{Probability: 0.3}

	if complete.Probability <= 0.5 {
		t.Errorf("expected complete.Probability > 0.5, got %v", complete.Probability)
	}
	if incomplete.Probability > 0.5 {
		t.Errorf("expected incomplete.Probability <= 0.5, got %v", incomplete.Probability)
	}
}
