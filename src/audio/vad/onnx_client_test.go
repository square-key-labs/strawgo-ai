package vad

import (
	"encoding/binary"
	"math"
	"net"
	"os"
	"testing"
)

// startMockVADServer starts a Unix socket server that reads one request frame
// and responds with the provided f32 value. Returns the socket path and a
// channel that closes when the server goroutine exits.
func startMockVADServer(t *testing.T, response float32) (sockPath string, done <-chan struct{}) {
	t.Helper()

	f, err := os.CreateTemp("", "mock-vad-*.sock")
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

		// Drain the incoming frame (header + payload).
		// Header: 1 byte msg_type + 4 bytes payload_len LE.
		var hdr [5]byte
		if err := readFull(conn, hdr[:]); err != nil {
			return
		}
		payloadLen := binary.LittleEndian.Uint32(hdr[1:5])
		payload := make([]byte, payloadLen)
		if err := readFull(conn, payload); err != nil {
			return
		}

		// Respond with f32 LE.
		var resp [4]byte
		binary.LittleEndian.PutUint32(resp[:], math.Float32bits(response))
		conn.Write(resp[:])
	}()

	return sockPath, ch
}

func TestOnnxVADClient_VoiceConfidence(t *testing.T) {
	const want float32 = 0.42

	sockPath, done := startMockVADServer(t, want)

	client, err := NewOnnxVADClient(sockPath)
	if err != nil {
		t.Fatalf("NewOnnxVADClient: %v", err)
	}
	defer client.Close()

	// 32 samples of silence (int16 LE zeros).
	audio := make([]byte, 64)

	got, err := client.VoiceConfidence(audio, 16000)
	if err != nil {
		t.Fatalf("VoiceConfidence: %v", err)
	}

	if got != want {
		t.Errorf("VoiceConfidence = %v, want %v", got, want)
	}

	<-done
}

func TestOnnxVADClient_8kHz(t *testing.T) {
	const want float32 = 0.42

	sockPath, done := startMockVADServer(t, want)

	client, err := NewOnnxVADClient(sockPath)
	if err != nil {
		t.Fatalf("NewOnnxVADClient: %v", err)
	}
	defer client.Close()

	audio := make([]byte, 64)

	got, err := client.VoiceConfidence(audio, 8000)
	if err != nil {
		t.Fatalf("VoiceConfidence: %v", err)
	}

	if got != want {
		t.Errorf("VoiceConfidence = %v, want %v", got, want)
	}

	<-done
}
