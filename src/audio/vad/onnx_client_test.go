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

// TestOnnxVADClient_Reconnect verifies that the client reconnects automatically
// after its connection is torn down mid-call.
func TestOnnxVADClient_Reconnect(t *testing.T) {
	const want float32 = 0.77

	// Server 1: accepts connection but immediately closes it without responding.
	// This simulates a worker crash mid-request.
	f, err := os.CreateTemp("", "mock-vad-reconnect-*.sock")
	if err != nil {
		t.Fatalf("create temp file: %v", err)
	}
	sockPath := f.Name()
	f.Close()
	os.Remove(sockPath)

	// Use a shared listener path that server 1 and then server 2 will bind to.
	ln1, err := net.Listen("unix", sockPath)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { os.Remove(sockPath) })

	server1Ready := make(chan struct{})
	server1Done := make(chan struct{})
	go func() {
		defer close(server1Done)
		close(server1Ready)
		conn, err := ln1.Accept()
		if err != nil {
			return
		}
		// Close immediately — no response — simulates crash.
		conn.Close()
		ln1.Close()
	}()

	<-server1Ready

	// Connect client BEFORE server 1 dies.
	client, err := NewOnnxVADClient(sockPath)
	if err != nil {
		t.Fatalf("NewOnnxVADClient: %v", err)
	}
	defer client.Close()

	// Wait for server 1 to die so the socket is gone.
	<-server1Done

	// Remove stale socket file — ln1.Close() does not unlink it on Linux/macOS.
	os.Remove(sockPath)

	// Start server 2 on the same path before the client retry fires (~50ms window).
	ln2, err := net.Listen("unix", sockPath)
	if err != nil {
		t.Fatalf("listen server2: %v", err)
	}
	t.Cleanup(func() { ln2.Close() })

	go func() {
		conn, err := ln2.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		var hdr [5]byte
		if err := readFull(conn, hdr[:]); err != nil {
			return
		}
		payloadLen := binary.LittleEndian.Uint32(hdr[1:5])
		payload := make([]byte, payloadLen)
		readFull(conn, payload)
		var resp [4]byte
		binary.LittleEndian.PutUint32(resp[:], math.Float32bits(want))
		conn.Write(resp[:])
	}()

	// Now call VoiceConfidence — the first attempt will fail (connection was closed by server 1),
	// the client will nil out conn and retry, reconnecting to server 2.
	audio := make([]byte, 64)
	got, err := client.VoiceConfidence(audio, 16000)
	if err != nil {
		t.Fatalf("VoiceConfidence after reconnect: %v", err)
	}
	if got != want {
		t.Errorf("VoiceConfidence = %v, want %v", got, want)
	}
}
