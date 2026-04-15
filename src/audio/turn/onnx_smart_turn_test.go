package turn

import (
	"encoding/binary"
	"math"
	"net"
	"os"
	"testing"
)

// startMockTurnServerOnce starts a mock turn server that handles one request
// and responds with the given probability.
func startMockTurnServerOnce(t *testing.T, probability float32) (sockPath string) {
	t.Helper()

	f, err := os.CreateTemp("", "mock-smart-turn-*.sock")
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

	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		// Drain header + payload
		var hdr [5]byte
		if err := turnReadFull(conn, hdr[:]); err != nil {
			return
		}
		payloadLen := binary.LittleEndian.Uint32(hdr[1:5])
		payload := make([]byte, payloadLen)
		turnReadFull(conn, payload)
		// Respond with f32 LE probability
		var resp [4]byte
		binary.LittleEndian.PutUint32(resp[:], math.Float32bits(probability))
		conn.Write(resp[:])
	}()

	return sockPath
}

func TestOnnxSmartTurn_DialFail(t *testing.T) {
	_, err := NewOnnxSmartTurn(OnnxSmartTurnConfig{
		SockPath: "/tmp/nonexistent-onnx-worker.sock",
	})
	if err == nil {
		t.Fatal("expected error when socket does not exist, got nil")
	}
}

func TestOnnxSmartTurn_AnalyzeNoAudio(t *testing.T) {
	// Server won't receive any call because GetAudioSegment returns empty
	sockPath := startMockTurnServerOnce(t, 0.9)

	st, err := NewOnnxSmartTurn(OnnxSmartTurnConfig{SockPath: sockPath})
	if err != nil {
		t.Fatalf("NewOnnxSmartTurn: %v", err)
	}
	defer st.Close()

	state, metrics, err := st.AnalyzeEndOfTurn()
	if err != nil {
		t.Fatalf("AnalyzeEndOfTurn: %v", err)
	}
	if state != TurnIncomplete {
		t.Errorf("expected TurnIncomplete with no audio, got %v", state)
	}
	if metrics != nil {
		t.Errorf("expected nil metrics with no audio, got %+v", metrics)
	}
}

func TestOnnxSmartTurn_AnalyzeTurnComplete(t *testing.T) {
	const prob float32 = 0.82 // > 0.5 → TurnComplete

	sockPath := startMockTurnServerOnce(t, prob)

	st, err := NewOnnxSmartTurn(OnnxSmartTurnConfig{SockPath: sockPath})
	if err != nil {
		t.Fatalf("NewOnnxSmartTurn: %v", err)
	}
	defer st.Close()

	// Buffer 1s of fake int16 PCM at 16kHz (32000 bytes) as "speech"
	audio := make([]byte, 32000)
	st.AppendAudio(audio, true)

	state, metrics, err := st.AnalyzeEndOfTurn()
	if err != nil {
		t.Fatalf("AnalyzeEndOfTurn: %v", err)
	}
	if state != TurnComplete {
		t.Errorf("expected TurnComplete (prob=%.2f > 0.5), got %v", prob, state)
	}
	if metrics == nil {
		t.Fatal("expected non-nil metrics")
	}
	if !metrics.IsComplete {
		t.Errorf("metrics.IsComplete should be true")
	}
	if math.Abs(float64(metrics.Probability)-float64(prob)) > 0.001 {
		t.Errorf("metrics.Probability = %.4f, want %.4f", metrics.Probability, prob)
	}
}

func TestOnnxSmartTurn_AnalyzeTurnIncomplete(t *testing.T) {
	const prob float32 = 0.23 // <= 0.5 → TurnIncomplete

	sockPath := startMockTurnServerOnce(t, prob)

	st, err := NewOnnxSmartTurn(OnnxSmartTurnConfig{SockPath: sockPath})
	if err != nil {
		t.Fatalf("NewOnnxSmartTurn: %v", err)
	}
	defer st.Close()

	audio := make([]byte, 32000)
	st.AppendAudio(audio, true)

	state, metrics, err := st.AnalyzeEndOfTurn()
	if err != nil {
		t.Fatalf("AnalyzeEndOfTurn: %v", err)
	}
	if state != TurnIncomplete {
		t.Errorf("expected TurnIncomplete (prob=%.2f <= 0.5), got %v", prob, state)
	}
	if metrics == nil {
		t.Fatal("expected non-nil metrics")
	}
	if metrics.IsComplete {
		t.Errorf("metrics.IsComplete should be false")
	}
}

func TestOnnxSmartTurn_Close(t *testing.T) {
	sockPath := startMockTurnServerOnce(t, 0.5)

	st, err := NewOnnxSmartTurn(OnnxSmartTurnConfig{SockPath: sockPath})
	if err != nil {
		t.Fatalf("NewOnnxSmartTurn: %v", err)
	}

	if err := st.Close(); err != nil {
		t.Errorf("Close() returned error: %v", err)
	}
	// Second close should be safe
	if err := st.Close(); err != nil {
		t.Errorf("second Close() returned error: %v", err)
	}
}
