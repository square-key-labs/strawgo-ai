package worker

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// localBinaryPath returns the path to the locally-compiled onnx-worker binary
// (produced by `cargo build --release` in the onnx-worker/ directory).
// Used by integration tests that need a real worker process.
func localBinaryPath(t *testing.T) string {
	t.Helper()
	// Go test CWD is the package directory (src/worker).
	// Repo root is 2 levels up: src/worker → src → repo root.
	abs, err := filepath.Abs("../..")
	if err != nil {
		t.Skipf("cannot resolve repo root: %v", err)
	}
	bin := filepath.Join(abs, "onnx-worker", "target", "release", "onnx-worker")
	if _, err := os.Stat(bin); err != nil {
		t.Skipf("local onnx-worker binary not found at %s (run: cd onnx-worker && cargo build --release)", bin)
	}
	return bin
}

// localModelPaths returns paths to the cached ONNX model files.
func localModelPaths(t *testing.T) (vadModel, turnModel string) {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("cannot determine home dir: %v", err)
	}
	vad := filepath.Join(home, ".cache", "strawgo", "models", "silero_vad.onnx")
	turn := filepath.Join(home, ".cache", "strawgo", "models", "smart-turn-v3.1-cpu.onnx")
	if _, err := os.Stat(vad); err != nil {
		t.Skipf("VAD model not found at %s (run once with internet to auto-download)", vad)
	}
	if _, err := os.Stat(turn); err != nil {
		t.Skipf("Smart Turn model not found at %s (run once with internet to auto-download)", turn)
	}
	return vad, turn
}

// TestSupervisor_StartStop verifies that Start() launches the worker, the socket
// appears, and Stop() removes the socket and shuts down cleanly.
func TestSupervisor_StartStop(t *testing.T) {
	bin := localBinaryPath(t)
	localModelPaths(t) // skip if models absent — Start() downloads them, but that takes time

	s, err := Start(bin)
	if err != nil {
		t.Fatalf("Start(): %v", err)
	}

	sockPath := s.SocketPath()
	if sockPath == "" {
		t.Fatal("SocketPath() returned empty string")
	}
	if _, err := os.Stat(sockPath); err != nil {
		t.Fatalf("socket file should exist after Start(), stat: %v", err)
	}

	s.Stop()

	// Socket must be removed by Stop()
	if _, err := os.Stat(sockPath); err == nil {
		t.Error("socket file still exists after Stop() — expected it to be removed")
	}
}

// TestSupervisor_CrashRecovery sends SIGKILL to the worker and verifies that
// WatchAndRestart restarts it within the expected window (backoff=500ms + 5s
// readiness timeout = ~5.5s max).
func TestSupervisor_CrashRecovery(t *testing.T) {
	bin := localBinaryPath(t)
	localModelPaths(t)

	s, err := Start(bin)
	if err != nil {
		t.Fatalf("Start(): %v", err)
	}
	defer s.Stop()

	sockPath := s.SocketPath()

	// Kill worker, which removes the socket file.
	s.KillWorkerForTesting()

	// Wait for socket to disappear (confirming the kill took effect).
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(sockPath); err != nil {
			break // socket gone — kill registered
		}
		time.Sleep(20 * time.Millisecond)
	}

	// Now wait for the supervisor to restart and socket to reappear.
	// Max: 500ms backoff + 5s readiness + margin = 8s
	restartDeadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(restartDeadline) {
		if _, err := os.Stat(sockPath); err == nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	if _, err := os.Stat(sockPath); err != nil {
		t.Errorf("socket did not reappear after crash within 8s: %v", err)
	}
}
