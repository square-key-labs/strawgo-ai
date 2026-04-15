package worker

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/models"
)

// Supervisor manages the lifecycle of the onnx-worker subprocess.
type Supervisor struct {
	binaryPath    string
	vadModelPath  string
	turnModelPath string
	sockPath      string
	cmd           *exec.Cmd
	mu            sync.Mutex
	stopCh        chan struct{} // closed on Stop()
	stoppedCh     chan struct{} // closed when WatchAndRestart exits
	watchOnce     sync.Once   // ensures WatchAndRestart goroutine runs at most once
}

// Start starts the onnx-worker process and waits for it to be ready.
//
// binaryPath is the path to the onnx-worker binary. Pass an empty string to
// let Start resolve it automatically in this order:
//
//  1. PATH — uses the system-installed "onnx-worker" if present.
//  2. Cache — uses a previously auto-downloaded binary at
//     ~/.cache/strawgo/bin/onnx-worker.
//  3. Download — fetches the latest release binary for the current OS/arch
//     from GitHub Releases and caches it for future calls.
//
// ONNX model files (silero_vad.onnx, smart-turn-v3.1-cpu.onnx) are always
// downloaded automatically to ~/.cache/strawgo/models/ if not already present,
// regardless of how the binary was resolved.
func Start(binaryPath string) (*Supervisor, error) {
	// 1. Resolve binary — auto-download if not supplied and not in PATH/cache
	if binaryPath == "" {
		var err error
		binaryPath, err = EnsureWorkerBinary("latest")
		if err != nil {
			return nil, fmt.Errorf("onnx-worker binary unavailable: %w", err)
		}
	}

	// 2. Ensure models are downloaded
	vadModel, err := models.EnsureModel(models.SileroVADURL, models.SileroVADFile)
	if err != nil {
		return nil, fmt.Errorf("ensure VAD model: %w", err)
	}
	turnModel, err := models.EnsureModel(models.SmartTurnURL, models.SmartTurnFile)
	if err != nil {
		return nil, fmt.Errorf("ensure turn model: %w", err)
	}

	// 3. Build socket path
	sockPath := filepath.Join(os.TempDir(), fmt.Sprintf("onnx-worker-%d.sock", os.Getpid()))

	// 4. Launch the process
	cmd := exec.Command(binaryPath,
		"--vad-model", vadModel,
		"--turn-model", turnModel,
		"--socket", sockPath,
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start onnx-worker: %w", err)
	}

	// 5. Wait for readiness (socket file appears)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(sockPath); err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := os.Stat(sockPath); err != nil {
		cmd.Process.Kill()
		_ = cmd.Wait() // reap to avoid zombie
		return nil, fmt.Errorf("onnx-worker socket not ready after 5s: %s", sockPath)
	}

	// 6. Return initialised Supervisor
	s := &Supervisor{
		binaryPath:    binaryPath,
		vadModelPath:  vadModel,
		turnModelPath: turnModel,
		sockPath:      sockPath,
		cmd:           cmd,
		stopCh:        make(chan struct{}),
		stoppedCh:     make(chan struct{}),
	}

	// Auto-start the crash-recovery watcher so Stop() can safely wait on stoppedCh.
	go s.WatchAndRestart()

	return s, nil
}

// SocketPath returns the Unix socket path that clients should connect to.
func (s *Supervisor) SocketPath() string {
	return s.sockPath
}

// WatchAndRestart watches the worker process and restarts it on crash.
// It is safe to call multiple times; only the first call runs the watcher.
// Stops when Stop() is called. Max 3 restarts with backoff: 500ms, 1s, 2s.
func (s *Supervisor) WatchAndRestart() {
	started := false
	s.watchOnce.Do(func() { started = true })
	if !started {
		return // already running via auto-start in Start()
	}
	defer close(s.stoppedCh)

	backoffs := []time.Duration{500 * time.Millisecond, time.Second, 2 * time.Second}
	attempts := 0

	for {
		// Wait for process exit or stop signal
		doneCh := make(chan error, 1)
		s.mu.Lock()
		cmd := s.cmd
		s.mu.Unlock()

		go func() { doneCh <- cmd.Wait() }()

		select {
		case <-s.stopCh:
			return
		case err := <-doneCh:
			// Check whether Stop() was called concurrently
			select {
			case <-s.stopCh:
				return
			default:
			}

			logger.Error("[Worker] onnx-worker exited unexpectedly: %v", err)
			if attempts >= len(backoffs) {
				logger.Error("[Worker] onnx-worker restart limit reached, giving up")
				return
			}
			backoff := backoffs[attempts]
			attempts++
			logger.Info("[Worker] Restarting onnx-worker in %v (attempt %d)", backoff, attempts)
			// Interruptible backoff — honour Stop() during the wait window.
			select {
			case <-s.stopCh:
				return
			case <-time.After(backoff):
			}
			// Re-check stop after backoff before spawning a new child.
			select {
			case <-s.stopCh:
				return
			default:
			}
			if err := s.restart(); err != nil {
				logger.Error("[Worker] Failed to restart onnx-worker: %v", err)
				return
			}
		}
	}
}

// restart relaunches the onnx-worker binary and waits for the socket to appear.
// Must not be called with s.mu held.
func (s *Supervisor) restart() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Clean up stale socket so the readiness check starts fresh
	os.Remove(s.sockPath)

	cmd := exec.Command(s.binaryPath,
		"--vad-model", s.vadModelPath,
		"--turn-model", s.turnModelPath,
		"--socket", s.sockPath,
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	s.cmd = cmd

	// Wait for readiness
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(s.sockPath); err == nil {
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	cmd.Process.Kill()
	_ = cmd.Wait() // reap to avoid zombie
	return fmt.Errorf("worker did not become ready after restart")
}

// KillWorkerForTesting sends SIGKILL to the worker process without triggering Stop.
// Used in integration tests to simulate a crash and verify supervisor restart.
func (s *Supervisor) KillWorkerForTesting() {
	s.mu.Lock()
	proc := s.cmd.Process
	s.mu.Unlock()
	if proc != nil {
		proc.Signal(syscall.SIGKILL)
	}
}

// Stop sends SIGTERM to the worker, waits up to 2 s, then SIGKILLs it,
// and waits for WatchAndRestart to return.
func (s *Supervisor) Stop() {
	close(s.stopCh)

	s.mu.Lock()
	proc := s.cmd.Process
	s.mu.Unlock()

	if proc != nil {
		// Graceful shutdown attempt
		proc.Signal(syscall.SIGTERM)

		done := make(chan struct{})
		go func() {
			proc.Wait() //nolint:errcheck
			close(done)
		}()

		select {
		case <-done:
			// Exited cleanly within the grace period
		case <-time.After(2 * time.Second):
			// Force kill after grace period
			proc.Kill()
		}
	}

	<-s.stoppedCh
	os.Remove(s.sockPath)
}
