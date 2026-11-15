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

const readyTimeout = 15 * time.Second

type Supervisor struct {
	binaryPath    string
	vadModelPath  string
	turnModelPath string
	sockPath      string
	cmd           *exec.Cmd
	mu            sync.Mutex
	stopCh        chan struct{} // closed on Stop()
	stoppedCh     chan struct{} // closed when WatchAndRestart exits
	watchOnce     sync.Once     // ensures WatchAndRestart goroutine runs at most once
}

func Start(binaryPath string) (*Supervisor, error) {
	if binaryPath == "" {
		var err error
		binaryPath, err = EnsureWorkerBinary("latest")
		if err != nil {
			return nil, fmt.Errorf("onnx-worker binary unavailable: %w", err)
		}
	}

	vadModel, err := models.EnsureModel(models.SileroVADURL, models.SileroVADFile)
	if err != nil {
		return nil, fmt.Errorf("ensure VAD model: %w", err)
	}
	turnModel, err := models.EnsureModel(models.SmartTurnURL, models.SmartTurnFile)
	if err != nil {
		return nil, fmt.Errorf("ensure turn model: %w", err)
	}

	sockPath := filepath.Join(os.TempDir(), fmt.Sprintf("onnx-worker-%d.sock", os.Getpid()))
	cmd, err := launch(binaryPath, vadModel, turnModel, sockPath)
	if err != nil {
		return nil, err
	}

	s := &Supervisor{
		binaryPath:    binaryPath,
		vadModelPath:  vadModel,
		turnModelPath: turnModel,
		sockPath:      sockPath,
		cmd:           cmd,
		stopCh:        make(chan struct{}),
		stoppedCh:     make(chan struct{}),
	}

	go s.WatchAndRestart()

	return s, nil
}

func (s *Supervisor) SocketPath() string {
	return s.sockPath
}

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
		doneCh := make(chan error, 1)
		s.mu.Lock()
		cmd := s.cmd
		s.mu.Unlock()

		go func() { doneCh <- cmd.Wait() }()

		select {
		case <-s.stopCh:
			return
		case err := <-doneCh:
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
			select {
			case <-s.stopCh:
				return
			case <-time.After(backoff):
			}
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

func launch(binaryPath, vadModel, turnModel, sockPath string) (*exec.Cmd, error) {
	_ = os.Remove(sockPath)
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
	deadline := time.Now().Add(readyTimeout)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(sockPath); err == nil {
			return cmd, nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	_ = cmd.Process.Kill()
	_ = cmd.Wait()
	return nil, fmt.Errorf("onnx-worker socket not ready after %s: %s", readyTimeout, sockPath)
}

func (s *Supervisor) restart() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	cmd, err := launch(s.binaryPath, s.vadModelPath, s.turnModelPath, s.sockPath)
	if err != nil {
		return err
	}
	s.cmd = cmd
	return nil
}

func (s *Supervisor) KillWorkerForTesting() {
	s.mu.Lock()
	proc := s.cmd.Process
	s.mu.Unlock()
	if proc != nil {
		proc.Signal(syscall.SIGKILL)
	}
}

func (s *Supervisor) Stop() {
	close(s.stopCh)

	s.mu.Lock()
	proc := s.cmd.Process
	s.mu.Unlock()

	if proc != nil {
		proc.Signal(syscall.SIGTERM)

		done := make(chan struct{})
		go func() {
			proc.Wait() //nolint:errcheck
			close(done)
		}()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			proc.Kill()
		}
	}

	<-s.stoppedCh
	os.Remove(s.sockPath)
}
