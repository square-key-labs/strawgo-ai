package models

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/logger"
)

const (
	cacheSubDir = ".cache/strawgo/models"

	// SileroVADURL is the HuggingFace download URL for the Silero VAD ONNX model.
	SileroVADURL  = "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx"
	SileroVADFile = "silero_vad.onnx"

	// SmartTurnURL is the HuggingFace download URL for the Smart Turn v3.1 ONNX model (third-party hosted).
	SmartTurnURL  = "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.1-cpu.onnx"
	SmartTurnFile = "smart-turn-v3.1-cpu.onnx"
)

var downloadMu sync.Mutex

// CacheDir returns the default model cache directory (~/.cache/strawgo/models/).
func CacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "strawgo", "models")
	}
	return filepath.Join(home, cacheSubDir)
}

// EnsureModel checks the cache for a model file and downloads it if missing.
// Returns the path to the cached model file.
func EnsureModel(url, filename string) (string, error) {
	cacheDir := CacheDir()
	modelPath := filepath.Join(cacheDir, filename)

	// Fast path: already cached
	if _, err := os.Stat(modelPath); err == nil {
		return modelPath, nil
	}

	// Serialize downloads to avoid duplicate concurrent fetches
	downloadMu.Lock()
	defer downloadMu.Unlock()

	// Double-check after acquiring lock
	if _, err := os.Stat(modelPath); err == nil {
		return modelPath, nil
	}

	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create cache directory %s: %w", cacheDir, err)
	}

	logger.Info("[Models] Downloading %s from %s ...", filename, url)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to download %s: %w", filename, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to download %s: HTTP %d", filename, resp.StatusCode)
	}

	// Write to temp file, then atomic rename
	tmpPath := modelPath + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}

	written, err := io.Copy(f, resp.Body)
	f.Close()
	if err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("failed to write model data: %w", err)
	}

	// Check for truncated download
	if cl := resp.ContentLength; cl > 0 && written != cl {
		os.Remove(tmpPath)
		return "", fmt.Errorf("download truncated: expected %d bytes, got %d", cl, written)
	}

	if err := os.Rename(tmpPath, modelPath); err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("failed to finalize model file: %w", err)
	}

	logger.Info("[Models] Downloaded %s (%d bytes) → %s", filename, written, modelPath)
	return modelPath, nil
}
