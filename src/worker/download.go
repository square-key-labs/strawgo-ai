package worker

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/logger"
)

const (
	githubOwner   = "square-key-labs"
	githubRepo    = "strawgo-ai"
	workerBinDir  = ".cache/strawgo/bin"
	workerBinName = "onnx-worker"
)

var workerDownloadMu sync.Mutex

// workerAssetName returns the GitHub Releases asset filename for the current
// OS and CPU architecture, e.g. "onnx-worker-linux-amd64".
func workerAssetName() (string, error) {
	osMap := map[string]string{
		"darwin": "darwin",
		"linux":  "linux",
	}
	archMap := map[string]string{
		"amd64": "amd64",
		"arm64": "arm64",
	}

	goos, ok := osMap[runtime.GOOS]
	if !ok {
		return "", fmt.Errorf(
			"onnx-worker has no pre-built binary for OS %q (supported: linux, darwin); "+
				"build from source: cd onnx-worker && cargo build --release",
			runtime.GOOS,
		)
	}
	goarch, ok := archMap[runtime.GOARCH]
	if !ok {
		return "", fmt.Errorf(
			"onnx-worker has no pre-built binary for arch %q (supported: amd64, arm64); "+
				"build from source: cd onnx-worker && cargo build --release",
			runtime.GOARCH,
		)
	}

	// darwin/amd64 (Intel Mac) has no pre-built binary — macos-13 runners are
	// unavailable on our CI. Build from source:
	//   cd onnx-worker && cargo build --release
	if goos == "darwin" && goarch == "amd64" {
		return "", fmt.Errorf(
			"onnx-worker has no pre-built binary for Intel Mac (darwin/amd64); " +
				"build from source: cd onnx-worker && cargo build --release",
		)
	}

	return fmt.Sprintf("onnx-worker-%s-%s", goos, goarch), nil
}

// workerDownloadURL constructs the GitHub Releases download URL.
// version may be empty or "latest" to get the latest release, or a specific
// tag such as "v0.2.0".
func workerDownloadURL(version string) (string, error) {
	asset, err := workerAssetName()
	if err != nil {
		return "", err
	}

	if version == "" || version == "latest" {
		return fmt.Sprintf(
			"https://github.com/%s/%s/releases/latest/download/%s",
			githubOwner, githubRepo, asset,
		), nil
	}
	return fmt.Sprintf(
		"https://github.com/%s/%s/releases/download/%s/%s",
		githubOwner, githubRepo, version, asset,
	), nil
}

// workerCachePath returns the path where the onnx-worker binary is cached
// (~/.cache/strawgo/bin/onnx-worker). Falls back to the OS temp directory if
// the user home directory cannot be determined.
func workerCachePath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "strawgo", "bin", workerBinName)
	}
	return filepath.Join(home, workerBinDir, workerBinName)
}

// EnsureWorkerBinary returns a path to the onnx-worker binary, downloading it
// from GitHub Releases if necessary. Resolution order:
//
//  1. PATH — if "onnx-worker" is already on the system PATH, use it as-is.
//  2. Cache — ~/.cache/strawgo/bin/onnx-worker if previously downloaded.
//  3. Download — fetch the binary for the current OS/arch from GitHub Releases
//     and cache it for future calls.
//
// version controls which release is fetched. Pass "" or "latest" for the most
// recent release, or a specific tag like "v0.2.0" to pin a version.
//
// Downloads are serialised by a package-level mutex so concurrent calls from
// multiple goroutines do not trigger duplicate fetches.
func EnsureWorkerBinary(version string) (string, error) {
	// 1. PATH fast path — no download needed
	if path, err := exec.LookPath(workerBinName); err == nil {
		return path, nil
	}

	// 2. Cache fast path — already downloaded
	cachePath := workerCachePath()
	if _, err := os.Stat(cachePath); err == nil {
		return cachePath, nil
	}

	// 3. Download — serialise to avoid concurrent fetches
	workerDownloadMu.Lock()
	defer workerDownloadMu.Unlock()

	// Double-check after acquiring the lock
	if _, err := os.Stat(cachePath); err == nil {
		return cachePath, nil
	}

	url, err := workerDownloadURL(version)
	if err != nil {
		return "", err
	}

	logger.Info("[Worker] onnx-worker not found — downloading from %s", url)

	if err := os.MkdirAll(filepath.Dir(cachePath), 0755); err != nil {
		return "", fmt.Errorf("create worker cache dir: %w", err)
	}

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return "", fmt.Errorf("download onnx-worker: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf(
			"download onnx-worker: HTTP %d from %s\n"+
				"hint: check that a release exists for this OS/arch at "+
				"https://github.com/%s/%s/releases",
			resp.StatusCode, url, githubOwner, githubRepo,
		)
	}

	// Write to a temp file then atomically rename — avoids a corrupt binary
	// if the process is interrupted mid-download.
	tmpPath := cachePath + ".tmp"
	f, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0755)
	if err != nil {
		return "", fmt.Errorf("create temp file for onnx-worker: %w", err)
	}

	written, err := io.Copy(f, resp.Body)
	f.Close()
	if err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("write onnx-worker binary: %w", err)
	}

	if cl := resp.ContentLength; cl > 0 && written != cl {
		os.Remove(tmpPath)
		return "", fmt.Errorf(
			"download truncated: expected %d bytes, got %d", cl, written,
		)
	}

	if err := os.Rename(tmpPath, cachePath); err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("finalize onnx-worker binary: %w", err)
	}

	logger.Info("[Worker] Downloaded onnx-worker (%d bytes) → %s", written, cachePath)
	return cachePath, nil
}
