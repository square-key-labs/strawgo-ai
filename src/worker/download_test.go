package worker

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestWorkerAssetName_CurrentPlatform(t *testing.T) {
	name, err := workerAssetName()

	switch {
	case runtime.GOOS == "darwin" && runtime.GOARCH == "amd64":
		// Intel Mac — no prebuilt binary available
		if err == nil {
			t.Fatal("expected error for darwin/amd64, got nil")
		}
		if !strings.Contains(err.Error(), "Intel Mac") {
			t.Errorf("error should mention Intel Mac, got: %v", err)
		}

	case runtime.GOOS == "darwin" && runtime.GOARCH == "arm64":
		if err != nil {
			t.Fatalf("unexpected error for darwin/arm64: %v", err)
		}
		if name != "onnx-worker-darwin-arm64" {
			t.Errorf("expected onnx-worker-darwin-arm64, got %q", name)
		}

	case runtime.GOOS == "linux" && runtime.GOARCH == "amd64":
		if err != nil {
			t.Fatalf("unexpected error for linux/amd64: %v", err)
		}
		if name != "onnx-worker-linux-amd64" {
			t.Errorf("expected onnx-worker-linux-amd64, got %q", name)
		}

	case runtime.GOOS == "linux" && runtime.GOARCH == "arm64":
		if err != nil {
			t.Fatalf("unexpected error for linux/arm64: %v", err)
		}
		if name != "onnx-worker-linux-arm64" {
			t.Errorf("expected onnx-worker-linux-arm64, got %q", name)
		}

	default:
		// Unsupported platform — must return an error
		if err == nil {
			t.Fatalf("expected error for %s/%s, got nil (name=%q)", runtime.GOOS, runtime.GOARCH, name)
		}
	}
}

func TestWorkerDownloadURL_Latest(t *testing.T) {
	if _, err := workerAssetName(); err != nil {
		t.Skipf("skipping on unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	for _, version := range []string{"", "latest"} {
		url, err := workerDownloadURL(version)
		if err != nil {
			t.Fatalf("workerDownloadURL(%q): %v", version, err)
		}
		want := "https://github.com/square-key-labs/strawgo-ai/releases/latest/download/onnx-worker-"
		if !strings.HasPrefix(url, want) {
			t.Errorf("URL %q does not start with %q", url, want)
		}
	}
}

func TestWorkerDownloadURL_Pinned(t *testing.T) {
	if _, err := workerAssetName(); err != nil {
		t.Skipf("skipping on unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	url, err := workerDownloadURL("v0.1.0")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "https://github.com/square-key-labs/strawgo-ai/releases/download/v0.1.0/onnx-worker-"
	if !strings.HasPrefix(url, want) {
		t.Errorf("URL %q does not start with %q", url, want)
	}
	if strings.Contains(url, "/latest/") {
		t.Errorf("pinned URL should not contain /latest/, got %q", url)
	}
}

func TestWorkerCachePath(t *testing.T) {
	path := workerCachePath()
	if !filepath.IsAbs(path) {
		t.Errorf("cache path should be absolute, got %q", path)
	}
	wantSuffix := filepath.Join(".cache", "strawgo", "bin", "onnx-worker")
	altSuffix := filepath.Join("strawgo", "bin", "onnx-worker") // temp fallback
	if !strings.HasSuffix(path, wantSuffix) && !strings.HasSuffix(path, altSuffix) {
		t.Errorf("unexpected cache path: %q", path)
	}
}

// TestEnsureWorkerBinary_CacheHit plants a fake binary in a temp HOME and
// verifies that EnsureWorkerBinary returns it without hitting the network.
func TestEnsureWorkerBinary_CacheHit(t *testing.T) {
	if _, err := workerAssetName(); err != nil {
		t.Skipf("skipping on unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	// If onnx-worker is on PATH, EnsureWorkerBinary returns early before ever
	// checking the cache — we can't exercise the cache branch in that case.
	if _, err := exec.LookPath(workerBinName); err == nil {
		t.Skip("onnx-worker is already on PATH; cache-hit branch not reachable")
	}

	// Point HOME at a temp dir so workerCachePath() builds there.
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	// Plant the fake binary at the expected cache location.
	cacheDir := filepath.Join(dir, workerBinDir)
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		t.Fatal(err)
	}
	cachedBin := filepath.Join(cacheDir, workerBinName)
	if err := os.WriteFile(cachedBin, []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatal(err)
	}

	got, err := EnsureWorkerBinary("latest")
	if err != nil {
		t.Fatalf("EnsureWorkerBinary with warm cache: %v", err)
	}
	if got != cachedBin {
		t.Errorf("expected %q, got %q", cachedBin, got)
	}
}
