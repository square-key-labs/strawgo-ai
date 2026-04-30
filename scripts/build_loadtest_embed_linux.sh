#!/usr/bin/env bash
# Build cmd/loadtest-embed for Linux x86_64.
#
# Cross-compiling cgo from macOS to linux/amd64 needs a glibc cross-toolchain
# (musl-cross / x86_64-linux-gnu-gcc / `zig cc`) which most macs don't have.
# This script tries the available paths in order:
#
#   1. If we're already on Linux x86_64: build natively.
#   2. If `zig` is on PATH: use `CC="zig cc -target x86_64-linux-gnu"`.
#   3. If Docker is available: build in a golang:bookworm container.
#   4. Otherwise: fail with instructions.
#
# Output: ./loadtest-embed-linux (Linux ELF amd64).
#
# Required at runtime on the target VM:
#   - libonnxruntime.so.1.25 (extract from
#     onnxruntime-linux-x64-1.25.1.tgz, place at /usr/local/lib or pass
#     -lib explicitly)
#   - testdata/models/silero_vad.onnx
#   - testdata/sine_440_500ms_16k.pcm
#
# Usage:
#   ./scripts/build_loadtest_embed_linux.sh
#   ./loadtest-embed-linux -lib /usr/local/lib/libonnxruntime.so -model ~/silero_vad.onnx -dur 20 -levels 1,5,10,25,50,100,200

set -euo pipefail

cd "$(dirname "$0")/.."
OUT=./loadtest-embed-linux

build_native() {
  echo "→ native linux/amd64 build"
  CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -o "$OUT" ./cmd/loadtest-embed
}

build_zig() {
  echo "→ zig cc cross-build (linux/amd64, glibc)"
  CC="zig cc -target x86_64-linux-gnu" \
  CXX="zig c++ -target x86_64-linux-gnu" \
  CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
    go build -o "$OUT" ./cmd/loadtest-embed
}

build_docker() {
  echo "→ docker cross-build via golang:1.25-bookworm"
  # Map this repo into the container, run `go build`, write binary back into ./
  docker run --rm \
    -v "$(pwd)":/src \
    -w /src \
    -e CGO_ENABLED=1 \
    -e GOOS=linux \
    -e GOARCH=amd64 \
    -e GOMODCACHE=/src/.docker-gomod \
    -e GOCACHE=/src/.docker-gocache \
    --platform=linux/amd64 \
    golang:1.25-bookworm \
    bash -c "apt-get update -qq && apt-get install -y -qq build-essential >/dev/null && \
             go build -o $OUT ./cmd/loadtest-embed"
}

# 1. Native?
if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
  build_native
  echo "✓ built $OUT"
  file "$OUT" || true
  exit 0
fi

# 2. zig?
if command -v zig >/dev/null 2>&1; then
  build_zig
  echo "✓ built $OUT (via zig)"
  file "$OUT" || true
  exit 0
fi

# 3. docker?  (must have daemon running, not just CLI installed)
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  build_docker
  echo "✓ built $OUT (via docker)"
  file "$OUT" || true
  exit 0
fi

echo "ERROR: no cross-compile path available." >&2
echo "Install one of:" >&2
echo "  brew install zig          # then re-run" >&2
echo "  Docker Desktop            # then re-run" >&2
echo "Or copy this repo to the Linux VM and run:" >&2
echo "  CGO_ENABLED=1 go build ./cmd/loadtest-embed" >&2
exit 1
