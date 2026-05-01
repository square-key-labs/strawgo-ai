#!/usr/bin/env bash
# Build cmd/loadtest-pipeline for Linux x86_64.
# Mirrors scripts/build_loadtest_embed_linux.sh.
#
# Output: ./loadtest-pipeline-linux (Linux ELF amd64).
#
# Required at runtime on the target VM:
#   - libonnxruntime.so.1.25 (extract from ORT 1.25.1 tgz)
#   - testdata/models/silero_vad.onnx
#   - testdata/models/gtcrn_simple.onnx
#   - testdata/models/smart-turn-v3.1-cpu.onnx (or pass -smart-turn-model)
#   - testdata/sine_440_500ms_16k.pcm
#
# Usage:
#   ./scripts/build_loadtest_pipeline_linux.sh
#   ./loadtest-pipeline-linux -lib /usr/local/lib/libonnxruntime.so -dur 20 -levels 1,5,10,25,50,100,200

set -euo pipefail

cd "$(dirname "$0")/.."
OUT=./loadtest-pipeline-linux

build_native() {
  echo "→ native linux/amd64 build"
  CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -o "$OUT" ./cmd/loadtest-pipeline
}

build_zig() {
  echo "→ zig cc cross-build (linux/amd64, glibc)"
  CC="zig cc -target x86_64-linux-gnu" \
  CXX="zig c++ -target x86_64-linux-gnu" \
  CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
    go build -o "$OUT" ./cmd/loadtest-pipeline
}

build_docker() {
  echo "→ docker cross-build via golang:1.25-bookworm"
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
             go build -o $OUT ./cmd/loadtest-pipeline"
}

if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
  build_native
  echo "✓ built $OUT"
  file "$OUT" || true
  exit 0
fi

if command -v zig >/dev/null 2>&1; then
  build_zig
  echo "✓ built $OUT (via zig)"
  file "$OUT" || true
  exit 0
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  build_docker
  echo "✓ built $OUT (via docker)"
  file "$OUT" || true
  exit 0
fi

echo "ERROR: no cross-compile path available." >&2
exit 1
