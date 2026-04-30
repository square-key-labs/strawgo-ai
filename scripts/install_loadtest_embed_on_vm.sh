#!/usr/bin/env bash
# Build cmd/loadtest-embed on the GCE VM (linux/amd64). Run this *on the VM*,
# not on macOS. Cross-compiling cgo from macOS requires a glibc toolchain
# (zig / x86_64-linux-gnu-gcc / docker) and the simplest path is to build
# natively on the target.
#
# Prereqs on the VM:
#   - Go 1.25+ installed
#   - This repo cloned (or rsync'd from a macOS worktree)
#
# This script:
#   1. Downloads ORT 1.25.1 Linux x64 if /usr/local/lib/libonnxruntime.so isn't already 1.25-compatible.
#   2. Downloads silero_vad.onnx if not already at testdata/models/.
#   3. Builds ./loadtest-embed-linux.
#
# Usage on VM:
#   cd ~/strawgo-worktree
#   ./scripts/install_loadtest_embed_on_vm.sh
#   ./loadtest-embed-linux -lib /usr/local/lib/libonnxruntime.so -dur 20 -levels 1,5,10,25,50,100,200

set -euo pipefail
cd "$(dirname "$0")/.."

ORT_VER=1.25.1
ORT_TGZ="onnxruntime-linux-x64-${ORT_VER}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_TGZ}"

# --- ORT ---
NEED_ORT=1
if [ -f /usr/local/lib/libonnxruntime.so.${ORT_VER} ]; then
  NEED_ORT=0
fi
if [ -f ./lib/libonnxruntime.so ] && \
   strings ./lib/libonnxruntime.so 2>/dev/null | grep -q "ORT v${ORT_VER}"; then
  NEED_ORT=0
fi

if [ "$NEED_ORT" = "1" ]; then
  echo "→ Downloading ORT $ORT_VER..."
  cd /tmp
  curl -sL "$ORT_URL" -o "$ORT_TGZ"
  tar -xzf "$ORT_TGZ"
  cd - >/dev/null

  mkdir -p ./lib
  cp -P /tmp/onnxruntime-linux-x64-${ORT_VER}/lib/libonnxruntime.so* ./lib/
  echo "  ./lib/libonnxruntime.so* installed"

  if [ "${INSTALL_SYSTEM:-0}" = "1" ]; then
    sudo cp -P /tmp/onnxruntime-linux-x64-${ORT_VER}/lib/libonnxruntime.so* /usr/local/lib/
    sudo ldconfig
    echo "  /usr/local/lib/libonnxruntime.so installed (system)"
  fi
fi

# --- silero_vad.onnx ---
if [ ! -f ./testdata/models/silero_vad.onnx ]; then
  mkdir -p ./testdata/models
  if [ -f ~/silero_vad.onnx ]; then
    cp ~/silero_vad.onnx ./testdata/models/
    echo "→ silero_vad.onnx copied from ~/"
  else
    # Use the same source pipecat does
    echo "→ Downloading silero_vad.onnx..."
    curl -sL \
      "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx?download=true" \
      -o ./testdata/models/silero_vad.onnx
  fi
fi

# --- build ---
echo "→ Building ./loadtest-embed-linux..."
CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
  CGO_LDFLAGS="-L$(pwd)/lib -Wl,-rpath,$(pwd)/lib" \
  go build -o ./loadtest-embed-linux ./cmd/loadtest-embed

echo
echo "✓ Built ./loadtest-embed-linux"
file ./loadtest-embed-linux

echo
echo "Smoke test:"
ORT_DYLIB_PATH="$(pwd)/lib/libonnxruntime.so" ./loadtest-embed-linux \
  -lib "$(pwd)/lib/libonnxruntime.so" \
  -dur 5 \
  -levels 1
