#!/bin/bash
# Install ONNX Runtime for StrawGo VAD support
# This script downloads and installs the ONNX Runtime library

set -e

echo "🔧 Installing ONNX Runtime for StrawGo VAD..."

# Detect OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Darwin" ]; then
    # macOS
    if [ "$ARCH" = "arm64" ]; then
        echo "📦 Detected: macOS ARM64 (Apple Silicon)"
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-osx-arm64-1.19.2.tgz"
        ONNX_DIR="onnxruntime-osx-arm64-1.19.2"
    else
        echo "📦 Detected: macOS x86_64 (Intel)"
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-osx-x86_64-1.19.2.tgz"
        ONNX_DIR="onnxruntime-osx-x86_64-1.19.2"
    fi
elif [ "$OS" = "Linux" ]; then
    # Linux
    if [ "$ARCH" = "aarch64" ]; then
        echo "📦 Detected: Linux ARM64"
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-aarch64-1.19.2.tgz"
        ONNX_DIR="onnxruntime-linux-aarch64-1.19.2"
    else
        echo "📦 Detected: Linux x86_64"
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz"
        ONNX_DIR="onnxruntime-linux-x64-1.19.2"
    fi
else
    echo "❌ Unsupported OS: $OS"
    exit 1
fi

# Download and extract
echo "⬇️  Downloading ONNX Runtime from GitHub..."
cd /tmp
curl -L "$ONNX_URL" -o onnxruntime.tgz
echo "📦 Extracting..."
tar -xzf onnxruntime.tgz

# Install to project lib directory (no sudo needed)
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$PROJECT_ROOT/lib"

echo "📂 Installing to $PROJECT_ROOT/lib/"
if [ "$OS" = "Darwin" ]; then
    cp "$ONNX_DIR"/lib/libonnxruntime.*.dylib "$PROJECT_ROOT/lib/"
else
    cp "$ONNX_DIR"/lib/libonnxruntime.so* "$PROJECT_ROOT/lib/"
fi

# Optional: Install system-wide (requires sudo)
read -p "🔐 Install system-wide? (requires sudo) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ "$OS" = "Darwin" ]; then
        sudo mkdir -p /usr/local/lib
        sudo cp "$ONNX_DIR"/lib/libonnxruntime.*.dylib /usr/local/lib/
        echo "✅ Installed to /usr/local/lib/"
    else
        sudo mkdir -p /usr/local/lib
        sudo cp "$ONNX_DIR"/lib/libonnxruntime.so* /usr/local/lib/
        sudo ldconfig
        echo "✅ Installed to /usr/local/lib/"
    fi
fi

# Cleanup
rm -rf /tmp/onnxruntime.tgz /tmp/"$ONNX_DIR"

echo ""
echo "✅ ONNX Runtime installed successfully!"
echo ""
echo "📍 Library location: $PROJECT_ROOT/lib/"
echo ""
echo "🚀 You can now use SileroVAD:"
echo "   go run examples/asterisk_flow.go"
echo ""
