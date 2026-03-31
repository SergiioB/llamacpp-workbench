#!/bin/bash
# Build llama.cpp with CUDA support for NVIDIA GPUs
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/../third_party/llama.cpp"
BUILD_DIR="${LLAMA_DIR}/build-cuda"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    mkdir -p "$(dirname "$LLAMA_DIR")"
    git clone https://github.com/ggerganov/llama.cpp.git "$(dirname "$LLAMA_DIR")/llama.cpp"
fi

echo "Building llama.cpp with CUDA support in ${BUILD_DIR}..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_CLI=ON

cmake --build . --config Release -j$(nproc)

echo ""
echo "CUDA build complete!"
echo "Binary location: ${BUILD_DIR}/bin/llama-server"
echo ""
echo "To use with llama-webui:"
echo "  1. Set LLAMA_WEBUI_LLAMA_SERVER=${BUILD_DIR}/bin/llama-server"
echo "  2. Or run: cp ${BUILD_DIR}/bin/llama-server ${BUILD_DIR}/bin/llama-cli"
