#!/usr/bin/env bash
# Build a MODERN upstream llama.cpp CUDA server for the Q4_0-lossless GGUF.
#
# The Microsoft BitNet fork (ggml 0.15.3) is only needed for the TQ2_0 ternary
# quant. The Q4_0-lossless pack is a STANDARD GGUF quant, so a current llama.cpp
# (2026 kernels, much faster CUDA) can run it on the A10G for a big speedup.
#
# Usage:
#   bash build-modern-llama-cpp.sh [srcdir] [branch|tag]
#     srcdir   default: /home/ubuntu/llama.cpp
#     branch   default: master   (use a release tag for stability)
#     CUDA_ARCH env: default native (auto-detect); e.g. 100 for Blackwell,
#                   86 for A10G, 89 for RTX 40 series, 120 for RTX 50 series.
set -euo pipefail

SRC="${1:-/home/ubuntu/llama.cpp}"
BRANCH="${2:-master}"
BUILD="$SRC/build_cuda"
ARCH="${CUDA_ARCH:-native}"

command -v nvcc >/dev/null 2>&1 || { echo "ERROR: CUDA toolkit not found (nvcc)"; exit 1; }
echo "nvcc: $(nvcc --version | grep release)"
echo "CUDA arch: $ARCH (set CUDA_ARCH=100 for Blackwell, 86 for A10G if native fails)"

if [[ ! -d "$SRC/.git" ]]; then
    git clone --depth 1 --branch "$BRANCH" https://github.com/ggerganov/llama.cpp.git "$SRC"
else
    cd "$SRC"
    git fetch --depth 1 origin "$BRANCH"
    git checkout -f FETCH_HEAD 2>/dev/null || git checkout -f origin/"$BRANCH" 2>/dev/null || git checkout -f "$BRANCH"
fi

cd "$SRC"
echo "Building upstream llama.cpp @ $(git rev-parse --short HEAD) ..."
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    -DLLAMA_CURL=OFF \
    -DLLAMA_NATIVE=ON
cmake --build "$BUILD" --target llama-server -j"$(nproc)"

echo
echo "Done: $BUILD/bin/llama-server"
echo "Quick check (Q4_0-lossless):"
echo "  timeout 30 $BUILD/bin/llama-server -m /home/ubuntu/falcon3/ser/ser/bt/bt/models/standard/Ternary-Bonsai-8B-Q4_0-lossless.gguf -ngl 999 -c 8192 --port 8095 --no-webui 2>&1 | grep -iE 'offload|cuda|model loaded'"
echo "Stress test:"
echo "  LLAMA_SERVER=$BUILD/bin/llama-server bash stress-test.sh"
