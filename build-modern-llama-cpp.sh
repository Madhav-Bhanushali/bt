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
ARCH="${CUDA_ARCH:-native}"
# Step-1 A/B: build the same source with different matmul dispatch policies.
#   baseline  default dispatch (MMQ where it fits, else cuBLAS)      -> build_cuda
#   cublas    force every matmul through cuBLAS (diagnostic)          -> build_cuda_cublas
#   mmq       force MMQ kernels for quantized matmul (diagnostic)     -> build_cuda_mmq
# NOTE: GGML_CUDA_FORCE_MMQ / FORCE_CUBLAS are COMPILE-TIME flags, not env vars.
MODE="${MODE:-baseline}"
case "$MODE" in
    baseline) MODE_FLAGS="" ;                                BUILDDIR="build_cuda" ;;
    cublas)   MODE_FLAGS="-DGGML_CUDA_FORCE_CUBLAS=ON" ;     BUILDDIR="build_cuda_cublas" ;;
    mmq)      MODE_FLAGS="-DGGML_CUDA_FORCE_MMQ=ON" ;        BUILDDIR="build_cuda_mmq" ;;
    *) echo "ERROR: unknown MODE='$MODE' (use baseline|cublas|mmq)"; exit 1 ;;
esac
BUILD="$SRC/$BUILDDIR"

# --- build tools (cmake, g++, make) -----------------------------------------
if ! command -v cmake >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
    if [[ "$(id -u)" == "0" ]]; then
        echo "Build tools missing - installing cmake + build-essential ..."
        apt-get update -qq && apt-get install -y cmake build-essential ninja-build
    else
        echo "ERROR: cmake/g++ not found. Run (as root): apt install -y cmake build-essential"
        exit 1
    fi
fi

# --- CUDA toolkit ------------------------------------------------------------
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH=/usr/local/cuda/bin:$PATH   # installed but not on PATH
fi
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

# --- Blackwell smpbo clamp patch (OPTIMIZATION_PLAN Step 1) ------------------
# The NVIDIA driver bug (sharedMemPerBlockOptin -> 0 or 0x100000001 on Blackwell)
# makes llama.cpp reject every MMQ tile config and silently fall back to slow
# cuBLAS. No upstream clamp exists yet; apply ours. Disable with SMPBO_PATCH=0.
# (Upstream PR #22338 fixed SOFT_MAX only; #26141 guards but still leaves the
# batched path on cuBLAS for broken drivers.)
if [[ "${SMPBO_PATCH:-1}" == "1" ]]; then
    if ! python3 - <<'PY'
import io
p = "ggml/src/ggml-cuda/ggml-cuda.cu"
s = io.open(p, encoding="utf-8").read()
old = """        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;"""
new = """        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        // Blackwell driver bug: sharedMemPerBlockOptin can report 0 or 0x100000001.
        // Clamp to the base limit so the MMQ dispatcher keeps working.
        if (info.devices[id].smpbo == 0 || info.devices[id].smpbo > 1024*1024) {
            info.devices[id].smpbo = prop.sharedMemPerBlock;
        }
        info.devices[id].cc = 100*prop.major + 10*prop.minor;"""
assert old in s, "smpbo patch target not found in ggml-cuda.cu"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("patched smpbo clamp into ggml-cuda.cu")
PY
    then
        echo "ERROR: smpbo patch failed; aborting build (set SMPBO_PATCH=0 to skip)"; exit 1
    fi
else
    echo "SMPBO_PATCH=0 - skipping the Blackwell shared-memory clamp"
fi
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    $MODE_FLAGS \
    -DLLAMA_CURL=OFF \
    -DLLAMA_NATIVE=ON
cmake --build "$BUILD" --target llama-server -j"$(nproc)"

echo
echo "Done ($MODE): $BUILD/bin/llama-server"
echo "Quick check (Q4_0-lossless):"
echo "  timeout 30 $BUILD/bin/llama-server -m /home/ubuntu/falcon3/ser/ser/bt/bt/models/standard/Ternary-Bonsai-8B-Q4_0-lossless.gguf -ngl 999 -c 8192 --port 8095 --no-webui 2>&1 | grep -iE 'offload|cuda|model loaded'"
echo "Stress test:"
echo "  LLAMA_SERVER=$BUILD/bin/llama-server bash stress-test.sh"
