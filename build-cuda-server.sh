#!/usr/bin/env bash
# ============================================================================
# Build a CUDA-enabled llama-server for the A10G (and other NVIDIA GPUs).
#
# The bt harness needs a llama-server that was compiled WITH GPU support -
# the existing build_server is CPU-only ("no usable GPU found"). This script
# configures + builds `build_cuda` inside the BitNet source repo (found via
# ancestor/sibling walk-up, e.g. ser/ser/bt/bt -> ser/ser/final).
#
#   bash build-cuda-server.sh [--arch 86] [--jobs N]
#
# After it finishes, run the stress test with:
#   LLAMA_SERVER=<repo>/build_cuda/bin/llama-server bash stress-test.sh
# ============================================================================

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${CUDA_ARCH:-86}"        # A10G = compute capability 8.6 (GA10B)
JOBS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        -h|--help) sed -n 's/^# \{0,1\}//p' "${BASH_SOURCE[0]}" | head -n 18; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

sibling_roots() {
    local p="$ROOT" prev=""
    local i
    for ((i=0; i<6; i++)); do
        p="$(dirname "$p")"
        [[ "$p" == "$prev" ]] && break
        prev="$p"
        echo "$p"
        if [[ -d "$p/final" ]]; then
            echo "$p/final"
        fi
    done
}

# Find the BitNet source repo (has CMakeLists.txt + 3rdparty/llama.cpp).
SRC=""
local_src="$ROOT"
if [[ -f "$local_src/CMakeLists.txt" ]]; then
    SRC="$local_src"
else
    local root
    while read -r root; do
        [[ -n "$root" ]] || continue
        if [[ -f "$root/CMakeLists.txt" && -d "$root/3rdparty/llama.cpp" ]]; then
            SRC="$root"
            break
        fi
    done < <(sibling_roots)
fi
if [[ -z "$SRC" ]]; then
    echo "ERROR: BitNet source repo (CMakeLists.txt + 3rdparty/llama.cpp) not found."
    echo "This script must run inside or next to the source repo (e.g. the 'final' repo)."
    exit 1
fi
echo "Source repo : $SRC"

# nvcc must be present to compile CUDA kernels.
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found - the CUDA toolkit is not installed."
    echo "On Ubuntu: sudo apt install -y nvidia-cuda-toolkit"
    echo "or install the NVIDIA CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
    exit 1
fi
echo "CUDA compiler: $(nvcc --version | grep -i release | head -n 1)"

if [[ "$JOBS" -eq 0 ]]; then
    JOBS="$(nproc 2>/dev/null || echo 4)"
fi

BDIR="$SRC/build_cuda"
if [[ ! -f "$BDIR/CMakeCache.txt" ]]; then
    echo
    echo "Configuring CUDA build in $BDIR (arch sm_$ARCH)..."
    echo
    cmake -S "$SRC" -B "$BDIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMON=ON
    if [[ $? -ne 0 ]]; then
        echo "ERROR: cmake configure failed."
        exit 1
    fi
fi

echo
echo "Building llama-server (this can take a while)..."
echo
cmake --build "$BDIR" --target llama-server --config Release -j "$JOBS"
if [[ $? -ne 0 ]]; then
    echo "ERROR: build failed."
    exit 1
fi

EXE="$BDIR/bin/llama-server"
[[ -f "$EXE.exe" ]] && EXE="$EXE.exe"
if [[ -x "$EXE" ]]; then
    echo
    echo "BUILD OK: $EXE"
    echo
    echo "Now run the GPU stress test with:"
    echo "  LLAMA_SERVER=$EXE bash stress-test.sh"
    echo
    echo "Quick GPU check:"
    echo "  timeout 20 $EXE -m models/standard/Ternary-Bonsai-8B-TQ2_0.gguf -ngl 999 --port 8095 --no-webui 2>&1 | grep -iE 'offload|cuda|error' | head"
else
    echo "ERROR: built but binary not found at $EXE"
    exit 1
fi