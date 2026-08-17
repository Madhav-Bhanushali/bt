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

# --- Q2_0 MMVQ vec-dot VDR patch (OPTIMIZATION_PLAN final status: kernel lever) --
# Single-stream decode is stuck at ~37% BW efficiency (vs 55% for Q4_0) because
# VDR_Q2_0_Q8_1_MMVQ=1 makes each thread process only a 32-element half-block,
# while every other MMVQ quant (Q4_0/Q4_1/Q8_0) uses VDR=2. Bumping to 2 makes
# each thread handle a full 64-element block per loop iteration: half the loop
# overhead, larger independent loads per iteration -> better latency hiding.
# Same math, purely a code-path change. Disable with Q2VDR_PATCH=0.
if [[ "${Q2VDR_PATCH:-1}" == "1" ]]; then
    if ! python3 - <<'PY'
import io
p = "ggml/src/ggml-cuda/vecdotq.cuh"
s = io.open(p, encoding="utf-8").read()

s2 = s.replace(
    "#define VDR_Q2_0_Q8_1_MMVQ 1  // Process one 32-element chunk at a time for parallelism",
    "#define VDR_Q2_0_Q8_1_MMVQ 2  // Process a full 64-element block per thread")

old_fn = """static __device__ __forceinline__ float vec_dot_q2_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_0 * bq2_0 = (const block_q2_0 *) vbq + kbx;

    // Q2_0 (group 64): 64 elements with ONE scale, 2 bits per element (4 elements per byte)
    // Q8_1: 32 elements per block with individual scales
    // iqs selects which of the 2 chunks of 32 elements to process (0-1)

    const float     d2 = bq2_0->d;
    const int16_t * qs = (const int16_t *) bq2_0->qs + iqs * 4;

    // Process only the chunk specified by iqs
    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q  = qs[j];
        const int u  = get_int_b4(bq8_1_chunk->qs, j*2+0);
        const int v  = get_int_b4(bq8_1_chunk->qs, j*2+1);

        // unpack even and odd crumbs into byte values
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        // unshuffle values
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);

        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }

    // Apply Q2_0's single scale and this chunk's Q8_1 scale
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}"""
new_fn = """template <int vdr>
static __device__ __forceinline__ float vec_dot_q2_0_q8_1_impl(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_0 * bq2_0 = (const block_q2_0 *) vbq + kbx;

    // Q2_0 (group 64): 64 elements with ONE scale, 2 bits per element (4 elements per byte)
    // Q8_1: 32 elements per block with individual scales
    // iqs selects the first of vdr consecutive 32-element chunks to process (0..2-vdr)

    const float     d2 = bq2_0->d;
    const int16_t * qs = (const int16_t *) bq2_0->qs + iqs * 4;

    float sum = 0.0f;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const block_q8_1 * bq8_1_chunk = bq8_1 + iqs + i;
        const int16_t    * qc          = qs + i * 4;

        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q  = qc[j];
            const int u  = get_int_b4(bq8_1_chunk->qs, j*2+0);
            const int v  = get_int_b4(bq8_1_chunk->qs, j*2+1);

            // unpack even and odd crumbs into byte values
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            // unshuffle values
            const int qx = __byte_perm(qe, qo, 0x5140);
            const int qy = __byte_perm(qe, qo, 0x7362);

            sumi = ggml_cuda_dp4a(u, qx, sumi);
            sumi = ggml_cuda_dp4a(v, qy, sumi);
        }

        // Apply Q2_0's single scale and this chunk's Q8_1 scale
        const float d8 = __low2float(bq8_1_chunk->ds);
        sum += d2 * d8 * sumi;
    }

    return sum;
}

static __device__ __forceinline__ float vec_dot_q2_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(vbq, bq8_1, kbx, iqs);
}"""
assert old_fn in s, "q2_0 vec_dot target not found in vecdotq.cuh"
assert s2 != s, "q2_0 VDR define not found in vecdotq.cuh"
s = s2.replace(old_fn, new_fn, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("patched Q2_0 MMVQ vec-dot: VDR 1 -> 2 (full 64-element block per thread)")
PY
    then
        echo "ERROR: Q2VDR patch failed; aborting build (set Q2VDR_PATCH=0 to skip)"; exit 1
    fi
else
    echo "Q2VDR_PATCH=0 - skipping the Q2_0 vec-dot unroll change"
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
