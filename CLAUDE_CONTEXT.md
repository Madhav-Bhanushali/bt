# Claude Handoff: Ternary Bonsai 8B — GPU Inference Optimization

> Purpose: give Claude the full context so you can ask for optimization
> suggestions. Everything below is verified/measured; commands are current.

## 1. Goal
Run **Ternary-Bonsai-8B** (native ternary LLM, `PrismML`) for a bank-loan
collection agent on the fastest hardware we have. Constraints from the owner:
- **predict = 64** (must NOT be reduced — reply length is fixed)
- Target: **≥ 500 tokens/sec generation**, max parallelism, max VRAM use
- **Do not lose accuracy / efficiency / consistency** — quant must be lossless

## 2. Model
- 8.19 B params, qwen3-style architecture (36 layers, 8 KV heads, head_dim 128)
- Weights are exactly `{-1, 0, +1}` (ternary)
- GGUFs we use (all lossless for this model):
  | Variant | Size | CUDA kernel? | Source |
  |---|---|---|---|
  | **Q2_0_g64** (recommended) | 2.15 GiB | ✅ merged in llama.cpp 2026-07-30 (PR #24448 CPU, #25707 CUDA) | `prism-ml/Ternary-Bonsai-8B-gguf` |
  | Q4_0-lossless | 4.61 GiB | ✅ | `Minarut/Ternary-Bonsai-8B-GGUF-llamacpp-compatible` |
  | TQ2_0 | 2.12 GiB | ❌ NO kernel (CPU matmul even in VRAM) | Minarut |

## 3. Hardware — current box
- **NVIDIA RTX PRO 6000 Blackwell Server Edition** — 96 GB GDDR7, **1.79 TB/s**,
  sm_100. Driver 610.57.04, CUDA toolkit 13.0 (`CUDA_ARCH=100`).
- 64 CPU threads, 30 GB+ RAM, GPU idle (VLLM engines stopped).
- (Earlier work was on an A10G / 600 GB/s; Q4_0 ceiling there was 88 tok/s.)

## 4. Stack (all in repo `github.com/Madhav-Bhanushali/bt`)
- `build-modern-llama-cpp.sh` — clones **ggml-org/llama.cpp master**, builds
  `llama-server` with CUDA. `CUDA_ARCH=100` for Blackwell (auto-installs
  cmake/build-essential/nvcc check). Binary: `~/llama.cpp/build_cuda/bin/llama-server`.
- `download-model.sh Q2_0` — pulls the 2.15 GiB Q2_0_g64 from prism-ml.
- `stress-test.sh` — concurrency/latency sweep (levels 1–32), VRAM pre-flight,
  port checks, warmup request, auto-parallel slots (cap 32), ctx default 16384,
  `-fa on`, `-ngl 999`, `cache_prompt`, optional `--sustain N`.
- The Microsoft BitNet fork (`final` repo) is **no longer needed** (only for TQ2_0).

## 5. Measured results (Blackwell, Q2_0_g64, 32 slots/16384 ctx pending re-run)
| Concurrency | Gen tok/s | Lat p50 | TTFT p50 | Req/s | Success |
|---|---|---|---|---|---|
| 1 | **307.6** | 179 ms | 4 ms | 5.19 | 100% |
| 2 | 232.5 | 257 ms | 5 ms | 7.48 | 100% |
| 4 | 159.8 | 368 ms | 6 ms | 10.69 | 100% |
| 8 | 95.4 | 592 ms | 10 ms | 13.04 | 100% |
| 16 | 71.0 | 815 ms | 12 ms | 13.45 | 100% |
| 32 | 70.8 | 1507 ms | 13 ms | 15.27 | 100% |

Aggregate at conc 32 ≈ **~800 tok/s** (was 12 slots; new 32-slot build pending).

## 6. Bottleneck analysis
- **Single-stream gen is kernel/bandwidth-bound, NOT resource-bound.** Parallelism,
  threads (64), and VRAM do not raise conc-1 tok/s.
- Bandwidth math for Q2_0 (2.15 GB/token read): 1792 GB/s → 833 tok/s at 100% BW,
  ~460 at Q4_0's 55% efficiency, **500+ needs ~60% efficiency**.
- Q4_0-lossless hits 213.8 tok/s = **55%** BW efficiency; Q2_0_g64 hits 307.6 = **37%**.
  → The new Q2_0 CUDA kernel is **under-tuned on Blackwell** (MMVQ vec-dot path).
- **Known Blackwell driver bug:** `sharedMemPerBlockOptin` returns corrupted value
  (0 / 0x100000001), breaks llama.cpp MMQ batched kernels. Fixed for SOFT_MAX
  (PR #22338), guarded to cuBLAS fallback (PR #26141). Single-stream uses MMVQ, so
  the 307 stands. Batched/prefill may also be degraded on Blackwell.

## 7. Open questions we want suggestions on
1. **Any kernel/path that beats Q2_0 MMVQ on Blackwell for single-stream gen?**
   (e.g., forcing MMQ for token batches, tensor-core / mma ternary kernels,
   cuBLAS GEMV vs vec-dot, `GGML_CUDA_*` env toggles, CUDA graphs in server.)
2. **Is the PrismML fork (`PrismML-Eng/llama.cpp`, g128 Q2_0 CUDA kernels)
   worth switching to** for better Blackwell tuning? (Their g128 kernels are
   production-hardened; ~+6% on L40S vs mainline g64.) Requires their
   `Ternary-Bonsai-8B-Q2_0.gguf` (g128) file.
3. **Config knobs to try:** ubatch 4096, threads 16 vs 64, KV quant
   (`-ctk/-ctv`), `-fa on/off`, bigger ctx (VRAM is essentially free).
4. **Is 500+ tok/s single-stream realistic on this GPU + llama.cpp master?**
   If not, is the target better defined as aggregate throughput (already ~800+
   tok/s and rising with 32 slots)?
5. **Alternative servers:** vLLM ternary/BitNet support vs llama-server for this
   workload (latency-focused, predict 64, continuous batching).

## 8. How to run (on the box)
```bash
cd ~/bonsai/bt && git pull
CUDA_ARCH=100 bash build-modern-llama-cpp.sh ~/llama.cpp   # rebuild if needed
LLAMA_SERVER=~/llama.cpp/build_cuda/bin/llama-server \
  bash stress-test.sh --model standard/Ternary-Bonsai-8B-Q2_0_g64.gguf
```
Results append to `stress_results.txt` (repo root, gitignored).