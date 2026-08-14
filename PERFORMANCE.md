# Ternary Bonsai 8B — Performance Metrics

Measured on the GPU box (`gpu-vm`) with the concurrency/latency stress test
(`stress-test.sh`). Model: `Minarut/Ternary-Bonsai-8B-GGUF-llamacpp-compatible`.

## Hardware

| Resource | Value |
|---|---|
| GPU | NVIDIA A10G (23028 MiB, ~18364 MiB free) |
| CPU threads | 8 |
| RAM | 30 GB |
| Driver / CUDA | 580.173.02 / 13.0 (toolkit 12.4) |

## Stack comparison (worst → best)

| Metric | 1. TQ2_0 CPU<br>(fork `build_server`) | 2. TQ2_0 GPU<br>(fork `build_cuda`, no CUDA kernel) | 3. Q4_0 GPU<br>(fork `build_cuda`) | 4. Q4_0 GPU<br>(modern llama.cpp) |
|---|---|---|---|---|
| Model file | 2.12 GB TQ2_0 | 2.12 GB TQ2_0 | 4.61 GB Q4_0-lossless | 4.61 GB Q4_0-lossless |
| Gen speed @ conc 1 | 9.5 tok/s | 17.7–19.1 tok/s | 88.1 tok/s | **88.1 tok/s** |
| Reply latency p50 @ conc 1 (predict 64) | 6.8 s | 3.4–3.7 s | 0.55 s | **0.55 s** |
| Reply latency p50 @ conc 8 | 13.1 s | 12.3 s | 1.57 s | **1.49 s** |
| Max throughput | 0.61 req/s | ~0.65 req/s | 5.43 req/s | **6.15 req/s** |
| Best concurrency | 32 | 8+ | 32 | **32** |
| Failures | HTTP-000 ×4 @ conc 4 | none | none | **none (96/96 @ 32)** |
| TTFT p50 | 92–275 ms | 54–117 ms | 12–31 ms | **12–33 ms** |
| GPU VRAM used | 0 | ~1.8 GB | ~6.2 GB | **~6.0 GB** |

## Best configuration (current recommendation)

| Setting | Value |
|---|---|
| Server | `/home/ubuntu/llama.cpp/build_cuda/bin/llama-server` (modern llama.cpp, CUDA) |
| Model | `models/standard/Ternary-Bonsai-8B-Q4_0-lossless.gguf` |
| Offload | `-ngl 999` (all layers) |
| Flash attention | `-fa on` |
| Parallel slots | 12 |
| Context | 8192 (total, ~682 tokens/slot) |
| Predict | 64 |
| Threads | 8 |
| Batch | 4096 / 2048 |
| cache_prompt | on |

```
LLAMA_SERVER=/home/ubuntu/llama.cpp/build_cuda/bin/llama-server bash stress-test.sh
```

## Key takeaways

- 8B Q4_0 single-stream generation is **memory-bandwidth bound** at ~88 tok/s on the
  A10G — identical for the 2025 BitNet fork and 2026 llama.cpp. That's a hard
  ~0.55 s floor per 64-token reply.
- Throughput scales to ~6.15 req/s at 32 concurrent (100% success) — queue-bound at
  that point (p50 queue ~1.9 s), expected for a stress test.
- The ternary TQ2_0 quant has **no CUDA kernel** in llama.cpp; it runs on CPU even
  when loaded into VRAM (~18 tok/s). Q4_0-lossless encodes the identical {-1,0,1}
  weights losslessly and is fully GPU-accelerated.

## Individual run data

Detailed per-concurrency results are appended to `stress_results.txt` on the server.
