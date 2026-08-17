# Ternary Bonsai 8B — Performance Metrics

Measured on the production GPU box (`dev-markytics-6000ada`, RTX PRO 6000 Blackwell
Server Edition) with `stress-test.sh`. See `OPTIMIZATION_PLAN.md` and
`CLAUDE_CONTEXT.md` for the full testing history and closure verdict.

## Hardware (current box)

| Resource | Value |
|---|---|
| GPU | RTX PRO 6000 Blackwell Server Edition (96 GB, 1.79 TB/s, sm_100) |
| CPU threads | 64 |
| Driver | 610.57.04 (CUDA UMD 13.3; toolkit 13.0) |
| Server | modern llama.cpp master, CUDA build (`~/llama.cpp/build_cuda/bin/llama-server`) |
| Model | `standard/Ternary-Bonsai-8B-Q2_0_g64.gguf` (2.15 GB, lossless) |

## Baseline (running config: `parallel=31/ctx=16384`, predict 64)

| Concurrency | Gen tok/s | Lat p50 | TTFT p50 | Req/s | Success |
|---|---|---|---|---|---|
| 1 | **307.6** | 179 ms | 4 ms | 5.19 | 100% |
| 2 | 232.5 | 257 ms | 5 ms | 7.48 | 100% |
| 4 | 159.8 | 368 ms | 6 ms | 10.69 | 100% |
| 8 | 95.4 | 592 ms | 10 ms | 13.04 | 100% |
| 16 | 71.0 | 815 ms | 12 ms | 13.45 | 100% |
| 32 | 70.8 | 1507 ms | 13 ms | 15.27 | 100% |

Aggregate at conc 32 ≈ **~800–1160 tok/s** (queue p50 ~23 ms at 31 slots).
Single-stream ceiling ≈ **37%** of theoretical BW-limited throughput.

## Consolidated A/B comparison (all dispatch/config variants — plan Steps 1–6)

| Variant | conc1 tok/s | conc32 aggregate | Verdict |
|---|---|---|---|
| baseline | 311.4 | 1156 | running config |
| force_cublas | 308.3 | 717 | 38% worse — ruled out |
| force_mmq | 315.4 | 1160 | matches baseline |
| graphs_on (`GGML_CUDA_GRAPH_OPT=1`) | 301.1 | 1155 | matches baseline |
| prismml_g128 fork | 315.9 | 1128 | slightly worse, not adopted |
| ub_4096 | 311.7 | — | no-op (ruled out) |
| threads16 | 311.5 | — | no-op (ruled out) |
| kv q8_0 (`-ctk/-ctv`) | 285.5 | — | 8% worse (ruled out) |
| q2vdr2 custom kernel | 289.3 | — | slower AND garbled output — rejected |
| clean_master_25603 | 311.2 | — | = baseline; #25603 already present |

## Final conclusion

**~310 tok/s single-stream is the confirmed practical ceiling** for
Ternary-Bonsai-8B / Q2_0 / llama.cpp / RTX PRO 6000 Blackwell as of this round.
Every lever (dispatch, CUDA graphs, batch/thread config, KV quant, alternate
fork, the one real upstream kernel fix) has been tested; none move it. Closing
the gap to 500 tok/s requires a materially better CUDA 2-bit GEMV kernel than
exists anywhere in the open-source ecosystem today — a research contribution,
out of scope for this deployment pass.

Options to bring to the model owner:
1. Accept ~310 tok/s single-stream as the number.
2. Treat 500 tok/s as an **aggregate/system** throughput target — already
   exceeded (~1150–1160 tok/s at conc32).

## Best configuration (keep)

| Setting | Value |
|---|---|
| Server | `~/llama.cpp/build_cuda/bin/llama-server` (mainline master, no custom kernel patches) |
| Model | `standard/Ternary-Bonsai-8B-Q2_0_g64.gguf` |
| Offload | `-ngl 999` (all layers) |
| Flash attention | `-fa on` |
| Parallel slots | 31 |
| Context | 16384 (total) |
| Predict | 64 |
| Threads | 64 |
| Batch | 4096 / 2048 |

Before benchmarking any new build, run `verify-output.sh` (output-legibility gate).

```
LLAMA_SERVER=~/llama.cpp/build_cuda/bin/llama-server bash stress-test.sh \
  --model standard/Ternary-Bonsai-8B-Q2_0_g64.gguf --tag <name>
```

Detailed per-concurrency results are appended to `stress_results.txt` on the server.
