# Ternary Bonsai 8B — Model Config

Saved configuration for the **Ternary Bonsai 8B** model used in the loan-collection benchmark harness (`test.sh` in the `final` repo).

See [`ternary-8b-config.json`](ternary-8b-config.json) for the full config: Hugging Face source, inference settings, stop tokens, `chat_template_kwargs`, and system prompt.

## Key facts

| Field | Value |
|---|---|
| HF repo | `Minarut/Ternary-Bonsai-8B-GGUF-llamacpp-compatible` |
| HF file | `Ternary-Bonsai-8B-TQ2_0.gguf` |
| Quant | `TQ2_0` (native ternary, plain llama.cpp builds) |
| Context | 8192 |
| Max predict | 64 |
| Thinking | disabled via `chat_template_kwargs: {"enable_thinking": false}` |
| Cache prompt | on (fast per-test prompt eval) |

## GPU: use the Q4_0-lossless variant

`TQ2_0` has **no CUDA kernel** in the llama.cpp fork — the weights sit in VRAM but the ternary matmul runs on CPU (~10–19 tok/s), so the A10G stays idle. The same model pack ships a **`Q4_0-lossless`** GGUF (4.61 GB) that encodes the identical {-1, 0, 1} weights losslessly and is fully CUDA-accelerated:

```bash
bash download-model.sh Q4_0-lossless
LLAMA_SERVER=/home/ubuntu/falcon3/ser/ser/final/build_cuda/bin/llama-server bash stress-test.sh   # defaults to Q4_0-lossless now
# or explicitly: bash stress-test.sh --model standard/Ternary-Bonsai-8B-Q4_0-lossless.gguf
```

## Repo layout (ternary model focus)

| Path | Purpose |
|---|---|
| `test.sh` | benchmark harness (runs `--model ternary-8b` and others) |
| `sp.txt` | system prompt loaded by `test.sh` |
| `download-model.sh` | downloads the 2.0 GB GGUF into `models/standard/` |
| `ternary-8b-config.json` | full model config (source, inference, stop tokens, prompt) |

## The model binary is downloaded, not committed

GitHub rejects files over 100 MB, so the 2.0 GB GGUF is **not in git**. Get it with:

```bash
bash download-model.sh        # or just: bash test.sh --model ternary-8b
```

Source: `Minarut/Ternary-Bonsai-8B-GGUF-llamacpp-compatible` → `Ternary-Bonsai-8B-TQ2_0.gguf`.

> **Why TQ2_0 and not prism Q2_0:** the `prism-ml/Ternary-Bonsai-8B-gguf` `Q2_0.gguf` uses a custom g128 quantization that requires a forked llama.cpp and fails to load on plain builds (`tensor 'output_norm.weight' has offset ...`). The llamacpp-compatible `TQ2_0` pack loads cleanly.