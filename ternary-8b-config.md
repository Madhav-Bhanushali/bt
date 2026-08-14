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

## Full model included locally

The complete model binary is copied into this repo's working tree at
`models/standard/Ternary-Bonsai-8B-TQ2_0.gguf` (2.0 GB). It is **not tracked by git**
(see `.gitignore`) because GitHub rejects files over 100 MB; re-download it any time from
`Minarut/Ternary-Bonsai-8B-GGUF-llamacpp-compatible`.

> **Why TQ2_0 and not prism Q2_0:** the `prism-ml/Ternary-Bonsai-8B-gguf` `Q2_0.gguf` uses a custom g128 quantization that requires a forked llama.cpp and fails to load on plain builds (`tensor 'output_norm.weight' has offset ...`). The llamacpp-compatible `TQ2_0` pack loads cleanly.