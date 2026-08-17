#!/usr/bin/env bash
# Download a Ternary Bonsai 8B GGUF.
#
# FAST PATH (recommended, modern llama.cpp master >= 2026-07-30):
#   Q2_0 / Q2_0_g64   Ternary-Bonsai-8B-Q2_0_g64.gguf (2.15 GiB)
#                     lossless 2-bit ternary, dedicated CUDA kernels -> ~2.7x
#                     faster generation than Q4_0-lossless on the same GPU.
#
# Minarut pack (llamacpp-compatible / Microsoft fork):
#   TQ2_0             pure ternary (CPU-only in llama.cpp)
#   Q4_0-lossless     CUDA-accelerated 4-bit pack of the same ternary weights
#   TQ2_0-Q6out       2.47 GiB TQ2_0 with Q6 out
#
# Usage:
#   bash download-model.sh Q2_0
#   bash download-model.sh Q4_0-lossless
#   bash download-model.sh <full filename>
set -euo pipefail

MINARUT_REPO="Minarut/Ternary-Bonsai-8B-GGUF-llamacpp-compatible"
PRISMML_REPO="prism-ml/Ternary-Bonsai-8B-gguf"
REPO="$MINARUT_REPO"
FILE="${1:-Ternary-Bonsai-8B-TQ2_0.gguf}"

case "$FILE" in
    Q2_0|Q2_0_g64|Q2_0-lossless)
        REPO="$PRISMML_REPO"; FILE="Ternary-Bonsai-8B-Q2_0_g64.gguf" ;;
    Q2_0-g128|g128)
        REPO="$PRISMML_REPO"; FILE="Ternary-Bonsai-8B-Q2_0.gguf" ;;   # PrismML fork format (group 128)
    F16)
        REPO="$PRISMML_REPO"; FILE="Ternary-Bonsai-8B-F16.gguf" ;;
    Q4_0|Q4_0-lossless)
        FILE="Ternary-Bonsai-8B-Q4_0-lossless.gguf" ;;
    TQ2_0)
        FILE="Ternary-Bonsai-8B-TQ2_0.gguf" ;;
    TQ2_0-Q6out|Q6out)
        FILE="Ternary-Bonsai-8B-TQ2_0-Q6out.gguf" ;;
    *.gguf) : ;;                      # full filename passed directly
    *) echo "Unknown variant: $1 (use Q2_0 | Q4_0-lossless | TQ2_0 | TQ2_0-Q6out | F16)"; exit 1 ;;
esac

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models/standard/$FILE"

if [[ -f "$DEST" ]]; then
    echo "Already present: $DEST"
    exit 0
fi

mkdir -p "$(dirname "$DEST")"
echo "Downloading $REPO / $FILE ..."
curl -L --fail --progress-bar -o "$DEST" "https://huggingface.co/$REPO/resolve/main/$FILE"
echo "Done: $DEST"
