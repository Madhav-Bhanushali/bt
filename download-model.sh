#!/usr/bin/env bash
# Download a Ternary Bonsai 8B GGUF (llamacpp-compatible pack).
# The GGUFs are too large for GitHub, so they live here as plain downloads.
# Usage:
#   bash download-model.sh                  # TQ2_0 (pure ternary; CPU-only in llama.cpp)
#   bash download-model.sh Q4_0-lossless    # Q4_0 (CUDA-accelerated on GPU)
set -euo pipefail

REPO="Minarut/Ternary-Bonsai-8B-GGUF-llamacpp-compatible"
FILE="${1:-Ternary-Bonsai-8B-TQ2_0.gguf}"
case "$FILE" in
    Q4_0|Q4_0-lossless)              FILE="Ternary-Bonsai-8B-Q4_0-lossless.gguf" ;;
    TQ2_0)                           FILE="Ternary-Bonsai-8B-TQ2_0.gguf" ;;
    TQ2_0-Q6out|Q6out)               FILE="Ternary-Bonsai-8B-TQ2_0-Q6out.gguf" ;;
    *.gguf) : ;;                      # full filename passed directly
    *) echo "Unknown variant: $1 (use TQ2_0 | Q4_0-lossless | TQ2_0-Q6out)"; exit 1 ;;
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
