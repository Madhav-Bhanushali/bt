#!/usr/bin/env bash
# Download a Ternary Bonsai 8B GGUF (llamacpp-compatible pack).
# The GGUFs are too large for GitHub, so they live here as plain downloads.
# Usage:
#   bash download-model.sh                  # TQ2_0 (pure ternary; CPU-only in llama.cpp)
#   bash download-model.sh Q4_0-lossless    # Q4_0 (CUDA-accelerated on GPU)
set -euo pipefail

REPO="Minarut/Ternary-Bonsai-8B-GGUF-llamacpp-compatible"
FILE="${1:-Ternary-Bonsai-8B-TQ2_0.gguf}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models/standard/$FILE"

if [[ -f "$DEST" ]]; then
    echo "Already present: $DEST"
    exit 0
fi

mkdir -p "$(dirname "$DEST")"
echo "Downloading $REPO / $FILE ..."
curl -L --fail --progress-bar -o "$DEST" "https://huggingface.co/$REPO/resolve/main/$FILE"
echo "Done: $DEST"
