#!/usr/bin/env bash
# Download the Ternary Bonsai 8B model (TQ2_0, llamacpp-compatible pack).
# The GGUF is too large for GitHub, so it lives here as a plain download.
# Usage: bash download-model.sh
set -euo pipefail

REPO="Minarut/Ternary-Bonsai-8B-GGUF-llamacpp-compatible"
FILE="Ternary-Bonsai-8B-TQ2_0.gguf"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models/standard/$FILE"

if [[ -f "$DEST" ]]; then
    echo "Already present: $DEST"
    exit 0
fi

mkdir -p "$(dirname "$DEST")"
echo "Downloading $REPO / $FILE ..."
curl -L --fail --progress-bar -o "$DEST" "https://huggingface.co/$REPO/resolve/main/$FILE"
echo "Done: $DEST"
