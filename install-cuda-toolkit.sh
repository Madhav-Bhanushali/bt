#!/usr/bin/env bash
# Install the NVIDIA CUDA toolkit via the NVIDIA apt repo (Blackwell sm_100
# needs CUDA >= 12.8). Auto-detects the Ubuntu version.
#
# Usage:  sudo bash install-cuda-toolkit.sh
set -euo pipefail

[[ "$(id -u)" == "0" ]] || { echo "ERROR: run as root (sudo)"; exit 1; }
command -v apt-get >/dev/null 2>&1 || { echo "ERROR: not a Debian/Ubuntu system"; exit 1; }

VER="$(. /etc/os-release && echo "$VERSION_ID")"
case "$VER" in
    24.04) SUB="ubuntu2404" ;;
    22.04) SUB="ubuntu2204" ;;
    20.04) SUB="ubuntu2004" ;;
    *) echo "ERROR: unsupported Ubuntu version: $VER"; exit 1 ;;
esac

echo "Ubuntu $VER -> $SUB"
KEYRING="https://developer.download.nvidia.com/compute/cuda/repos/$SUB/x86_64/cuda-keyring_1.1-1_all.deb"
cd /tmp
echo "Adding NVIDIA apt repo ($SUB) ..."
wget -q "$KEYRING" -O cuda-keyring.deb
dpkg -i cuda-keyring.deb
apt-get update -qq

echo "Installing CUDA toolkit 13.0 ..."
apt-get install -y cuda-toolkit-13-0

echo "Persisting CUDA on PATH for all shells ..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' > /etc/profile.d/cuda.sh
. /etc/profile.d/cuda.sh

echo
echo "Done. CUDA is on PATH now and in every new shell:"
echo '  nvcc --version'