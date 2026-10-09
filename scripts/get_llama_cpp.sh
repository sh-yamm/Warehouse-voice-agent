#!/usr/bin/env bash
# Download the llama.cpp Windows CUDA 12.4 build into tools/llama.cpp
set -euo pipefail
BUILD="${1:-b11514}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/tools/llama.cpp-$BUILD"
mkdir -p "$DEST"
cd "$DEST"
BASE="https://github.com/ggml-org/llama.cpp/releases/download/$BUILD"
for f in "llama-$BUILD-bin-win-cuda-12.4-x64.zip" "cudart-llama-bin-win-cuda-12.4-x64.zip"; do
  [ -f "$f" ] || curl -fL -o "$f" "$BASE/$f"
  unzip -o -q "$f"
done
"$(find "$DEST" -name llama-server.exe | head -1)" --version
