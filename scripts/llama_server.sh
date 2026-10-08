#!/usr/bin/env bash
# Usage: scripts/llama_server.sh models/llm/Qwen3.5-4B-Q4_K_M.gguf
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="$(find "$ROOT/tools/llama.cpp" -name llama-server.exe | head -1)"
exec "$SERVER" -m "$1" -ngl 99 -c 8192 --parallel 1 --host 127.0.0.1 --port 8080 --jinja --cache-reuse 256
