#!/usr/bin/env bash
# Usage: scripts/llama_server.sh models/llm/Qwen3.5-2B-Q4_K_M.gguf [chat-template.jinja]
# Qwen3.5 GGUFs need scripts/chat_templates/qwen3.5-multi-system.jinja for the live agent
# (its stock template rejects the per-node system messages Pipecat Flows sends); it is used by default for them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="$(find "$ROOT/tools/${LLAMA_DIR:-llama.cpp-b6050}" -name llama-server.exe | head -1)"
TEMPLATE="${2:-}"
if [ -z "$TEMPLATE" ] && [[ "$(basename "$1")" == Qwen3.5-* ]]; then
  TEMPLATE="$ROOT/scripts/chat_templates/qwen3.5-multi-system.jinja"
fi
EXTRA=()
[ -n "$TEMPLATE" ] && EXTRA=(--chat-template-file "$TEMPLATE")
exec "$SERVER" -m "$1" -ngl 99 -c 8192 --parallel 1 --host 127.0.0.1 --port 8080 --jinja --cache-reuse 256 "${EXTRA[@]}"
