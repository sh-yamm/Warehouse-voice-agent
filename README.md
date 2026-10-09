# Warehouse-voice-agent

A low-latency, fully local voice agent that calls customers about pending micro-warehouse
e-commerce orders, checks live stock and delivery-slot capacity in a database, and confirms a
delivery time over the call.

Open-weight models only (streaming STT, small LLM via llama.cpp, Kokoro TTS), with end-of-turn
detection, barge-in handling and per-turn latency instrumentation.

Design: [docs/specs/2026-10-09-voice-caller-agent-design.md](docs/specs/2026-10-09-voice-caller-agent-design.md)

## Setup (Windows, Git Bash)

    py -3.11 -m venv .venv
    .venv/Scripts/python -m pip install -e ".[dev]"
    .venv/Scripts/python -m pytest

## Talk to the agent (local, browser)

    .venv/Scripts/python -m pip install -e ".[dev,bench,agent]"
    .venv/Scripts/python -m voiceagent.db.seed --db data/warehouse.db
    bash scripts/llama_server.sh models/llm/Qwen3.5-2B-Q4_K_M.gguf        # terminal 1
    .venv/Scripts/python -m voiceagent.agent.bot -t webrtc --order-id 1   # terminal 2
    .venv/Scripts/python -m voiceagent.agent.dashboard                    # terminal 3

Open http://localhost:7860/client, click Connect and answer the call. Per-turn latency: http://localhost:7861.
