# warehouse-voice-agent

A low-latency, fully local voice agent that calls customers about pending micro-warehouse
e-commerce orders, checks live stock and delivery-slot capacity in a database, and confirms a
delivery time over the call.

Open-weight models only (streaming STT, small LLM via llama.cpp, Kokoro TTS), with end-of-turn
detection, barge-in handling and per-turn latency instrumentation.

Design: [docs/specs/2026-10-09-voice-caller-agent-design.md](docs/specs/2026-10-09-voice-caller-agent-design.md)
