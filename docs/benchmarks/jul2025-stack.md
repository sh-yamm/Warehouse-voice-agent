# July-2025 stack: measured results

Date: 2026-10-10. Hardware: RTX 4060 Laptop 8 GB, Windows 11.
Branch: `stack-jul2025`. The application code is the same as `main` (intent reader + deterministic dialog). Only the framework, models and runtime are the versions available on 31 July 2025.

## Environment

- **Python:** `uv pip install --exclude-newer 2025-08-01` (all transitive dependencies), with torch 2.7.1 from the cu128 index.
- **Resolved versions:**
  - Pipecat 0.0.77
  - faster-whisper 1.1.1 / CTranslate2 4.6.0
  - transformers 4.54.1
  - onnxruntime 1.20.1
  - kokoro 0.9.4
  - aiortc 1.11
  - av 14.2.0, pinned: 14.3 and 14.4 have no Windows wheels for Python 3.11
- **Models, pinned to their last commit before 2025-08-01:**
  - Qwen3-1.7B / Qwen3-4B GGUF (unsloth, 2025-06-08)
  - Smart Turn v2 (2025-07-25; the repo changed again in September 2025)
  - Kokoro-82M (2025-04-10)
  - distil-whisper small.en (2024)
- **llama.cpp:** b6050, published 2025-07-31 18:29 UTC.

## Intent reader (`bench/intent_eval.py`, 48 cases)

| Model | Accuracy | Value matches label | Latency p50 / p95 |
|---|---|---|---|
| Qwen3-1.7B Q4_K_M | 0.854 | 1.000 | 257.7 / 408.9 ms |
| Qwen3-4B Q4_K_M | 0.854 | 1.000 | 456.5 / 667.1 ms |

**Chosen: Qwen3-1.7B** (same accuracy, about half the latency). Data-changing misreads, and how they're handled:
- **"Skip the bread then." → `substitute`.** New guard: a substitute is never applied when the utterance says skip, without, drop, remove or leave. It becomes "send without it" instead. A negated substitute asks again.
- **"Nothing, thanks." → `add_note`.** Already guarded: a filler-only note means no instructions.

## Scripted WebRTC calls (`scripts/scripted_call.py`)

There were 8 calls over 2 rounds, all with the correct outcome and 0 unrequested changes:
- **A:** happy path, booked
- **B:** shortage, then "send the rest without it", then an out-of-stock add-item correctly refused; booked
- **C:** cancel, confirmed
- **D:** substitute, address change read back and confirmed, note; booked

| Round | Change | Replies | Median | p95 |
|---|---|---|---|---|
| 1 | as ported | 25 | 1,170 ms | 2,263 ms |
| 2 | user-aggregator `aggregation_timeout` 0.5 s → 0.1 s | 25 | **864 ms** | 2,565 ms |

Round-2 stage medians:
- 200 ms VAD silence
- 146 ms Whisper (TTFB)
- 241 ms intent reader (TTFB)
- ~277 ms for the rest: Smart Turn v2 (~72 ms measured offline on the GPU), the 0.1 s aggregator wait, and pipeline hops

The slowest replies came from the first, cold intent-reader requests (up to 2.4 s) and occasional slow Whisper decodes (400–590 ms).

Greetings started 10–27 ms after connect (pre-synthesized).

One harness run failed: the test caller's offer timed out while the first bot was still loading models. It was re-run successfully and isn't counted in the table.

## Compared with main

| | July-2025 stack | main |
|---|---|---|
| Median reply | 864 ms | 735–841 ms |
| p95 | ~2.6 s | ~1.7 s |
| Intent accuracy | 85.4 % (Qwen3-1.7B) | 93.8 % (Qwen3.5-2B) |
| Unrequested changes | 0 | 0 |
