# Phase 2 live call results

Date: 2026-10-09. Hardware: RTX 4060 Laptop 8 GB, Windows 11.
Stack: Pipecat 1.12, Silero VAD + Smart Turn v3.2 (CPU), faster-whisper distil-small.en (greedy), Qwen3.5-2B Q4_K_M via llama-server (patched chat template), Kokoro-82M (PyTorch).

## How these calls were made

- A scripted WebRTC client (aiortc) dials the bot exactly as the browser does.
- It plays Phase 1 corpus clips as the customer's lines, at fixed times, and records the agent's audio.
- Calls with a human speaking (the plan's Task 8 table) are still to do.

## Scripted happy-path call (order 2, no shortages)

Customer lines, in order:
1. "Yeah, that's me."
2. "Yes, that sounds good."
3. "Tomorrow evening works for me."
4. "Yes, that sounds good."
5. "That's all, thank you."
6. "Yes, please confirm that."

**Result:** the call ran start to finish. The outcome was `scheduled`, the order was booked for "tomorrow, 5 to 6 PM", and the transcript was saved.

| Turn | Total | Endpointing | STT | Turn detection | LLM (2 passes + tool) | TTS |
|---|---|---|---|---|---|---|
| greeting (from connect) | 950 ms | n/a | n/a | n/a | n/a (fixed text) | — |
| response 1 | 5659 ms | 200 | 482 | **2681** (cold) | ≈ 1030 | 949 |
| response 2 | 5596 ms | 200 | 479 | 31 | ≈ 1530 | 873 |
| response 3 | 5534 ms | 200 | 322 | 24 | ≈ 1570 | 732 |
| response 4 | 2765 ms | 200 | 450 | 28 | ≈ 800 | 662 |

The LLM column covers two inferences per turn: the first picks a tool, and the second speaks after seeing the tool result. Totals also include sentence aggregation (146–277 ms) and pipeline hops.

## Problems found

1. **Unrequested tool calls with invented arguments (critical).** Qwen3.5-2B called these without being asked:
   - `add_item("Aashirvaad sugar", 2)`: added to the order
   - `add_item("Haldiram salted peanuts", 3)`
   - `update_address("23 MG Road, Bengaluru")`: an address it made up
   - `add_note("none")`
   - `resolve_shortage` on an order with no shortage

   The domain layer kept stock and slots consistent, but the order content and address were changed without the customer asking.
2. **Responses take about 5.5 s, against a 700 ms p50 target.** The causes, largest first:
   1. Two LLM passes on every turn that calls a tool.
   2. TTS at 0.7–0.95 s for the first sentence.
   3. Sentence aggregation, which waits for a full sentence before TTS starts.
   4. Smart Turn's first inference, which takes 2.7 s cold.
3. **Bugs found by the scripted calls and fixed, each with a regression test:**
   - The greeting was silent: a cold Kokoro start exceeded the TTS context's 3 s timeout.
   - Qwen3.5's chat template rejected the per-node system messages, so every turn failed with HTTP 500.
   - The call row stayed open when Flows ended the call.
   - A time the customer volunteered early was booked as a callback.
   - The LLM chose "wait for restock" on the customer's behalf.
   - A slot held before choosing "wait" could be booked before the restock.
   - The seed's 3 days of slots left nothing to book after a restock. It now generates 7 days.

## What this means for the next steps

- The pipeline, conversation graph, tools and recording work end to end.
- **Model quality decides correctness.** A 2B model given tools at every node invents actions. Options:
  - a stronger model (Qwen3-4B-2507 fits if its context is reduced to 4k)
  - fewer tools per node, with `add_item` and `update_address` gated behind explicit customer intent
  - fine-tuning the 2B on this graph's tool traces (Plan 5)
- **Latency needs architecture work, not tuning:**
  - Remove the second LLM pass for transitions, e.g. canned or templated speech after deterministic tools.
  - Stream TTS by clause.
  - Warm up Smart Turn.
  - Pre-synthesize common openers.
