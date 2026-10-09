# Voice Caller Agent for Micro-Warehouse Deliveries — Design Spec

Date: 2026-10-09
Status: Draft for review

## 1. Goal

An outbound, audio-only AI agent that calls a customer about a pending e-commerce order, confirms
identity, walks through the order (backed by live warehouse stock), negotiates a delivery slot,
captures address/notes, confirms, and writes the booking to a database — with **voice-to-voice
latency as the primary optimisation target**.

### Constraints
- Zero spend: open-weight models only, all inference local (RTX 4060 Laptop, 8 GB VRAM, Windows 11).
- Must stream LLM tokens, detect end of turn, and handle barge-in (user interrupting the agent).
- English v1; STT/TTS behind swappable services so other languages (e.g. Hindi) can be added later.
- Transport v1: browser call over WebRTC. Telephony (Asterisk + SIP softphone) is out of scope for v1.

### Success criteria
| Metric | Target |
|---|---|
| Voice-to-voice latency (user stops speaking → first agent audio frame sent), p50 | ≤ 700 ms |
| Voice-to-voice latency, p95 | ≤ 1200 ms |
| Barge-in: user speech onset → agent playback stopped | ≤ 200 ms |
| Text-eval task success over A+B+C scenarios | ≥ 90 % |
| Hallucinated facts (item/price/slot not in DB) | 0 in eval set |

Latency is measured on the reference laptop with one concurrent call. Browser/network overhead is
calibrated once with a loopback recording and reported separately.

### Scope
- **A. Happy path:** greet → verify identity → present order → preferred window → check slot
  capacity → offer nearest alternatives if full → confirm → write booking → close.
- **B. Common detours:** call back later, wrong person answers, address change / delivery notes,
  cancel order.
- **C. Inventory-aware:** shortage on an item (partial ship / substitute / wait for restock),
  "can you add X?" with live stock check.
- **Out of scope (stretch, D):** human escalation for angry/out-of-scope callers, payment/COD
  questions, silence/noise re-prompt-and-hangup, PSTN/SIP telephony, non-English.

## 2. Architecture

```
Browser (mic/speaker, browser AEC)
   │ WebRTC (Pipecat SmallWebRTCTransport)
   ▼
Pipecat pipeline (one per call)
  transport.in → Silero VAD → Smart Turn (end-of-turn) → streaming STT
  → user context aggregator → LLM (llama.cpp server, OpenAI-compatible, streaming)
  → clause aggregator → Kokoro TTS (streaming) → transport.out → assistant context aggregator
  + FlowManager (conversation graph; prompt + tools per node)
  + LatencyObserver (per-turn stage timestamps → DB + JSONL)
        │ tool calls (in-process async)          ▲ pre-call context
        ▼                                        │
  Domain services (pure Python, no LLM) ── Repository ── SQLite
        ▲
  Post-call worker (off critical path): outcome, summary, transcript persistence
```

### Design decisions
1. **The LLM reads intent; code acts and speaks.**
   - **Per turn:** exactly one short LLM call returns `{"intent", "value"}`. A JSON schema limits `intent` to the intents valid in the current dialog state.
   - **Action and reply:** a deterministic dialog manager runs the tools and speaks templated replies. Each reply opens with a short acknowledgement ("Sure.", "Got it.") whose audio is pre-synthesized, so speech starts immediately.
   - **Grounding:** extracted values (product, address, time, note) must be grounded in the customer's own words, or nothing is changed.
   - **Questions:** answered from call facts. Anything off-topic gets a fixed polite refusal; there is no free-form generation.
   - **Why:** Phase 2 used Pipecat Flows with LLM function calling. Qwen3.5-2B made tool calls nobody asked for, with invented arguments, and each turn needed two LLM passes (about 5.5 s per reply). See `docs/benchmarks/phase2-live-calls.md`.
2. **Pre-call context injection.** The call is outbound, so customer, order lines with stock status
   and the next 6 free slots are loaded before dialling and placed in the static prefix of the
   system prompt. Most turns need no tool call; the static prefix keeps llama.cpp's prompt cache hot.
3. **Deterministic time parsing.** Phrases like "tomorrow after 5" are normalised to a time window in
   code (a rule-based parser); the LLM never does date arithmetic.
4. **Barge-in** uses Pipecat's interruption frames (cancel LLM, flush TTS, drop queued audio,
   truncate assistant message to what was actually spoken). Backchannel guard: user speech
   < 300 ms or matching a backchannel list ("mm-hmm", "okay", "yeah") while the agent speaks does
   not interrupt.
5. **Preemptive generation (phase 5).** Start LLM generation on a medium-confidence end-of-turn
   signal; discard if the user continues; TTS only starts after turn confirmation. Draft survival
   rate is logged.
6. **Swappable speech services.** STT/TTS/LLM selected via `config.yaml`; each is a Pipecat service.
7. **VRAM budget (8 GB, ~1.4 GB used by display):** STT ≈ 1–1.5 GB, LLM (≈4B, Q4_K_M + KV) ≈ 3–3.5 GB,
   Kokoro ≈ 0.5 GB; Silero VAD and Smart Turn run on CPU.

### Model choices (initial, confirmed by Phase 1 benchmarks)
| Stage | Primary | Fallbacks |
|---|---|---|
| VAD | Silero VAD | — |
| End of turn | Smart Turn v3.x (ONNX, CPU) | VAD-silence only |
| STT | NVIDIA Nemotron Speech Streaming 0.6B (cache-aware, 160 ms chunks) | Parakeet TDT 0.6B v2 / Moonshine streaming via sherpa-onnx; faster-whisper distil-small.en |
| LLM | Qwen3.5-4B Instruct, non-thinking, GGUF Q4_K_M via llama.cpp server | Qwen3.5-2B; Qwen3-4B |
| TTS | Kokoro-82M | Piper |

Risk: NeMo on native Windows. Mitigation: run STT under WSL2, or use the sherpa-onnx path which
runs natively on Windows. Phase 1 decides.

### Period-accurate stack (later, separate branch)
`main` uses the latest stack above. Later, a `stack-jul2025` branch will rebuild the same system
using only components public by 31 July 2025: Pipecat 0.0.77 + `pipecat-ai-flows`, Smart Turn v2,
Parakeet TDT 0.6B v2 (re-decoded per utterance) or Kyutai STT 1B, Qwen3-4B/1.7B (non-thinking),
Kokoro-82M. Service interfaces and config are kept stack-agnostic so that branch only swaps
services and versions. Latency results always state which stack produced them.

## 3. Dialog states and intents

| State | Agent asks | Intents accepted (plus global: add_item, callback, cancel, question, unclear) | Goes to |
|---|---|---|---|
| `greet` | "...Am I speaking with {first name}?" | confirm, deny, wrong_person | `shortage` / `schedule` / closed |
| `shortage` | "The {item} is out of stock. Would you like {substitute} instead, the rest without it, or to wait until {restock}?" | substitute, send_available, wait_restock, deny | next shortage / `schedule` |
| `schedule` | "When would you like it delivered?" / "I can deliver {slot}. Shall I book that?" / "I have {a}, {b} or {c}." | give_time, confirm, deny, pick_option | `address` |
| `address` | "Should we deliver to {address}?" | confirm, deny, change_address | `note` |
| `note` | "Any instructions for the driver?" | confirm, deny, add_note, change_address | `confirm` |
| `confirm` | "To confirm, {items}, delivered {slot}, to {address}. Shall I book it?" | confirm, deny, give_time, change_address, add_note | closed (booked) / `schedule` |
| `confirm_cancel` | "Just to confirm, do you want to cancel the whole order?" | confirm, deny | closed (cancelled) / previous state |

Rules:
- **Slot holds:** a slot is held for 5 minutes when offered and committed only on `confirm`.
- **Changes need grounding:** an order-changing action (add_item, change_address, add_note) runs only if its extracted value is grounded in the utterance. "Yes", "no" and "unclear" never change the order.
- **Cancelling needs two steps:** an explicit cancel request, then a second "yes".
- **Replies are templates.** They are filled from call facts and tool results.

## 4. Data model (SQLite)

```
warehouses(id, name, zone)
products(sku, name, spoken_name, category, unit_price)
stock(warehouse_id, sku, on_hand, reserved, restock_eta)   -- available = on_hand - reserved
customers(id, name, phone, default_address)
orders(id, customer_id, warehouse_id, status, address, notes, slot_id, callback_at, created_at)
  -- status: pending_schedule | scheduled | cancelled | callback | needs_human
order_lines(order_id, sku, qty, reserved_qty, substitute_sku, substitute_qty, resolution)
  -- resolution: NULL | partial | substitute | wait
delivery_slots(id, warehouse_id, start_ts, end_ts, capacity, booked)
slot_holds(slot_id, order_id UNIQUE, expires_at)   -- held seats are derived from unexpired holds
calls(id, order_id, started_at, ended_at, outcome, summary, transcript_json)        -- added with the pipeline
turn_metrics(call_id, turn_idx, vad_stop_ms, eot_ms, stt_final_ms, llm_ttft_ms,
             first_clause_ms, tts_first_audio_ms, v2v_ms, interrupted, preempt_used)  -- added with the pipeline
```

- All access goes through a `Repository` class; swapping to Postgres changes only that module.
- Stock and slot mutations run in a single transaction.
- Seed script (fixed RNG seed): ~50 products, 200 customers, 300 pending orders, 3 days of slots,
  with a deliberate share of shortages and full slots.

## 5. Measurement

- `LatencyObserver` timestamps: VAD stop, end-of-turn decision, final transcript, LLM first token,
  first clause to TTS, first TTS audio frame out. Barge-in: user speech onset → playback stopped.
- Per turn → `turn_metrics` + JSONL. `report.py` prints p50/p95 and per-stage breakdown.
- Browser debug panel shows live per-turn latency.

## 6. Evaluation

1. **Component benchmarks (`bench/`):** STT WER and latency on ~50 delivery-domain utterances;
   LLM TTFT and tokens/s per quantisation; TTS time-to-first-audio; Smart Turn accuracy.
2. **Text-only conversation eval:** ~60 scripted scenarios over A+B+C, driven by a simulated
   customer (scripted or persona-prompted local LLM). Assertions on final DB state, tool-call
   validity, hallucination, and response length.
3. **Audio end-to-end eval:** ~10 scenarios via a headless client with pre-generated customer audio,
   including barge-in and backchannel cases.

## 7. Fine-tuning and quantisation (phase 6)

- Data: 2–4k synthetic dialogues from the text-eval harness with varied personas and seeds; only
  DB-assertion-passing episodes kept; samples are (node prompt + context + history) → (reply and/or
  tool call).
- QLoRA with Unsloth on 2B and 4B bases (local GPU; Kaggle T4 fallback); merge; quantise to GGUF
  Q4_K_M and Q5_K_M.
- Results table: base vs fine-tuned × size × quant on task success, tool-call accuracy,
  hallucination rate, TTFT and voice-to-voice latency.
- Cost model: GPU-hours per call vs human agent cost per call, with all assumptions stated.

## 8. Phases

1. Benchmark harness and STT selection.
2. DB, domain layer, seed data, unit tests.
3. Pipeline + conversation graph + latency observer + browser client (working demo).
4. Text-eval harness, then audio eval.
5. Latency work: preemptive generation, prompt-cache tuning, clause chunking.
6. Fine-tune and quantise, results table.

## 9. Environment

- Python 3.12 virtualenv (system Python 3.14 lacks wheels for several ML packages); WSL2 if NeMo is
  required.
- llama.cpp server (CUDA build) run as a separate process.
- Model weights and generated data are git-ignored.
