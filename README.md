# Warehouse-voice-agent: July-2025 stack (`stack-jul2025` branch)

**The same fully local voice caller agent as [`main`](https://github.com/sh-yamm/Warehouse-voice-agent), rebuilt using only software and model weights that were publicly available on 31 July 2025.**
- **Every Python dependency** was resolved with `uv --exclude-newer 2025-08-01`, transitive ones included.
- **Every model** is pinned to its last Hugging Face commit before August 2025.
- **llama.cpp** is the last build published on 31 July 2025 (b6050).

The agent phones a micro-warehouse grocery customer, confirms who it's talking to, walks through the order and anything out of stock, books a delivery slot from live capacity, confirms the address and driver instructions, and writes everything to SQLite. It runs on one laptop (RTX 4060, 8 GB) with open-weight models only.

> *"Hi, this is **Avio** calling from **AvioStack** about your grocery order. Am I speaking with Arjun?"*

| | July-2025 stack (this branch) | main (October-2026 stack) |
|---|---|---|
| **Median reply latency** (customer stops → agent starts) | **864 ms** (25 replies) | 735–841 ms |
| **p95** | ≈ 2.6 s | ≈ 1.7 s |
| **Greeting after connect** | 10–27 ms (pre-synthesized) | 20–85 ms |
| **Order changes the customer didn't ask for** | **0** across 8 scripted calls | 0 |
| **Intent reader** (48 labelled utterances) | Qwen3-1.7B: **85.4 %**, 258 ms p50 | Qwen3.5-2B: 93.8 %, 324 ms |
| **Tests** | 194 pass (domain, dialog, intent reader, pipeline glue, services, metrics) | 200 |
| **Cost** | ₹0, all local | ₹0 |

---

## Contents
1. [Demo](#1-demo)
2. [Why this branch exists, and the stack](#2-why-this-branch-exists-and-the-stack)
3. [What the agent does](#3-what-the-agent-does)
4. [Architecture](#4-architecture)
5. [How one turn works (data flow)](#5-how-one-turn-works-data-flow)
6. [The dialog: why the LLM doesn't call tools](#6-the-dialog-why-the-llm-doesnt-call-tools)
7. [Business logic and data model](#7-business-logic-and-data-model)
8. [Model choices on a July-2025 budget](#8-model-choices-on-a-july-2025-budget)
9. [Results](#9-results)
10. [Porting notes: Pipecat 1.12 → 0.0.77](#10-porting-notes-pipecat-112--0077)
11. [Repository structure](#11-repository-structure)
12. [Running it yourself](#12-running-it-yourself)
13. [Testing and measurement tools](#13-testing-and-measurement-tools)
14. [Limitations and next steps](#14-limitations-and-next-steps)

---

## 1. Demo

🔊 **[Listen to a recorded call on this stack (MP3, 2 min)](docs/demo/demo-call.mp3).** Both voices are mixed into one file. The agent is Avio, running on Pipecat 0.0.77 with Smart Turn v2, Qwen3-1.7B on llama.cpp b6050, faster-whisper 1.1.1 and Kokoro 0.9.4. The customer lines are synthesized speech, sent over a real WebRTC connection by `scripts/scripted_call.py`.

The call covers:
- an out-of-stock item with the closest substitute
- a delivery slot held from live capacity
- an address change read back before saving
- a driver note
- the booking

![Recorded demo call transcript with per-reply latency](docs/assets/demo-transcript.svg)

---

## 2. Why this branch exists, and the stack

`main` uses the newest open models available when it was built (October 2026). This branch answers a different question: **how good would this agent be with what existed in mid-2025?** It's the same application code: domain layer, dialog manager, grounding and guards, tests. Only the framework, the model weights and the runtime differ.

| Component | Version on this branch | Released | On main instead |
|---|---|---|---|
| Voice framework | **Pipecat 0.0.77** | 2025-07-31 | Pipecat 1.12 |
| WebRTC UI | pipecat-ai-small-webrtc-prebuilt 1.0.0 | 2025-07-25 | prebuilt in the 1.x runner |
| End-of-turn model | **Smart Turn v2** (wav2vec2, ~95M params, GPU), snapshot `849c530` | 2025-07-25 | Smart Turn v3.2 (8 MB, CPU, Sept 2025+) |
| VAD | Silero VAD (bundled) | — | same |
| Speech → text | **faster-whisper 1.1.1**, CTranslate2 4.6, distil-small.en | 2025-01 / 2024 | faster-whisper 1.2 |
| Intent reader | **Qwen3-1.7B Q4_K_M**, unsloth GGUF `d7f544e` | 2025-06-08 | Qwen3.5-2B (2026) |
| LLM runtime | **llama.cpp b6050** (CUDA 12.4 Windows build) | 2025-07-31 | llama.cpp b11514 |
| Text → speech | **Kokoro-82M**, `kokoro` 0.9.4 / misaki 0.9.4 | 2025-04-05 | same |
| ML runtime | **torch 2.7.1** + CUDA 12.8, transformers 4.54.1, onnxruntime 1.20.1 | 2025-06 / 07 | torch 2.11 |
| Web | fastapi 0.116.1, uvicorn 0.35.0, aiortc 1.11, av 14.2.0 | ≤ 2025-07 | newer |

How "period-accurate" was enforced:
- **Python:** `uv pip install --exclude-newer 2025-08-01`. The only exception is torch 2.7.1, installed from PyTorch's CUDA index, which carries no upload dates; it was released in June 2025.
- **Models:** `scripts/download_models_jul2025.py` downloads each Hugging Face repo **at its last commit before 1 August 2025**. Smart Turn v2 was updated in September 2025, so the July snapshot matters.
- **llama.cpp:** `bash scripts/get_llama_cpp.sh b6050` fetches release b6050, published 2025-07-31 18:29 UTC.
- **One pin forced by platform:** `av==14.2.0`. PyAV 14.3 and 14.4 shipped no Windows wheels for Python 3.11.

Not available in July 2025, so not used here:
- Smart Turn v3
- NVIDIA Nemotron streaming ASR and Parakeet Realtime EOU
- Qwen3.5 and Qwen3-4B-Instruct-2507 (released 5 August 2025)
- Pipecat 1.x (its `UserBotLatencyObserver` breakdowns and turn strategies)
- Pipecat's built-in Kokoro service

---

## 3. What the agent does

The agent calls one customer about one pending order:

1. **Identity.** "Am I speaking with Arjun?" If not → the call is marked *wrong person* and ends politely.
2. **Order and shortages.** It reads the order aloud. If an item is short, it offers the most similar in-stock **substitute**, **the rest without it**, or **waiting for the restock**.
3. **Scheduling.** The customer's words ("tomorrow evening", "after 6", "the second one") are parsed by a rule-based time parser. The nearest free slot is **held for 5 minutes**; if the time is full, it offers the three nearest alternatives.
4. **Address.** A new address is **read back and saved only after an explicit yes**.
5. **Driver instructions,** then a **read-back and booking** on the final yes.

At any point the customer can **add an item** (checked against live stock), **ask a question** ("how much is the total?"), ask for a **callback**, **cancel** (which needs a second yes), or **interrupt** the agent mid-sentence.

---

## 4. Architecture

![System architecture on the July-2025 stack](docs/assets/architecture.svg)

| Layer | Component on this branch | What it does |
|---|---|---|
| Transport | Pipecat 0.0.77 `SmallWebRTCTransport`, prebuilt UI at `/client` | Browser audio in and out; echo cancellation in the browser. |
| Turn-taking | Silero VAD (200 ms) + **Smart Turn v2** (`LocalSmartTurnAnalyzerV2`, July snapshot, ~72 ms on the GPU) | Decides from the audio itself when the customer has finished. In 0.0.77 both are set on `TransportParams`. |
| Speech → text | faster-whisper distil-small.en, fp16, greedy (`GreedyWhisperSTTService`) | Pre-warmed with a throwaway decode when the customer starts speaking. |
| Dialog glue | `DialogProcessor` | One intent read per turn. Joins turns cut short by an interruption and drops replies that went stale. Uses the 0.0.77 frame types (`OpenAILLMContextFrame`, `StartInterruptionFrame`). |
| **The only LLM** | **Qwen3-1.7B Q4_K_M** on llama.cpp b6050 | Returns `{"intent", "value"}` only, restricted by a JSON schema to the intents valid in the current state. |
| Dialog logic | `DialogManager` | State machine, grounding checks, safety guards, templated replies. Identical to main, plus one guard (see §6). |
| Business rules | `CallSession` → domain services → SQLite | Stock, substitutions, 5-minute slot holds, booking, cancellation, callbacks. Identical to main. |
| Text → speech | Kokoro-82M (PyTorch, CUDA), custom 0.0.77 `TTSService` | Openers and the greeting are pre-synthesized, so first audio is instant. |
| Observability | `ResponseLatencyObserver` (written for this branch), barge-in observer, `CallRecorder`, dashboard | 0.0.77 has no latency breakdown observer. This one measures silence → first bot speech, adds the VAD wait, and keeps each service's TTFB metric. |

---

## 5. How one turn works (data flow)

![Latency of one turn on the July-2025 stack, compared with main](docs/assets/latency.svg)

1. **The customer stops talking.** Silero waits **200 ms**, then **Smart Turn v2** (about 72 ms on the GPU) decides whether the turn is complete. If it isn't sure, the system waits up to 1.2 s more.
2. **Speech → text.** Whisper, pre-warmed while the customer was speaking, takes a median of **146 ms**.
3. **The user aggregator** pushes one `OpenAILLMContextFrame`. Its late-transcript wait was cut from 0.0.77's default 0.5 s to **0.1 s**: `DialogProcessor` already joins late messages into the next turn.
4. **The intent reader** (Qwen3-1.7B, JSON schema) takes a median of **241 ms**:
   ```
   AGENT: Should we deliver to 230 BTM Layout, Bengaluru?
   CUSTOMER: No, deliver it to 42 Church Street, Bengaluru.
   →  {"intent": "change_address", "value": "42 Church Street, Bengaluru"}
   ```
5. **`DialogManager`** runs in under 2 ms. It validates the intent for the current state, checks the value is grounded in what the customer said, applies the guards, calls the tools, and returns template sentences.
6. **Each sentence becomes a `TTSSpeakFrame`.** The opener is cached and plays at once while Kokoro renders the rest.

---

## 6. The dialog: why the LLM doesn't call tools

![Dialog state machine](docs/assets/dialog-states.svg)

The design is the same as main's, and so is the reason for it. An earlier version used LLM function calling (Pipecat Flows). The small model invented tool calls (adding items, changing the address) and needed two LLM passes per turn, about 5.5 s per reply ([phase2-live-calls.md](docs/benchmarks/phase2-live-calls.md)).

| Concern | How it's handled |
|---|---|
| The LLM taking actions nobody asked for | The LLM only returns an intent label and the words that matter. Code decides and acts. |
| Intents that make no sense right now | A per-state JSON-schema enum; anything else → "Sorry, I didn't catch that." |
| Invented arguments | **Grounding:** values must come from the customer's own words. Quantities must be said. |
| Destructive actions | **Cancel** needs a second, non-negated yes. A **new address** is read back first. A **callback** needs a "not now" cue. A **substitute** is never applied if the customer says "skip", "without" or "no". This guard was added on this branch because Qwen3-1.7B read "Skip the bread then" as *substitute*. |
| Latency | One short LLM call per turn; template replies; cached openers. |

---

## 7. Business logic and data model

Identical to main. Plain Python over SQLite, with no knowledge of the LLM or audio, unit-tested with a fixed clock.

```
products(sku, name, spoken_name, category, unit_price)       stock(warehouse_id, sku, on_hand, reserved, restock_eta)
customers(id, name, phone, default_address)                  orders(id, customer_id, warehouse_id, status, address, notes, slot_id, ...)
order_lines(order_id, sku, qty, reserved_qty, substitute_sku, substitute_qty, resolution)
delivery_slots(id, warehouse_id, start_ts, end_ts, capacity, booked)    slot_holds(slot_id, order_id UNIQUE, expires_at)
calls(id, order_id, started_at, ended_at, outcome, transcript_json)     turn_metrics(id, call_id, kind, total_ms, breakdown_json, recorded_at)
```

- **Slot holds:** 5 minutes. Other orders' holds count against capacity.
- **Restock date:** booking re-checks it after a "wait for restock" choice.
- **Leaving `scheduled`** frees the seat and every stock reservation.
- **Time parser:** handles "tomorrow after 5 p.m.", "half past five", "next Monday", "as soon as possible" and ordinals.
- **Substitutes:** ranked by name similarity.
- **Seed data:** reproducible.

---

## 8. Model choices on a July-2025 budget

| Stage | Options available by 31 July 2025 | Chosen | Why |
|---|---|---|---|
| End of turn | Smart Turn v1 (Mar 2025), **v2 (Jul 2025)**, LiveKit's text-based detector | **Smart Turn v2** | Newest audio model available; ~72 ms on the GPU. Its 95M parameters cost more than v3's 8 MB. |
| Speech → text | Whisper / distil-whisper, Parakeet TDT 0.6B v2 (May 2025), Kyutai STT 1B (Jun 2025) | **distil-small.en** | On main's Phase 1 benchmarks: 1.9 % WER at ~98 ms. Parakeet on CPU took ~400 ms. Kyutai doesn't fit alongside everything else in 8 GB. |
| Intent reader | Qwen3 0.6B / **1.7B** / 4B (Apr 2025), Llama 3.2, Gemma 3n | **Qwen3-1.7B** | Same accuracy as Qwen3-4B (85.4 %) at almost half the latency (258 vs 457 ms p50). |
| Text → speech | **Kokoro-82M** (Jan 2025), Kyutai TTS 1.6B (Jun 2025), Chatterbox | **Kokoro** | Fastest open model on this GPU; openers are cached anyway. |

Intent-reader evaluation (`bench/intent_eval.py`, 48 labelled utterances, llama.cpp b6050):

| Model | Accuracy | Latency p50 / p95 | Harmful misreads |
|---|---|---|---|
| **Qwen3-1.7B Q4_K_M** | **85.4 %** | **258 / 409 ms** | "Skip the bread then" → substitute (now guarded); "Nothing, thanks" → note (guarded) |
| Qwen3-4B Q4_K_M | 85.4 % | 457 / 667 ms | "Skip the bread then" → substitute; "tell him to call first" → callback (guarded) |
| *(main) Qwen3.5-2B* | *93.8 %* | *324 / 435 ms* | *guarded* |

The remaining 1.7B misses that change nothing: "Who is this?" / "Hmm, let me think" read as questions, and "the second one" read as a time (handled the same way). See `bench/results/intent_qwen3-*-jul2025.json`.

---

## 9. Results

### Scripted calls on this stack (8 calls, 2 rounds)
| Call | Scenario | Result |
|---|---|---|
| A | happy path | booked ✔ |
| B | shortage ("send the rest without it") + add item (out of stock) | partial + booked; milk correctly refused ✔ |
| C | cancel ("cancel the whole order" → "yes") | cancelled after confirmation ✔ |
| D | substitute + address change + note (the demo) | substitute applied, address saved after read-back, booked ✔ |

**Unrequested order changes: 0.**

### Latency
| Run | Median | p95 | Notes |
|---|---|---|---|
| Round 1 | 1,170 ms | 2,263 ms | 0.0.77's default 0.5 s aggregator wait |
| **Round 2** | **864 ms** | 2,565 ms | aggregator wait cut to 0.1 s |
| *main, for reference* | *735–841 ms* | *~1,750 ms* | *Pipecat 1.12, Smart Turn v3.2, Qwen3.5-2B* |

Round-2 stage medians:
- 200 ms VAD silence
- 146 ms Whisper
- 241 ms intent reader
- ~277 ms for Smart Turn v2, the 0.1 s aggregator wait and pipeline hops
- ~0 ms first audio (cached opener)

The p95 comes from the intent reader's first, cold requests (up to 2.4 s) and occasional slow Whisper decodes.

**Verdict.** A mid-2025 stack gets within ~100 ms of the newest one at the median, with the same safety properties. The gap is mostly turn detection: v2 is heavier than v3. The newer stack's real advantage is intent accuracy (93.8 % vs 85.4 %), which the deterministic guards turn into a safety margin rather than a correctness requirement.

---

## 10. Porting notes: Pipecat 1.12 → 0.0.77

What changed in the code, for anyone moving between the two:

| Area | Pipecat 1.12 (main) | Pipecat 0.0.77 (this branch) |
|---|---|---|
| Run a pipeline | `PipelineWorker` + `WorkerRunner` | `PipelineTask` + `PipelineRunner` |
| Context frame | `LLMContextFrame` / `LLMContext` | `OpenAILLMContextFrame` / `OpenAILLMContext` |
| Interruption | `InterruptionFrame` | `StartInterruptionFrame` |
| VAD and turn analyzer | `LLMUserAggregatorParams(vad_analyzer=…, user_turn_strategies=…)` | `TransportParams(vad_analyzer=…, turn_analyzer=…)` |
| User aggregator | `LLMContextAggregatorPair` | `LLMUserContextAggregator(OpenAILLMContext(), params=LLMUserAggregatorParams(aggregation_timeout=0.1))` |
| TTS service | `run_tts(text, context_id)`, `Settings` objects | `run_tts(text)`; the service yields `TTSStartedFrame`/`TTSStoppedFrame` itself |
| Whisper | `WhisperModel` imported at module level | imported inside `_load()` (tests patch `faster_whisper.WhisperModel`) |
| Latency breakdown | `UserBotLatencyObserver` (per-stage contributions) | none: `ResponseLatencyObserver` written here (silence → bot speech + TTFB metrics) |
| Runner CLI | `main(parser)` accepts custom arguments | `main()` parses `sys.argv` itself: `--order-id`, `--db` and `--llm-url` are split off first |
| Tests | `run_test` returns all frames | `run_test` returns frames only with an exact expected list, and installs SIGINT handlers that Windows can't use: a local helper and a test fixture cover both |

---

## 11. Repository structure

```
src/voiceagent/
├── agent/          bot.py (0.0.77 wiring) · processor.py · intent.py · dialog.py · nlu.py · tools.py
│                   services.py (Whisper pre-warm, Kokoro TTS, Smart Turn v2) · metrics.py (own latency observer) · dashboard.py
├── domain/         timeparse · inventory · scheduling · orders · context · models     (same as main)
└── db/             schema.sql · repository.py · seed.py                                 (same as main)
bench/              intent_eval.py + intent_cases.json; Phase 1 model benchmarks; results in bench/results/
scripts/            download_models_jul2025.py · get_llama_cpp.sh (takes a build) · llama_server.sh (defaults to b6050)
                    scripted_call.py (WebRTC test caller with two-voice MP3) · chat_templates/
tests/              domain/ · agent/ · bench/
docs/               specs/ · superpowers/plans/ · benchmarks/ (incl. jul2025-stack.md) · assets/ (SVGs) · demo/demo-call.mp3
```

---

## 12. Running it yourself

**You need:** Windows with an NVIDIA GPU (8 GB+), Python **3.11**, Git Bash.

```bash
# 1. a separate, period-accurate environment
py -3.11 -m venv .venv-jul2025
.venv-jul2025/Scripts/python -m pip install uv
.venv-jul2025/Scripts/python -m uv pip install --python .venv-jul2025/Scripts/python.exe \
    torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv-jul2025/Scripts/python -m uv pip install --python .venv-jul2025/Scripts/python.exe \
    --exclude-newer 2025-08-01 -e ".[dev,agent]"

# 2. models at their July-2025 revisions + llama.cpp b6050
.venv-jul2025/Scripts/python scripts/download_models_jul2025.py
bash scripts/get_llama_cpp.sh b6050

# 3. data
.venv-jul2025/Scripts/python -m voiceagent.db.seed --db data/warehouse.db
```

**Start a call** (three terminals):
```bash
bash scripts/llama_server.sh models/jul2025/llm/Qwen3-1.7B-Q4_K_M.gguf       # 1: intent reader (llama.cpp b6050)
.venv-jul2025/Scripts/python -m voiceagent.agent.bot --order-id 2           # 2: the agent (Pipecat 0.0.77)
.venv-jul2025/Scripts/python -m voiceagent.agent.dashboard                  # 3: optional, http://localhost:7861
```

Open **http://localhost:7860/client**, click **Connect**, allow the microphone, and **use headphones**.

> **PowerShell:** use `.\.venv-jul2025\Scripts\python.exe` and run the commands from the repo folder. Red log lines in PowerShell are normal.

**Settings that matter:**

| Where | Setting | Default | Effect |
|---|---|---|---|
| `bot.py` | `VAD_STOP_SECS` | 0.2 s | silence before turn analysis |
| `bot.make_user_aggregator` | `aggregation_timeout` | 0.1 s | wait for late transcripts (0.0.77 default 0.5 s) |
| `services.make_turn_analyzer` | `SmartTurnParams(stop_secs=…)` | 1.2 s | max wait when a turn looks unfinished |
| `dialog.py` | `AGENT_NAME`, `COMPANY`, `ACKS` | Avio, AvioStack | persona and cached openers |
| `nlu.grounded` | `min_share` | 0.6 | how much of a value must be the customer's own words |

---

## 13. Testing and measurement tools

```bash
.venv-jul2025/Scripts/python -m pytest tests/domain tests/agent        # 194 tests on the July-2025 stack
.venv/Scripts/python -m bench.intent_eval --model-name qwen3-1.7b-q4km-jul2025   # with llama-server running
.venv/Scripts/python scripts/scripted_call.py --url http://localhost:7860/api/offer \
    --mix-out call.mp3 --duration 120 "20:Yes, speaking." "34:Tomorrow evening." "48:Yes, book that."
```

- **Unit tests** run every dialog path against an in-memory database. The intent reader is tested against mocked HTTP, and the pipeline glue in a real 0.0.77 pipeline.
- **The scripted caller** dials the agent over WebRTC exactly like the browser and writes a two-voice MP3.
- **The dashboard and `turn_metrics`** give per-call and per-turn latency.

---

## 14. Limitations and next steps

- **p95 ≈ 2.6 s.** The intent reader's first cold requests are slow. A warm-up request at bot start (as main does for Whisper and Kokoro) would remove most of it.
- **Turn detection** is the biggest gap to main. Smart Turn v2 is heavier and less accurate on short answers than v3, which arrived after this branch's cutoff.
- **Intent accuracy is 85.4 %, versus 93.8 % on main.** Every data-changing misread found so far is caught by a deterministic guard. A small fine-tune of Qwen3-1.7B on the dialog's intent set is the obvious lever if real calls show systematic errors.
- **Same open items as main:** only synthetic customer voices in the scripted calls, browser WebRTC rather than telephony, and the small follow-ups listed in main's README.

---

<sub>Built with Pipecat 0.0.77, faster-whisper, Smart Turn v2, Silero VAD, llama.cpp b6050, Qwen3 and Kokoro, all available by 31 July 2025, all open-weight and running locally.</sub>
