# Phase 3: intent-reader dialog, measured calls

Date: 2026-10-09. Hardware: RTX 4060 Laptop 8 GB, Windows 11, "Performance" power plan.

Stack:
- Pipecat 1.12 with Silero VAD and Smart Turn v3.2 (CPU, silence fallback 1.2 s)
- faster-whisper distil-small.en, greedy, pre-warmed at speech start
- Qwen3.5-2B Q4_K_M (llama-server) as the **intent reader only**
- deterministic `DialogManager`
- Kokoro-82M (PyTorch) with cached openers

## What changed since Phase 2

**Phase 2 design:** Pipecat Flows with LLM function calling. Two LLM passes per turn, and the LLM wrote the replies.

**Phase 3 design:**
- **One LLM call per turn.** It returns `{"intent", "value"}`, restricted by JSON schema to the intents valid in the current state.
- **Code runs the action and speaks a template.** Every reply opens with a pre-synthesized acknowledgement.
- **Extracted values must be grounded** in the customer's words.
- **Guards protect the actions that change or end the call:** cancel needs a second, non-negated "yes"; a callback needs a "not now" cue; a note made only of filler words means "no instructions".

## Scripted calls

Each call used a WebRTC test caller (`scripts/scripted_call.py`) speaking Kokoro-synthesized customer lines on a fixed schedule. Database freshly seeded.

| Call | Scenario | Customer said | Result | Correct? |
|---|---|---|---|---|
| A | happy path (order 2) | that's me / tomorrow evening / yes / yes / no instructions / book it | booked tomorrow 5–6 PM | yes |
| B | shortage + add item (order 1) | yes / send the rest without it / add two Amul milk / tomorrow after 6 / yes / yes / leave it with the guard / book it | chips dropped, booked tomorrow 7–8 PM, note saved | yes (see below) |
| C | cancel (order 3) | yes / cancel the whole order / yes, cancel it | asked to confirm, then cancelled | yes |

**Unrequested order changes: 0.** Phase 2 had 5 in one call.

In call B, voice activity detection split "Can you also add two Amul milk?" into two segments, and Whisper heard the second as "to amyl milk". The intent reader extracted `add_item` with value "tomorrow". The grounding check rejected it, so nothing was added and the agent asked "Which product would you like to add?". The scripted customer didn't answer, so the item was never added. That is the correct, safe behaviour.

## Latency (16 measured responses, round 2)

"Response" = customer stops speaking → first agent audio frame sent.

| | Phase 2 | Phase 3 round 1 | **Phase 3 round 2** |
|---|---|---|---|
| p50 | ~5,500 ms | 1,004 ms | **735 ms** |
| p95 | ~5,700 ms | 3,742 ms | **1,769 ms** |
| Whisper decode in-call | — (not logged) | 280–740 ms | **99–267 ms** |
| greeting after connect | 950 ms | 41–47 ms | **18–47 ms** |

Typical round-2 response breakdown, from the pipeline's latency observer:
- endpointing wait: 200 ms (VAD `stop_secs`)
- transcription: p50 146 ms
- intent reader plus pipeline hops: p50 379 ms (the reader alone was 312–324 ms p50 in the offline evaluation)
- first audio: ≈ 0 ms (cached acknowledgement)

### Fixes between rounds 1 and 2

1. **Smart Turn short-answer fallback.** A bare "Yes." was judged incomplete and waited out the 3 s silence fallback, giving 3.7 s turns. The fallback is now 1.2 s.
2. **Whisper idle penalty.**
   - After a few idle seconds, a decode took ~400 ms; back to back it took ~120 ms. This was measured offline: 3 s idle before each clip gives ~400 ms.
   - A GPU keep-warm thread did not help, which points at CPU/driver power states rather than GPU clocks alone.
   - A throwaway 0.5 s decode when the customer starts speaking did help, bringing it back to ~140 ms. That is now built into `GreedyWhisperSTTService`.

### Remaining outliers (p95)

Two of 16 responses took ~1.77 s:
- one where VAD split an utterance into two segments
- one first-turn reply

Neither involved the intent reader timing out.

## Intent reader evaluation (`bench/intent_eval.py`, 48 labelled utterances)

| Model | Accuracy | Grounded values | Latency p50 / p95 |
|---|---|---|---|
| Qwen3.5-2B Q4_K_M (prompt v1) | 0.875 | 1.000 | 318 / 379 ms |
| **Qwen3.5-2B Q4_K_M (prompt v2)** | **0.938** | 0.979 | 324 / 435 ms |
| Qwen3-4B-Instruct-2507 Q4_K_M (prompt v2) | 0.896 | 1.000 | 321 / 437 ms |

Remaining 2B misses, and what the dialog does with each:
- "Six to seven is good." read as `give_time` instead of `pick_option`. Harmless: the same slot is found.
- "I'm busy now, call me later." had a paraphrased value. Harmless: grounding drops it and the callback defaults to +1 h.
- "Nothing, thanks." read as `add_note`. **Guarded:** a filler-only note is treated as "no instructions".
- "Yes. Oh and tell him to call first." read as `callback`. **Guarded:** with no "not now" cue, it becomes a driver note.

**Fine-tuning decision: not now.**
- Both models are under the 95% bar, so the plan's rule technically permits fine-tuning.
- But every remaining 2B miss is harmless or caught by a deterministic guard, the 4B is worse, and 48 utterances is too small a set to fine-tune on without overfitting.
- **Revisit when** real call transcripts (human voices, accents, noise) show systematic intent errors. The evaluation harness is ready for that.

## Still above target

- **p50 is 735 ms against a 700 ms target.** The remaining parts are:
  - the fixed 200 ms VAD wait
  - about 150 ms for STT
  - about 380 ms for the intent reader plus hops
- **Next levers:**
  - run the intent reader on Whisper's text while Smart Turn is still deciding, in parallel instead of after
  - a smaller or faster reader model
  - a shorter `stop_secs` once tested on human speech
- **p95 is dominated by segmentation edge cases** (split utterances), not model speed.
- **All of this used synthesized customer voices.** Human recordings, with Indian-accented English, background noise and real pauses, are the next validation step.
