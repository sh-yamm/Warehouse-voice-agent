# Phase 1 model decision

Date: 2026-10-09   Hardware: RTX 4060 Laptop 8 GB (≈1.4 GB used by the display), Windows 11 (WDDM driver), on AC power
Corpus: 65 Kokoro-synthesized clips (50 complete, 15 incomplete). No human recordings yet, so every "human" column is n/a.
Raw numbers: [phase1-results.md](phase1-results.md)

| Stage | Choice | Key numbers | Rejected (why) |
|---|---|---|---|
| STT | **faster-whisper distil-small.en** (CUDA, fp16), whole-utterance decode at end of turn | WER 1.9 %, post-speech p50 98 ms / p95 152 ms, +585 MB VRAM | Nemotron-160: WER 6.0 % and RTF 1.3, so it cannot keep up in real time on this machine. Nemotron-560: WER 6.0 %, p50 241 ms, +2.9 GB. Parakeet TDT v2 on CPU: WER 3.5 % but p50 400 ms. Nemotron-80: crashes in the transformers processor (empty first chunk). |
| End of turn | **Smart Turn v3.2 (CPU)**, kept with a flag | p50 61 ms / p95 145 ms; complete-turn recall 0.98; incomplete recall 0.07 on TTS clips | p95 misses the plan's ≤ 50 ms bar. No alternative was measured; the GPU ONNX variant and thread tuning are latency-plan items. Incomplete recall is meaningless on TTS prosody and needs human recordings. |
| LLM | **Qwen3.5-2B Q4_K_M** (llama-server, non-thinking) | cold TTFT 237 ms; warm TTFT p50 176 / p95 218 ms; first clause p50 313 / p95 448 ms; 87 tok/s; GPU used peak 5.5 GB | Qwen3-4B-Instruct-2507: best warm TTFT (105 ms) but first-clause p95 811 ms, 27 tok/s, and a 7.6 GB peak, so it cannot share the GPU with STT + TTS. Qwen3.5-4B: TTFT p50 260 / p95 682 ms, first clause p50 526 ms, 7.2 GB peak. |
| TTS | **Kokoro v1.0, PyTorch** (`kokoro` package, CUDA) | first clause p50 395 ms / p95 463 ms, +889 MB | kokoro-onnx on CUDA: p50 612 ms (STFT falls back to CPU, 129 ms per run). |

## Correctness (LLM)

All three models invented at least one fact when given only the context block and no tools:
- Qwen3.5-4B said "one potato chip left"; the context says 0 of 2.
- Qwen3.5-2B moved today's slot 49 to "tomorrow".
- Qwen3-4B-2507 promised delivery of out-of-stock chips.

The plan's rule ("among models whose replies contained no invented items") selects nothing, so the choice above is made on latency and VRAM. Correctness has to come from the pipeline design: per-node tools and prompts (Pipecat Flows), deterministic business rules, and slot and stock facts returned by tools. After that, from the fine-tune. A fine-tuned 2B is now the main path, not an experiment.

## Budget check

Estimated voice-to-voice p50: 200 (VAD silence) + 61 (Smart Turn) + 98 (STT) + 313 (LLM first clause) + 395 (TTS first clause) = **1067 ms**. That is about 370 ms over the 700 ms target.

Estimated VRAM: 585 (STT) + ≈4100 (LLM: 5545 peak minus ≈1.4 GB baseline) + 889 (TTS) ≈ 5.6 GB, plus 1.4 GB for the display ≈ 7.0 GB of 8.2 GB.

## Observations

- **Windows WDDM kernel-launch overhead is the dominant latency tax.** The GPU sustains 28 fp16 TFLOPS, but each tiny kernel launch costs ≈ 48 µs. Kokoro (thousands of small kernels per clause) and streaming Nemotron (per-chunk generate) suffer most. llama.cpp is less affected because it uses CUDA graphs.
- **The biggest remaining costs are TTS (395 ms) and the LLM first clause (313 ms).** Levers for the latency plan:
  1. Pre-synthesized audio for frequent openers ("Sure.", "Okay.", "Got it.", "One moment.") played while the real clause renders.
  2. Shorter first clauses, enforced by prompt and fine-tune.
  3. Preemptive LLM generation on medium end-of-turn confidence.
  4. CUDA-graph capture of Kokoro's decoder with bucketed lengths.
  5. Re-measuring the whole stack under WSL2/Linux to quantify the WDDM tax.
- **Qwen3.5 (hybrid linear attention) reuses llama.cpp's prompt cache worse than Qwen3-2507** (standard attention): warm TTFT 260 vs 105 ms at the same size. A fine-tuned Qwen3-1.7B (standard attention) is worth adding to the fine-tuning comparison.
- **The STT choice holds only for short turns.** Whisper's whole-utterance decode cost grows with turn length. Turns longer than about 6 s (addresses) should be re-checked once human recordings exist.
- **Next measurement to do:** record the human corpus (`python -m bench.record_corpus`), then re-run `stt_bench --engine whisper-distil-small`, `turn_bench` and `report`.
