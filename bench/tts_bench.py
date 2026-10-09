"""Kokoro TTS benchmark. Usage: python -m bench.tts_bench [--engine onnx|torch]"""
from __future__ import annotations

import argparse
import time

from bench.common import GpuMemorySampler, first_clause_end, summarize, write_result
from bench.tts_engine import load_kokoro

REPLIES = [
    "Hi, this is Asha calling from QuickMart about your grocery order. Am I speaking with Priya?",
    "Great, thanks. Your order has two Amul milk packets, eggs, and brown bread.",
    "The brown bread is out of stock today, but I can send whole wheat bread instead.",
    "Sure, I have a slot tomorrow between five and six in the evening. Shall I book it?",
    "That slot is full, but six to seven or four to five tomorrow are open.",
    "Done, your delivery is booked for tomorrow, five to six PM.",
    "Got it, I'll ask them to leave it with the security guard.",
    "No problem, I'll call you back in an hour.",
    "Sorry, could you say that again?",
    "Thank you, have a nice day.",
]
VOICE = "af_heart"


def load_synth(engine: str):
    """Return (synth(text) -> (samples, sample_rate), device label)."""
    if engine == "onnx":
        kokoro, provider = load_kokoro()
        return (lambda text: kokoro.create(text, voice=VOICE, lang="en-us")), provider
    import numpy as np
    import torch
    from kokoro import KModel, KPipeline

    model = KModel(repo_id="hexgrad/Kokoro-82M").to("cuda").eval()
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)

    def synth(text: str):
        chunks = [r.audio.cpu().numpy() for r in pipeline(text, voice=VOICE)]
        torch.cuda.synchronize()
        return np.concatenate(chunks), 24000

    return synth, "cuda (torch)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="onnx", choices=["onnx", "torch"])
    args = parser.parse_args()
    with GpuMemorySampler() as gpu:
        synth, provider = load_synth(args.engine)
        synth("Warming up the voice.")
        first_audio, rtfs = [], []
        for reply in REPLIES:
            clause = reply[: first_clause_end(reply) or len(reply)]
            t0 = time.perf_counter()
            synth(clause)
            first_audio.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            samples, sr = synth(reply)
            rtfs.append((time.perf_counter() - t0) / (len(samples) / sr))
    result = {"model": f"kokoro-v1.0-{args.engine}", "voice": VOICE, "first_clause_audio_ms": summarize(first_audio),
              "rtf": summarize(rtfs), "vram_delta_mb": gpu.delta_mb}
    path = write_result(f"tts_kokoro-{args.engine}", result, device=provider)
    print(f"kokoro ({provider}): first clause p50 {result['first_clause_audio_ms']['p50']} ms  "
          f"p95 {result['first_clause_audio_ms']['p95']} ms -> {path}")


if __name__ == "__main__":
    main()
