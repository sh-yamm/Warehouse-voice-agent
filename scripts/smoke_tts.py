"""GPU smoke test: synthesize one clause with KokoroTorchTTSService. Usage: python scripts/smoke_tts.py"""
import time

from voiceagent.agent.services import KokoroTorchTTSService

tts = KokoroTorchTTSService()
tts._synthesize("Warming up.")
t0 = time.perf_counter()
samples = tts._synthesize("Sure, I have a slot tomorrow between five and six.")
print(f"{len(samples) / 24000:.2f}s of audio in {(time.perf_counter() - t0) * 1000:.0f} ms")
