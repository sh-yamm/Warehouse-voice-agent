"""The only place STT, TTS and turn services are constructed (swap models here).

July-2025 stack: Pipecat 0.0.77 TTS API (run_tts(text), explicit TTSStarted/Stopped frames) and Smart Turn v2.
"""
from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import AsyncGenerator

from pathlib import Path

import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (ErrorFrame, Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame,
                                   VADUserStartedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.services.whisper import stt as whisper_stt

KOKORO_SAMPLE_RATE = 24000
# Smart Turn v2 snapshot as of 2025-07-25 (scripts/download_models_jul2025.py); the hub repo changed afterwards.
SMART_TURN_V2_DIR = Path(__file__).resolve().parents[3] / "models" / "jul2025" / "smart-turn-v2"


def float_to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class GreedyWhisperSTTService(whisper_stt.WhisperSTTService):
    """Whisper with greedy decoding, as benchmarked in Phase 1 (Pipecat's default beam search is slower)."""

    def _load(self):
        import torch  # noqa: F401  (torch/lib ships cublas64_12.dll, which CTranslate2 needs on Windows)

        super()._load()
        transcribe = functools.partial(self._model.transcribe, beam_size=1, best_of=1, without_timestamps=True,
                                       condition_on_previous_text=False)

        def timed_transcribe(audio, **kwargs):
            started = time.perf_counter()
            segments, info = transcribe(audio, **kwargs)
            segments = list(segments)  # decoding is lazy; run it here so the timing is real
            logger.info(f"whisper decode {(time.perf_counter() - started) * 1000:.0f} ms "
                        f"for {len(audio) / 16000:.1f} s audio")
            return segments, info

        self._model.transcribe = timed_transcribe
        self._greedy_transcribe = transcribe
        self._prewarming = False
        # First CUDA decode is slow (kernel/cuBLAS init); pay it at load, not on the customer's first turn.
        self._model.transcribe(np.zeros(16000, dtype=np.float32), language="en")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # After a few idle seconds a decode takes ~400 ms instead of ~120 ms on this laptop (CPU/GPU power states;
        # keeping the GPU busy alone did not help). A throwaway decode when the customer starts talking warms
        # CTranslate2 up, so the real decode after they stop is fast again.
        if isinstance(frame, VADUserStartedSpeakingFrame) and self._model is not None and not self._prewarming:
            self._prewarming = True
            asyncio.get_running_loop().run_in_executor(None, self._prewarm)
        await super().process_frame(frame, direction)

    def _prewarm(self) -> None:
        try:
            segments, _ = self._greedy_transcribe(np.zeros(8000, dtype=np.float32), language="en")
            list(segments)
        finally:
            self._prewarming = False


class KokoroTorchTTSService(TTSService):
    """Kokoro-82M on PyTorch/CUDA: 395 ms first clause vs 612 ms for kokoro-onnx in Phase 1."""

    def __init__(self, *, voice: str = "af_heart", device: str = "cuda", cache_phrases: tuple[str, ...] = (),
                 **kwargs):
        super().__init__(**kwargs)
        from kokoro import KModel, KPipeline

        self._voice = voice
        model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
        self._pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
        self._resampler = create_stream_resampler()
        self._cache: dict[str, np.ndarray] = {}
        # The first synthesis after loading takes seconds (CUDA init). TTSService gives up on a context after
        # stop_frame_timeout_s (3 s) without audio, which silenced the greeting, so warm up here.
        self._synthesize("Warming up.")
        for phrase in cache_phrases:
            self._cache[phrase.strip()] = self._synthesize(phrase)

    def can_generate_metrics(self) -> bool:
        return True

    def _synthesize(self, text: str) -> np.ndarray:
        chunks = [r.audio.cpu().numpy() for r in self._pipeline(text, voice=self._voice) if r.audio is not None]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def _audio_for(self, text: str) -> np.ndarray:
        cached = self._cache.get(text.strip())
        return cached if cached is not None else self._synthesize(text)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()
            cached = self._cache.get(text.strip())
            samples = cached if cached is not None else await asyncio.to_thread(self._synthesize, text)
            await self.stop_ttfb_metrics()
            if samples.size:
                audio = await self._resampler.resample(float_to_pcm16(samples), KOKORO_SAMPLE_RATE, self.sample_rate)
                yield TTSAudioRawFrame(audio, self.sample_rate, 1)
        except Exception as exc:
            logger.exception("Kokoro synthesis failed")
            yield ErrorFrame(error=f"Kokoro synthesis failed: {exc}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()


async def make_turn_analyzer():
    """Smart Turn v2 (2025-07-25 snapshot), warmed up so its first real inference is fast.

    stop_secs caps how long a turn the model judged "incomplete" waits in silence before it is released anyway
    (3 s default made a bare "Yes." take 3.7 s on the main branch).
    """
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2

    analyzer = LocalSmartTurnAnalyzerV2(smart_turn_model_path=str(SMART_TURN_V2_DIR),
                                        params=SmartTurnParams(stop_secs=1.2))
    await analyzer._predict_endpoint(np.zeros(16000 * 2, dtype=np.float32))
    return analyzer
