"""The only place STT, TTS and LLM services are constructed (swap models here)."""
from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.services.whisper import stt as whisper_stt
from pipecat.transcriptions.language import Language

KOKORO_SAMPLE_RATE = 24000


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
        # First CUDA decode is slow (kernel/cuBLAS init); pay it at load, not on the customer's first turn.
        self._model.transcribe(np.zeros(16000, dtype=np.float32), language="en")


class KokoroTorchTTSService(TTSService):
    """Kokoro-82M on PyTorch/CUDA: 395 ms first clause vs 612 ms for kokoro-onnx in Phase 1."""

    def __init__(self, *, voice: str = "af_heart", device: str = "cuda", cache_phrases: tuple[str, ...] = (),
                 **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True,
                         settings=TTSSettings(model="kokoro-82m", voice=voice, language=Language.EN), **kwargs)
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

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_tts_usage_metrics(text)
            cached = self._cache.get(text.strip())
            samples = cached if cached is not None else await asyncio.to_thread(self._synthesize, text)
            await self.stop_ttfb_metrics()
            if samples.size:
                audio = await self._resampler.resample(float_to_pcm16(samples), KOKORO_SAMPLE_RATE, self.sample_rate)
                yield TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1,
                                       context_id=context_id)
        except Exception as exc:
            logger.exception("Kokoro synthesis failed")
            yield ErrorFrame(error=f"Kokoro synthesis failed: {exc}")
        finally:
            await self.stop_ttfb_metrics()


def make_turn_analyzer():
    """Smart Turn v3.2 on CPU, warmed up (its first inference took 2.7 s in Phase 2).

    stop_secs caps how long a turn the model judged "incomplete" waits in silence before it is released anyway.
    The default 3 s made a bare "Yes." take 3.7 s in Phase 3 calls.
    """
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    analyzer = LocalSmartTurnAnalyzerV3(cpu_count=4, params=SmartTurnParams(stop_secs=1.2))
    analyzer._predict_endpoint(np.zeros(16000 * 2, dtype=np.float32))
    return analyzer
