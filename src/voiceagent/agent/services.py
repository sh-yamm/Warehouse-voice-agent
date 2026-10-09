"""The only place STT, TTS and LLM services are constructed (swap models here)."""
from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.openai.llm import OpenAILLMService
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
        self._model.transcribe = functools.partial(
            self._model.transcribe, beam_size=1, best_of=1, without_timestamps=True,
            condition_on_previous_text=False,
        )


class KokoroTorchTTSService(TTSService):
    """Kokoro-82M on PyTorch/CUDA: 395 ms first clause vs 612 ms for kokoro-onnx in Phase 1."""

    def __init__(self, *, voice: str = "af_heart", device: str = "cuda", **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True,
                         settings=TTSSettings(model="kokoro-82m", voice=voice, language=Language.EN), **kwargs)
        from kokoro import KModel, KPipeline

        self._voice = voice
        model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
        self._pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
        self._resampler = create_stream_resampler()

    def can_generate_metrics(self) -> bool:
        return True

    def _synthesize(self, text: str) -> np.ndarray:
        chunks = [r.audio.cpu().numpy() for r in self._pipeline(text, voice=self._voice) if r.audio is not None]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_tts_usage_metrics(text)
            samples = await asyncio.to_thread(self._synthesize, text)
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


def make_llm(base_url: str = "http://127.0.0.1:8080/v1") -> OpenAILLMService:
    """Qwen3.5-2B (or whatever GGUF llama-server was started with), non-thinking, prompt cache on."""
    return OpenAILLMService(
        base_url=base_url,
        api_key="local",
        settings=OpenAILLMService.Settings(
            model="local",
            temperature=0.3,
            max_tokens=120,
            extra={"extra_body": {"cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}},
        ),
    )
