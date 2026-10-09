import asyncio

import numpy as np

from voiceagent.agent import services


def test_float_to_pcm16_clips_and_scales():
    pcm = np.frombuffer(services.float_to_pcm16(np.array([0.0, 0.5, 2.0, -2.0], dtype=np.float32)), dtype=np.int16)
    assert pcm.tolist() == [0, 16383, 32767, -32767]


class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return [], None


def test_greedy_whisper_forces_greedy_decoding(monkeypatch):
    monkeypatch.setattr(services.whisper_stt, "WhisperModel", FakeWhisperModel)

    async def build():
        return services.GreedyWhisperSTTService(device="cpu")

    stt = asyncio.run(build())
    stt._model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
    assert stt._model.calls[-1] == {"beam_size": 1, "best_of": 1, "without_timestamps": True,
                                   "condition_on_previous_text": False, "language": "en"}


def test_make_llm_targets_local_server():
    llm = services.make_llm("http://127.0.0.1:8080/v1")
    assert str(llm._client.base_url).startswith("http://127.0.0.1:8080/v1")
    extra = llm._settings.extra["extra_body"]
    assert extra["chat_template_kwargs"] == {"enable_thinking": False} and extra["cache_prompt"] is True


def test_greedy_whisper_warms_up_on_load(monkeypatch):
    monkeypatch.setattr(services.whisper_stt, "WhisperModel", FakeWhisperModel)

    async def build():
        return services.GreedyWhisperSTTService(device="cpu")

    assert len(asyncio.run(build())._model.calls) == 1


def test_kokoro_torch_warms_up_on_construction(monkeypatch):
    import sys
    import types

    synthesized = []

    class FakeModel:
        def __init__(self, repo_id):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakePipeline:
        def __init__(self, lang_code, repo_id, model):
            pass

        def __call__(self, text, voice):
            synthesized.append(text)
            return []

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KModel=FakeModel, KPipeline=FakePipeline))

    async def build():
        return services.KokoroTorchTTSService()

    asyncio.run(build())
    assert len(synthesized) == 1
