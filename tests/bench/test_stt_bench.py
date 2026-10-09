import numpy as np

from bench import stt_bench


class FakeEngine:
    name = "fake"
    device = "cpu"

    def warmup(self):
        pass

    def transcribe(self, audio):
        return "deliver it at 5 pm", 42.0


def test_run_engine_aggregates(monkeypatch):
    rows = [
        {"id": "c1", "path": "x.wav", "text": "Deliver it at 5 p.m.", "source": "tts"},
        {"id": "c2", "path": "y.wav", "text": "Deliver it at six p.m.", "source": "human"},
    ]
    monkeypatch.setattr(stt_bench, "load_wav16k", lambda path: np.zeros(16000, dtype=np.float32))
    result = stt_bench.run_engine(FakeEngine(), rows)
    assert result["engine"] == "fake"
    assert result["wer_by_source"] == {"tts": 0.0, "human": 0.2}
    assert result["wer_all"] == 0.1
    assert result["post_speech_ms"]["p50"] == 42.0
    assert len(result["clips"]) == 2
