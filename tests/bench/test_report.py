from bench.report import render


def test_render_has_a_row_per_result():
    results = [
        {"name": "stt_parakeet-cpu", "device": "cpu", "engine": "parakeet-cpu", "wer_all": 0.081,
         "wer_by_source": {"tts": 0.081}, "post_speech_ms": {"p50": 90.0, "p95": 140.0},
         "rtf": {"mean": 0.05}, "vram_delta_mb": 0},
        {"name": "tts_kokoro", "device": "CUDAExecutionProvider", "model": "kokoro-v1.0",
         "first_clause_audio_ms": {"p50": 60.0, "p95": 80.0}, "rtf": {"mean": 0.04}, "vram_delta_mb": 500},
        {"name": "turn_smart-turn-v3.2", "device": "CPUExecutionProvider", "model": "smart-turn-v3.2-cpu",
         "latency_ms": {"p50": 12.0, "p95": 15.0}, "overall": {"accuracy": 0.9},
         "by_source": {"tts": {"accuracy": 0.9, "incomplete_recall": 0.6}}},
        {"name": "llm_qwen3.5-4b-q4km", "device": "cuda", "model": "qwen3.5-4b-q4km", "cold_ttft_ms": 400.0,
         "warm": {"ttft_ms": {"p50": 80.0, "p95": 120.0}, "first_clause_ms": {"p50": 200.0, "p95": 300.0},
                  "tokens_per_s": {"p50": 70.0}}, "gpu_used_mb_peak": 5000},
    ]
    md = render(results)
    assert "| parakeet-cpu | cpu | 8.1% |" in md
    assert "| kokoro-v1.0 | CUDAExecutionProvider | 60.0 | 80.0 |" in md
    assert "| smart-turn-v3.2-cpu | 12.0 | 15.0 | 0.9 |" in md
    assert "| qwen3.5-4b-q4km | 400.0 | 80.0 | 120.0 | 200.0 | 300.0 | 70.0 |" in md
