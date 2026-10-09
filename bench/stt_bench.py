"""STT benchmark. Usage: python -m bench.stt_bench --engine nemotron-160 [--source all]"""
from __future__ import annotations

import argparse
import time
from threading import Thread

import numpy as np

from bench.common import (MODELS_DIR, GpuMemorySampler, corpus_wer, load_manifest, load_wav16k, summarize,
                          write_result)

NEMOTRON_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
NEMOTRON_LOOKAHEAD = {"nemotron-80": 0, "nemotron-160": 1, "nemotron-560": 6}


class NemotronStreaming:
    """Cache-aware streaming RNNT via transformers (pattern from the model card)."""

    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        self.name, self.device = name, "cuda"
        self.processor = AutoProcessor.from_pretrained(NEMOTRON_ID)
        self.model = AutoModelForRNNT.from_pretrained(NEMOTRON_ID, dtype=torch.float32).to("cuda").eval()
        self.processor.set_num_lookahead_tokens(NEMOTRON_LOOKAHEAD[name])
        self.sr = self.processor.feature_extractor.sampling_rate
        self.streaming_latency_ms = self.processor.streaming_latency_ms

    def warmup(self) -> None:
        self.transcribe(np.zeros(self.sr, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        from transformers import TextIteratorStreamer

        p, model = self.processor, self.model
        # trailing silence flushes the right context, as the VAD silence does in the live pipeline
        audio = np.concatenate([audio, np.zeros(2 * p.num_samples_per_audio_chunk, dtype=np.float32)])
        first = p(audio[: p.num_samples_first_audio_chunk], sampling_rate=self.sr, is_streaming=True,
                  is_first_audio_chunk=True, return_tensors="pt").to(model.device, dtype=model.dtype)
        marks: dict[str, float] = {}

        def features():
            yield first.input_features[:, : p.num_mel_frames_first_audio_chunk, :]
            mel_idx = p.num_mel_frames_first_audio_chunk
            hop, n_fft = p.feature_extractor.hop_length, p.feature_extractor.n_fft
            start = mel_idx * hop - n_fft // 2
            while (end := start + p.num_samples_per_audio_chunk) < audio.shape[0]:
                chunk = p(audio[start:end], sampling_rate=self.sr, is_streaming=True, is_first_audio_chunk=False,
                          return_tensors="pt").to(model.device, dtype=model.dtype)
                marks["last_yield"] = time.perf_counter()
                yield chunk.input_features
                mel_idx += p.num_mel_frames_per_audio_chunk
                start = mel_idx * hop - n_fft // 2

        streamer = TextIteratorStreamer(p.tokenizer, skip_special_tokens=True)
        thread = Thread(target=model.generate,
                        kwargs={**first, "input_features": features(), "streamer": streamer})
        thread.start()
        text = "".join(streamer)
        thread.join()
        done = time.perf_counter()
        return text.strip(), (done - marks.get("last_yield", done)) * 1000


class ParakeetSherpaCpu:
    def __init__(self, name: str):
        import sherpa_onnx

        d = MODELS_DIR / "parakeet-tdt-0.6b-v2-int8"
        self.name, self.device = name, "cpu"
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
            num_threads=4, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
            model_type="nemo_transducer",
        )

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip(), (time.perf_counter() - t0) * 1000


class FasterWhisper:
    def __init__(self, name: str):
        import torch  # noqa: F401  (puts torch/lib, which ships cublas64_12.dll, on the Windows DLL path)
        from faster_whisper import WhisperModel

        self.name, self.device = name, "cuda"
        self.model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, without_timestamps=True,
                                            vad_filter=False, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments)
        return text.strip(), (time.perf_counter() - t0) * 1000


def build_engine(name: str):
    if name in NEMOTRON_LOOKAHEAD:
        return NemotronStreaming(name)
    if name == "parakeet-cpu":
        return ParakeetSherpaCpu(name)
    if name == "whisper-distil-small":
        return FasterWhisper(name)
    raise SystemExit(f"unknown engine {name}")


def run_engine(engine, rows: list[dict]) -> dict:
    clips = []
    for row in rows:
        audio = load_wav16k(row["path"])
        t0 = time.perf_counter()
        text, post_ms = engine.transcribe(audio)
        total_s = time.perf_counter() - t0
        clips.append({"id": row["id"], "source": row["source"], "ref": row["text"], "hyp": text,
                      "post_speech_ms": round(post_ms, 1), "rtf": round(total_s / max(len(audio) / 16000, 1e-6), 3)})
    by_source = {}
    for source in dict.fromkeys(c["source"] for c in clips):
        by_source[source] = round(corpus_wer([(c["ref"], c["hyp"]) for c in clips if c["source"] == source]), 4)
    return {
        "engine": engine.name,
        "wer_all": round(corpus_wer([(c["ref"], c["hyp"]) for c in clips]), 4),
        "wer_by_source": by_source,
        "post_speech_ms": summarize([c["post_speech_ms"] for c in clips]),
        "rtf": summarize([c["rtf"] for c in clips]),
        "clips": clips,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True,
                        choices=[*NEMOTRON_LOOKAHEAD, "parakeet-cpu", "whisper-distil-small"])
    parser.add_argument("--source", default="all", choices=["all", "tts", "human"])
    args = parser.parse_args()
    rows = load_manifest(args.source)
    with GpuMemorySampler() as gpu:
        engine = build_engine(args.engine)
        engine.warmup()
        result = run_engine(engine, rows)
    result["vram_delta_mb"] = gpu.delta_mb
    result["streaming_latency_ms"] = getattr(engine, "streaming_latency_ms", None)
    path = write_result(f"stt_{args.engine}", result, device=engine.device)
    print(f"{args.engine}: WER {result['wer_all']:.3f}  post-speech p50 {result['post_speech_ms']['p50']} ms  "
          f"p95 {result['post_speech_ms']['p95']} ms  VRAM +{gpu.delta_mb} MB -> {path}")


if __name__ == "__main__":
    main()
