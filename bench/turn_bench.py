"""Smart Turn v3.2 benchmark (CPU). Usage: python -m bench.turn_bench [--source all]"""
from __future__ import annotations

import argparse
import time

import numpy as np

from bench.common import MODELS_DIR, load_manifest, load_wav16k, summarize, write_result

MODEL_PATH = MODELS_DIR / "smart-turn" / "smart-turn-v3.2-cpu.onnx"
WINDOW_S = 8


class SmartTurn:
    """Inference adapted from pipecat-ai/smart-turn inference.py."""

    def __init__(self, path=MODEL_PATH):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.features = WhisperFeatureExtractor(chunk_length=WINDOW_S)

    def predict(self, audio: np.ndarray) -> float:
        audio = audio[-WINDOW_S * 16000:]
        inputs = self.features(audio, sampling_rate=16000, return_tensors="np", padding="max_length",
                               max_length=WINDOW_S * 16000, truncation=True, do_normalize=True)
        features = inputs.input_features.squeeze(0).astype(np.float32)[None, ...]
        return float(self.session.run(None, {"input_features": features})[0][0].item())


def score(rows: list[dict], threshold: float = 0.5) -> dict:
    def recall(label: str) -> float:
        subset = [r for r in rows if r["label"] == label]
        hits = [(r["prob"] > threshold) == (label == "complete") for r in subset]
        return round(sum(hits) / len(hits), 3) if hits else None

    correct = [(r["prob"] > threshold) == (r["label"] == "complete") for r in rows]
    return {"accuracy": round(sum(correct) / len(correct), 3), "complete_recall": recall("complete"),
            "incomplete_recall": recall("incomplete"), "n": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all", choices=["all", "tts", "human"])
    args = parser.parse_args()
    model = SmartTurn()
    model.predict(np.zeros(16000, dtype=np.float32))  # warmup
    rows, latencies = [], []
    for row in load_manifest(args.source):
        audio = load_wav16k(row["path"])
        t0 = time.perf_counter()
        prob = model.predict(audio)
        latencies.append((time.perf_counter() - t0) * 1000)
        rows.append({"id": row["id"], "source": row["source"], "label": row["label"], "prob": round(prob, 4)})
    result = {
        "model": "smart-turn-v3.2-cpu",
        "latency_ms": summarize(latencies),
        "overall": score(rows),
        "by_source": {s: score([r for r in rows if r["source"] == s]) for s in dict.fromkeys(r["source"] for r in rows)},
        "note": "TTS clips are read with finished-sentence prosody; trust the 'human' scores for incomplete turns.",
        "clips": rows,
    }
    path = write_result("turn_smart-turn-v3.2", result, device="CPUExecutionProvider")
    print(f"smart-turn: p50 {result['latency_ms']['p50']} ms  acc {result['overall']['accuracy']} -> {path}")


if __name__ == "__main__":
    main()
