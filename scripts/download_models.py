"""Download model files into models/. Usage: python scripts/download_models.py [turn] [tts] [stt] [llm]"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = Path(__file__).resolve().parents[1] / "models"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
LLMS = [
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf"),
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
]


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def main(groups: set[str]) -> None:
    if "turn" in groups:
        hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx", local_dir=MODELS / "smart-turn")
    if "tts" in groups:
        for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
            fetch(KOKORO_URL + name, MODELS / "kokoro" / name)
    if "stt" in groups:
        for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
            hf_hub_download("csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", name,
                            local_dir=MODELS / "parakeet-tdt-0.6b-v2-int8")
        snapshot_download("nvidia/nemotron-speech-streaming-en-0.6b",
                          ignore_patterns=["*.nemo", "*.gguf", "figures/*"])
    if "llm" in groups:
        for repo_id, filename in LLMS:
            hf_hub_download(repo_id, filename, local_dir=MODELS / "llm")
    print("done")


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"turn", "tts", "stt", "llm"})
