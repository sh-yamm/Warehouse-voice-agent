"""Download the July-2025 model set at the revisions that existed on 31 July 2025.

    python scripts/download_models_jul2025.py

Kokoro-82M (last commit 2025-04-10) and distil-whisper small.en (2024-01-22) have not changed since, so they are
loaded from their repos as usual. llama.cpp: bash scripts/get_llama_cpp.sh b6050  (published 2025-07-31).
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = Path(__file__).resolve().parents[1] / "models" / "jul2025"
PINS = {
    # repo: (revision, published, file or None for the whole snapshot)
    "unsloth/Qwen3-1.7B-GGUF": ("d7f544eead69", "2025-06-08", "Qwen3-1.7B-Q4_K_M.gguf"),
    "unsloth/Qwen3-4B-GGUF": ("22c9fc8a8c77", "2025-06-08", "Qwen3-4B-Q4_K_M.gguf"),
    "pipecat-ai/smart-turn-v2": ("849c530ae792", "2025-07-25", None),
}


def main() -> None:
    for repo, (revision, published, filename) in PINS.items():
        print(f"{repo} @ {revision} ({published})")
        if filename:
            hf_hub_download(repo, filename, revision=revision, local_dir=MODELS / "llm")
        else:
            snapshot_download(repo, revision=revision, local_dir=MODELS / repo.split("/")[1])
    print("done")


if __name__ == "__main__":
    main()
