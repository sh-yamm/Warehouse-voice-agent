"""Record yourself reading corpus phrases. Usage: python -m bench.record_corpus [--ids c01 c05 ...]

For each phrase: press Enter to start, read it naturally, press Enter to stop.
Incomplete phrases: stop mid-thought exactly where the text ends.
"""
from __future__ import annotations

import argparse
import json
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

from bench.common import CORPUS_DIR, MANIFEST_PATH


def record_until_enter() -> np.ndarray:
    chunks: list[np.ndarray] = []
    stop = threading.Event()

    def callback(indata, frames, time_info, status):
        chunks.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=callback):
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        stop.wait()
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", help="phrase ids to record (default: all)")
    args = parser.parse_args()
    phrases = json.loads((CORPUS_DIR / "phrases.json").read_text(encoding="utf-8"))
    if args.ids:
        phrases = [p for p in phrases if p["id"] in set(args.ids)]
    rows = []
    if MANIFEST_PATH.exists():
        rows = [json.loads(l) for l in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    recorded_ids = {p["id"] for p in phrases}
    rows = [r for r in rows if not (r["source"] == "human" and r["id"] in recorded_ids)]
    (CORPUS_DIR / "wav").mkdir(parents=True, exist_ok=True)
    for phrase in phrases:
        input(f"\n[{phrase['id']}] ({phrase['label']}) \"{phrase['text']}\"  -- Enter to start")
        print("recording... Enter to stop")
        audio = record_until_enter()
        rel = f"wav/{phrase['id']}_human.wav"
        sf.write(CORPUS_DIR / rel, audio, 16000)
        rows.append({**phrase, "path": rel, "voice": "human", "source": "human"})
        print(f"saved {rel} ({len(audio) / 16000:.1f}s)")
    MANIFEST_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
