"""Synthesize the benchmark corpus with Kokoro. Usage: python -m bench.make_corpus"""
from __future__ import annotations

import json
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from bench.common import CORPUS_DIR, MANIFEST_PATH
from bench.tts_engine import load_kokoro

VOICES = ["af_heart", "am_michael", "bf_emma", "bm_george"]
PAD = np.zeros(int(0.2 * 16000), dtype=np.float32)


def to16k(samples: np.ndarray, sr: int) -> np.ndarray:
    g = gcd(16000, sr)
    return resample_poly(samples, 16000 // g, sr // g).astype(np.float32)


def main() -> None:
    kokoro, provider = load_kokoro()
    phrases = json.loads((CORPUS_DIR / "phrases.json").read_text(encoding="utf-8"))
    wav_dir = CORPUS_DIR / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    if MANIFEST_PATH.exists():
        existing = [json.loads(l) for l in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in existing if r["source"] != "tts"]
    for i, phrase in enumerate(phrases):
        voice = VOICES[i % len(VOICES)]
        lang = "en-us" if voice.startswith("a") else "en-gb"
        samples, sr = kokoro.create(phrase["text"], voice=voice, lang=lang)
        audio = np.concatenate([PAD, to16k(samples, sr), PAD])
        rel = f"wav/{phrase['id']}_{voice}.wav"
        sf.write(CORPUS_DIR / rel, audio, 16000)
        rows.append({**phrase, "path": rel, "voice": voice, "source": "tts"})
    MANIFEST_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(phrases)} tts clips ({provider}); manifest has {len(rows)} rows")


if __name__ == "__main__":
    main()
