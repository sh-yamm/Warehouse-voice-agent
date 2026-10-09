from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CORPUS_DIR = ROOT / "bench" / "corpus"
RESULTS_DIR = ROOT / "bench" / "results"
MANIFEST_PATH = CORPUS_DIR / "manifest.jsonl"

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50",
}


def percentile(values, p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(math.ceil(p / 100 * len(ordered)) - 1, 0)
    return ordered[min(index, len(ordered) - 1)]


def summarize(values) -> dict:
    values = list(values)
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    r = lambda v: round(float(v), 1)
    return {"n": len(values), "p50": r(percentile(values, 50)), "p95": r(percentile(values, 95)),
            "mean": r(sum(values) / len(values)), "min": r(min(values)), "max": r(max(values))}


def normalize_text(s: str) -> str:
    t = s.lower().replace("p.m.", "pm").replace("a.m.", "am")
    t = t.replace("o'clock", " ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t).replace("'", "")
    return " ".join(_NUM_WORDS.get(w, w) for w in t.split())


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1], len(r)


def corpus_wer(pairs) -> float:
    edits = words = 0
    for ref, hyp in pairs:
        e, n = word_errors(ref, hyp)
        edits, words = edits + e, words + n
    return edits / words if words else 0.0


def _end_after(text: str, pattern: str, min_words: int) -> int | None:
    for m in re.finditer(pattern, text):
        if len(text[: m.end()].split()) >= min_words:
            return m.end()
    return None


def first_clause_end(text: str) -> int | None:
    return _end_after(text, r"[,;:.!?]", 2)


def first_sentence_end(text: str) -> int | None:
    return _end_after(text, r"[.!?]", 1)


class GpuMemorySampler:
    """Samples total GPU memory used (MB) via nvidia-smi while the block runs."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.baseline_mb: int | None = None
        self.peak_mb: int | None = None

    @staticmethod
    def read_mb() -> int | None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            value = self.read_mb()
            if value is not None and (self.peak_mb is None or value > self.peak_mb):
                self.peak_mb = value

    def __enter__(self) -> "GpuMemorySampler":
        self.baseline_mb = self.peak_mb = self.read_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def delta_mb(self) -> int | None:
        if self.baseline_mb is None or self.peak_mb is None:
            return None
        return self.peak_mb - self.baseline_mb


def write_result(name: str, payload: dict, device: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"name": name, "device": device, "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "machine": platform.node(), **payload}
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_manifest(source: str = "all") -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise SystemExit("bench/corpus/manifest.jsonl missing: run `python -m bench.make_corpus` first")
    rows = [json.loads(l) for l in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows if source == "all" else [r for r in rows if r["source"] == source]


def load_wav16k(path):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(CORPUS_DIR / path, dtype="float32", always_2d=True)
    if sr != 16000:
        raise ValueError(f"{path}: expected 16 kHz, got {sr}")
    return np.ascontiguousarray(audio[:, 0])
