#!/usr/bin/env python3
"""
whisper_inference/transcribe.py

Streaming Whisper tiny.en inference with all optimisations from bench.py:
  - Chunked mid-speech flushing (parallel background transcription)
  - VAD-based silence detection
  - CUDA + int8 compute
  - Single serialised worker thread (no GPU contention)
  - Results saved to results.json after each utterance

Usage:
  python transcribe.py
  python transcribe.py --chunk-ms 600 --silence-ms 700
  python transcribe.py --output my_session.json
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.table import Table

# ── Audio constants ────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000
FRAME_MS      = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples per frame

# ── VAD defaults ──────────────────────────────────────────────────────────────
SPEECH_RMS_DEFAULT = 0.018
MIN_SPEECH_MS      = 200
SILENCE_MS_DEFAULT = 800
PADDING_MS         = 80
CHUNK_MS_DEFAULT   = 800   # mid-speech flush interval — lower = lower final latency

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_REPO   = "tiny.en"
COMPUTE_TYPE = "int8"       # fastest on any CUDA GPU; use float16 if you have >= 4 GB VRAM
DEVICE       = "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Utterance:
    id:              int
    timestamp:       str    # ISO-8601 wall-clock when utterance ended
    audio_secs:      float
    post_silence_ms: float  # silence-detected → full transcript ready  ← THE key latency
    final_chunk_ms:  float  # time to transcribe only the last chunk
    n_chunks:        int
    text:            str


@dataclass
class Session:
    model:       str
    compute:     str
    device:      str
    chunk_ms:    int
    silence_ms:  int
    started_at:  str
    utterances:  list[Utterance] = field(default_factory=list)

    # ── aggregate stats ───────────────────────────────────────────────────────
    def avg(self, attr: str) -> float:
        vals = [getattr(u, attr) for u in self.utterances]
        return sum(vals) / len(vals) if vals else 0.0

    def pct(self, attr: str, p: float) -> float:
        vals = sorted(getattr(u, attr) for u in self.utterances)
        if not vals:
            return 0.0
        return vals[min(int(len(vals) * p), len(vals) - 1)]

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.utterances:
            d["summary"] = {
                "total_utterances":   len(self.utterances),
                "avg_post_silence_ms": round(self.avg("post_silence_ms"), 1),
                "p50_post_silence_ms": round(self.pct("post_silence_ms", 0.50), 1),
                "p95_post_silence_ms": round(self.pct("post_silence_ms", 0.95), 1),
                "avg_final_chunk_ms":  round(self.avg("final_chunk_ms"), 1),
                "p95_final_chunk_ms":  round(self.pct("final_chunk_ms", 0.95), 1),
                "avg_audio_secs":      round(self.avg("audio_secs"), 2),
            }
        return d


# ─────────────────────────────────────────────────────────────────────────────
# VAD  — emits (chunk_audio, is_final)
# ─────────────────────────────────────────────────────────────────────────────

class VAD:
    """Energy-based VAD with mid-speech chunked flushing."""

    def __init__(self, speech_rms: float, silence_ms: int, chunk_ms: int):
        self.speech_rms        = speech_rms
        self.silence_frames    = silence_ms // FRAME_MS
        self.min_speech_frames = MIN_SPEECH_MS // FRAME_MS
        self.pad_frames        = PADDING_MS // FRAME_MS
        self.chunk_frames      = chunk_ms // FRAME_MS

        self._buf:    list[np.ndarray] = []
        self._pre:    list[np.ndarray] = []   # pre-roll padding
        self._silent  = 0
        self._speech  = 0
        self._active  = False
        self._frames_since_flush = 0

    @property
    def is_active(self) -> bool:
        return self._active

    def push(self, frame: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """
        Feed one audio frame.
        Returns:
          (audio, False)  — mid-speech chunk ready, keep recording
          (audio, True)   — end-of-utterance
          (None,  False)  — nothing to do yet
        """
        rms       = float(np.sqrt(np.mean(frame ** 2)))
        is_speech = rms >= self.speech_rms

        if is_speech:
            if not self._active:
                self._buf    = list(self._pre)
                self._active = True
                self._frames_since_flush = len(self._buf)
            self._buf.append(frame)
            self._speech += 1
            self._silent  = 0
            self._frames_since_flush += 1

            # mid-speech flush → send chunk to worker while user keeps talking
            if self._frames_since_flush >= self.chunk_frames:
                chunk = np.concatenate(self._buf)
                self._buf = []
                self._frames_since_flush = 0
                return chunk, False
        else:
            self._pre.append(frame)
            if len(self._pre) > self.pad_frames:
                self._pre.pop(0)

            if self._active:
                self._buf.append(frame)
                self._silent += 1
                if self._silent >= self.silence_frames:
                    keep = max(
                        len(self._buf) - self._silent + (PADDING_MS // FRAME_MS),
                        self.min_speech_frames,
                    )
                    utterance = None
                    if self._speech >= self.min_speech_frames:
                        utterance = np.concatenate(self._buf[:keep])
                    # reset
                    self._buf    = []
                    self._pre    = []
                    self._silent = 0
                    self._speech = 0
                    self._active = False
                    self._frames_since_flush = 0
                    return utterance, True

        return None, False


# ─────────────────────────────────────────────────────────────────────────────
# Transcription worker  — single serialised background thread
# ─────────────────────────────────────────────────────────────────────────────

class _Job(NamedTuple):
    audio:    np.ndarray
    is_final: bool
    result_q: queue.Queue


class TranscriptionWorker:
    """
    All faster-whisper calls run in one thread to avoid GPU context switching.

    Mid-speech chunks (is_final=False) → transcribed in background while user
    keeps talking. By the time silence is detected, most of the sentence is
    already done; only the last chunk remains.

    submit_final() blocks until the last chunk AND any queued partials are done,
    then returns the assembled full transcript.
    """

    def __init__(self, transcribe_fn):
        self._fn     = transcribe_fn
        self._q:     queue.Queue[_Job] = queue.Queue()
        self._parts: list[str]         = []
        self._lock   = threading.Lock()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        while True:
            job  = self._q.get()
            text = self._fn(job.audio)
            with self._lock:
                self._parts.append(text)
            job.result_q.put(text)
            self._q.task_done()

    def submit_partial(self, audio: np.ndarray) -> None:
        """Fire-and-forget mid-speech chunk."""
        rq: queue.Queue[str] = queue.Queue()
        self._q.put(_Job(audio, False, rq))

    def submit_final(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Submit last chunk; block until it and all pending partials finish.
        Returns (full_transcript, final_chunk_latency_secs).
        """
        rq: queue.Queue[str] = queue.Queue()
        self._q.put(_Job(audio, True, rq))
        t0 = time.perf_counter()
        rq.get()           # wait for this specific job
        self._q.join()     # drain any stragglers
        final_lat = time.perf_counter() - t0

        with self._lock:
            full_text = " ".join(p for p in self._parts if p).strip()
            self._parts.clear()

        return full_text, final_lat

    def clear(self):
        with self._lock:
            self._parts.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(compute_type: str) -> callable:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[ERROR] faster-whisper not installed.")
        print("  pip install faster-whisper")
        sys.exit(1)

    print(f"Loading faster-whisper [tiny.en]  (compute={compute_type}, device={DEVICE})")
    t0    = time.perf_counter()
    model = WhisperModel(MODEL_REPO, device=DEVICE, compute_type=compute_type)
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s — warming up CUDA kernels...")

    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
    t1    = time.perf_counter()
    list(model.transcribe(dummy, language="en", beam_size=1, without_timestamps=True)[0])
    print(f"Warmup done in {time.perf_counter() - t1:.2f}s — ready.\n")

    def transcribe(audio: np.ndarray) -> str:
        segs, _ = model.transcribe(
            audio,
            language="en",
            beam_size=1,          # greedy — fastest, negligible quality loss for short audio
            best_of=1,
            without_timestamps=True,
            vad_filter=False,     # we have our own VAD above
        )
        return " ".join(s.text.strip() for s in segs).strip()

    return transcribe


# ─────────────────────────────────────────────────────────────────────────────
# Rich display
# ─────────────────────────────────────────────────────────────────────────────

def _lat_color(ms: float) -> str:
    if ms < 150: return "bold green"
    if ms < 300: return "bold yellow"
    return "bold red"


def build_table(session: Session, state: str, partial: str = "") -> Table:
    state_label = {
        "listening":  "[bold green]● LISTENING[/bold green]",
        "recording":  "[bold red]● RECORDING[/bold red]",
        "processing": "[bold yellow]● PROCESSING[/bold yellow]",
        "loading":    "[bold cyan]● LOADING MODEL...[/bold cyan]",
    }.get(state, state)

    tbl = Table(
        title=f"[cyan]whisper tiny.en[/cyan]   {state_label}",
        show_header=True, header_style="bold",
        min_width=110, pad_edge=True,
    )
    tbl.add_column("#",              width=4,  style="dim")
    tbl.add_column("Audio",          width=7)
    tbl.add_column("Post-silence ↓", width=14, justify="right")
    tbl.add_column("Last chunk",     width=11, justify="right")
    tbl.add_column("Chunks",         width=7,  justify="right")
    tbl.add_column("Transcript")

    for u in session.utterances:
        ps = _lat_color(u.post_silence_ms)
        fc = _lat_color(u.final_chunk_ms)
        tbl.add_row(
            str(u.id),
            f"{u.audio_secs:.2f}s",
            f"[{ps}]{u.post_silence_ms:.0f}ms[/{ps}]",
            f"[{fc}]{u.final_chunk_ms:.0f}ms[/{fc}]",
            str(u.n_chunks),
            u.text,
        )

    if session.utterances:
        tbl.add_section()
        ap = session.avg("post_silence_ms")
        ps = _lat_color(ap)
        tbl.add_row(
            "avg", "",
            f"[{ps}]{ap:.0f}ms[/{ps}]",
            f"{session.avg('final_chunk_ms'):.0f}ms",
            "",
            f"[dim]P95 post-silence={session.pct('post_silence_ms', 0.95):.0f}ms"
            f"  P95 last-chunk={session.pct('final_chunk_ms', 0.95):.0f}ms[/dim]",
        )

    if partial:
        tbl.add_section()
        tbl.add_row("…", "", "", "", "", f"[dim italic]{partial}[/dim italic]")

    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# JSON persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_results(session: Session, path: Path) -> None:
    path.write_text(json.dumps(session.to_dict(), indent=2, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(
    silence_ms:  int,
    speech_rms:  float,
    compute_type: str,
    chunk_ms:    int,
    output_path: Path,
):
    console = Console()

    session = Session(
        model      = MODEL_REPO,
        compute    = compute_type,
        device     = DEVICE,
        chunk_ms   = chunk_ms,
        silence_ms = silence_ms,
        started_at = datetime.now().isoformat(timespec="seconds"),
    )

    # ── load model ────────────────────────────────────────────────────────────
    with Live(build_table(session, "loading"), console=console, refresh_per_second=4):
        fn = load_model(compute_type)

    worker = TranscriptionWorker(fn)
    vad    = VAD(speech_rms=speech_rms, silence_ms=silence_ms, chunk_ms=chunk_ms)
    aq: queue.Queue[np.ndarray] = queue.Queue()

    state         = "listening"
    partial       = ""
    speech_start  = 0.0
    utt_id        = 0

    def audio_cb(indata, frames, time_info, status):
        aq.put(indata[:, 0].copy())

    console.print(
        f"[dim]chunk_ms={chunk_ms}  silence_ms={silence_ms}  "
        f"speech_rms={speech_rms}  output={output_path}  Ctrl+C to stop[/dim]\n"
    )

    with Live(build_table(session, state), console=console, refresh_per_second=20) as live:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
            blocksize=FRAME_SAMPLES, callback=audio_cb,
        ):
            try:
                while True:
                    try:
                        frame = aq.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    chunk, is_final = vad.push(frame)

                    # ── state machine ─────────────────────────────────────────
                    new_state = (
                        "processing" if (chunk is not None and is_final)
                        else "recording" if vad.is_active
                        else "listening"
                    )
                    if new_state != state:
                        if new_state == "recording":
                            speech_start = time.perf_counter()
                            partial      = ""
                        state = new_state

                    # ── dispatch audio ────────────────────────────────────────
                    if chunk is not None:
                        if not is_final:
                            # mid-speech: background transcription runs in parallel
                            # with the user still talking → almost zero extra latency
                            worker.submit_partial(chunk)
                            partial = "[transcribing…]"
                            state   = "recording"
                        else:
                            # silence crossed — start the post-silence clock NOW
                            t_silence = time.perf_counter()
                            state     = "processing"
                            live.update(build_table(session, state, partial))

                            audio_secs = time.perf_counter() - speech_start

                            # block only until the last (small) chunk is done
                            text, final_chunk_lat = worker.submit_final(chunk)

                            post_silence_lat = time.perf_counter() - t_silence
                            n_chunks = max(1, round(audio_secs / (chunk_ms / 1000)))

                            if text:
                                utt_id += 1
                                utt = Utterance(
                                    id              = utt_id,
                                    timestamp       = datetime.now().isoformat(timespec="milliseconds"),
                                    audio_secs      = round(audio_secs, 3),
                                    post_silence_ms = round(post_silence_lat * 1000, 1),
                                    final_chunk_ms  = round(final_chunk_lat * 1000, 1),
                                    n_chunks        = n_chunks,
                                    text            = text,
                                )
                                session.utterances.append(utt)

                                # persist after every utterance — never lose data
                                save_results(session, output_path)

                            partial = ""
                            state   = "listening"

                    live.update(build_table(session, state, partial))

            except KeyboardInterrupt:
                pass

    # ── final save + summary ─────────────────────────────────────────────────
    save_results(session, output_path)

    if session.utterances:
        console.print(
            f"\n[bold]Session saved → {output_path}[/bold]\n"
            f"  turns            = {len(session.utterances)}\n"
            f"  avg post-silence = {session.avg('post_silence_ms'):.0f} ms\n"
            f"  P95 post-silence = {session.pct('post_silence_ms', 0.95):.0f} ms\n"
            f"  avg last-chunk   = {session.avg('final_chunk_ms'):.0f} ms\n"
        )
    else:
        console.print("[dim]No utterances recorded.[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Whisper tiny.en streaming inference — saves results to JSON"
    )
    parser.add_argument(
        "--compute-type", default=COMPUTE_TYPE,
        choices=["int8", "float16", "int8_float16"],
        help=f"CTranslate2 compute type (default: {COMPUTE_TYPE})",
    )
    parser.add_argument(
        "--silence-ms", default=SILENCE_MS_DEFAULT, type=int,
        help=f"Silence duration to end utterance (default: {SILENCE_MS_DEFAULT} ms)",
    )
    parser.add_argument(
        "--speech-rms", default=SPEECH_RMS_DEFAULT, type=float,
        help=f"RMS amplitude threshold for speech detection (default: {SPEECH_RMS_DEFAULT})",
    )
    parser.add_argument(
        "--chunk-ms", default=CHUNK_MS_DEFAULT, type=int,
        help=f"Mid-speech flush interval (default: {CHUNK_MS_DEFAULT} ms) — lower = less final latency",
    )
    parser.add_argument(
        "--output", default="results.json",
        help="Output JSON file path (default: results.json)",
    )
    args = parser.parse_args()

    run(
        silence_ms   = args.silence_ms,
        speech_rms   = args.speech_rms,
        compute_type = args.compute_type,
        chunk_ms     = args.chunk_ms,
        output_path  = Path(args.output),
    )


if __name__ == "__main__":
    main()