"""Per-turn latency, barge-in latency and call outcome, persisted to SQLite.

July-2025 stack: Pipecat 0.0.77 has no latency observer with a per-part breakdown (that arrived in 1.x), so
ResponseLatencyObserver measures "VAD heard silence" -> "bot started speaking" itself, adds the VAD silence it
waited, and keeps the TTFB metrics services report in between. Observers see each frame once per hop, so frames
are de-duplicated by id.
"""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import datetime

from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, MetricsFrame,
                                   VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed

from voiceagent.db.repository import Repository


class BargeInTimer:
    """User speech onset while the agent is talking -> agent audio stopped (ms)."""

    def __init__(self, now: Callable[[], float] = time.perf_counter):
        self._now = now
        self._bot_speaking = False
        self._interrupted_at: float | None = None

    def bot_started(self) -> None:
        self._bot_speaking = True
        self._interrupted_at = None

    def user_started(self) -> None:
        if self._bot_speaking and self._interrupted_at is None:
            self._interrupted_at = self._now()

    def bot_stopped(self) -> float | None:
        self._bot_speaking = False
        if self._interrupted_at is None:
            return None
        elapsed_ms = (self._now() - self._interrupted_at) * 1000
        self._interrupted_at = None
        return elapsed_ms


class BargeInObserver(BaseObserver):
    def __init__(self, on_barge_in: Callable[[float], Awaitable[None]], **kwargs):
        super().__init__(**kwargs)
        self._timer = BargeInTimer()
        self._on_barge_in = on_barge_in

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, BotStartedSpeakingFrame):
            self._timer.bot_started()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._timer.user_started()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            elapsed_ms = self._timer.bot_stopped()
            if elapsed_ms is not None:
                await self._on_barge_in(elapsed_ms)


class ResponseLatencyObserver(BaseObserver):
    def __init__(self, on_response: Callable[[float, dict], Awaitable[None]],
                 on_greeting: Callable[[float], Awaitable[None]] | None = None, vad_stop_secs: float = 0.2,
                 now: Callable[[], float] = time.perf_counter, **kwargs):
        super().__init__(**kwargs)
        self._on_response, self._on_greeting = on_response, on_greeting
        self._vad_ms = vad_stop_secs * 1000
        self._now = now
        self._stopped_at: float | None = None
        self._last_stop_id = None
        self._seen_metrics: set = set()
        self._ttfb: list[list] = []
        self._connected_at: float | None = None

    def mark_connected(self) -> None:
        self._connected_at = self._now()

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if frame.id != self._last_stop_id:
                self._last_stop_id, self._stopped_at, self._ttfb = frame.id, self._now(), []
        elif isinstance(frame, MetricsFrame) and self._stopped_at is not None and frame.id not in self._seen_metrics:
            self._seen_metrics.add(frame.id)
            for item in frame.data:
                if isinstance(item, TTFBMetricsData) and item.value > 0:
                    self._ttfb.append([item.processor, round(item.value * 1000, 1)])
        elif isinstance(frame, BotStartedSpeakingFrame):
            if self._connected_at is not None:
                greeting_ms, self._connected_at = (self._now() - self._connected_at) * 1000, None
                if self._on_greeting:
                    await self._on_greeting(greeting_ms)
            if self._stopped_at is not None:
                after = round((self._now() - self._stopped_at) * 1000, 1)
                self._stopped_at = None
                breakdown = {
                    "contributions": [["endpointing_wait", "silence wait (VAD)", round(self._vad_ms, 1)],
                                      ["after_silence", "turn detection + STT + intent + first audio", after]],
                    "ttfb": self._ttfb,
                }
                await self._on_response(round(self._vad_ms + after, 1), breakdown)


class CallRecorder:
    """Persists one call: a calls row, a turn_metrics row per measured turn or barge-in, outcome and transcript."""

    def __init__(self, repo: Repository, order_id: int, clock: Callable[[], datetime] = datetime.now):
        self.repo = repo
        self.clock = clock
        self.call_id = repo.start_call(order_id, clock())
        self._finished = False
        self.latency: ResponseLatencyObserver | None = None

    def record_response(self, total_ms: float, breakdown: dict) -> None:
        self.repo.add_turn_metric(self.call_id, "response", total_ms, breakdown, self.clock())

    def record_greeting(self, elapsed_ms: float) -> None:
        self.repo.add_turn_metric(self.call_id, "greeting", elapsed_ms, {}, self.clock())

    def record_barge_in(self, elapsed_ms: float) -> None:
        self.repo.add_turn_metric(self.call_id, "barge_in", elapsed_ms, {}, self.clock())

    def finish(self, outcome: str | None, messages: list) -> None:
        """Close the call row once; later calls (disconnect after the call ended itself) are ignored."""
        if self._finished:
            return
        self._finished = True
        transcript = [{"role": m["role"], "content": m["content"]} for m in messages
                      if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                      and isinstance(m.get("content"), str)]
        self.repo.finish_call(self.call_id, self.clock(), outcome or "abandoned", transcript)

    def observers(self) -> list[BaseObserver]:
        async def on_response(total_ms: float, breakdown: dict):
            self.record_response(total_ms, breakdown)

        async def on_greeting(elapsed_ms: float):
            self.record_greeting(elapsed_ms)

        async def on_barge_in(elapsed_ms: float):
            self.record_barge_in(elapsed_ms)

        self.latency = ResponseLatencyObserver(on_response=on_response, on_greeting=on_greeting)
        return [self.latency, BargeInObserver(on_barge_in)]
