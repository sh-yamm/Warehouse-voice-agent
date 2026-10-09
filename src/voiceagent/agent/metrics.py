"""Per-turn latency, barge-in latency and call outcome, persisted to SQLite."""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import datetime

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, VADUserStartedSpeakingFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import LatencyBreakdown, MeasuredFrom, UserBotLatencyObserver

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
        super().__init__(observe_every_push=False, **kwargs)
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


def breakdown_to_dict(breakdown: LatencyBreakdown) -> dict:
    ms = lambda secs: round(secs * 1000, 1)
    return {
        "contributions": [[c.key, c.label, ms(c.duration_secs)] for c in breakdown.contributions],
        "ttfb": [[t.processor, ms(t.duration_secs)] for t in breakdown.ttfb],
        "user_turn_ms": ms(breakdown.user_turn_secs) if breakdown.user_turn_secs is not None else None,
    }


class CallRecorder:
    """Persists one call: a calls row, a turn_metrics row per measured turn or barge-in, outcome and transcript."""

    def __init__(self, repo: Repository, order_id: int, clock: Callable[[], datetime] = datetime.now):
        self.repo = repo
        self.clock = clock
        self.call_id = repo.start_call(order_id, clock())
        self._finished = False

    def record_breakdown(self, breakdown: LatencyBreakdown) -> None:
        kind = "greeting" if breakdown.measured_from == MeasuredFrom.CLIENT_CONNECTED else "response"
        self.repo.add_turn_metric(self.call_id, kind, breakdown.total_secs * 1000, breakdown_to_dict(breakdown),
                                  self.clock())

    def record_barge_in(self, elapsed_ms: float) -> None:
        self.repo.add_turn_metric(self.call_id, "barge_in", elapsed_ms, {}, self.clock())

    def finish(self, outcome: str | None, messages: list) -> None:
        """Close the call row once; later calls (disconnect after end_conversation) are ignored."""
        if self._finished:
            return
        self._finished = True
        transcript = [{"role": m["role"], "content": m["content"]} for m in messages
                      if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                      and isinstance(m.get("content"), str)]
        self.repo.finish_call(self.call_id, self.clock(), outcome or "abandoned", transcript)

    def observers(self) -> list[BaseObserver]:
        latency = UserBotLatencyObserver()

        @latency.event_handler("on_latency_breakdown")
        async def _on_breakdown(observer, breakdown):
            self.record_breakdown(breakdown)

        async def _on_barge_in(elapsed_ms: float):
            self.record_barge_in(elapsed_ms)

        return [latency, BargeInObserver(_on_barge_in)]
