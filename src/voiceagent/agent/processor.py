"""Sits where the LLM service would: one intent read per customer turn, then a deterministic spoken reply."""
from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.frames.frames import (EndTaskFrame, Frame, InterruptionFrame, LLMContextFrame, TTSSpeakFrame,
                                   UserStartedSpeakingFrame, UserStoppedSpeakingFrame, VADUserStartedSpeakingFrame,
                                   VADUserStoppedSpeakingFrame)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voiceagent.agent.dialog import DialogManager


def _text(message) -> str | None:
    if not isinstance(message, dict) or message.get("role") != "user":
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict)).strip()
    return None


def user_texts(context: LLMContext) -> list[str]:
    return [t for t in (_text(m) for m in context.get_messages()) if t is not None]


def last_user_text(context: LLMContext) -> str:
    texts = user_texts(context)
    return texts[-1] if texts else ""


class DialogProcessor(FrameProcessor):
    """Answers every customer turn exactly once.

    - Turns that were cut short (an interruption cancelled the reply) are not lost: all user messages not yet
      answered are joined and read together on the next turn ("Saturday..." + "...at ten").
    - If an interruption brings no new words (a cough, a door), the pending turn is answered after
      `recovery_secs` of quiet instead of leaving the call silent.
    """

    def __init__(self, manager: DialogManager, classifier, recovery_secs: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        self._manager = manager
        self._classifier = classifier
        self._recovery_secs = recovery_secs
        self._generation = 0
        self._context: LLMContext | None = None
        self._answered = 0
        self._lock = asyncio.Lock()
        self._watchdog: asyncio.Task | None = None

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self._on_interruption()
            await self.push_frame(frame, direction)
        elif isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            self._cancel_watchdog()  # they are talking; their own turn will arrive
            await self.push_frame(frame, direction)
        elif isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            if self._unanswered():
                self._arm_watchdog()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMContextFrame):
            if not getattr(frame, "speculation", False):
                self._cancel_watchdog()
                self._context = frame.context
                await self._respond()
        else:
            await self.push_frame(frame, direction)

    def _unanswered(self) -> list[str]:
        return user_texts(self._context)[self._answered:] if self._context is not None else []

    def _on_interruption(self) -> None:
        self._generation += 1
        if self._unanswered():
            self._arm_watchdog()

    def _arm_watchdog(self) -> None:
        self._cancel_watchdog()
        self._watchdog = self.create_task(self._recover())

    def _cancel_watchdog(self) -> None:
        if self._watchdog is not None and not self._watchdog.done():
            self._watchdog.cancel()
        self._watchdog = None

    async def _recover(self) -> None:
        await asyncio.sleep(self._recovery_secs)
        self._watchdog = None
        if self._unanswered():
            logger.info("answering a turn whose reply was interrupted without new words")
            await self._respond()

    async def _respond(self) -> None:
        async with self._lock:
            pending = self._unanswered()
            if not pending:
                return
            utterance = " ".join(pending)
            seen = self._answered + len(pending)
            generation = self._generation
            await self.start_ttfb_metrics()
            intent = await self._classifier.classify(self._manager.question, utterance,
                                                     self._manager.allowed_intents())
            await self.stop_ttfb_metrics()
            if generation != self._generation:
                logger.info(f"held reply to {utterance!r}: customer interrupted")
                return
            self._answered = seen
            state = self._manager.state
            reply = self._manager.handle(intent, utterance)
            logger.info(f"turn state={state} intent={intent.name} value={intent.value!r} utterance={utterance!r} "
                        f"-> {self._manager.state}: {' '.join(reply.sentences)}")
            for sentence in reply.sentences:
                await self.push_frame(TTSSpeakFrame(sentence))
            if reply.end_call:
                await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
