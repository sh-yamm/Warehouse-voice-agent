"""Sits where the LLM service would: one intent read per customer turn, then a deterministic spoken reply."""
from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import EndTaskFrame, Frame, InterruptionFrame, LLMContextFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voiceagent.agent.dialog import DialogManager


def last_user_text(context: LLMContext) -> str:
    for message in reversed(context.get_messages()):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict)).strip()
    return ""


class DialogProcessor(FrameProcessor):
    def __init__(self, manager: DialogManager, classifier, **kwargs):
        super().__init__(**kwargs)
        self._manager = manager
        self._classifier = classifier
        self._generation = 0

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self._generation += 1
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMContextFrame):
            if not getattr(frame, "speculation", False):
                await self._respond(frame.context)
        else:
            await self.push_frame(frame, direction)

    async def _respond(self, context: LLMContext) -> None:
        utterance = last_user_text(context)
        if not utterance:
            return
        generation = self._generation
        await self.start_ttfb_metrics()
        intent = await self._classifier.classify(self._manager.question, utterance, self._manager.allowed_intents())
        await self.stop_ttfb_metrics()
        if generation != self._generation:
            logger.info(f"dropped reply to {utterance!r}: customer interrupted")
            return
        state = self._manager.state
        reply = self._manager.handle(intent, utterance)
        logger.info(f"turn state={state} intent={intent.name} value={intent.value!r} utterance={utterance!r} "
                    f"-> {self._manager.state}: {' '.join(reply.sentences)}")
        for sentence in reply.sentences:
            await self.push_frame(TTSSpeakFrame(sentence))
        if reply.end_call:
            await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
