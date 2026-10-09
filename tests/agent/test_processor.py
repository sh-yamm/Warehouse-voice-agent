import asyncio

import pytest
from pipecat.frames.frames import EndTaskFrame, LLMContextFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test

from voiceagent.agent.dialog import DialogManager
from voiceagent.agent.intent import Intent
from voiceagent.agent.processor import DialogProcessor, last_user_text
from voiceagent.agent.tools import CallSession

from conftest import NOW


class FakeClassifier:
    def __init__(self, intent, on_classify=None):
        self.intent, self.on_classify, self.calls = intent, on_classify, []

    async def classify(self, agent_said, customer_said, allowed):
        self.calls.append((agent_said, customer_said, allowed))
        if self.on_classify:
            self.on_classify()
        return self.intent


@pytest.fixture
def manager(world, repo):
    return DialogManager(CallSession(repo, 1, clock=lambda: NOW))


def context(text):
    return LLMContext(messages=[{"role": "user", "content": text}])


def spoken(frames):
    return [f.text for f in frames if isinstance(f, TTSSpeakFrame)]


def test_last_user_text():
    ctx = LLMContext(messages=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "x"},
                               {"role": "user", "content": [{"type": "text", "text": "second"}]}])
    assert last_user_text(ctx) == "second"
    assert last_user_text(LLMContext()) == ""


def test_speaks_reply_sentences(manager):
    classifier = FakeClassifier(Intent("confirm"))
    down, up = asyncio.run(run_test(DialogProcessor(manager, classifier),
                                    frames_to_send=[LLMContextFrame(context=context("Yes, this is Priya."))]))
    assert spoken(down)[:2] == ["Great.", "Your order has 2 Amul milk, 1 brown bread and 1 eggs."]
    agent_said, customer_said, allowed = classifier.calls[0]
    assert customer_said == "Yes, this is Priya." and "confirm" in allowed
    assert agent_said.endswith("Am I speaking with Priya?")


def test_end_call_pushes_end_task_upstream(manager):
    down, up = asyncio.run(run_test(DialogProcessor(manager, FakeClassifier(Intent("wrong_person"))),
                                    frames_to_send=[LLMContextFrame(context=context("Wrong number."))]))
    assert any(isinstance(f, EndTaskFrame) for f in up)


def test_reply_dropped_after_interruption(manager):
    processor = None

    def barge_in():
        processor._generation += 1  # what an InterruptionFrame does while the reader is running

    processor = DialogProcessor(manager, FakeClassifier(Intent("confirm"), on_classify=barge_in))
    down, up = asyncio.run(run_test(processor, frames_to_send=[LLMContextFrame(context=context("Yes."))]))
    assert spoken(down) == [] and manager.state == "greet"
