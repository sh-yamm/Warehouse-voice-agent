# Intent-Reader Dialog Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove both failures found in Phase 2:
- **Unrequested, invented tool calls:** the agent changed orders nobody asked to change.
- **Slow replies:** about 5.5 s per turn.

The fix: the LLM only reads the customer's intent, and deterministic code acts and speaks templated replies with cached openers.

**Architecture:**
- **One LLM call per user turn.** It returns `{"intent", "value"}`, with `intent` restricted by JSON schema to the intents valid in the current dialog state.
- **`DialogManager`** (pure Python, fully unit-tested) checks that extracted values are grounded in the customer's words, calls the existing `CallSession` tools, and returns template sentences.
- **`DialogProcessor`** replaces the LLM service in the Pipecat pipeline. It turns the reply into `TTSSpeakFrame`s, one per sentence. The first sentence is a short acknowledgement whose audio Kokoro has already cached.
- **Removed:** Pipecat Flows and LLM function calling.

**Tech Stack:** Python 3.11, Pipecat 1.12 (no Flows), llama-server JSON-schema output with Qwen3.5-2B, faster-whisper, Kokoro (PyTorch), pytest, httpx `MockTransport`, `pipecat.tests.utils.run_test`.

**Spec:** `docs/specs/2026-10-09-voice-caller-agent-design.md` (decision 1 and section 3 were rewritten for this design). Evidence: `docs/benchmarks/phase2-live-calls.md`.

## Global Constraints

- **Exactly one LLM call per customer turn.** No LLM-generated reply text and no LLM function calling.
- **Grounding:** an order-changing action (add item, change address, add note) runs only if its extracted value is grounded in the customer's utterance. That means ≥ 60% of its content words were said. "Yes", "no" and "unclear" never change the order.
- **Two-step cancel:** a cancel request, then an explicit "yes".
- **Replies are template sentences.** The first sentence of every reply is one of `ACKS` (pre-synthesized).
- **Business rules** stay in the Plan 1 services, reached through `CallSession`. No SQL in agent code.
- **The intent-reader system prompt is a constant,** so llama-server's prompt cache can reuse it across turns and calls.
- Python 3.11 venv at `.venv/Scripts/python`; commands run from the repo root in Git Bash.
- **Commit messages** contain no co-author or tool-attribution trailers.

## Review Focus

1. **The model echoes the agent's words as the value.** For example, "Amul milk and eggs" taken from the agent's own sentence. That must not count as grounded. Pinned by `test_grounded_rejects_agent_words` (Task 1) and `test_add_item_ungrounded_changes_nothing` (Task 3).
2. **The intent is valid somewhere but not in this state,** such as `change_address` during the greeting. It must be treated as unclear, with no change. Pinned by `test_intent_not_allowed_in_state_is_unclear` (Task 3).
3. **llama-server is down, or returns malformed JSON or an out-of-enum intent.** The turn degrades to "Sorry, I didn't catch that" instead of crashing. Pinned by `test_failures_are_unclear` (Task 2).
4. **The customer barges in while the intent reader is still running.** The stale reply must not be spoken. Pinned by `test_reply_dropped_after_interruption` (Task 4).
5. **The customer picks "the first one" from listed slots.** The number words in that phrase must not be read as a 1 PM time. Pinned by `test_pick_option_by_ordinal_and_by_time` (Task 3).

---

## File Structure

```
src/voiceagent/agent/nlu.py         grounded(), split_quantity()
src/voiceagent/agent/intent.py      Intent, IntentClassifier (the one LLM call)
src/voiceagent/agent/dialog.py      DialogManager, Reply, STATE_INTENTS, ACKS
src/voiceagent/agent/processor.py   DialogProcessor (Pipecat FrameProcessor), last_user_text()
src/voiceagent/agent/services.py    + Kokoro phrase cache, Smart Turn warm-up, Whisper decode timing; - make_llm
src/voiceagent/agent/bot.py         pipeline without LLM service / Flows
src/voiceagent/agent/flow.py        DELETED (and tests/agent/test_flow.py)
bench/corpus/intent_cases.json      48 labeled customer utterances
bench/intent_eval.py                intent accuracy + latency against llama-server
scripts/scripted_call.py            WebRTC test caller with spoken text cues
tests/agent/test_nlu.py, test_intent.py, test_dialog.py, test_processor.py, test_intent_eval.py
docs/benchmarks/phase3-dialog-calls.md
```

---

### Task 1: Grounding and quantity helpers

**Files:**
- Create: `src/voiceagent/agent/nlu.py`, `tests/agent/test_nlu.py`

**Interfaces:**
- Produces:
  - `grounded(value: str, utterance: str, min_share: float = 0.6) -> bool`
  - `split_quantity(value: str) -> tuple[int, str]`

- [ ] **Step 1: Write the failing tests**

`tests/agent/test_nlu.py`:
```python
from voiceagent.agent.nlu import grounded, split_quantity


def test_grounded_accepts_customer_words():
    assert grounded("two packets of Amul milk", "Can you also add two packets of Amul milk?")
    assert grounded("42 Church Street, Bengaluru", "No, it's 42 Church Street, Bengaluru.")
    assert grounded("tomorrow evening", "Tomorrow evening works for me.")
    assert grounded("leave it with the security guard", "Please leave it with the security guard.")


def test_grounded_rejects_agent_words():
    assert not grounded("Amul milk and eggs", "Yes, that sounds good.")
    assert not grounded("Aashirvaad sugar", "Yes, that sounds good.")
    assert not grounded("23 MG Road, Bengaluru", "Yes, that's right.")
    assert not grounded("", "Anything at all")
    assert not grounded("the", "the")


def test_split_quantity():
    assert split_quantity("two packets of Amul milk") == (2, "Amul milk")
    assert split_quantity("3 Haldiram salted peanuts") == (3, "Haldiram salted peanuts")
    assert split_quantity("Amul milk") == (1, "Amul milk")
    assert split_quantity("an extra pack of eggs") == (1, "eggs")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/agent/test_nlu.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.agent.nlu'`.

- [ ] **Step 3: Implement**

`src/voiceagent/agent/nlu.py`:
```python
"""Deterministic checks on values the intent reader extracted."""
from __future__ import annotations

import re

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "a": 1, "an": 1, "another": 1}
_FILLERS = {"packet", "packets", "pack", "packs", "of", "unit", "units", "piece", "pieces", "more", "also",
            "please", "some", "extra"}
_STOPWORDS = {"the", "of", "to", "and", "or", "please", "can", "could", "you", "also", "add", "some", "my",
              "it", "is", "its", "for", "me", "want", "like", "would", "packet", "packets", "pack", "packs",
              "more", "with", "at", "in", "on", "an", "extra"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def grounded(value: str, utterance: str, min_share: float = 0.6) -> bool:
    """True when most content words of `value` were actually said by the customer."""
    words = [w for w in _words(value) if len(w) > 1 and w not in _STOPWORDS]
    if not words:
        return False
    said = set(_words(utterance))
    return sum(w in said for w in words) / len(words) >= min_share


def split_quantity(value: str) -> tuple[int, str]:
    """'two packets of Amul milk' -> (2, 'Amul milk'). The quantity defaults to 1."""
    qty, found, rest = 1, False, []
    for token in value.split():
        word = re.sub(r"[^a-z0-9]", "", token.lower())
        if not found and (word.isdigit() or word in _NUMBER_WORDS):
            qty, found = (int(word) if word.isdigit() else _NUMBER_WORDS[word]), True
            continue
        if not word or word in _FILLERS:
            continue
        rest.append(token.strip(".,?!"))
    return max(qty, 1), " ".join(rest)
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/agent/test_nlu.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/agent/nlu.py tests/agent/test_nlu.py
git commit -m "Add grounding and quantity helpers for extracted values"
```

---

### Task 2: Intent reader (the one LLM call)

**Files:**
- Create: `src/voiceagent/agent/intent.py`, `tests/agent/test_intent.py`

**Interfaces:**
- Produces:
  - `Intent(name: str, value: str = "")`, frozen
  - `INTENT_DESCRIPTIONS: dict[str, str]`
  - `SYSTEM_PROMPT: str`
  - `IntentClassifier(base_url="http://127.0.0.1:8080/v1", client=None, timeout_s=10.0)` with:
    - `request_body(agent_said, customer_said, allowed) -> dict` (static)
    - `async classify(agent_said, customer_said, allowed) -> Intent`, which never raises. Any failure, or an intent not in `allowed`, gives `Intent("unclear")`.
    - `async aclose()`

- [ ] **Step 1: Write the failing tests**

`tests/agent/test_intent.py`:
```python
import asyncio
import json

import httpx

from voiceagent.agent.intent import Intent, IntentClassifier

ALLOWED = ["confirm", "give_time", "unclear"]


def classifier(handler):
    return IntentClassifier("http://llm/v1", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def answering(content, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    return handler


def test_classify_sends_allowed_enum_and_parses():
    seen = []
    c = classifier(answering('{"intent": "give_time", "value": "tomorrow evening"}', seen))
    result = asyncio.run(c.classify("When would you like it?", "Tomorrow evening works.", ALLOWED))
    assert result == Intent("give_time", "tomorrow evening")
    body = seen[0]
    assert body["response_format"]["json_schema"]["schema"]["properties"]["intent"]["enum"] == ALLOWED
    assert body["messages"][1]["content"] == "AGENT: When would you like it?\nCUSTOMER: Tomorrow evening works."
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0


def test_intent_outside_allowed_is_unclear():
    c = classifier(answering('{"intent": "cancel", "value": ""}'))
    assert asyncio.run(c.classify("a", "b", ALLOWED)) == Intent("unclear")


def test_failures_are_unclear():
    assert asyncio.run(classifier(answering("not json")).classify("a", "b", ALLOWED)) == Intent("unclear")
    down = classifier(lambda request: httpx.Response(500, json={"error": "boom"}))
    assert asyncio.run(down.classify("a", "b", ALLOWED)) == Intent("unclear")


def test_system_prompt_is_static():
    a = IntentClassifier.request_body("x", "y", ["confirm"])["messages"][0]
    b = IntentClassifier.request_body("p", "q", ["deny", "unclear"])["messages"][0]
    assert a == b and "Never copy the agent's words" in a["content"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/agent/test_intent.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.agent.intent'`.

- [ ] **Step 3: Implement**

`src/voiceagent/agent/intent.py`:
```python
"""The one LLM call per customer turn: name the intent and keep the customer's own words."""
from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
from loguru import logger

INTENT_DESCRIPTIONS = {
    "confirm": "yes, agrees, that's me, that's right, that's all, go ahead",
    "deny": "no, disagrees, that's not right",
    "give_time": "names a delivery day or time",
    "pick_option": "picks one of the options the agent just listed (first, second, last, or by naming it)",
    "substitute": "accepts the suggested substitute product",
    "send_available": "send the order without the missing item, or only what is available",
    "wait_restock": "wait until the missing item is back in stock",
    "add_item": "asks to add a product to the order",
    "change_address": "gives a different delivery address",
    "add_note": "gives delivery instructions for the driver",
    "callback": "busy now, or asks to be called back later",
    "cancel": "wants to cancel the whole order",
    "wrong_person": "is not the person the agent asked for",
    "question": "asks a question",
    "unclear": "anything else, or unintelligible",
}

SYSTEM_PROMPT = (
    "Classify the customer's last utterance on a grocery delivery phone call. "
    'Reply with JSON {"intent": ..., "value": ...}.\n'
    "value = the customer's own words for the time, product with quantity, address, instruction, chosen option "
    "or question; \"\" if none. Never copy the agent's words.\n"
    "Intents:\n" + "\n".join(f"- {name}: {meaning}" for name, meaning in INTENT_DESCRIPTIONS.items())
)


@dataclass(frozen=True)
class Intent:
    name: str
    value: str = ""


class IntentClassifier:
    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", client: httpx.AsyncClient | None = None,
                 timeout_s: float = 10.0):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    @staticmethod
    def request_body(agent_said: str, customer_said: str, allowed: list[str]) -> dict:
        schema = {"type": "object",
                  "properties": {"intent": {"type": "string", "enum": allowed},
                                 "value": {"type": "string", "maxLength": 80}},
                  "required": ["intent", "value"]}
        return {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": f"AGENT: {agent_said}\nCUSTOMER: {customer_said}"}],
            "temperature": 0,
            "max_tokens": 48,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "turn", "schema": schema}},
        }

    async def classify(self, agent_said: str, customer_said: str, allowed: list[str]) -> Intent:
        try:
            response = await self._client.post(self._url, json=self.request_body(agent_said, customer_said, allowed))
            response.raise_for_status()
            data = json.loads(response.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning(f"intent reader failed, treating the turn as unclear: {exc}")
            return Intent("unclear")
        name = data.get("intent") if isinstance(data, dict) else None
        if name not in allowed:
            return Intent("unclear")
        return Intent(name, str(data.get("value") or "").strip())

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/agent/test_intent.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/agent/intent.py tests/agent/test_intent.py
git commit -m "Add schema-constrained intent reader"
```

---

### Task 3: Dialog manager

**Files:**
- Create: `src/voiceagent/agent/dialog.py`, `tests/agent/test_dialog.py`

**Interfaces:**
- Consumes: `CallSession` (Plan 2), `Intent` (Task 2), `grounded` and `split_quantity` (Task 1), `parse_time_window`, `spoken_day`, `Shortage`.
- Produces:
  - `Reply(sentences: list[str], end_call: bool = False)`
  - `ACKS`
  - `STATE_INTENTS: dict[str, list[str]]`
  - `DialogManager(session)` with:
    - attributes `state`, `question` (last sentence the agent said), `transcript`, `customer_name`, `first_name`, `alternatives`
    - methods `greeting() -> str`, `allowed_intents() -> list[str]`, `handle(intent, utterance) -> Reply`

  States: `greet`, `shortage`, `schedule`, `address`, `note`, `confirm`, `confirm_cancel`, `closed`.

- [ ] **Step 1: Write the failing tests**

`tests/agent/test_dialog.py`:
```python
from datetime import timedelta

import pytest

from voiceagent.agent.dialog import ACKS, DialogManager
from voiceagent.agent.intent import Intent
from voiceagent.agent.tools import CallSession

from conftest import NOW


@pytest.fixture
def clock():
    return {"now": NOW}


@pytest.fixture
def dm(world, repo, clock):
    return DialogManager(CallSession(repo, 1, clock=lambda: clock["now"]))


def say(dm, name, utterance, value=""):
    return dm.handle(Intent(name, value), utterance)


def to_schedule(dm):
    say(dm, "confirm", "Yes, this is Priya.")
    say(dm, "substitute", "Sure, the whole wheat one.")
    assert dm.state == "schedule"


def test_greeting_and_allowed_intents(dm):
    assert dm.greeting() == ("Hi, this is Asha calling from QuickMart about your grocery order. "
                             "Am I speaking with Priya?")
    assert dm.state == "greet" and "change_address" not in dm.allowed_intents()
    assert dm.transcript == [{"role": "assistant", "content": dm.greeting()}]


def test_confirm_identity_presents_order_and_shortage(dm):
    reply = say(dm, "confirm", "Yes, this is Priya.")
    assert reply.sentences == [
        "Great.", "Your order has 2 Amul milk, 1 brown bread and 1 eggs.", "The brown bread is out of stock.",
        "Would you like whole wheat bread instead, the rest without it or to wait until it's back tomorrow?"]
    assert dm.state == "shortage"


def test_every_reply_starts_with_a_cached_ack(dm):
    for name, utterance in [("confirm", "Yes."), ("substitute", "The whole wheat one."),
                            ("give_time", "Tomorrow after 5."), ("unclear", "Hmm?")]:
        assert say(dm, name, utterance, "tomorrow after 5" if name == "give_time" else "").sentences[0] in ACKS


def test_full_happy_path_books(dm, repo):
    to_schedule(dm)
    assert next(l for l in repo.get_order_lines(1) if l.sku == "BREAD1").resolution == "substitute"
    assert say(dm, "give_time", "Tomorrow after 5 please.", "tomorrow after 5").sentences == [
        "Okay.", "I can deliver tomorrow, 5 to 6 PM.", "Shall I book that?"]
    assert say(dm, "confirm", "Yes.").sentences == ["Great.", "Should we deliver to 12 MG Road, Bengaluru?"]
    assert say(dm, "confirm", "Yes, that's right.").sentences == ["Okay.", "Any instructions for the driver?"]
    reply = say(dm, "add_note", "Please leave it with the guard.", "leave it with the guard")
    assert reply.sentences == [
        "Noted.",
        "To confirm, 2 Amul milk, 1 whole wheat bread instead of brown bread and 1 eggs, delivered tomorrow, "
        "5 to 6 PM, to 12 MG Road, Bengaluru.",
        "Shall I book it?"]
    reply = say(dm, "confirm", "Yes, book it.")
    assert reply.sentences == ["Perfect.", "Your delivery is booked for tomorrow, 5 to 6 PM.", "Thank you, goodbye!"]
    assert reply.end_call and dm.state == "closed"
    order = repo.get_order(1)
    assert order.status == "scheduled" and order.notes == "leave it with the guard"


def test_ungrounded_time_value_falls_back_to_utterance(dm):
    to_schedule(dm)
    reply = say(dm, "give_time", "Tomorrow after 5 please.", "Friday")
    assert reply.sentences[1] == "I can deliver tomorrow, 5 to 6 PM."


def test_full_slot_offers_alternatives(dm):
    to_schedule(dm)
    assert say(dm, "give_time", "Today at 5 pm.", "today at 5 pm").sentences == [
        "Sorry.", "That time is full.", "I have today, 3 to 4 PM, today, 4 to 5 PM or today, 6 to 7 PM.",
        "Which works for you?"]


def test_pick_option_by_ordinal_and_by_time(dm, repo):
    to_schedule(dm)
    say(dm, "give_time", "Today at 5 pm.", "today at 5 pm")
    assert say(dm, "pick_option", "The first one.", "the first one").sentences == [
        "Great.", "today, 3 to 4 PM it is.", "Should we deliver to 12 MG Road, Bengaluru?"]
    dm.state = "schedule"
    say(dm, "give_time", "Today at 5 pm.", "today at 5 pm")
    assert say(dm, "pick_option", "Six to seven.", "six to seven").sentences[1] == "today, 6 to 7 PM it is."


def test_add_item_grounded(dm, repo):
    to_schedule(dm)
    reply = say(dm, "add_item", "Can you also add two Nandini milk?", "two Nandini milk")
    assert reply.sentences == ["Sure.", "I've added 2 Nandini milk.", "When would you like it delivered?"]
    assert any(l.sku == "MILK2" and l.qty == 2 for l in repo.get_order_lines(1))


def test_add_item_ungrounded_changes_nothing(dm, repo):
    to_schedule(dm)
    before = repo.get_order_lines(1)
    reply = say(dm, "add_item", "Yes, that sounds good.", "Aashirvaad sugar")
    assert reply.sentences == ["Sorry.", "Which product would you like to add?"]
    assert repo.get_order_lines(1) == before


def test_add_item_out_of_stock(dm):
    to_schedule(dm)
    assert say(dm, "add_item", "Add five basmati rice.", "five basmati rice").sentences[:2] == [
        "Sorry.", "We only have 2 basmati rice left."]


def test_intent_not_allowed_in_state_is_unclear(dm, repo):
    reply = say(dm, "change_address", "42 Church Street.", "42 Church Street")
    assert reply.sentences == ["Sorry.", "I didn't catch that.", "Am I speaking with Priya?"]
    assert repo.get_order(1).address == "12 MG Road, Bengaluru"


def test_change_address_grounded_and_ungrounded(dm, repo):
    to_schedule(dm)
    say(dm, "give_time", "Tomorrow after 5.", "tomorrow after 5")
    say(dm, "confirm", "Yes.")
    assert say(dm, "change_address", "Yes.", "23 MG Road, Bengaluru").sentences == [
        "Sorry.", "Could you tell me the full address again?"]
    assert repo.get_order(1).address == "12 MG Road, Bengaluru"
    reply = say(dm, "change_address", "No, it's 42 Church Street, Bengaluru.", "42 Church Street, Bengaluru")
    assert reply.sentences == ["Got it.", "I've updated the address to 42 Church Street, Bengaluru.",
                               "Any instructions for the driver?"]
    assert repo.get_order(1).address == "42 Church Street, Bengaluru" and dm.state == "note"


def test_deny_address_asks_for_it(dm):
    to_schedule(dm)
    say(dm, "give_time", "Tomorrow after 5.", "tomorrow after 5")
    say(dm, "confirm", "Yes.")
    assert say(dm, "deny", "No.").sentences == ["No problem.", "What's the correct address?"]


def test_cancel_needs_two_steps(dm, repo):
    to_schedule(dm)
    assert say(dm, "cancel", "Cancel it.").sentences == ["Okay.", "Just to confirm, do you want to cancel the whole order?"]
    assert say(dm, "deny", "No, keep it.").sentences == ["Okay.", "I won't cancel it.", "When would you like it delivered?"]
    assert repo.get_order(1).status == "pending_schedule"
    say(dm, "cancel", "Actually cancel it.")
    reply = say(dm, "confirm", "Yes.")
    assert reply.end_call and reply.sentences == ["Okay.", "Your order is cancelled.", "Goodbye!"]
    assert repo.get_order(1).status == "cancelled"


def test_callback(dm, repo):
    reply = say(dm, "callback", "I'm busy, call me tomorrow morning.", "tomorrow morning")
    assert reply.sentences == ["No problem.", "I'll call you back tomorrow 8 AM.", "Goodbye!"] and reply.end_call
    assert repo.get_order(1).status == "callback"


def test_wrong_person(dm, repo):
    reply = say(dm, "wrong_person", "No, this is her brother.")
    assert reply.end_call and dm.session.outcome == "wrong_person"


def test_question_total_then_repeats_question(dm):
    say(dm, "confirm", "Yes.")
    reply = say(dm, "question", "How much is the total?", "how much is the total")
    assert reply.sentences[0] == "Your total is about 202 rupees."
    assert reply.sentences[-1].startswith("Would you like whole wheat bread instead")


def test_off_topic_question(dm):
    reply = say(dm, "question", "What's the weather like?", "what's the weather like")
    assert reply.sentences[0] == "Sorry, I can only help with this delivery right now."


def test_confirm_after_hold_expiry_goes_back_to_schedule(dm, clock):
    to_schedule(dm)
    say(dm, "give_time", "Tomorrow after 5.", "tomorrow after 5")
    say(dm, "confirm", "Yes.")
    say(dm, "confirm", "Yes.")
    say(dm, "confirm", "No instructions.")
    clock["now"] = NOW + timedelta(minutes=6)
    reply = say(dm, "confirm", "Yes.")
    assert reply.sentences == ["Sorry.", "That slot is no longer available.", "What other time would suit you?"]
    assert dm.state == "schedule"


def test_changing_time_at_confirm_returns_to_read_back(dm):
    to_schedule(dm)
    say(dm, "give_time", "Tomorrow after 5.", "tomorrow after 5")
    say(dm, "confirm", "Yes.")
    say(dm, "confirm", "Yes.")
    say(dm, "confirm", "No instructions.")
    say(dm, "give_time", "Make it after 6.", "tomorrow after 6")
    reply = say(dm, "confirm", "Yes.")
    assert dm.state == "confirm" and reply.sentences[-1] == "Shall I book it?"
    assert "tomorrow, 6 to 7 PM" in reply.sentences[-2]


def test_transcript_records_both_sides(dm):
    say(dm, "confirm", "Yes, this is Priya.")
    assert [t["role"] for t in dm.transcript] == ["assistant", "user", "assistant"]
    assert dm.transcript[1]["content"] == "Yes, this is Priya."
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/agent/test_dialog.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.agent.dialog'`.

- [ ] **Step 3: Implement**

`src/voiceagent/agent/dialog.py`:
```python
"""Deterministic call dialog. The LLM only names the intent; this decides, acts and speaks."""
from __future__ import annotations

from dataclasses import dataclass

from voiceagent.agent.intent import Intent
from voiceagent.agent.nlu import grounded, split_quantity
from voiceagent.agent.tools import CallSession
from voiceagent.domain.context import spoken_day
from voiceagent.domain.inventory import Shortage
from voiceagent.domain.timeparse import parse_time_window

AGENT_NAME = "Asha"
COMPANY = "QuickMart"
# Every reply opens with one of these; the TTS service pre-synthesizes them so speech starts instantly.
ACKS = ("Sure.", "Got it.", "Okay.", "Great.", "Perfect.", "No problem.", "Sorry.", "Noted.")
GLOBAL_INTENTS = ["add_item", "callback", "cancel", "question", "unclear"]
STATE_INTENTS = {
    "greet": ["confirm", "deny", "wrong_person", "callback", "question", "unclear"],
    "shortage": ["substitute", "send_available", "wait_restock", "deny", *GLOBAL_INTENTS],
    "schedule": ["give_time", "confirm", "deny", "pick_option", *GLOBAL_INTENTS],
    "address": ["confirm", "deny", "change_address", *GLOBAL_INTENTS],
    "note": ["confirm", "deny", "add_note", "change_address", *GLOBAL_INTENTS],
    "confirm": ["confirm", "deny", "give_time", "change_address", "add_note", *GLOBAL_INTENTS],
    "confirm_cancel": ["confirm", "deny", "question", "unclear"],
    "closed": ["unclear"],
}
_ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}


def _join(items: list[str], last: str = "and") -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {last} {items[-1]}"


@dataclass
class Reply:
    sentences: list[str]
    end_call: bool = False


class DialogManager:
    def __init__(self, session: CallSession):
        self.session = session
        ctx = session.context()
        self.customer_name = ctx.customer_name
        self.first_name = ctx.customer_name.split()[0]
        self.state = "greet"
        self.alternatives: list[dict] = []
        self.substitute: dict | None = None
        self.awaiting_address = False
        self.details_done = False
        self._before_cancel = "greet"
        self.question = self.greeting()
        self.transcript: list[dict] = [{"role": "assistant", "content": self.greeting()}]

    # ── public ──────────────────────────────────────────────────────────────
    def greeting(self) -> str:
        return (f"Hi, this is {AGENT_NAME} calling from {COMPANY} about your grocery order. "
                f"Am I speaking with {self.first_name}?")

    def allowed_intents(self) -> list[str]:
        return STATE_INTENTS[self.state]

    def handle(self, intent: Intent, utterance: str) -> Reply:
        self.transcript.append({"role": "user", "content": utterance})
        if intent.name not in self.allowed_intents():
            intent = Intent("unclear")
        reply = self._dispatch(intent, utterance)
        if reply.sentences:
            self.question = reply.sentences[-1]
            self.transcript.append({"role": "assistant", "content": " ".join(reply.sentences)})
        return reply

    # ── routing ─────────────────────────────────────────────────────────────
    def _dispatch(self, intent: Intent, utterance: str) -> Reply:
        if self.state == "closed":
            return Reply([], end_call=True)
        if intent.name == "unclear":
            return Reply(["Sorry.", "I didn't catch that.", *self._ask()])
        if intent.name == "question":
            return Reply([self._answer(intent.value or utterance), *self._ask()])
        if intent.name == "callback":
            return self._callback(intent, utterance)
        if intent.name == "add_item":
            return self._add_item(intent, utterance)
        if intent.name == "cancel" and self.state != "confirm_cancel":
            self._before_cancel, self.state = self.state, "confirm_cancel"
            return Reply(["Okay.", "Just to confirm, do you want to cancel the whole order?"])
        return getattr(self, f"_on_{self.state}")(intent, utterance)

    # ── facts and questions ─────────────────────────────────────────────────
    def _shortages(self) -> list[Shortage]:
        return self.session.inventory.shortages(self.session.order_id)

    def _address(self) -> str:
        return self.session.repo.get_order(self.session.order_id).address

    def _held_slot(self) -> str | None:
        return self.session.order_summary()["slot"]

    def _options(self) -> str:
        return _join([a["slot"] for a in self.alternatives], "or")

    def _items(self) -> str:
        return _join([f"{line.qty} {line.name}" for line in self.session.repo.get_order_lines(self.session.order_id)])

    def _total(self) -> int:
        repo, total = self.session.repo, 0.0
        for line in repo.get_order_lines(self.session.order_id):
            total += line.reserved_qty * repo.get_product(line.sku).unit_price
            if line.substitute_sku:
                total += line.substitute_qty * repo.get_product(line.substitute_sku).unit_price
        return int(round(total))

    def _read_back(self) -> list[str]:
        summary = self.session.order_summary()
        return [f"To confirm, {_join(summary['items'])}, delivered {summary['slot']}, to {summary['address']}.",
                "Shall I book it?"]

    def _shortage_question(self, shortage: Shortage) -> list[str]:
        sub = shortage.substitutes[0] if shortage.substitutes else None
        self.substitute = {"sku": sub.sku, "name": sub.spoken_name} if sub else None
        status = ("is out of stock" if shortage.reserved == 0
                  else f"is short, we only have {shortage.reserved} of {shortage.wanted}")
        options = [f"{sub.spoken_name} instead"] if sub else []
        options.append("the rest without it" if shortage.reserved == 0 else f"just the {shortage.reserved} we have")
        if shortage.restock_eta:
            options.append(f"to wait until it's back {spoken_day(shortage.restock_eta.date(), self.session.clock())}")
        return [f"The {shortage.name} {status}.", f"Would you like {_join(options, 'or')}?"]

    def _ask(self) -> list[str]:
        """The question the agent is currently waiting on, regenerated from live facts."""
        if self.state == "greet":
            return [f"Am I speaking with {self.first_name}?"]
        if self.state == "shortage":
            shortages = self._shortages()
            if shortages:
                return self._shortage_question(shortages[0])
            self.state = "schedule"
        if self.state == "schedule":
            if self.session.held_slot_id:
                return [f"Shall I book {self._held_slot()}?"]
            if self.alternatives:
                return [f"I have {self._options()}.", "Which works for you?"]
            return ["When would you like it delivered?"]
        if self.state == "address":
            return ["What's the correct address?"] if self.awaiting_address else [f"Should we deliver to {self._address()}?"]
        if self.state == "note":
            return ["Any instructions for the driver?"]
        if self.state == "confirm":
            return self._read_back()
        if self.state == "confirm_cancel":
            return ["Do you want to cancel the whole order?"]
        return []

    def _answer(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ("total", "cost", "price", "how much", "pay")):
            return f"Your total is about {self._total()} rupees."
        if any(w in t for w in ("when", "time", "slot", "deliver")):
            slot = self._held_slot()
            return f"I'm holding {slot} for you." if slot else "We deliver between 8 AM and 9 PM."
        if any(w in t for w in ("order", "items", "what did", "what's in")):
            return f"Your order has {self._items()}."
        return "Sorry, I can only help with this delivery right now."

    # ── global intents ──────────────────────────────────────────────────────
    def _callback(self, intent: Intent, utterance: str) -> Reply:
        when = intent.value if grounded(intent.value, utterance) else ""
        result = self.session.schedule_callback(when)
        self.state = "closed"
        return Reply(["No problem.", f"I'll call you back {result['callback_at']}.", "Goodbye!"], end_call=True)

    def _add_item(self, intent: Intent, utterance: str) -> Reply:
        if not grounded(intent.value, utterance):
            return Reply(["Sorry.", "Which product would you like to add?"])
        qty, product = split_quantity(intent.value)
        result = self.session.add_item(product, qty)
        if result["status"] == "added":
            return Reply(["Sure.", f"I've added {qty} {result['item']}.", *self._ask()])
        if result["status"] == "insufficient":
            left = (f"We only have {result['available']} {result['item']} left." if result["available"]
                    else f"We're out of {result['item']}.")
            return Reply(["Sorry.", left, *self._ask()])
        if result["status"] == "not_found":
            return Reply(["Sorry.", f"I couldn't find {product}.", *self._ask()])
        return Reply(["Sorry.", "I couldn't add that.", *self._ask()])

    # ── states ──────────────────────────────────────────────────────────────
    def _on_greet(self, intent: Intent, utterance: str) -> Reply:
        if intent.name == "confirm":
            sentences = ["Great.", f"Your order has {self._items()}."]
            shortages = self._shortages()
            if shortages:
                self.state = "shortage"
                return Reply(sentences + self._shortage_question(shortages[0]))
            self.state = "schedule"
            return Reply(sentences + ["When would you like it delivered?"])
        self.session.wrong_person()
        self.state = "closed"
        return Reply(["Sorry for the trouble.", "I'll try again later. Goodbye!"], end_call=True)

    def _on_shortage(self, intent: Intent, utterance: str) -> Reply:
        shortages = self._shortages()
        if not shortages:
            self.state = "schedule"
            return Reply(["Okay.", *self._ask()])
        current = shortages[0]
        if intent.name == "substitute" and self.substitute:
            result = self.session.resolve_shortage(current.sku, "substitute", self.substitute["sku"])
        elif intent.name == "send_available":
            result = self.session.resolve_shortage(current.sku, "partial")
        elif intent.name == "wait_restock":
            result = self.session.resolve_shortage(current.sku, "wait")
        else:
            return Reply(["Sorry.", *self._shortage_question(current)])
        if result["status"] != "done":
            return Reply(["Sorry.", "I can't do that for this item.", *self._shortage_question(current)])
        remaining = self._shortages()
        if remaining:
            return Reply(["Got it.", *self._shortage_question(remaining[0])])
        self.state = "schedule"
        sentences = ["Got it."]
        if result.get("deliver_after"):
            sentences.append(f"We can deliver after {result['deliver_after']}.")
        return Reply(sentences + ["When would you like it delivered?"])

    def _check_time(self, intent: Intent, utterance: str) -> Reply:
        phrase = intent.value if grounded(intent.value, utterance) else utterance
        result = self.session.check_slot(phrase)
        if result["status"] == "unclear" and phrase != utterance:
            result = self.session.check_slot(utterance)
        self.state = "schedule"
        if result["status"] == "held":
            self.alternatives = []
            return Reply(["Okay.", f"I can deliver {result['slot']}.", "Shall I book that?"])
        if result["status"] == "unclear":
            return Reply(["Sorry.", "Which day and time suit you? We deliver between 8 AM and 9 PM."])
        self.alternatives = result["alternatives"]
        if not self.alternatives:
            return Reply(["Sorry.", "I have no free slots this week.", "Which other day would suit you?"])
        reason = "That time is full." if result["status"] == "full" else "We have no slots then."
        return Reply(["Sorry.", reason, f"I have {self._options()}.", "Which works for you?"])

    def _option_index(self, text: str) -> int | None:
        words = text.lower().replace(",", " ").split()
        for word in words:
            if word in _ORDINALS and _ORDINALS[word] < len(self.alternatives):
                return _ORDINALS[word]
            if word == "last":
                return len(self.alternatives) - 1
        window = parse_time_window(text, self.session.clock())
        if window:
            for index, alt in enumerate(self.alternatives):
                if window.start <= self.session.repo.get_slot(alt["slot_id"]).start < window.end:
                    return index
        return None

    def _after_slot_agreed(self, prefix: str | None) -> Reply:
        lead = ["Great.", *([prefix] if prefix else [])]
        if self.details_done:
            self.state = "confirm"
            return Reply(lead + self._read_back())
        self.state, self.awaiting_address = "address", False
        return Reply(lead + [f"Should we deliver to {self._address()}?"])

    def _on_schedule(self, intent: Intent, utterance: str) -> Reply:
        if intent.name == "give_time":
            return self._check_time(intent, utterance)
        if intent.name == "pick_option" and self.alternatives:
            index = self._option_index(intent.value or utterance)
            if index is None:
                return Reply(["Sorry.", f"Was that {self._options()}?"])
            result = self.session.choose_slot(self.alternatives[index]["slot_id"])
            if result["status"] != "held":
                self.alternatives = result["alternatives"]
                return Reply(["Sorry.", "That slot just filled up.", *self._ask()])
            self.alternatives = []
            return self._after_slot_agreed(f"{result['slot']} it is.")
        if intent.name == "confirm" and self.session.held_slot_id:
            return self._after_slot_agreed(None)
        if intent.name == "deny":
            return Reply(["No problem.", "What day and time would suit you?"])
        return Reply(["Okay.", *self._ask()])

    def _change_address(self, intent: Intent, utterance: str, then: str) -> Reply:
        if len(intent.value) < 5 or not grounded(intent.value, utterance):
            if self.state == "address":
                self.awaiting_address = True
            return Reply(["Sorry.", "Could you tell me the full address again?"])
        result = self.session.update_address(intent.value)
        if result["status"] != "updated":
            return Reply(["Sorry.", "Could you tell me the full address again?"])
        self.awaiting_address = False
        updated = f"I've updated the address to {result['address']}."
        if then == "note":
            self.state = "note"
            return Reply(["Got it.", updated, "Any instructions for the driver?"])
        self.state = "confirm"
        return Reply(["Got it.", updated, *self._read_back()])

    def _on_address(self, intent: Intent, utterance: str) -> Reply:
        if intent.name == "change_address":
            return self._change_address(intent, utterance, then="note")
        if intent.name == "confirm" and not self.awaiting_address:
            self.state = "note"
            return Reply(["Okay.", "Any instructions for the driver?"])
        self.awaiting_address = True
        return Reply(["No problem.", "What's the correct address?"])

    def _add_note(self, intent: Intent, utterance: str) -> Reply:
        if not grounded(intent.value, utterance):
            return Reply(["Sorry.", "What should I tell the driver?"])
        self.session.add_note(intent.value)
        self.details_done, self.state = True, "confirm"
        return Reply(["Noted.", *self._read_back()])

    def _on_note(self, intent: Intent, utterance: str) -> Reply:
        if intent.name == "add_note":
            return self._add_note(intent, utterance)
        if intent.name == "change_address":
            return self._change_address(intent, utterance, then="note")
        self.details_done, self.state = True, "confirm"
        return Reply(["Okay.", *self._read_back()])

    def _on_confirm(self, intent: Intent, utterance: str) -> Reply:
        if intent.name == "confirm":
            result = self.session.confirm_booking()
            if result["status"] == "booked":
                self.state = "closed"
                return Reply(["Perfect.", f"Your delivery is booked for {result['slot']}.", "Thank you, goodbye!"],
                             end_call=True)
            self.state, self.alternatives = "schedule", []
            return Reply(["Sorry.", "That slot is no longer available.", "What other time would suit you?"])
        if intent.name == "give_time":
            return self._check_time(intent, utterance)
        if intent.name == "change_address":
            return self._change_address(intent, utterance, then="confirm")
        if intent.name == "add_note":
            return self._add_note(intent, utterance)
        return Reply(["No problem.", "What would you like to change, the time or the address?"])

    def _on_confirm_cancel(self, intent: Intent, utterance: str) -> Reply:
        if intent.name == "confirm":
            self.session.cancel_order()
            self.state = "closed"
            return Reply(["Okay.", "Your order is cancelled.", "Goodbye!"], end_call=True)
        self.state = self._before_cancel
        return Reply(["Okay.", "I won't cancel it.", *self._ask()])
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/agent/test_dialog.py -q`
Expected: `21 passed`. Slot ids and spoken strings come from the Plan 1 fixture (today 08:00 = slot 1, tomorrow 08:00 = 14). If a string differs, fix the code to match the test, unless the test contradicts the spec's section 3 table.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/agent/dialog.py tests/agent/test_dialog.py
git commit -m "Add deterministic dialog manager with grounded actions and templated replies"
```

---

### Task 4: Dialog processor (pipeline glue)

**Files:**
- Create: `src/voiceagent/agent/processor.py`, `tests/agent/test_processor.py`

**Interfaces:**
- Consumes: `DialogManager` (Task 3), any object with `async classify(agent_said, customer_said, allowed) -> Intent` (Task 2).
- Produces:
  - `last_user_text(context: LLMContext) -> str`
  - `DialogProcessor(manager, classifier, **kwargs)`. On an `LLMContextFrame` that isn't a speculation, it pushes one `TTSSpeakFrame` per reply sentence, and an `EndTaskFrame` upstream when the reply ends the call. On an `InterruptionFrame`, it drops any reply still in flight.

- [ ] **Step 1: Write the failing tests**

`tests/agent/test_processor.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/agent/test_processor.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.agent.processor'`.

- [ ] **Step 3: Implement**

`src/voiceagent/agent/processor.py`:
```python
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/agent/test_processor.py -q`
Expected: `4 passed`. If `run_test` requires the `expected_down_frames` argument, pass `expected_down_frames=None` explicitly. If it rejects frames it didn't expect, read its docstring in `.venv/Lib/site-packages/pipecat/tests/utils.py` and adjust the call, not the processor.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/agent/processor.py tests/agent/test_processor.py
git commit -m "Add dialog processor that replaces the LLM service in the pipeline"
```

---

### Task 5: Faster speech services and pipeline rewiring

**Files:**
- Modify: `src/voiceagent/agent/services.py`, `tests/agent/test_services.py`, `src/voiceagent/agent/bot.py`, `tests/agent/test_bot.py`
- Delete: `src/voiceagent/agent/flow.py`, `tests/agent/test_flow.py`

**Interfaces:**
- Produces:
  - `KokoroTorchTTSService(*, voice="af_heart", device="cuda", cache_phrases=(), **kwargs)`. Phrases are synthesized at construction and served from memory.
  - `make_turn_analyzer() -> LocalSmartTurnAnalyzerV3`, warmed up.
  - `GreedyWhisperSTTService` logs `whisper decode {ms} ms for {secs} s audio` at INFO.
  - `make_llm` is removed.
  - `bot.run_bot` pipeline: `transport.input() → stt → user aggregator → DialogProcessor → tts → transport.output()`.

- [ ] **Step 1: Write the failing tests** (append to `tests/agent/test_services.py`, and delete `test_make_llm_targets_local_server`)

```python
def test_kokoro_cache_phrases_are_synthesized_once(monkeypatch):
    import sys
    import types

    synthesized = []

    class FakeModel:
        def __init__(self, repo_id):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeResult:
        def __init__(self):
            import torch
            self.audio = torch.zeros(2400)

    class FakePipeline:
        def __init__(self, lang_code, repo_id, model):
            pass

        def __call__(self, text, voice):
            synthesized.append(text)
            return [FakeResult()]

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KModel=FakeModel, KPipeline=FakePipeline))

    async def build():
        return services.KokoroTorchTTSService(cache_phrases=("Sure.", "Got it."))

    tts = asyncio.run(build())
    assert synthesized == ["Warming up.", "Sure.", "Got it."]
    assert tts._audio_for(" Sure. ").size == 2400
    assert synthesized == ["Warming up.", "Sure.", "Got it."]  # served from cache
    tts._audio_for("Something new.")
    assert synthesized[-1] == "Something new."


def test_turn_analyzer_is_warmed_up():
    analyzer = services.make_turn_analyzer()
    assert analyzer.__class__.__name__ == "LocalSmartTurnAnalyzerV3"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/agent/test_services.py -q`
Expected: the two new tests FAIL (`unexpected keyword argument 'cache_phrases'`, `no attribute 'make_turn_analyzer'`).

- [ ] **Step 3: Implement the service changes**

In `src/voiceagent/agent/services.py`:

(a) Delete `make_llm` and the `OpenAILLMService` import, and add `import time`.

(b) Replace `GreedyWhisperSTTService._load` with:
```python
    def _load(self):
        import torch  # noqa: F401  (torch/lib ships cublas64_12.dll, which CTranslate2 needs on Windows)

        super()._load()
        transcribe = functools.partial(self._model.transcribe, beam_size=1, best_of=1, without_timestamps=True,
                                       condition_on_previous_text=False)

        def timed_transcribe(audio, **kwargs):
            started = time.perf_counter()
            segments, info = transcribe(audio, **kwargs)
            segments = list(segments)  # decoding is lazy; run it here so the timing is real
            logger.info(f"whisper decode {(time.perf_counter() - started) * 1000:.0f} ms "
                        f"for {len(audio) / 16000:.1f} s audio")
            return segments, info

        self._model.transcribe = timed_transcribe
        # First CUDA decode is slow (kernel/cuBLAS init); pay it at load, not on the customer's first turn.
        self._model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
```
The existing test `test_greedy_whisper_forces_greedy_decoding` still holds, because the fake's recorded kwargs are unchanged.

(c) In `KokoroTorchTTSService`, change the constructor signature to `def __init__(self, *, voice: str = "af_heart", device: str = "cuda", cache_phrases: tuple[str, ...] = (), **kwargs):`. Replace the warm-up line with:
```python
        self._cache: dict[str, np.ndarray] = {}
        # The first synthesis after loading takes seconds (CUDA init). TTSService gives up on a context after
        # stop_frame_timeout_s (3 s) without audio, which silenced the greeting, so warm up here.
        self._synthesize("Warming up.")
        for phrase in cache_phrases:
            self._cache[phrase.strip()] = self._synthesize(phrase)
```
Then add:
```python
    def _audio_for(self, text: str) -> np.ndarray:
        cached = self._cache.get(text.strip())
        return cached if cached is not None else self._synthesize(text)
```
In `run_tts`, change `samples = await asyncio.to_thread(self._synthesize, text)` to:
```python
            cached = self._cache.get(text.strip())
            samples = cached if cached is not None else await asyncio.to_thread(self._synthesize, text)
```

(d) Add:
```python
def make_turn_analyzer():
    """Smart Turn v3.2 on CPU, warmed up (its first inference took 2.7 s in Phase 2)."""
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    analyzer = LocalSmartTurnAnalyzerV3(cpu_count=4)
    analyzer._predict_endpoint(np.zeros(16000 * 2, dtype=np.float32))
    return analyzer
```

- [ ] **Step 4: Rewire the bot**

Delete `src/voiceagent/agent/flow.py` and `tests/agent/test_flow.py`.

In `src/voiceagent/agent/bot.py`:
- **Imports:**
  - Replace `from pipecat.flows import FlowManager` with `from pipecat.frames.frames import TTSSpeakFrame`.
  - Delete the `LocalSmartTurnAnalyzerV3` import.
  - Replace `from voiceagent.agent.flow import DeliveryFlow` with:
    ```python
    from voiceagent.agent.dialog import ACKS, DialogManager
    from voiceagent.agent.intent import IntentClassifier
    from voiceagent.agent.processor import DialogProcessor
    ```
  - Change the services import to `GreedyWhisperSTTService, KokoroTorchTTSService, make_turn_analyzer`.
- **Replace the body of `run_bot` from `session = ...` down to and including the `on_client_disconnected` handler** with:
```python
    session = CallSession(repo, args.order_id)
    manager = DialogManager(session)
    classifier = IntentClassifier(args.llm_url)
    recorder = CallRecorder(repo, args.order_id)

    stt = GreedyWhisperSTTService(device="cuda", compute_type="float16",
                                  settings=GreedyWhisperSTTService.Settings(model="distil-small.en"))
    tts = KokoroTorchTTSService(voice="af_heart", cache_phrases=(*ACKS, manager.greeting()))

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=make_turn_analyzer())],
            ),
        ),
    )
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        DialogProcessor(manager, classifier),
        tts,
        transport.output(),
    ])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[*recorder.observers(), TranscriptionLogObserver()],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Calling about order {args.order_id} ({manager.customer_name})")
        await worker.queue_frames([TTSSpeakFrame(manager.greeting())])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        recorder.finish(session.outcome, manager.transcript)
        logger.info(f"Call {recorder.call_id} ended: {session.outcome or 'abandoned'}")
        await runner.cancel()
```
- **In the `finally:` block**, replace `recorder.finish(session.outcome, context.get_messages())` with `recorder.finish(session.outcome, manager.transcript)`, and add `await classifier.aclose()` before `repo.close()`.
- **Update the module docstring:** llama-server now serves the intent reader only.

- [ ] **Step 5: Run the whole suite and the server smoke test**

```bash
.venv/Scripts/python -m pytest -q
grep -rn "flow\b\|FlowManager\|make_llm\|DeliveryFlow" src tests || echo "no stale references"
.venv/Scripts/python -m voiceagent.agent.bot -t webrtc --order-id 1 --port 7876 > /tmp/bot.log 2>&1 &
BOT=$!; for i in $(seq 1 60); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:7876/client/)" = 200 ] && break; sleep 1; done
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7876/client/; kill $BOT
```
Expected: all tests pass, then `no stale references`, then `200`.

- [ ] **Step 6: Commit**

```bash
git add -A src/voiceagent/agent tests/agent
git commit -m "Replace Flows/function calling with the intent-reader dialog pipeline; cache openers, warm Smart Turn"
```

---

### Task 6: Intent accuracy evaluation (decides fine-tuning)

**Files:**
- Create: `bench/corpus/intent_cases.json`, `bench/intent_eval.py`, `tests/agent/test_intent_eval.py`

**Interfaces:**
- Consumes: `IntentClassifier` (Task 2), `STATE_INTENTS` (Task 3), `grounded` (Task 1).
- Produces:
  - `score(rows) -> dict` with `accuracy`, `value_grounded_rate` and `per_intent: {intent: {n, correct}}`, plus a `confusions` list
  - CLI `python -m bench.intent_eval --model-name <label>`, which writes `bench/results/intent_<label>.json`

- [ ] **Step 1: Write the failing test**

`tests/agent/test_intent_eval.py`:
```python
from bench.intent_eval import score


def test_score():
    rows = [
        {"expected": "confirm", "got": "confirm", "value_ok": True},
        {"expected": "give_time", "got": "give_time", "value_ok": True},
        {"expected": "add_item", "got": "confirm", "value_ok": False},
        {"expected": "add_item", "got": "add_item", "value_ok": False},
    ]
    s = score(rows)
    assert s["accuracy"] == 0.75 and s["value_grounded_rate"] == 0.5
    assert s["per_intent"]["add_item"] == {"n": 2, "correct": 1}
    assert s["confusions"] == [["add_item", "confirm", 1]]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/agent/test_intent_eval.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.intent_eval'`.

- [ ] **Step 3: Write the cases and the evaluator**

`bench/corpus/intent_cases.json` (state, what the agent just said, what the customer said, the expected intent, and words the value must contain, or `""` when no value is expected):
```json
[
  {"state": "greet", "agent": "Am I speaking with Priya?", "customer": "Yes, this is Priya.", "intent": "confirm", "value": ""},
  {"state": "greet", "agent": "Am I speaking with Priya?", "customer": "Yeah, speaking.", "intent": "confirm", "value": ""},
  {"state": "greet", "agent": "Am I speaking with Priya?", "customer": "No, you have the wrong number.", "intent": "wrong_person", "value": ""},
  {"state": "greet", "agent": "Am I speaking with Priya?", "customer": "This is her husband, she's not here.", "intent": "wrong_person", "value": ""},
  {"state": "greet", "agent": "Am I speaking with Priya?", "customer": "I'm driving, can you call me back in an hour?", "intent": "callback", "value": "in an hour"},
  {"state": "greet", "agent": "Am I speaking with Priya?", "customer": "Who is this?", "intent": "question", "value": ""},
  {"state": "greet", "agent": "Am I speaking with Priya?", "customer": "Mm-hmm.", "intent": "confirm", "value": ""},
  {"state": "greet", "agent": "Am I speaking with Priya?", "customer": "Sorry, what?", "intent": "unclear", "value": ""},
  {"state": "shortage", "agent": "Would you like whole wheat bread instead, the rest without it or to wait until it's back tomorrow?", "customer": "The whole wheat one is fine.", "intent": "substitute", "value": ""},
  {"state": "shortage", "agent": "Would you like whole wheat bread instead, the rest without it or to wait until it's back tomorrow?", "customer": "Just send the rest.", "intent": "send_available", "value": ""},
  {"state": "shortage", "agent": "Would you like whole wheat bread instead, the rest without it or to wait until it's back tomorrow?", "customer": "I'll wait for it to come back.", "intent": "wait_restock", "value": ""},
  {"state": "shortage", "agent": "Would you like whole wheat bread instead, the rest without it or to wait until it's back tomorrow?", "customer": "Skip the bread then.", "intent": "send_available", "value": ""},
  {"state": "shortage", "agent": "Would you like whole wheat bread instead, the rest without it or to wait until it's back tomorrow?", "customer": "Yes, the substitute is okay.", "intent": "substitute", "value": ""},
  {"state": "shortage", "agent": "Would you like whole wheat bread instead, the rest without it or to wait until it's back tomorrow?", "customer": "Can you add two Amul milk as well?", "intent": "add_item", "value": "Amul milk"},
  {"state": "shortage", "agent": "Would you like whole wheat bread instead, the rest without it or to wait until it's back tomorrow?", "customer": "Forget it, cancel the whole order.", "intent": "cancel", "value": ""},
  {"state": "shortage", "agent": "Would you like whole wheat bread instead, the rest without it or to wait until it's back tomorrow?", "customer": "How much is the total?", "intent": "question", "value": ""},
  {"state": "schedule", "agent": "When would you like it delivered?", "customer": "Tomorrow evening works for me.", "intent": "give_time", "value": "tomorrow evening"},
  {"state": "schedule", "agent": "When would you like it delivered?", "customer": "Can you do it between 2 and 4 tomorrow?", "intent": "give_time", "value": "2 and 4 tomorrow"},
  {"state": "schedule", "agent": "When would you like it delivered?", "customer": "Any time after 6.", "intent": "give_time", "value": "after 6"},
  {"state": "schedule", "agent": "When would you like it delivered?", "customer": "As soon as possible please.", "intent": "give_time", "value": "as soon as possible"},
  {"state": "schedule", "agent": "I can deliver tomorrow, 5 to 6 PM. Shall I book that?", "customer": "Yes, that works.", "intent": "confirm", "value": ""},
  {"state": "schedule", "agent": "I can deliver tomorrow, 5 to 6 PM. Shall I book that?", "customer": "No, that's too late.", "intent": "deny", "value": ""},
  {"state": "schedule", "agent": "I have today, 3 to 4 PM, today, 4 to 5 PM or today, 6 to 7 PM. Which works for you?", "customer": "The second one.", "intent": "pick_option", "value": "second"},
  {"state": "schedule", "agent": "I have today, 3 to 4 PM, today, 4 to 5 PM or today, 6 to 7 PM. Which works for you?", "customer": "Six to seven is good.", "intent": "pick_option", "value": "six to seven"},
  {"state": "schedule", "agent": "When would you like it delivered?", "customer": "Friday afternoon.", "intent": "give_time", "value": "Friday afternoon"},
  {"state": "schedule", "agent": "When would you like it delivered?", "customer": "I'm busy now, call me later.", "intent": "callback", "value": "later"},
  {"state": "address", "agent": "Should we deliver to 12 MG Road, Bengaluru?", "customer": "Yes, that's right.", "intent": "confirm", "value": ""},
  {"state": "address", "agent": "Should we deliver to 12 MG Road, Bengaluru?", "customer": "No, deliver to my office at 42 Church Street.", "intent": "change_address", "value": "42 Church Street"},
  {"state": "address", "agent": "Should we deliver to 12 MG Road, Bengaluru?", "customer": "No.", "intent": "deny", "value": ""},
  {"state": "address", "agent": "Should we deliver to 12 MG Road, Bengaluru?", "customer": "It's flat 304, B block, Prestige Towers, Indiranagar.", "intent": "change_address", "value": "Prestige Towers"},
  {"state": "address", "agent": "Should we deliver to 12 MG Road, Bengaluru?", "customer": "Yes.", "intent": "confirm", "value": ""},
  {"state": "address", "agent": "Should we deliver to 12 MG Road, Bengaluru?", "customer": "Also add a dozen eggs.", "intent": "add_item", "value": "eggs"},
  {"state": "note", "agent": "Any instructions for the driver?", "customer": "Please leave it with the security guard.", "intent": "add_note", "value": "security guard"},
  {"state": "note", "agent": "Any instructions for the driver?", "customer": "No, that's all.", "intent": "confirm", "value": ""},
  {"state": "note", "agent": "Any instructions for the driver?", "customer": "Call me when you reach the gate.", "intent": "add_note", "value": "reach the gate"},
  {"state": "note", "agent": "Any instructions for the driver?", "customer": "Nothing, thanks.", "intent": "confirm", "value": ""},
  {"state": "note", "agent": "Any instructions for the driver?", "customer": "Ring the bell twice, the dog is friendly.", "intent": "add_note", "value": "bell twice"},
  {"state": "note", "agent": "Any instructions for the driver?", "customer": "Don't ring the bell, the baby is sleeping.", "intent": "add_note", "value": "baby is sleeping"},
  {"state": "confirm", "agent": "Shall I book it?", "customer": "Yes, please book it.", "intent": "confirm", "value": ""},
  {"state": "confirm", "agent": "Shall I book it?", "customer": "Go ahead.", "intent": "confirm", "value": ""},
  {"state": "confirm", "agent": "Shall I book it?", "customer": "Actually, make it after 7 instead.", "intent": "give_time", "value": "after 7"},
  {"state": "confirm", "agent": "Shall I book it?", "customer": "Wait, the address is wrong, it's 9 Brigade Road.", "intent": "change_address", "value": "9 Brigade Road"},
  {"state": "confirm", "agent": "Shall I book it?", "customer": "No, something's not right.", "intent": "deny", "value": ""},
  {"state": "confirm", "agent": "Shall I book it?", "customer": "Yes. Oh and tell him to call first.", "intent": "add_note", "value": "call first"},
  {"state": "confirm", "agent": "Shall I book it?", "customer": "Cancel it, I don't need it anymore.", "intent": "cancel", "value": ""},
  {"state": "confirm_cancel", "agent": "Just to confirm, do you want to cancel the whole order?", "customer": "Yes, cancel it.", "intent": "confirm", "value": ""},
  {"state": "confirm_cancel", "agent": "Just to confirm, do you want to cancel the whole order?", "customer": "No no, keep it.", "intent": "deny", "value": ""},
  {"state": "confirm_cancel", "agent": "Just to confirm, do you want to cancel the whole order?", "customer": "Hmm, let me think.", "intent": "unclear", "value": ""}
]
```

`bench/intent_eval.py`:
```python
"""Intent reader accuracy against llama-server.

    bash scripts/llama_server.sh models/llm/Qwen3.5-2B-Q4_K_M.gguf
    python -m bench.intent_eval --model-name qwen3.5-2b-q4km
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter

from bench.common import CORPUS_DIR, summarize, write_result
from voiceagent.agent.dialog import STATE_INTENTS
from voiceagent.agent.intent import IntentClassifier
from voiceagent.agent.nlu import grounded


def score(rows: list[dict]) -> dict:
    per_intent: dict[str, dict] = {}
    for row in rows:
        stats = per_intent.setdefault(row["expected"], {"n": 0, "correct": 0})
        stats["n"] += 1
        stats["correct"] += row["got"] == row["expected"]
    confusions = Counter((r["expected"], r["got"]) for r in rows if r["got"] != r["expected"])
    return {
        "accuracy": round(sum(r["got"] == r["expected"] for r in rows) / len(rows), 3),
        "value_grounded_rate": round(sum(r["value_ok"] for r in rows) / len(rows), 3),
        "per_intent": per_intent,
        "confusions": [[e, g, n] for (e, g), n in confusions.most_common()],
    }


async def run(model_name: str, url: str) -> dict:
    cases = json.loads((CORPUS_DIR / "intent_cases.json").read_text(encoding="utf-8"))
    classifier = IntentClassifier(url)
    rows, latencies = [], []
    try:
        await classifier.classify("warm up", "warm up", ["unclear"])
        for case in cases:
            started = time.perf_counter()
            intent = await classifier.classify(case["agent"], case["customer"], STATE_INTENTS[case["state"]])
            latencies.append((time.perf_counter() - started) * 1000)
            value_ok = (not case["value"]) or (case["value"].lower() in intent.value.lower()
                                               and grounded(intent.value, case["customer"]))
            rows.append({**case, "expected": case["intent"], "got": intent.name, "got_value": intent.value,
                         "value_ok": value_ok})
    finally:
        await classifier.aclose()
    result = {"model": model_name, "latency_ms": summarize(latencies), **score(rows), "rows": rows}
    path = write_result(f"intent_{model_name}", result, device="cuda (llama-server)")
    for row in rows:
        if row["got"] != row["expected"] or not row["value_ok"]:
            print(f"MISS [{row['state']}] {row['customer']!r}: expected {row['expected']} got {row['got']} "
                  f"value={row['got_value']!r}")
    print(f"{model_name}: accuracy {result['accuracy']}, value ok {result['value_grounded_rate']}, "
          f"latency p50 {result['latency_ms']['p50']} ms p95 {result['latency_ms']['p95']} ms -> {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1")
    args = parser.parse_args()
    asyncio.run(run(args.model_name, args.url))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the evaluation**

```bash
.venv/Scripts/python -m pytest tests/agent/test_intent_eval.py -q
.venv/Scripts/python -m bench.intent_eval --model-name qwen3.5-2b-q4km
```
Expected: `1 passed`, then a list of MISS lines and a summary line.

**Decision rule.** Write it in Task 8's doc:
- **No fine-tuning needed** if accuracy ≥ 0.95 and there are no misses where a non-changing intent (confirm, deny, unclear, question) was read as a changing one (add_item, change_address, add_note, cancel, substitute, send_available, wait_restock).
- **Otherwise, the first fix is the prompt.** Clarify the confused descriptions in `INTENT_DESCRIPTIONS` and re-run.
- **If it is still below the bar, compare models.** Run the same evaluation on Qwen3-4B-Instruct-2507 (`bash scripts/llama_server.sh models/llm/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`) and record both.
- **Fine-tuning is justified only if** the 2B misses the bar and the 4B either misses it too or is too slow (p50 > 500 ms).

- [ ] **Step 5: Commit**

```bash
git add bench/corpus/intent_cases.json bench/intent_eval.py tests/agent/test_intent_eval.py bench/results/intent_*.json
git commit -m "Add intent reader evaluation set and results"
```

---

### Task 7: Scripted test caller

**Files:**
- Create: `scripts/scripted_call.py`

**Interfaces:**
- Produces: `python scripts/scripted_call.py --url http://localhost:7860/api/offer --out call.wav --duration 120 "22:Yeah, that's me." "36:..."`. Each cue is `seconds:text`. Cues are synthesized with Kokoro on the CPU before dialing, then spoken over WebRTC at their times. The script prints when the agent was speaking.

- [ ] **Step 1: Write the script**

`scripts/scripted_call.py`:
```python
"""Dial the agent over WebRTC like the browser does, speak scripted lines, record and time the agent's speech.

    python scripts/scripted_call.py --url http://localhost:7860/api/offer --out call.wav --duration 120 \
        "20:Yeah, that's me." "34:Tomorrow evening works for me."
"""
from __future__ import annotations

import argparse
import asyncio
import fractions
import time

import av
import httpx
import numpy as np
import soundfile as sf
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from scipy.signal import resample_poly

RATE = 48000
FRAME = 960  # 20 ms at 48 kHz


class ScriptedSpeaker(MediaStreamTrack):
    kind = "audio"

    def __init__(self, cues: list[tuple[float, np.ndarray, str]]):
        super().__init__()
        self.cues, self.t0, self.pts, self.queue = cues, None, 0, np.zeros(0, np.int16)

    async def recv(self):
        if self.t0 is None:
            self.t0 = time.time()
        await asyncio.sleep(max(0.0, self.t0 + self.pts / RATE - time.time()))
        now = time.time() - self.t0
        while self.cues and self.cues[0][0] <= now:
            _, pcm, text = self.cues.pop(0)
            self.queue = np.concatenate([self.queue, pcm])
            print(f"[{now:6.1f}s] customer: {text}", flush=True)
        chunk, self.queue = self.queue[:FRAME], self.queue[FRAME:]
        frame = av.AudioFrame.from_ndarray(np.pad(chunk, (0, FRAME - len(chunk))).reshape(1, -1), format="s16",
                                           layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = RATE, self.pts, fractions.Fraction(1, RATE)
        self.pts += FRAME
        return frame


def synthesize(lines: list[str]) -> list[np.ndarray]:
    from kokoro import KModel, KPipeline

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M",
                         model=KModel(repo_id="hexgrad/Kokoro-82M").to("cpu").eval())
    out = []
    for line in lines:
        audio = np.concatenate([r.audio.numpy() for r in pipeline(line, voice="am_michael")])
        out.append((np.clip(resample_poly(audio, 2, 1), -1, 1) * 32767).astype(np.int16))  # 24 kHz -> 48 kHz
    return out


async def call(url: str, out: str, duration: float, cues: list[tuple[float, str]]) -> None:
    pcms = synthesize([text for _, text in cues])
    pc = RTCPeerConnection()
    pc.addTrack(ScriptedSpeaker([(t, pcm, text) for (t, text), pcm in zip(cues, pcms)]))
    received: list[tuple[float, np.ndarray, int]] = []
    started = time.time()

    @pc.on("track")
    def on_track(track):
        async def pull():
            while True:
                try:
                    frame = await track.recv()
                except Exception:
                    return
                received.append((time.time() - started, frame.to_ndarray().astype(np.float32).mean(axis=0) / 32768,
                                 frame.sample_rate))
        asyncio.ensure_future(pull())

    await pc.setLocalDescription(await pc.createOffer())
    async with httpx.AsyncClient(timeout=120) as client:
        answer = (await client.post(url, json={"sdp": pc.localDescription.sdp,
                                               "type": pc.localDescription.type})).json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    print(f"[{time.time() - started:6.1f}s] connected", flush=True)
    await asyncio.sleep(duration)
    await pc.close()
    if not received:
        print("no agent audio received")
        return
    sf.write(out, np.concatenate([r[1] for r in received]), received[0][2])
    spans, begin = [], None
    for t, audio, _ in received:
        loud = float(np.sqrt(np.mean(audio ** 2))) > 0.01
        if loud and begin is None:
            begin = t
        elif not loud and begin is not None:
            if t - begin > 0.3:
                spans.append(f"{begin:.1f}-{t:.1f}")
            begin = None
    print("agent speaking (s):", ", ".join(spans))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860/api/offer")
    parser.add_argument("--out", default="call.wav")
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("cues", nargs="+", help='"seconds:text" spoken by the customer')
    args = parser.parse_args()
    cues = sorted((float(c.split(":", 1)[0]), c.split(":", 1)[1]) for c in args.cues)
    asyncio.run(call(args.url, args.out, args.duration, cues))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Check it parses its arguments**

Run: `.venv/Scripts/python scripts/scripted_call.py --help`
Expected: usage text listing `--url`, `--out`, `--duration` and `cues`.

- [ ] **Step 3: Commit**

```bash
git add scripts/scripted_call.py
git commit -m "Add scripted WebRTC test caller"
```

---

### Task 8: Measured calls and results

**Files:**
- Create: `docs/benchmarks/phase3-dialog-calls.md`

- [ ] **Step 1: Run three scripted calls**

Do these with llama-server (Qwen3.5-2B) running and a freshly seeded DB:
```bash
.venv/Scripts/python -m voiceagent.db.seed --db data/warehouse.db
```
Start the bot for each order:
```bash
.venv/Scripts/python -m voiceagent.agent.bot -t webrtc --order-id <id> --port 7880 > bot.log 2>&1 &
```
Run each call with `--url http://localhost:7880/api/offer`, then stop the bot. Leave about 12 s between cues; the bot loads models before answering the offer.

| Call | Order | Cues |
|---|---|---|
| A, happy path | first order without shortages | "20:Yeah, that's me." "32:Tomorrow evening works for me." "44:Yes, book that." "56:Yes, that's right." "68:No, that's all." "80:Yes, please book it." |
| B, shortage + add item | first order with a shortage | "20:Yes, speaking." "32:Send the rest without it." "44:Can you also add two Amul milk?" "56:Tomorrow after 6." "68:Yes." "80:Yes." "92:Leave it with the guard." "104:Yes, book it." |
| C, cancel | any | "20:Yes, this is me." "32:Actually I want to cancel the whole order." "44:Yes, cancel it." |

List the order ids with: `.venv/Scripts/python -c "from voiceagent.db.repository import Repository; from voiceagent.domain.inventory import InventoryService as I; r=Repository('data/warehouse.db'); i=I(r); ids=[o for (o,) in r.conn.execute('select id from orders order by id limit 40')]; print('no shortage', [o for o in ids if not i.shortages(o)][:3], 'shortage', [o for o in ids if i.shortages(o)][:3])"`

- [ ] **Step 2: Collect the numbers**

For each call, take the `tool ...`, `turn state=...` and `whisper decode ...` lines from the bot log, and the call's `turn_metrics` from `python -m voiceagent.agent.dashboard` (or `Repository.call_metrics`).

- [ ] **Step 3: Write `docs/benchmarks/phase3-dialog-calls.md`**

It should contain:
- **Outcome per call:** expected vs. actual order state, and every tool call made.
- **Unrequested changes:** list any change the customer didn't ask for. The target is none.
- **Response latency:** p50 and p95 across all `response` rows, plus one per-turn breakdown (endpointing, STT, turn detection, intent read, TTS).
- **Whisper decode times** from the log, explaining the Phase 2 STT figure.
- **The intent evaluation result** (Task 6), and the fine-tuning decision under its rule.
- **What is still above the 700 ms target, and why.**

```bash
git add docs/benchmarks/phase3-dialog-calls.md
git commit -m "Record phase 3 dialog-manager call results"
```
