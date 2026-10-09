"""Deterministic call dialog. The LLM only names the intent; this decides, acts and speaks."""
from __future__ import annotations

import re
from dataclasses import dataclass

from voiceagent.agent.intent import Intent
from voiceagent.agent.nlu import grounded, said_number, split_quantity
from voiceagent.agent.tools import CallSession
from voiceagent.domain.context import spoken_day
from voiceagent.domain.inventory import Shortage
from voiceagent.domain.timeparse import parse_time_window

AGENT_NAME = "Avio"
COMPANY = "AvioStack"
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
    "confirm_address": ["confirm", "deny", "change_address", "callback", "cancel", "question", "unclear"],
    "closed": ["unclear"],
}
_ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
# Cancelling is the one irreversible action: a "yes" that contains any of these is treated as a refusal.
_NEGATIONS = {"no", "not", "don't", "dont", "keep", "never", "nope", "wait", "stop"}
# Ending the call for a callback needs a real "not now" cue; otherwise it was probably a driver instruction.
_CALLBACK_CUES = ("later", "busy", "call back", "call me back", "callback", "another time", "driving", "meeting",
                  "not now", "can't talk", "cannot talk", "in an hour", "in a while")
# A "note" made only of these words means "no instructions".
_EMPTY_NOTE_WORDS = {"nothing", "none", "no", "nope", "thanks", "thank", "you", "that's", "thats", "all", "it",
                     "is", "fine", "ok", "okay", "nah", "not", "really"}


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
        # (candidate address, state to continue in) while the customer confirms a new address
        self.pending_address: tuple[str | None, str] | None = None
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
            # What the intent reader sees as "AGENT:": the whole question incl. listed options, not just the last line.
            self.question = " ".join(s for s in reply.sentences if s not in ACKS) or reply.sentences[-1]
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
            if any(cue in utterance.lower() for cue in _CALLBACK_CUES):
                return self._callback(intent, utterance)
            if "give_time" in self.allowed_intents() and parse_time_window(utterance, self.session.clock()):
                intent = Intent("give_time", intent.value)  # "tomorrow evening is better" is a delivery time
            elif "add_note" in self.allowed_intents():
                intent = Intent("add_note", intent.value or utterance)
            else:
                return Reply(["Sorry.", "I didn't catch that.", *self._ask()])
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
        if self.state == "confirm_address":
            value = self.pending_address[0] if self.pending_address else None
            return [f"Is the new address {value}?"] if value else ["Could you tell me the full address again?"]
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
        if not said_number(qty, utterance):
            qty = 1  # a quantity the customer never said (e.g. copied from "we only have 5 left")
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
        if any(cue in utterance.lower() for cue in _CALLBACK_CUES):
            return self._callback(intent, utterance)  # "No, I'm driving, call me later" is not a wrong number
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
        if self.alternatives:
            index = self._ordinal_index(utterance)
            if index is not None:  # "the second one" labelled as a time still means a listed option
                return self._pick(index)
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

    def _ordinal_index(self, text: str) -> int | None:
        for word in text.lower().replace(",", " ").replace(".", " ").split():
            if word in _ORDINALS and _ORDINALS[word] < len(self.alternatives):
                return _ORDINALS[word]
            if word == "last" and self.alternatives:
                return len(self.alternatives) - 1
        return None

    def _option_index(self, text: str) -> int | None:
        index = self._ordinal_index(text)
        if index is not None:
            return index
        window = parse_time_window(text, self.session.clock())
        if window:
            for index, alt in enumerate(self.alternatives):
                if window.start <= self.session.repo.get_slot(alt["slot_id"]).start < window.end:
                    return index
        return None

    def _pick(self, index: int) -> Reply:
        result = self.session.choose_slot(self.alternatives[index]["slot_id"])
        if result["status"] != "held":
            self.alternatives = result["alternatives"]
            return Reply(["Sorry.", "That slot just filled up.", *self._ask()])
        self.alternatives = []
        return self._after_slot_agreed(f"{result['slot']} it is.")

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
            text = intent.value if grounded(intent.value, utterance) else utterance
            index = self._option_index(text)
            if index is None:
                return Reply(["Sorry.", f"Was that {self._options()}?"])
            return self._pick(index)
        if intent.name == "confirm" and self.session.held_slot_id:
            return self._after_slot_agreed(None)
        if intent.name == "deny":
            return Reply(["No problem.", "What day and time would suit you?"])
        return Reply(["Okay.", *self._ask()])

    def _change_address(self, intent: Intent, utterance: str, then: str) -> Reply:
        if len(re.findall(r"[a-z0-9]+", intent.value.lower())) < 3 or not grounded(intent.value, utterance):
            if self.state == "address":
                self.awaiting_address = True
            return Reply(["Sorry.", "Could you tell me the full address again?"])
        # Speech recognition can turn an address into plausible-looking nonsense, and the address decides where
        # the order goes, so it is read back and saved only after an explicit yes.
        self.pending_address = (intent.value, then)
        self.state = "confirm_address"
        return Reply(["Okay.", f"Just to check, the new address is {intent.value}.", "Is that right?"])

    def _continue_after_address(self, lead: list[str], then: str) -> Reply:
        self.pending_address, self.awaiting_address = None, False
        if then == "note":
            self.state = "note"
            return Reply(lead + ["Any instructions for the driver?"])
        self.state = "confirm"
        return Reply(lead + self._read_back())

    def _on_confirm_address(self, intent: Intent, utterance: str) -> Reply:
        value, then = self.pending_address or (None, "note")
        if intent.name == "change_address":
            return self._change_address(intent, utterance, then)
        negated = bool(_NEGATIONS & set(utterance.lower().replace(",", " ").replace(".", " ").split()))
        if intent.name == "confirm" and value and not negated:
            result = self.session.update_address(value)
            if result["status"] == "updated":
                return self._continue_after_address(
                    ["Got it.", f"I've updated the address to {result['address']}."], then)
        if intent.name == "confirm" and not value:  # no candidate pending: "keep the old one" keeps it
            return self._continue_after_address(["Okay.", f"I'll keep {self._address()}."], then)
        self.pending_address = (None, then)  # a "no": forget the candidate, keep the saved address
        return Reply(["Sorry.", "Could you tell me the full address again?"])

    def _on_address(self, intent: Intent, utterance: str) -> Reply:
        if intent.name == "change_address":
            return self._change_address(intent, utterance, then="note")
        if intent.name == "confirm":
            self.awaiting_address = False  # "the old one is fine" while we waited for a new address
            self.state = "note"
            return Reply(["Okay.", "Any instructions for the driver?"])
        self.awaiting_address = True
        return Reply(["No problem.", "What's the correct address?"])

    def _add_note(self, intent: Intent, utterance: str) -> Reply:
        words = set(intent.value.lower().replace(",", " ").replace(".", " ").split())
        if words and words <= _EMPTY_NOTE_WORDS:
            self.details_done, self.state = True, "confirm"
            return Reply(["Okay.", *self._read_back()])
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
        negated = bool(_NEGATIONS & set(utterance.lower().replace(",", " ").replace(".", " ").split()))
        if intent.name == "confirm" and not negated:
            self.session.cancel_order()
            self.state = "closed"
            return Reply(["Okay.", "Your order is cancelled.", "Goodbye!"], end_call=True)
        self.state = self._before_cancel
        return Reply(["Okay.", "I won't cancel it.", *self._ask()])
