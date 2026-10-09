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
    say(dm, "give_time", "Make it tomorrow after 6.", "tomorrow after 6")
    reply = say(dm, "confirm", "Yes.")
    assert dm.state == "confirm" and reply.sentences[-1] == "Shall I book it?"
    assert "tomorrow, 6 to 7 PM" in reply.sentences[-2]


def test_transcript_records_both_sides(dm):
    say(dm, "confirm", "Yes, this is Priya.")
    assert [t["role"] for t in dm.transcript] == ["assistant", "user", "assistant"]
    assert dm.transcript[1]["content"] == "Yes, this is Priya."
