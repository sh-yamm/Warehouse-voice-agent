import pytest

from voiceagent.agent.tools import CallSession

from conftest import NOW


@pytest.fixture
def session(world, repo):
    return CallSession(repo, 1, clock=lambda: NOW)


def test_check_slot_holds_available_slot(session, repo):
    result = session.check_slot("tomorrow after 5")
    assert result == {"status": "held", "slot_id": 23, "slot": "tomorrow, 5 to 6 PM"}
    assert session.held_slot_id == 23
    assert repo.get_active_hold(1, NOW) == 23


def test_check_slot_full_offers_alternatives(session):
    result = session.check_slot("today at 5 pm")
    assert result["status"] == "full"
    assert [a["slot"] for a in result["alternatives"]] == [
        "today, 3 to 4 PM", "today, 4 to 5 PM", "today, 6 to 7 PM"]


def test_check_slot_unclear(session):
    assert session.check_slot("I don't know")["status"] == "unclear"


def test_check_slot_twice_keeps_single_hold(session, repo):
    session.check_slot("tomorrow after 5")
    session.check_slot("tomorrow morning")
    assert repo.active_holds(23, NOW) == 0
    assert repo.get_active_hold(1, NOW) == session.held_slot_id == 14


def test_choose_slot(session):
    assert session.choose_slot(9) == {"status": "held", "slot_id": 9, "slot": "today, 4 to 5 PM"}


def test_choose_slot_rejects_unknown_and_full_slots(session, repo):
    for slot_id in (10, 999):
        result = session.choose_slot(slot_id)
        assert result["status"] == "unavailable"
        assert len(result["alternatives"]) == 3
    assert repo.get_active_hold(1, NOW) is None


def test_confirm_booking(session, repo):
    assert session.confirm_booking()["status"] == "failed"
    session.check_slot("tomorrow after 5")
    assert session.confirm_booking() == {"status": "booked", "slot": "tomorrow, 5 to 6 PM"}
    assert session.outcome == "scheduled"
    assert repo.get_order(1).status == "scheduled"


def test_add_item(session, repo):
    added = session.add_item("nandini milk", 2)
    assert (added["status"], added["item"], added["available"]) == ("added", "Nandini milk", 3)
    short = session.add_item("basmati rice", 5)
    assert (short["status"], short["available"]) == ("insufficient", 2)
    assert session.add_item("toothpaste", 1) == {"status": "not_found"}


def test_add_item_coerces_and_rejects_quantity(session):
    assert session.add_item("amul milk", "2")["status"] == "added"
    assert session.add_item("amul milk", "two")["status"] == "error"
    assert session.add_item("amul milk", 0)["status"] == "error"


def test_resolve_shortage_substitute(session):
    assert session.resolve_shortage("BREAD1", "substitute", "BREAD2") == {
        "status": "done", "remaining_shortages": []}


def test_resolve_shortage_wait_reports_restock(session):
    result = session.resolve_shortage("BREAD1", "wait")
    assert result["status"] == "done" and result["deliver_after"] == "tomorrow 9 AM"


def test_resolve_shortage_bad_choice(session):
    assert session.resolve_shortage("BREAD1", "teleport")["status"] == "error"


def test_resolve_shortage_unknown_sku_fails(session, repo):
    assert session.resolve_shortage("NOPE", "partial")["status"] == "failed"
    assert [l.resolution for l in repo.get_order_lines(1)] == [None, None, None]


def test_update_address_and_note(session, repo):
    assert session.update_address("x")["status"] == "error"
    assert session.update_address("42 Church Street, Bengaluru") == {
        "status": "updated", "address": "42 Church Street, Bengaluru"}
    assert session.add_note("leave with the security guard") == {"status": "saved"}
    assert repo.get_order(1).notes == "leave with the security guard"
    assert session.add_note("  ")["status"] == "error"


def test_cancel_order(session):
    assert session.cancel_order() == {"status": "cancelled"}
    assert session.outcome == "cancelled"
    assert session.cancel_order() == {"status": "failed"}


def test_schedule_callback(session, repo):
    assert session.schedule_callback("tomorrow morning") == {"status": "scheduled", "callback_at": "tomorrow 8 AM"}
    assert session.outcome == "callback"
    assert session.schedule_callback("")["callback_at"] == "today 11 AM"


def test_wrong_person(session, repo):
    assert session.wrong_person() == {"status": "noted"}
    assert session.outcome == "wrong_person"


def test_order_summary(session):
    assert session.order_summary()["items"] == ["2 Amul milk", "1 brown bread, not available yet", "1 eggs"]
    session.resolve_shortage("BREAD1", "substitute", "BREAD2")
    session.check_slot("tomorrow after 5")
    summary = session.order_summary()
    assert summary["items"][1] == "1 whole wheat bread instead of brown bread"
    assert summary["slot"] == "tomorrow, 5 to 6 PM"
    assert summary["address"] == "12 MG Road, Bengaluru"


def test_wait_restock_releases_hold_before_restock(session, repo):
    session.check_slot("today at noon")
    result = session.resolve_shortage("BREAD1", "wait")
    assert result["held_slot_released"] is True
    assert session.held_slot_id is None and repo.get_active_hold(1, NOW) is None
