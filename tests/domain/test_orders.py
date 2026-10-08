from datetime import timedelta

import pytest

from voiceagent.domain.inventory import InventoryService
from voiceagent.domain.orders import OrderService
from voiceagent.domain.scheduling import SchedulingService


@pytest.fixture
def orders(world, repo):
    return OrderService(repo)


def test_cancel_releases_reservations(orders, repo):
    assert orders.cancel_order(1) is True
    assert repo.available_stock(1, "MILK1") == 10
    assert repo.available_stock(1, "EGGS12") == 20
    assert repo.get_order(1).status == "cancelled"
    assert orders.cancel_order(1) is False


def test_cancel_scheduled_order_frees_slot_and_substitute_stock(orders, repo, world, now):
    inv = InventoryService(repo)
    sched = SchedulingService(repo, inv)
    inv.apply_substitution(1, "BREAD1", "BREAD2")
    slot_id = world["slots"][(1, 18)]
    sched.hold_slot(1, slot_id, now)
    sched.confirm_booking(1, now)
    assert orders.cancel_order(1) is True
    assert repo.get_slot(slot_id).booked == 0
    assert repo.available_stock(1, "BREAD2") == 4
    assert repo.get_order(1).slot_id is None


def test_cancel_unknown_order(orders):
    assert orders.cancel_order(999) is False


def test_update_address_collapses_whitespace(orders, repo):
    orders.update_address(1, "  42   Church Street,  Bengaluru ")
    assert repo.get_order(1).address == "42 Church Street, Bengaluru"
    with pytest.raises(ValueError):
        orders.update_address(1, "  ")


def test_add_note_appends(orders, repo):
    orders.add_note(1, "leave with the security guard")
    orders.add_note(1, "call at the gate")
    assert repo.get_order(1).notes == "leave with the security guard; call at the gate"
    with pytest.raises(ValueError):
        orders.add_note(1, "   ")


def test_schedule_callback(orders, repo, now):
    orders.schedule_callback(1, now + timedelta(hours=1))
    order = repo.get_order(1)
    assert (order.status, order.callback_at) == ("callback", now + timedelta(hours=1))


def test_mark_wrong_person(orders, repo):
    orders.mark_wrong_person(1)
    order = repo.get_order(1)
    assert order.status == "callback" and "wrong person" in order.notes
