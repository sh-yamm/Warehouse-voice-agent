from datetime import timedelta

import pytest

from voiceagent.domain.inventory import InventoryService
from voiceagent.domain.models import TimeWindow
from voiceagent.domain.scheduling import SchedulingService


@pytest.fixture
def inv(world, repo):
    return InventoryService(repo)


@pytest.fixture
def sched(inv, repo):
    return SchedulingService(repo, inv)


def window(now, day, h1, h2):
    base = (now + timedelta(days=day)).replace(hour=0, minute=0)
    return TimeWindow(base + timedelta(hours=h1), base + timedelta(hours=h2))


def test_free_slots_start_now_and_skip_full(sched, now):
    hours = [o.slot.start.hour for o in sched.free_slots(1, now, limit=10)]
    assert hours[:3] == [10, 11, 12]
    assert 17 not in hours[:8]


def test_check_available(sched, now):
    check = sched.check_slot(1, window(now, 1, 17, 21), now)
    assert check.status == "available"
    assert check.option.slot.start == window(now, 1, 17, 18).start


def test_check_full_offers_three_nearest_sorted(sched, now):
    check = sched.check_slot(1, window(now, 0, 17, 18), now)
    assert check.status == "full" and check.option is None
    assert [o.slot.start.hour for o in check.alternatives] == [15, 16, 18]


def test_check_no_slots_on_day_without_slots(sched, now):
    check = sched.check_slot(1, window(now, 2, 8, 21), now)
    assert check.status == "no_slots"
    assert len(check.alternatives) == 3


def test_wait_restock_pushes_earliest_slot(sched, inv, now):
    inv.wait_restock(1, "BREAD1")
    first = sched.free_slots(1, now)[0]
    assert first.slot.start == (now + timedelta(days=1)).replace(hour=9)
    assert sched.check_slot(1, window(now, 0, 12, 14), now).status == "no_slots"


def test_hold_and_confirm(sched, repo, world, now):
    slot_id = world["slots"][(1, 18)]
    assert sched.hold_slot(1, slot_id, now) is True
    slot = sched.confirm_booking(1, now + timedelta(minutes=2))
    assert slot.id == slot_id and slot.booked == 1
    order = repo.get_order(1)
    assert (order.status, order.slot_id) == ("scheduled", slot_id)
    assert repo.get_active_hold(1, now) is None


def test_confirm_without_hold_fails(sched, now):
    assert sched.confirm_booking(1, now) is None


def test_confirm_after_expiry_fails(sched, repo, world, now):
    slot_id = world["slots"][(1, 18)]
    sched.hold_slot(1, slot_id, now)
    assert sched.confirm_booking(1, now + timedelta(minutes=6)) is None
    assert repo.get_slot(slot_id).booked == 0
    assert repo.get_order(1).status == "pending_schedule"


def test_hold_respects_other_orders_holds(sched, repo, world, now):
    slot_id = world["slots"][(0, 16)]
    repo.increment_booked(slot_id)  # one seat left
    assert sched.hold_slot(1, slot_id, now) is True
    assert sched.hold_slot(2, slot_id, now) is False
    assert sched.hold_slot(2, slot_id, now + timedelta(minutes=6)) is True  # first hold expired


def test_rehold_moves_hold(sched, repo, world, now):
    sched.hold_slot(1, world["slots"][(0, 12)], now)
    sched.hold_slot(1, world["slots"][(0, 13)], now)
    assert repo.get_active_hold(1, now) == world["slots"][(0, 13)]


def test_hold_rejects_full_and_unknown_slots(sched, world, now):
    assert sched.hold_slot(1, world["slots"][(0, 17)], now) is False
    assert sched.hold_slot(1, 9999, now) is False


def test_confirm_twice_fails(sched, world, now):
    sched.hold_slot(1, world["slots"][(1, 18)], now)
    assert sched.confirm_booking(1, now) is not None
    assert sched.confirm_booking(1, now) is None


def test_hold_rejected_on_cancelled_order(sched, repo, world, now):
    from voiceagent.domain.orders import OrderService
    OrderService(repo).cancel_order(1)
    assert sched.hold_slot(1, world["slots"][(1, 18)], now) is False


def test_hold_rejects_slot_already_started(sched, world, now):
    assert sched.hold_slot(1, world["slots"][(0, 10)], now.replace(minute=58)) is False


def test_shared_repository_threads_cannot_double_hold_last_seat(sched, repo, world, now, monkeypatch):
    import threading
    import time as _time
    slot_id = world["slots"][(0, 16)]
    repo.increment_booked(slot_id)  # one seat left
    original = repo.active_holds

    def slow_active_holds(*args, **kwargs):
        result = original(*args, **kwargs)
        _time.sleep(0.2)
        return result

    monkeypatch.setattr(repo, "active_holds", slow_active_holds)
    results = {}
    threads = [threading.Thread(target=lambda o=o: results.__setitem__(o, sched.hold_slot(o, slot_id, now)))
               for o in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results.values()) == [False, True]
