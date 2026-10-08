import sqlite3
from datetime import timedelta

import pytest


def test_get_order_includes_customer(world, repo):
    order = repo.get_order(1)
    assert order.customer_name == "Priya Sharma"
    assert order.status == "pending_schedule"
    assert order.slot_id is None and order.callback_at is None


def test_get_order_missing_returns_none(world, repo):
    assert repo.get_order(999) is None


def test_order_lines_in_insertion_order_with_spoken_names(world, repo):
    lines = repo.get_order_lines(1)
    assert [(l.sku, l.name, l.qty, l.reserved_qty) for l in lines] == [
        ("MILK1", "Amul milk", 2, 2), ("BREAD1", "brown bread", 1, 0), ("EGGS12", "eggs", 1, 1),
    ]


def test_available_stock_subtracts_reserved(world, repo):
    assert repo.available_stock(1, "MILK1") == 8
    assert repo.available_stock(1, "NOPE") == 0


def test_reserving_beyond_on_hand_raises(world, repo):
    with pytest.raises(sqlite3.IntegrityError):
        repo.reserve_stock(1, "RICE5", 3)


def test_reserve_unknown_stock_row_raises(world, repo):
    with pytest.raises(ValueError):
        repo.reserve_stock(1, "NOPE", 1)


def test_transaction_rolls_back_on_error(world, repo):
    with pytest.raises(RuntimeError):
        with repo.transaction():
            repo.reserve_stock(1, "MILK1", 1)
            raise RuntimeError("boom")
    assert repo.available_stock(1, "MILK1") == 8


def test_nested_transaction_joins_outer(world, repo):
    with pytest.raises(RuntimeError):
        with repo.transaction():
            with repo.transaction():
                repo.reserve_stock(1, "MILK1", 1)
            raise RuntimeError("boom")
    assert repo.available_stock(1, "MILK1") == 8


def test_update_order_rejects_unknown_field(world, repo):
    with pytest.raises(ValueError):
        repo.update_order(1, customer_id=2)


def test_update_order_round_trips_datetime(world, repo, now):
    repo.update_order(1, status="callback", callback_at=now + timedelta(hours=1))
    order = repo.get_order(1)
    assert order.status == "callback"
    assert order.callback_at == now + timedelta(hours=1)


def test_list_slots_is_half_open_and_ordered(world, repo, now):
    start = now.replace(hour=17)
    slots = repo.list_slots(1, start, start + timedelta(hours=2))
    assert [s.start.hour for s in slots] == [17, 18]


def test_holds_count_only_unexpired_and_respect_exclusion(world, repo, now):
    slot_id = world["slots"][(0, 12)]
    repo.put_hold(slot_id, 1, now + timedelta(minutes=5))
    repo.put_hold(slot_id, 2, now - timedelta(minutes=1))
    assert repo.active_holds(slot_id, now) == 1
    assert repo.active_holds(slot_id, now, exclude_order_id=1) == 0
    assert repo.get_active_hold(1, now) == slot_id
    assert repo.get_active_hold(2, now) is None


def test_put_hold_replaces_existing_hold_for_order(world, repo, now):
    repo.put_hold(world["slots"][(0, 12)], 1, now + timedelta(minutes=5))
    repo.put_hold(world["slots"][(0, 13)], 1, now + timedelta(minutes=5))
    assert repo.get_active_hold(1, now) == world["slots"][(0, 13)]
    assert repo.active_holds(world["slots"][(0, 12)], now) == 0


def test_booked_cannot_exceed_capacity(world, repo):
    with pytest.raises(sqlite3.IntegrityError):
        repo.increment_booked(world["slots"][(0, 17)])
