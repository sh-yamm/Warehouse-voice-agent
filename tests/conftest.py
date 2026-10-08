from datetime import datetime, timedelta

import pytest

from voiceagent.db.repository import Repository

NOW = datetime(2026, 10, 12, 10, 0)  # Monday


@pytest.fixture
def now():
    return NOW


@pytest.fixture
def repo():
    r = Repository(":memory:")
    r.init_schema()
    yield r
    r.close()


@pytest.fixture
def world(repo):
    return build_world(repo, NOW)


def build_world(repo, now):
    """Small fixed world: 1 warehouse, 6 products, 2 orders, hourly slots today and tomorrow."""
    repo.insert_warehouse(1, "Koramangala Hub", "south")
    for p in [
        ("MILK1", "Amul Taaza Milk 1L", "Amul milk", "dairy", 56.0),
        ("MILK2", "Nandini Toned Milk 1L", "Nandini milk", "dairy", 48.0),
        ("BREAD1", "Brown Bread 400g", "brown bread", "bakery", 45.0),
        ("BREAD2", "Whole Wheat Bread 400g", "whole wheat bread", "bakery", 50.0),
        ("EGGS12", "Farm Eggs 12 pack", "eggs", "eggs", 90.0),
        ("RICE5", "Basmati Rice 5kg", "basmati rice", "staples", 520.0),
    ]:
        repo.insert_product(*p)
    bread_eta = (now + timedelta(days=1)).replace(hour=9, minute=0)
    for sku, on_hand, eta in [
        ("MILK1", 10, None), ("MILK2", 5, None), ("BREAD1", 0, bread_eta),
        ("BREAD2", 4, None), ("EGGS12", 20, None), ("RICE5", 3, None),
    ]:
        repo.insert_stock(1, sku, on_hand, eta)
    repo.insert_customer(1, "Priya Sharma", "+919800000001", "12 MG Road, Bengaluru")
    repo.insert_customer(2, "Arjun Rao", "+919800000002", "4 Church Street, Bengaluru")
    repo.insert_order(1, 1, 1, "12 MG Road, Bengaluru", now - timedelta(hours=2))
    repo.insert_order(2, 2, 1, "4 Church Street, Bengaluru", now - timedelta(hours=1))
    for order_id, sku, qty, reserved in [
        (1, "MILK1", 2, 2), (1, "BREAD1", 1, 0), (1, "EGGS12", 1, 1), (2, "RICE5", 1, 1),
    ]:
        repo.insert_order_line(order_id, sku, qty, reserved)
        if reserved:
            repo.reserve_stock(1, sku, reserved)
    slots = {}
    slot_id = 1
    for day in (0, 1):
        base = (now + timedelta(days=day)).replace(hour=0, minute=0, second=0, microsecond=0)
        for hour in range(8, 21):
            start = base + timedelta(hours=hour)
            booked = 2 if (day == 0 and hour == 17) else 0
            repo.insert_slot(slot_id, 1, start, start + timedelta(hours=1), capacity=2, booked=booked)
            slots[(day, hour)] = slot_id
            slot_id += 1
    return {"slots": slots}
