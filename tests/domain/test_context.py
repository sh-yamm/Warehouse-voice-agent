from datetime import datetime

import pytest

from voiceagent.domain.context import build_call_context, spoken_slot, spoken_time
from voiceagent.domain.inventory import InventoryService
from voiceagent.domain.models import Slot
from voiceagent.domain.scheduling import SchedulingService


@pytest.fixture
def services(world, repo):
    inv = InventoryService(repo)
    return inv, SchedulingService(repo, inv)


def test_spoken_time():
    assert spoken_time(datetime(2026, 10, 12, 17, 0)) == "5 PM"
    assert spoken_time(datetime(2026, 10, 12, 17, 30)) == "5:30 PM"
    assert spoken_time(datetime(2026, 10, 12, 12, 0)) == "12 PM"
    assert spoken_time(datetime(2026, 10, 12, 9, 5)) == "9:05 AM"


def test_spoken_slot(now):
    slot = Slot(1, 1, datetime(2026, 10, 13, 17), datetime(2026, 10, 13, 18), 2, 0)
    assert spoken_slot(slot, now) == "tomorrow, 5 to 6 PM"
    slot = Slot(1, 1, datetime(2026, 10, 12, 11), datetime(2026, 10, 12, 12), 2, 0)
    assert spoken_slot(slot, now) == "today, 11 AM to 12 PM"
    slot = Slot(1, 1, datetime(2026, 10, 15, 9), datetime(2026, 10, 15, 10), 2, 0)
    assert spoken_slot(slot, now) == "Thursday, 9 to 10 AM"


def test_context_lists_items_shortage_and_slots(repo, services, now):
    inv, sched = services
    ctx = build_call_context(repo, inv, sched, 1, now)
    text = ctx.prompt_text
    assert ctx.customer_name == "Priya Sharma"
    assert "CUSTOMER: Priya Sharma" in text
    assert "DELIVERY ADDRESS: 12 MG Road, Bengaluru" in text
    assert "- 2 x Amul milk [MILK1]: ready" in text
    assert ("- 1 x brown bread [BREAD1]: SHORT, only 0 of 1 available, restock tomorrow 9 AM; "
            "substitutes: whole wheat bread [BREAD2]") in text
    assert "NEXT FREE SLOTS:\n- slot 3: today, 10 to 11 AM" in text
    assert len(ctx.slot_options) == 6


def test_context_reflects_resolutions(repo, services, now):
    inv, sched = services
    inv.apply_substitution(1, "BREAD1", "BREAD2")
    text = build_call_context(repo, inv, sched, 1, now).prompt_text
    assert "- 1 x brown bread [BREAD1]: substituted with [BREAD2]" in text


def test_context_is_deterministic(repo, services, now):
    inv, sched = services
    a = build_call_context(repo, inv, sched, 1, now).prompt_text
    b = build_call_context(repo, inv, sched, 1, now).prompt_text
    assert a == b


def test_context_unknown_order(repo, services, now):
    inv, sched = services
    with pytest.raises(ValueError):
        build_call_context(repo, inv, sched, 999, now)
