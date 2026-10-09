from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from voiceagent.db.repository import Repository
from voiceagent.domain.inventory import InventoryService
from voiceagent.domain.models import Order, Slot, TimeWindow


@dataclass(frozen=True)
class SlotOption:
    slot: Slot
    available: int


@dataclass(frozen=True)
class SlotCheck:
    status: Literal["available", "full", "no_slots"]
    option: SlotOption | None
    alternatives: list[SlotOption]


class SchedulingService:
    HOLD_TTL = timedelta(minutes=5)
    HORIZON = timedelta(days=7)

    def __init__(self, repo: Repository, inventory: InventoryService):
        self.repo = repo
        self.inventory = inventory

    def _earliest(self, order_id: int, now: datetime) -> datetime:
        eta = self.inventory.earliest_delivery(order_id)
        return max(now, eta) if eta else now

    def _option(self, slot: Slot, order_id: int, now: datetime) -> SlotOption:
        held = self.repo.active_holds(slot.id, now, exclude_order_id=order_id)
        return SlotOption(slot, max(slot.capacity - slot.booked - held, 0))

    def _open_options(self, order: Order, now: datetime, start: datetime, end: datetime) -> list[SlotOption]:
        start = max(start, self._earliest(order.id, now))
        options = [self._option(s, order.id, now) for s in self.repo.list_slots(order.warehouse_id, start, end)]
        return [o for o in options if o.available > 0]

    def free_slots(self, order_id: int, now: datetime, limit: int = 6) -> list[SlotOption]:
        order = self.repo.get_order(order_id)
        if order is None:
            return []
        return self._open_options(order, now, now, now + self.HORIZON)[:limit]

    def check_slot(self, order_id: int, window: TimeWindow, now: datetime) -> SlotCheck:
        order = self.repo.get_order(order_id)
        if order is None:
            return SlotCheck("no_slots", None, [])
        in_window = self._open_options(order, now, window.start, window.end)
        if in_window:
            return SlotCheck("available", in_window[0], in_window[1:3])
        any_in_window = self.repo.list_slots(order.warehouse_id, max(window.start, self._earliest(order_id, now)),
                                             window.end)
        candidates = self._open_options(order, now, now, now + self.HORIZON)
        candidates.sort(key=lambda o: abs((o.slot.start - window.start).total_seconds()))
        alternatives = sorted(candidates[:3], key=lambda o: o.slot.start)
        return SlotCheck("full" if any_in_window else "no_slots", None, alternatives)

    def hold_slot(self, order_id: int, slot_id: int, now: datetime) -> bool:
        order, slot = self.repo.get_order(order_id), self.repo.get_slot(slot_id)
        if order is None or slot is None or slot.warehouse_id != order.warehouse_id or not order.is_open:
            return False
        with self.repo.transaction():
            if slot.start < self._earliest(order_id, now) or slot.end <= now:
                return False
            if self._option(self.repo.get_slot(slot_id), order_id, now).available <= 0:
                return False
            self.repo.put_hold(slot_id, order_id, now + self.HOLD_TTL)
        return True

    def confirm_booking(self, order_id: int, now: datetime) -> Slot | None:
        with self.repo.transaction():
            slot_id = self.repo.get_active_hold(order_id, now)
            order = self.repo.get_order(order_id)
            if slot_id is None or order is None or not order.is_open:
                return None
            self.repo.increment_booked(slot_id)
            self.repo.delete_hold(order_id)
            self.repo.update_order(order_id, status="scheduled", slot_id=slot_id, callback_at=None)
        return self.repo.get_slot(slot_id)
