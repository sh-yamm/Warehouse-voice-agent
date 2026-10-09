from __future__ import annotations

import functools
from collections.abc import Callable
from datetime import datetime, timedelta

from loguru import logger

from voiceagent.db.repository import Repository
from voiceagent.domain.context import CallContext, build_call_context, spoken_day, spoken_slot, spoken_time
from voiceagent.domain.inventory import InventoryService
from voiceagent.domain.orders import OrderService
from voiceagent.domain.scheduling import SchedulingService, SlotOption
from voiceagent.domain.timeparse import parse_time_window

SHORTAGE_CHOICES = ("substitute", "partial", "wait")


def _logged(method):
    """Log every tool call with its arguments and result (the call's audit trail)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        logger.info(f"tool {method.__name__} args={list(args)} kwargs={kwargs} -> {result}")
        return result

    return wrapper


class CallSession:
    """Everything one call may do, as small tools that return JSON-serializable dicts.

    The LLM only chooses which tool to call and what to say; all rules live in the Plan 1 services.
    """

    def __init__(self, repo: Repository, order_id: int, clock: Callable[[], datetime] = datetime.now):
        self.repo = repo
        self.order_id = order_id
        self.clock = clock
        self.inventory = InventoryService(repo)
        self.scheduling = SchedulingService(repo, self.inventory)
        self.orders = OrderService(repo)
        self.outcome: str | None = None
        self.held_slot_id: int | None = None

    def context(self) -> CallContext:
        return build_call_context(self.repo, self.inventory, self.scheduling, self.order_id, self.clock())

    def _option(self, option: SlotOption) -> dict:
        return {"slot_id": option.slot.id, "slot": spoken_slot(option.slot, self.clock())}

    def _alternatives(self, options: list[SlotOption]) -> list[dict]:
        return [self._option(o) for o in options]

    # ── scheduling ─────────────────────────────────────────────────────────
    @_logged
    def check_slot(self, preferred_time: str) -> dict:
        now = self.clock()
        window = parse_time_window(preferred_time, now)
        if window is None:
            return {"status": "unclear", "hint": "Ask for a day and a time between 8 AM and 9 PM."}
        check = self.scheduling.check_slot(self.order_id, window, now)
        status = check.status
        if status == "available":
            if self.scheduling.hold_slot(self.order_id, check.option.slot.id, now):
                self.held_slot_id = check.option.slot.id
                return {"status": "held", **self._option(check.option)}
            status = "full"
        return {"status": status, "alternatives": self._alternatives(check.alternatives)}

    @_logged
    def choose_slot(self, slot_id: int) -> dict:
        now = self.clock()
        slot = self.repo.get_slot(slot_id)
        if slot is None or not self.scheduling.hold_slot(self.order_id, slot_id, now):
            return {"status": "unavailable",
                    "alternatives": self._alternatives(self.scheduling.free_slots(self.order_id, now, limit=3))}
        self.held_slot_id = slot_id
        return {"status": "held", "slot_id": slot_id, "slot": spoken_slot(slot, now)}

    @_logged
    def confirm_booking(self) -> dict:
        slot = self.scheduling.confirm_booking(self.order_id, self.clock())
        if slot is None:
            return {"status": "failed", "hint": "The slot hold expired or no slot was chosen. Check the time again."}
        self.outcome = "scheduled"
        self.held_slot_id = None
        return {"status": "booked", "slot": spoken_slot(slot, self.clock())}

    # ── items ──────────────────────────────────────────────────────────────
    @_logged
    def add_item(self, product_name: str, quantity) -> dict:
        try:
            qty = int(quantity)
        except (TypeError, ValueError):
            return {"status": "error", "message": "quantity must be a whole number"}
        order = self.repo.get_order(self.order_id)
        matches = self.inventory.find_products(order.warehouse_id, product_name)
        if not matches:
            return {"status": "not_found"}
        best = matches[0]
        try:
            result = self.inventory.add_item(self.order_id, best.sku, qty)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": result.status, "item": best.spoken_name, "quantity": qty,
                "available": result.available, "unit_price": best.unit_price}

    @_logged
    def resolve_shortage(self, sku: str, choice: str, substitute_sku: str = "") -> dict:
        now = self.clock()
        deliver_after = None
        if choice == "substitute":
            ok = self.inventory.apply_substitution(self.order_id, sku, substitute_sku)
        elif choice == "partial":
            ok = self.inventory.set_partial(self.order_id, sku)
        elif choice == "wait":
            eta = self.inventory.wait_restock(self.order_id, sku)
            ok = eta is not None
            if eta is not None:
                deliver_after = f"{spoken_day(eta.date(), now)} {spoken_time(eta)}"
        else:
            return {"status": "error", "message": f"choice must be one of {', '.join(SHORTAGE_CHOICES)}"}
        result = {"status": "done" if ok else "failed",
                  "remaining_shortages": [s.sku for s in self.inventory.shortages(self.order_id)]}
        if deliver_after:
            result["deliver_after"] = deliver_after
        return result

    # ── order details ──────────────────────────────────────────────────────
    @_logged
    def update_address(self, address: str) -> dict:
        try:
            self.orders.update_address(self.order_id, address)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "updated", "address": self.repo.get_order(self.order_id).address}

    @_logged
    def add_note(self, note: str) -> dict:
        try:
            self.orders.add_note(self.order_id, note)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "saved"}

    @_logged
    def cancel_order(self) -> dict:
        ok = self.orders.cancel_order(self.order_id)
        if ok:
            self.outcome = "cancelled"
            self.held_slot_id = None
        return {"status": "cancelled" if ok else "failed"}

    @_logged
    def schedule_callback(self, when: str) -> dict:
        now = self.clock()
        window = parse_time_window(when, now) if when and when.strip() else None
        at = window.start if window else now + timedelta(hours=1)
        self.orders.schedule_callback(self.order_id, at)
        self.outcome = "callback"
        return {"status": "scheduled", "callback_at": f"{spoken_day(at.date(), now)} {spoken_time(at)}"}

    @_logged
    def wrong_person(self) -> dict:
        self.orders.mark_wrong_person(self.order_id)
        self.outcome = "wrong_person"
        return {"status": "noted"}

    def order_summary(self) -> dict:
        order = self.repo.get_order(self.order_id)
        items = []
        for line in self.repo.get_order_lines(self.order_id):
            if line.resolution == "substitute":
                substitute = self.repo.get_product(line.substitute_sku).spoken_name
                if line.reserved_qty:
                    items.append(f"{line.reserved_qty} {line.name} and {line.substitute_qty} {substitute}")
                else:
                    items.append(f"{line.substitute_qty} {substitute} instead of {line.name}")
            elif line.resolution == "partial":
                items.append(f"{line.reserved_qty} of {line.qty} {line.name}")
            elif line.resolution == "wait":
                items.append(f"{line.qty} {line.name}, after restock")
            elif line.reserved_qty < line.qty:
                items.append(f"{line.qty} {line.name}, not available yet")
            else:
                items.append(f"{line.qty} {line.name}")
        slot = self.repo.get_slot(self.held_slot_id) if self.held_slot_id else None
        return {"items": items, "address": order.address, "notes": order.notes,
                "slot": spoken_slot(slot, self.clock()) if slot else None}
