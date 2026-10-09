from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from voiceagent.db.repository import Repository
from voiceagent.domain.inventory import InventoryService
from voiceagent.domain.models import Slot
from voiceagent.domain.scheduling import SchedulingService, SlotOption


def spoken_time(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    suffix = "AM" if dt.hour < 12 else "PM"
    return f"{hour} {suffix}" if dt.minute == 0 else f"{hour}:{dt.minute:02d} {suffix}"


def spoken_day(day: date, now: datetime) -> str:
    delta = (day - now.date()).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    return day.strftime("%A")


def spoken_slot(slot: Slot, now: datetime) -> str:
    start, end = spoken_time(slot.start), spoken_time(slot.end)
    if (slot.start.hour < 12) == (slot.end.hour < 12):
        start = start.rsplit(" ", 1)[0]
    return f"{spoken_day(slot.start.date(), now)}, {start} to {end}"


@dataclass(frozen=True)
class CallContext:
    order_id: int
    customer_name: str
    prompt_text: str
    slot_options: list[SlotOption]


def build_call_context(repo: Repository, inventory: InventoryService, scheduling: SchedulingService,
                       order_id: int, now: datetime) -> CallContext:
    order = repo.get_order(order_id)
    if order is None:
        raise ValueError(f"unknown order {order_id}")
    shortages = {s.sku: s for s in inventory.shortages(order_id)}
    item_lines = []
    for line in repo.get_order_lines(order_id):
        if line.sku in shortages:
            s = shortages[line.sku]
            status = f"SHORT, only {s.reserved} of {s.wanted} available"
            if s.restock_eta:
                status += f", restock {spoken_day(s.restock_eta.date(), now)} {spoken_time(s.restock_eta)}"
            subs = ", ".join(f"{m.spoken_name} [{m.sku}]" for m in s.substitutes) or "none"
            status += f"; substitutes: {subs}"
        elif line.resolution == "substitute":
            status = f"substituted with [{line.substitute_sku}]"
        elif line.resolution == "partial":
            status = f"partial, sending {line.reserved_qty}"
        elif line.resolution == "wait":
            status = "waiting for restock"
        else:
            status = "ready"
        item_lines.append(f"- {line.qty} x {line.name} [{line.sku}]: {status}")

    options = scheduling.free_slots(order_id, now)
    slot_lines = [f"- slot {o.slot.id}: {spoken_slot(o.slot, now)}" for o in options] or ["- none in the next 7 days"]
    text = "\n".join([
        f"CUSTOMER: {order.customer_name}",
        f"DELIVERY ADDRESS: {order.address}",
        f"NOTES: {order.notes or 'none'}",
        f"ORDER #{order.id}:",
        *item_lines,
        "NEXT FREE SLOTS:",
        *slot_lines,
    ])
    return CallContext(order.id, order.customer_name, text, options)
