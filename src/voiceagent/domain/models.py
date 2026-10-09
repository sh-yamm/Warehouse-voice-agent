from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

DELIVERY_DAY_START_HOUR = 8
DELIVERY_DAY_END_HOUR = 21
OPEN_ORDER_STATUSES = frozenset({"pending_schedule", "callback"})


@dataclass(frozen=True)
class TimeWindow:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    spoken_name: str
    category: str
    unit_price: float


@dataclass(frozen=True)
class Slot:
    id: int
    warehouse_id: int
    start: datetime
    end: datetime
    capacity: int
    booked: int


@dataclass(frozen=True)
class OrderLine:
    order_id: int
    sku: str
    name: str
    qty: int
    reserved_qty: int
    substitute_sku: str | None
    substitute_qty: int
    resolution: str | None  # None | "partial" | "substitute" | "wait"


@dataclass(frozen=True)
class Order:
    id: int
    customer_id: int
    customer_name: str
    phone: str
    warehouse_id: int
    status: str  # pending_schedule | scheduled | cancelled | callback | needs_human
    address: str
    notes: str
    slot_id: int | None
    callback_at: datetime | None

    @property
    def is_open(self) -> bool:
        """Open orders can still be changed or scheduled; scheduled and cancelled ones cannot."""
        return self.status in OPEN_ORDER_STATUSES
