from __future__ import annotations

from datetime import datetime

from voiceagent.db.repository import Repository


class OrderService:
    def __init__(self, repo: Repository):
        self.repo = repo

    def cancel_order(self, order_id: int) -> bool:
        with self.repo.transaction():
            order = self.repo.get_order(order_id)
            if order is None or order.status == "cancelled":
                return False
            for line in self.repo.get_order_lines(order_id):
                if line.reserved_qty:
                    self.repo.release_stock(order.warehouse_id, line.sku, line.reserved_qty)
                if line.substitute_sku and line.substitute_qty:
                    self.repo.release_stock(order.warehouse_id, line.substitute_sku, line.substitute_qty)
                self.repo.update_order_line(order_id, line.sku, reserved_qty=0, substitute_qty=0)
            if order.status == "scheduled" and order.slot_id is not None:
                self.repo.decrement_booked(order.slot_id)
            self.repo.delete_hold(order_id)
            self.repo.update_order(order_id, status="cancelled", slot_id=None)
        return True

    def update_address(self, order_id: int, address: str) -> None:
        cleaned = " ".join(address.split())
        if len(cleaned) < 5:
            raise ValueError("address too short")
        self.repo.update_order(order_id, address=cleaned)

    def add_note(self, order_id: int, note: str) -> None:
        cleaned = " ".join(note.split())
        if not cleaned:
            raise ValueError("empty note")
        existing = self.repo.get_order(order_id).notes
        self.repo.update_order(order_id, notes=f"{existing}; {cleaned}" if existing else cleaned)

    def schedule_callback(self, order_id: int, when: datetime) -> None:
        self.repo.update_order(order_id, status="callback", callback_at=when)

    def mark_wrong_person(self, order_id: int) -> None:
        with self.repo.transaction():
            self.add_note(order_id, "wrong person answered")
            self.repo.update_order(order_id, status="callback", callback_at=None)
