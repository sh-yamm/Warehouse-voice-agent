from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from voiceagent.domain.models import Order, OrderLine, Product, Slot

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_ORDER_FIELDS = {"status", "address", "notes", "slot_id", "callback_at"}
_LINE_FIELDS = {"qty", "reserved_qty", "substitute_sku", "substitute_qty", "resolution"}


def _ts(value):
    return value.isoformat(timespec="seconds") if isinstance(value, datetime) else value


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Repository:
    """All SQL lives here. Services call these methods inside `transaction()`."""

    def __init__(self, path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._depth = 0
        self._lock = threading.RLock()  # one connection is shared across threads; serialize transactions

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    def close(self) -> None:
        self.conn.close()

    def dump(self) -> str:
        return "\n".join(self.conn.iterdump())

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            yield from self._transaction()

    def _transaction(self) -> Iterator[None]:
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self.conn.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")
        finally:
            self._depth = 0

    # ── inserts ────────────────────────────────────────────────────────────
    def insert_warehouse(self, id: int, name: str, zone: str) -> None:
        self.conn.execute("INSERT INTO warehouses (id, name, zone) VALUES (?, ?, ?)", (id, name, zone))

    def insert_product(self, sku: str, name: str, spoken_name: str, category: str, unit_price: float) -> None:
        self.conn.execute(
            "INSERT INTO products (sku, name, spoken_name, category, unit_price) VALUES (?, ?, ?, ?, ?)",
            (sku, name, spoken_name, category, unit_price),
        )

    def insert_stock(self, warehouse_id: int, sku: str, on_hand: int, restock_eta: datetime | None = None) -> None:
        self.conn.execute(
            "INSERT INTO stock (warehouse_id, sku, on_hand, reserved, restock_eta) VALUES (?, ?, ?, 0, ?)",
            (warehouse_id, sku, on_hand, _ts(restock_eta)),
        )

    def insert_customer(self, id: int, name: str, phone: str, default_address: str) -> None:
        self.conn.execute(
            "INSERT INTO customers (id, name, phone, default_address) VALUES (?, ?, ?, ?)",
            (id, name, phone, default_address),
        )

    def insert_order(self, id: int, customer_id: int, warehouse_id: int, address: str, created_at: datetime) -> None:
        self.conn.execute(
            "INSERT INTO orders (id, customer_id, warehouse_id, address, created_at) VALUES (?, ?, ?, ?, ?)",
            (id, customer_id, warehouse_id, address, _ts(created_at)),
        )

    def insert_order_line(self, order_id: int, sku: str, qty: int, reserved_qty: int) -> None:
        self.conn.execute(
            "INSERT INTO order_lines (order_id, sku, qty, reserved_qty) VALUES (?, ?, ?, ?)",
            (order_id, sku, qty, reserved_qty),
        )

    def insert_slot(self, id: int, warehouse_id: int, start: datetime, end: datetime,
                    capacity: int, booked: int = 0) -> None:
        self.conn.execute(
            "INSERT INTO delivery_slots (id, warehouse_id, start_ts, end_ts, capacity, booked)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (id, warehouse_id, _ts(start), _ts(end), capacity, booked),
        )

    # ── reads ──────────────────────────────────────────────────────────────
    def get_order(self, order_id: int) -> Order | None:
        row = self.conn.execute(
            "SELECT o.*, c.name AS customer_name, c.phone FROM orders o"
            " JOIN customers c ON c.id = o.customer_id WHERE o.id = ?",
            (order_id,),
        ).fetchone()
        if row is None:
            return None
        return Order(
            id=row["id"], customer_id=row["customer_id"], customer_name=row["customer_name"],
            phone=row["phone"], warehouse_id=row["warehouse_id"], status=row["status"],
            address=row["address"], notes=row["notes"], slot_id=row["slot_id"],
            callback_at=_dt(row["callback_at"]),
        )

    def get_order_lines(self, order_id: int) -> list[OrderLine]:
        rows = self.conn.execute(
            "SELECT l.*, p.spoken_name FROM order_lines l JOIN products p ON p.sku = l.sku"
            " WHERE l.order_id = ? ORDER BY l.rowid",
            (order_id,),
        ).fetchall()
        return [
            OrderLine(
                order_id=r["order_id"], sku=r["sku"], name=r["spoken_name"], qty=r["qty"],
                reserved_qty=r["reserved_qty"], substitute_sku=r["substitute_sku"],
                substitute_qty=r["substitute_qty"], resolution=r["resolution"],
            )
            for r in rows
        ]

    def get_product(self, sku: str) -> Product | None:
        row = self.conn.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
        return Product(**dict(row)) if row else None

    def list_products(self) -> list[Product]:
        return [Product(**dict(r)) for r in self.conn.execute("SELECT * FROM products ORDER BY sku")]

    def available_stock(self, warehouse_id: int, sku: str) -> int:
        row = self.conn.execute(
            "SELECT on_hand - reserved AS available FROM stock WHERE warehouse_id = ? AND sku = ?",
            (warehouse_id, sku),
        ).fetchone()
        return row["available"] if row else 0

    def restock_eta(self, warehouse_id: int, sku: str) -> datetime | None:
        row = self.conn.execute(
            "SELECT restock_eta FROM stock WHERE warehouse_id = ? AND sku = ?", (warehouse_id, sku)
        ).fetchone()
        return _dt(row["restock_eta"]) if row else None

    @staticmethod
    def _slot(row: sqlite3.Row) -> Slot:
        return Slot(
            id=row["id"], warehouse_id=row["warehouse_id"], start=_dt(row["start_ts"]),
            end=_dt(row["end_ts"]), capacity=row["capacity"], booked=row["booked"],
        )

    def get_slot(self, slot_id: int) -> Slot | None:
        row = self.conn.execute("SELECT * FROM delivery_slots WHERE id = ?", (slot_id,)).fetchone()
        return self._slot(row) if row else None

    def list_slots(self, warehouse_id: int, start: datetime, end: datetime) -> list[Slot]:
        rows = self.conn.execute(
            "SELECT * FROM delivery_slots WHERE warehouse_id = ? AND start_ts >= ? AND start_ts < ?"
            " ORDER BY start_ts",
            (warehouse_id, _ts(start), _ts(end)),
        ).fetchall()
        return [self._slot(r) for r in rows]

    def active_holds(self, slot_id: int, now: datetime, exclude_order_id: int | None = None) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM slot_holds WHERE slot_id = ? AND expires_at > ? AND order_id IS NOT ?",
            (slot_id, _ts(now), exclude_order_id),
        ).fetchone()
        return row["n"]

    def get_active_hold(self, order_id: int, now: datetime) -> int | None:
        row = self.conn.execute(
            "SELECT slot_id FROM slot_holds WHERE order_id = ? AND expires_at > ?", (order_id, _ts(now))
        ).fetchone()
        return row["slot_id"] if row else None

    # ── writes ─────────────────────────────────────────────────────────────
    def _adjust_reserved(self, warehouse_id: int, sku: str, delta: int) -> None:
        cur = self.conn.execute(
            "UPDATE stock SET reserved = reserved + ? WHERE warehouse_id = ? AND sku = ?",
            (delta, warehouse_id, sku),
        )
        if cur.rowcount != 1:
            raise ValueError(f"no stock row for warehouse {warehouse_id} sku {sku}")

    def reserve_stock(self, warehouse_id: int, sku: str, qty: int) -> None:
        self._adjust_reserved(warehouse_id, sku, qty)

    def release_stock(self, warehouse_id: int, sku: str, qty: int) -> None:
        self._adjust_reserved(warehouse_id, sku, -qty)

    def _update(self, table: str, allowed: set[str], where: str, args: tuple, fields: dict) -> None:
        unknown = set(fields) - allowed
        if unknown or not fields:
            raise ValueError(f"invalid fields for {table}: {sorted(unknown) or 'none given'}")
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE {table} SET {assignments} WHERE {where}",
            tuple(_ts(v) for v in fields.values()) + args,
        )

    def update_order(self, order_id: int, **fields) -> None:
        self._update("orders", _ORDER_FIELDS, "id = ?", (order_id,), fields)

    def update_order_line(self, order_id: int, sku: str, **fields) -> None:
        self._update("order_lines", _LINE_FIELDS, "order_id = ? AND sku = ?", (order_id, sku), fields)

    def put_hold(self, slot_id: int, order_id: int, expires_at: datetime) -> None:
        self.conn.execute(
            "INSERT INTO slot_holds (slot_id, order_id, expires_at) VALUES (?, ?, ?)"
            " ON CONFLICT(order_id) DO UPDATE SET slot_id = excluded.slot_id, expires_at = excluded.expires_at",
            (slot_id, order_id, _ts(expires_at)),
        )

    def delete_hold(self, order_id: int) -> None:
        self.conn.execute("DELETE FROM slot_holds WHERE order_id = ?", (order_id,))

    def increment_booked(self, slot_id: int) -> None:
        self.conn.execute("UPDATE delivery_slots SET booked = booked + 1 WHERE id = ?", (slot_id,))

    def decrement_booked(self, slot_id: int) -> None:
        self.conn.execute("UPDATE delivery_slots SET booked = booked - 1 WHERE id = ? AND booked > 0", (slot_id,))

    # ── calls ──────────────────────────────────────────────────────────────
    def start_call(self, order_id: int, started_at: datetime) -> int:
        cur = self.conn.execute("INSERT INTO calls (order_id, started_at) VALUES (?, ?)",
                                (order_id, _ts(started_at)))
        return cur.lastrowid

    def add_turn_metric(self, call_id: int, kind: str, total_ms: float, breakdown: dict,
                        recorded_at: datetime) -> None:
        self.conn.execute(
            "INSERT INTO turn_metrics (call_id, kind, total_ms, breakdown_json, recorded_at) VALUES (?, ?, ?, ?, ?)",
            (call_id, kind, round(float(total_ms), 1), json.dumps(breakdown), _ts(recorded_at)),
        )

    def finish_call(self, call_id: int, ended_at: datetime, outcome: str, transcript: list[dict]) -> None:
        self.conn.execute("UPDATE calls SET ended_at = ?, outcome = ?, transcript_json = ? WHERE id = ?",
                          (_ts(ended_at), outcome, json.dumps(transcript), call_id))

    _CALL_SELECT = ("SELECT c.id, c.order_id, cu.name AS customer, c.started_at, c.ended_at, c.outcome,"
                    " c.transcript_json FROM calls c JOIN orders o ON o.id = c.order_id"
                    " JOIN customers cu ON cu.id = o.customer_id")

    @staticmethod
    def _call(row: sqlite3.Row, with_transcript: bool) -> dict:
        call = {"id": row["id"], "order_id": row["order_id"], "customer": row["customer"],
                "started_at": _dt(row["started_at"]), "ended_at": _dt(row["ended_at"]), "outcome": row["outcome"]}
        if with_transcript:
            call["transcript"] = json.loads(row["transcript_json"])
        return call

    def get_call(self, call_id: int) -> dict | None:
        row = self.conn.execute(self._CALL_SELECT + " WHERE c.id = ?", (call_id,)).fetchone()
        return self._call(row, True) if row else None

    def list_calls(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(self._CALL_SELECT + " ORDER BY c.id DESC LIMIT ?", (limit,)).fetchall()
        return [self._call(r, False) for r in rows]

    def call_metrics(self, call_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT kind, total_ms, breakdown_json, recorded_at FROM turn_metrics WHERE call_id = ? ORDER BY id",
            (call_id,),
        ).fetchall()
        return [{"kind": r["kind"], "total_ms": r["total_ms"], "breakdown": json.loads(r["breakdown_json"]),
                 "recorded_at": _dt(r["recorded_at"])} for r in rows]
