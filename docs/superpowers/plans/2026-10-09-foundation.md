# Foundation (Domain Layer + Model Benchmarks) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the database and deterministic domain layer the voice agent will call, plus a benchmark harness that measures STT, end-of-turn, TTS and LLM latency on the target laptop and picks the STT and LLM models.

**Architecture:** A `voiceagent` Python package (src layout) holds a SQLite `Repository` (the only module with SQL) and pure-Python services (time parsing, inventory, scheduling, orders, pre-call context) with all clocks injected. A separate `bench` package runs each candidate model in isolation, records per-utterance latency and accuracy to JSON, and a report script turns results into a markdown table.

**Tech Stack:** Python 3.11, sqlite3, pytest; bench: torch (CUDA), transformers ≥ 5.13 (Nemotron streaming), sherpa-onnx (Parakeet TDT v2 int8), faster-whisper, onnxruntime-gpu, kokoro-onnx, llama.cpp `llama-server` (CUDA 12.4 Windows build), httpx.

**Spec:** `docs/specs/2026-10-09-voice-caller-agent-design.md`

## Global Constraints

- Python `>=3.11,<3.13` (kokoro-onnx requires `<3.14`; use the installed Python 3.11 via `py -3.11`).
- Zero spend: open-weight models only, all inference local on the RTX 4060 Laptop (8 GB VRAM).
- Delivery hours are 08:00–21:00 local time; all datetimes are naive local time stored as ISO strings `YYYY-MM-DDTHH:MM:SS`.
- The `Repository` is the only module that contains SQL.
- Stock and slot mutations run inside `Repository.transaction()`.
- Slots are held for 5 minutes when offered and committed only by `confirm_booking`.
- The agent never states an item, quantity or slot that is not in the pre-call context or a tool result (the context builder is the single source of those facts).
- Model weights, generated audio, databases and `tools/` are git-ignored; benchmark result JSON files are committed.
- Commit messages contain no co-author or tool-attribution trailers.
- All commands are run from the repo root (`D:\aviostack\intern-work`) in Git Bash; the venv Python is `.venv/Scripts/python`.

## Review Focus

1. STT output forms of times ("5 p.m.", "five thirty", "5:30 PM", "noon") must parse to the same window — pinned by the parametrized tests in Task 4.
2. Two calls racing for the last seat of a slot must not both get a hold — pinned by `test_hold_respects_other_orders_holds` in Task 6.
3. A hold that expires mid-call must make `confirm_booking` fail rather than overbook — pinned by `test_confirm_after_expiry_fails` in Task 6.
4. Cancelling an already scheduled order must free the slot seat and every stock reservation, including substitutes — pinned by `test_cancel_scheduled_order_frees_slot_and_substitute_stock` in Task 7.
5. A benchmark that silently falls back from CUDA to CPU would produce misleading latency — every bench result records the actual execution provider/device, asserted in Task 9's `test_write_result_includes_device`.

---

## File Structure

```
pyproject.toml                         package + optional deps (dev, bench)
src/voiceagent/__init__.py             version
src/voiceagent/domain/models.py        dataclasses + delivery-hour constants
src/voiceagent/db/schema.sql           SQLite schema
src/voiceagent/db/repository.py        Repository: connection, transactions, all SQL
src/voiceagent/db/seed.py              deterministic synthetic world + CLI
src/voiceagent/domain/timeparse.py     parse_time_window(phrase, now)
src/voiceagent/domain/inventory.py     InventoryService
src/voiceagent/domain/scheduling.py    SchedulingService
src/voiceagent/domain/orders.py        OrderService
src/voiceagent/domain/context.py       spoken formatting + build_call_context
tests/conftest.py                      repo + small fixed world fixtures
tests/domain/test_*.py                 unit tests per module
tests/bench/test_common.py             bench helper tests
bench/__init__.py
bench/common.py                        paths, percentiles, WER, GPU sampler, result writer, audio IO
bench/corpus/phrases.json              65 customer utterances (50 complete, 15 incomplete)
bench/tts_engine.py                    Kokoro loader (GPU if available)
bench/make_corpus.py                   synthesize corpus WAVs + manifest
bench/record_corpus.py                 record your own voice for the corpus
bench/stt_bench.py                     STT engines + benchmark
bench/turn_bench.py                    Smart Turn v3.2 benchmark
bench/tts_bench.py                     Kokoro benchmark
bench/llm_bench.py                     llama-server streaming benchmark
bench/report.py                        results/*.json -> docs/benchmarks/phase1-results.md
bench/results/*.json                   committed results
scripts/download_models.py             fetch model files into models/
scripts/get_llama_cpp.sh               fetch llama.cpp CUDA build into tools/
scripts/llama_server.sh                start llama-server for a GGUF
docs/benchmarks/phase1-decision.md     model choices with numbers
```

---

### Task 1: Project scaffold and environment

**Files:**
- Create: `pyproject.toml`, `src/voiceagent/__init__.py`, `tests/test_smoke.py`, `bench/__init__.py`
- Modify: `.gitignore`, `README.md`

**Interfaces:**
- Produces: importable package `voiceagent` with `voiceagent.__version__ == "0.1.0"`; importable package `bench`.

- [ ] **Step 1: Write the failing test**

`tests/test_smoke.py`:
```python
import voiceagent


def test_version():
    assert voiceagent.__version__ == "0.1.0"
```

- [ ] **Step 2: Create the venv and confirm the test fails**

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip pytest
.venv/Scripts/python -m pytest tests/test_smoke.py -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent'`.

- [ ] **Step 3: Write the scaffold**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "warehouse-voice-agent"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8"]
bench = [
  "numpy>=1.26",
  "scipy>=1.11",
  "soundfile>=0.12",
  "sounddevice>=0.4.6",
  "httpx>=0.27",
  "huggingface_hub>=0.30",
  "transformers>=5.13",
  "faster-whisper>=1.2",
  "sherpa-onnx>=1.12",
  "kokoro-onnx>=0.6",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "."]
```

`src/voiceagent/__init__.py`:
```python
__version__ = "0.1.0"
```

`bench/__init__.py`: empty file.

Append to `.gitignore`:
```
data/
tools/
bench/corpus/wav/
```

Replace the first line of `README.md` (`# warehouse-voice-agent`) with `# Warehouse-voice-agent`, and append:
```markdown

## Setup (Windows, Git Bash)

    py -3.11 -m venv .venv
    .venv/Scripts/python -m pip install -e ".[dev]"
    .venv/Scripts/python -m pytest
```

- [ ] **Step 4: Install and run the test**

```bash
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest tests/test_smoke.py -q
```
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/voiceagent/__init__.py bench/__init__.py tests/test_smoke.py .gitignore README.md
git commit -m "Add project scaffold and dev environment"
```

---

### Task 2: Domain models, schema and Repository

**Files:**
- Create: `src/voiceagent/domain/__init__.py` (empty), `src/voiceagent/domain/models.py`, `src/voiceagent/db/__init__.py` (empty), `src/voiceagent/db/schema.sql`, `src/voiceagent/db/repository.py`, `tests/conftest.py`, `tests/domain/test_repository.py` (test directories get no `__init__.py`; test file basenames are unique)

**Interfaces:**
- Produces (used by every later task):
  - `voiceagent.domain.models`: `DELIVERY_DAY_START_HOUR = 8`, `DELIVERY_DAY_END_HOUR = 21`, frozen dataclasses `TimeWindow(start: datetime, end: datetime)`, `Product(sku, name, spoken_name, category, unit_price)`, `Slot(id, warehouse_id, start, end, capacity, booked)`, `OrderLine(order_id, sku, name, qty, reserved_qty, substitute_sku, substitute_qty, resolution)`, `Order(id, customer_id, customer_name, phone, warehouse_id, status, address, notes, slot_id, callback_at)`.
  - `Repository(path=":memory:")` with `init_schema()`, `close()`, `transaction()` (context manager, nestable), `dump() -> str`; inserts `insert_warehouse(id, name, zone)`, `insert_product(sku, name, spoken_name, category, unit_price)`, `insert_stock(warehouse_id, sku, on_hand, restock_eta=None)`, `insert_customer(id, name, phone, default_address)`, `insert_order(id, customer_id, warehouse_id, address, created_at)`, `insert_order_line(order_id, sku, qty, reserved_qty)`, `insert_slot(id, warehouse_id, start, end, capacity, booked=0)`; reads `get_order(order_id) -> Order | None`, `get_order_lines(order_id) -> list[OrderLine]`, `get_product(sku) -> Product | None`, `list_products() -> list[Product]`, `available_stock(warehouse_id, sku) -> int`, `restock_eta(warehouse_id, sku) -> datetime | None`, `get_slot(slot_id) -> Slot | None`, `list_slots(warehouse_id, start, end) -> list[Slot]` (slots with `start <= slot.start < end`, ordered by start), `active_holds(slot_id, now, exclude_order_id=None) -> int`, `get_active_hold(order_id, now) -> int | None`; writes `reserve_stock(warehouse_id, sku, qty)`, `release_stock(warehouse_id, sku, qty)`, `update_order(order_id, **fields)`, `update_order_line(order_id, sku, **fields)`, `put_hold(slot_id, order_id, expires_at)`, `delete_hold(order_id)`, `increment_booked(slot_id)`, `decrement_booked(slot_id)`.
  - Test fixtures `repo`, `now` (`datetime(2026, 10, 12, 10, 0)`, a Monday) and `world` (returns `{"slots": {(day_offset, hour): slot_id}}`).

- [ ] **Step 1: Write the models, schema and fixtures (no logic yet)**

`src/voiceagent/domain/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

DELIVERY_DAY_START_HOUR = 8
DELIVERY_DAY_END_HOUR = 21


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
```

`src/voiceagent/db/schema.sql`:
```sql
CREATE TABLE IF NOT EXISTS warehouses (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    zone TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    spoken_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS stock (
    warehouse_id INTEGER NOT NULL REFERENCES warehouses(id),
    sku TEXT NOT NULL REFERENCES products(sku),
    on_hand INTEGER NOT NULL CHECK (on_hand >= 0),
    reserved INTEGER NOT NULL DEFAULT 0 CHECK (reserved >= 0 AND reserved <= on_hand),
    restock_eta TEXT,
    PRIMARY KEY (warehouse_id, sku)
);
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    default_address TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS delivery_slots (
    id INTEGER PRIMARY KEY,
    warehouse_id INTEGER NOT NULL REFERENCES warehouses(id),
    start_ts TEXT NOT NULL,
    end_ts TEXT NOT NULL,
    capacity INTEGER NOT NULL CHECK (capacity > 0),
    booked INTEGER NOT NULL DEFAULT 0 CHECK (booked >= 0 AND booked <= capacity)
);
CREATE INDEX IF NOT EXISTS idx_slots_wh_start ON delivery_slots(warehouse_id, start_ts);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    warehouse_id INTEGER NOT NULL REFERENCES warehouses(id),
    status TEXT NOT NULL DEFAULT 'pending_schedule',
    address TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    slot_id INTEGER REFERENCES delivery_slots(id),
    callback_at TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS order_lines (
    order_id INTEGER NOT NULL REFERENCES orders(id),
    sku TEXT NOT NULL REFERENCES products(sku),
    qty INTEGER NOT NULL CHECK (qty > 0),
    reserved_qty INTEGER NOT NULL CHECK (reserved_qty >= 0 AND reserved_qty <= qty),
    substitute_sku TEXT REFERENCES products(sku),
    substitute_qty INTEGER NOT NULL DEFAULT 0,
    resolution TEXT,
    PRIMARY KEY (order_id, sku)
);
CREATE TABLE IF NOT EXISTS slot_holds (
    slot_id INTEGER NOT NULL REFERENCES delivery_slots(id),
    order_id INTEGER NOT NULL UNIQUE REFERENCES orders(id),
    expires_at TEXT NOT NULL
);
```

`tests/conftest.py`:
```python
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
```

- [ ] **Step 2: Write the failing repository tests**

`tests/domain/test_repository.py`:
```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/domain/test_repository.py -q`
Expected: errors with `ModuleNotFoundError: No module named 'voiceagent.db.repository'`.

- [ ] **Step 4: Implement the Repository**

`src/voiceagent/db/repository.py`:
```python
from __future__ import annotations

import sqlite3
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

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    def close(self) -> None:
        self.conn.close()

    def dump(self) -> str:
        return "\n".join(self.conn.iterdump())

    @contextmanager
    def transaction(self) -> Iterator[None]:
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/domain/test_repository.py -q`
Expected: `14 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/voiceagent/domain src/voiceagent/db tests/conftest.py tests/domain
git commit -m "Add domain models, SQLite schema and repository"
```

---

### Task 3: Deterministic seed data

**Files:**
- Create: `src/voiceagent/db/seed.py`, `tests/domain/test_seed.py`

**Interfaces:**
- Consumes: `Repository` from Task 2.
- Produces: `generate_world(repo, now, seed=7, n_customers=200, n_orders=300, days=3) -> None`; CLI `python -m voiceagent.db.seed --db data/warehouse.db [--seed 7] [--now 2026-10-12T10:00]`.

- [ ] **Step 1: Write the failing tests**

`tests/domain/test_seed.py`:
```python
from voiceagent.db.repository import Repository
from voiceagent.db.seed import generate_world


def _fresh():
    r = Repository(":memory:")
    r.init_schema()
    return r


def test_seed_is_deterministic(now):
    a, b = _fresh(), _fresh()
    generate_world(a, now, seed=7)
    generate_world(b, now, seed=7)
    assert a.dump() == b.dump()


def test_different_seed_differs(now):
    a, b = _fresh(), _fresh()
    generate_world(a, now, seed=7)
    generate_world(b, now, seed=8)
    assert a.dump() != b.dump()


def test_seed_invariants(now):
    r = _fresh()
    generate_world(r, now, seed=7)
    q = lambda sql: r.conn.execute(sql).fetchone()[0]
    assert q("SELECT COUNT(*) FROM products") == 50
    assert q("SELECT COUNT(*) FROM orders") == 300
    # stock.reserved equals the sum of line reservations per warehouse and sku
    mismatches = q(
        """SELECT COUNT(*) FROM stock s WHERE s.reserved != (
               SELECT COALESCE(SUM(l.reserved_qty), 0) FROM order_lines l
               JOIN orders o ON o.id = l.order_id
               WHERE o.warehouse_id = s.warehouse_id AND l.sku = s.sku)"""
    )
    assert mismatches == 0
    assert q("SELECT COUNT(*) FROM order_lines WHERE reserved_qty < qty") >= 10
    assert q("SELECT COUNT(*) FROM delivery_slots WHERE booked = capacity") >= 5
    assert q("SELECT COUNT(*) FROM stock WHERE on_hand = 0 AND restock_eta IS NOT NULL") >= 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/domain/test_seed.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.db.seed'`.

- [ ] **Step 3: Implement the seed**

`src/voiceagent/db/seed.py`:
```python
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

from voiceagent.db.repository import Repository

CATALOG = [  # (category, item, unit, base_price)
    ("dairy", "Toned Milk", "1L", 50), ("dairy", "Curd", "400g", 35), ("dairy", "Paneer", "200g", 90),
    ("dairy", "Butter", "100g", 56), ("bakery", "Brown Bread", "400g", 45), ("bakery", "Pav Buns", "6 pack", 30),
    ("eggs", "Farm Eggs", "12 pack", 90), ("staples", "Basmati Rice", "5kg", 520), ("staples", "Atta", "5kg", 260),
    ("staples", "Toor Dal", "1kg", 160), ("staples", "Sugar", "1kg", 48), ("oils", "Sunflower Oil", "1L", 150),
    ("oils", "Mustard Oil", "1L", 170), ("snacks", "Potato Chips", "150g", 50), ("snacks", "Salted Peanuts", "200g", 60),
    ("beverages", "Tea Powder", "500g", 280), ("beverages", "Instant Coffee", "100g", 300),
    ("beverages", "Orange Juice", "1L", 120), ("produce", "Onions", "1kg", 40), ("produce", "Tomatoes", "1kg", 35),
    ("produce", "Bananas", "1 dozen", 60), ("household", "Dishwash Liquid", "500ml", 110),
    ("household", "Detergent Powder", "1kg", 140), ("personal", "Toothpaste", "150g", 95),
    ("personal", "Bath Soap", "4 pack", 160),
]
BRANDS = {
    "dairy": ["Amul", "Nandini"], "bakery": ["Modern", "Harvest"], "eggs": ["Eggoz", "Country"],
    "staples": ["Aashirvaad", "Tata"], "oils": ["Fortune", "Saffola"], "snacks": ["Lays", "Haldiram"],
    "beverages": ["Tata", "Bru"], "produce": ["Fresh", "Farm"], "household": ["Vim", "Surf"],
    "personal": ["Colgate", "Dove"],
}
FIRST_NAMES = ["Priya", "Arjun", "Ananya", "Rahul", "Kavya", "Vikram", "Sneha", "Rohan", "Meera", "Aditya",
               "Divya", "Karthik", "Pooja", "Suresh", "Neha", "Manoj", "Lakshmi", "Imran", "Fatima", "John"]
LAST_NAMES = ["Sharma", "Rao", "Iyer", "Reddy", "Nair", "Patel", "Khan", "Gupta", "Menon", "Das"]
STREETS = ["MG Road", "Church Street", "Indiranagar 100 Feet Road", "HSR Layout Sector 2",
           "Koramangala 5th Block", "Jayanagar 4th Block", "Whitefield Main Road", "BTM Layout 2nd Stage"]
WAREHOUSES = [(1, "Koramangala Hub", "south"), (2, "Indiranagar Hub", "east")]


def generate_world(repo: Repository, now: datetime, seed: int = 7, n_customers: int = 200,
                   n_orders: int = 300, days: int = 3) -> None:
    rng = random.Random(seed)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    with repo.transaction():
        for warehouse in WAREHOUSES:
            repo.insert_warehouse(*warehouse)

        skus = []
        for i, (category, item, unit, price) in enumerate(CATALOG):
            for j, brand in enumerate(BRANDS[category]):
                sku = f"{category[:3].upper()}{i:02d}{j}"
                repo.insert_product(sku, f"{brand} {item} {unit}", f"{brand} {item.lower()}", category,
                                    float(price + 5 * j))
                skus.append(sku)

        for warehouse_id, _, _ in WAREHOUSES:
            for sku in skus:
                if rng.random() < 0.12:
                    eta = today + timedelta(days=rng.randint(1, 3), hours=9)
                    repo.insert_stock(warehouse_id, sku, 0, eta)
                else:
                    repo.insert_stock(warehouse_id, sku, rng.randint(2, 40))

        addresses = {}
        for customer_id in range(1, n_customers + 1):
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            addresses[customer_id] = f"{rng.randint(1, 250)} {rng.choice(STREETS)}, Bengaluru"
            repo.insert_customer(customer_id, name, f"+9198{customer_id:08d}", addresses[customer_id])

        for order_id in range(1, n_orders + 1):
            customer_id = rng.randint(1, n_customers)
            warehouse_id = rng.choice(WAREHOUSES)[0]
            repo.insert_order(order_id, customer_id, warehouse_id, addresses[customer_id],
                              now - timedelta(minutes=rng.randint(5, 600)))
            for sku in rng.sample(skus, rng.randint(1, 4)):
                qty = rng.randint(1, 3)
                reserved = min(qty, repo.available_stock(warehouse_id, sku))
                repo.insert_order_line(order_id, sku, qty, reserved)
                if reserved:
                    repo.reserve_stock(warehouse_id, sku, reserved)

        slot_id = 1
        for warehouse_id, _, _ in WAREHOUSES:
            for day in range(days):
                for hour in range(8, 21):
                    start = today + timedelta(days=day, hours=hour)
                    capacity = rng.randint(3, 6)
                    booked = capacity if rng.random() < 0.2 else rng.randint(0, capacity - 1)
                    repo.insert_slot(slot_id, warehouse_id, start, start + timedelta(hours=1), capacity, booked)
                    slot_id += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic synthetic warehouse database")
    parser.add_argument("--db", default="data/warehouse.db")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--now", help="ISO datetime used as 'now' (default: current minute)")
    args = parser.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else datetime.now().replace(second=0, microsecond=0)
    path = Path(args.db)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    repo = Repository(path)
    repo.init_schema()
    generate_world(repo, now, seed=args.seed)
    q = lambda sql: repo.conn.execute(sql).fetchone()[0]
    print(f"wrote {path}: {q('SELECT COUNT(*) FROM orders')} orders, "
          f"{q('SELECT COUNT(*) FROM order_lines WHERE reserved_qty < qty')} short lines, "
          f"{q('SELECT COUNT(*) FROM delivery_slots WHERE booked = capacity')} full slots")
    repo.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and the CLI**

```bash
.venv/Scripts/python -m pytest tests/domain/test_seed.py -q
.venv/Scripts/python -m voiceagent.db.seed --db data/warehouse.db --now 2026-10-12T10:00
```
Expected: `3 passed`; CLI prints `wrote data/warehouse.db: 300 orders, N short lines, M full slots` with N ≥ 10 and M ≥ 5.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/db/seed.py tests/domain/test_seed.py
git commit -m "Add deterministic synthetic warehouse seed"
```

---

### Task 4: Time-window parser

**Files:**
- Create: `src/voiceagent/domain/timeparse.py`, `tests/domain/test_timeparse.py`

**Interfaces:**
- Consumes: `TimeWindow`, `DELIVERY_DAY_START_HOUR`, `DELIVERY_DAY_END_HOUR` from Task 2.
- Produces: `parse_time_window(phrase: str, now: datetime) -> TimeWindow | None`. Returns `None` when the phrase has no usable time or the window falls outside 08:00–21:00. Windows are clamped to delivery hours. If no day is given and the window has already ended today, it rolls to tomorrow.

- [ ] **Step 1: Write the failing tests**

`tests/domain/test_timeparse.py`:
```python
from datetime import datetime

import pytest

from voiceagent.domain.timeparse import parse_time_window

NOW = datetime(2026, 10, 12, 10, 0)  # Monday


def d(day, h, m=0):
    return datetime(2026, 10, day, h, m)


@pytest.mark.parametrize("phrase, start, end", [
    ("tomorrow after 5", d(13, 17), d(13, 21)),
    ("Tomorrow after 5 p.m.", d(13, 17), d(13, 21)),
    ("tomorrow after 5 PM.", d(13, 17), d(13, 21)),
    ("today evening", d(12, 17), d(12, 21)),
    ("between 2 and 4 tomorrow", d(13, 14), d(13, 16)),
    ("5 to 7 pm", d(12, 17), d(12, 19)),
    ("around 6", d(12, 17), d(12, 19)),
    ("in the morning", d(12, 8), d(12, 12)),
    ("Friday afternoon", d(16, 12), d(16, 17)),
    ("day after tomorrow at 11", d(14, 11), d(14, 12)),
    ("half past five", d(12, 17, 30), d(12, 18, 30)),
    ("five thirty pm", d(12, 17, 30), d(12, 18, 30)),
    ("5:30 PM", d(12, 17, 30), d(12, 18, 30)),
    ("as soon as possible", d(12, 10), d(12, 21)),
    ("anytime tomorrow", d(13, 8), d(13, 21)),
    ("tomorrow", d(13, 8), d(13, 21)),
    ("before noon", d(12, 8), d(12, 12)),
    ("by 6", d(12, 8), d(12, 18)),
    ("after 7 o'clock tonight", d(12, 19), d(12, 21)),
    ("seven to eight", d(12, 19), d(12, 20)),
])
def test_parses(phrase, start, end):
    window = parse_time_window(phrase, NOW)
    assert window is not None, phrase
    assert (window.start, window.end) == (start, end)


@pytest.mark.parametrize("phrase", [
    "I don't know",
    "",
    "11 at night",
    "after 9 pm",
    "on the 15th",
])
def test_unparseable_or_outside_hours(phrase):
    assert parse_time_window(phrase, NOW) is None


def test_rolls_to_tomorrow_when_window_passed():
    late = datetime(2026, 10, 12, 15, 0)
    window = parse_time_window("in the morning", late)
    assert (window.start, window.end) == (d(13, 8), d(13, 12))


def test_asap_after_close_is_tomorrow():
    late = datetime(2026, 10, 12, 21, 30)
    window = parse_time_window("asap", late)
    assert (window.start, window.end) == (d(13, 8), d(13, 21))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/domain/test_timeparse.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.domain.timeparse'`.

- [ ] **Step 3: Implement the parser**

`src/voiceagent/domain/timeparse.py`:
```python
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

from voiceagent.domain.models import DELIVERY_DAY_END_HOUR as CLOSE
from voiceagent.domain.models import DELIVERY_DAY_START_HOUR as OPEN
from voiceagent.domain.models import TimeWindow

_NUMBER_WORDS = {
    "forty five": "45", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "fifteen": "15", "thirty": "30",
}
_NUMBER_RE = re.compile(r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b")
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_PARTS = [  # (word, (start_hour, end_hour), am/pm hint)
    ("morning", (8, 12), "am"), ("afternoon", (12, 17), "pm"), ("evening", (17, 21), "pm"),
    ("tonight", (17, 21), "pm"), ("night", (19, 21), "pm"),
]
_T = r"(\d{1,2})(?::(\d{2}))?(?:\s*(am|pm))?(?![a-z0-9])"
_RANGE_RES = [
    re.compile(rf"\b(?:between|from)\s+{_T}\s+(?:and|to|till|until)\s+{_T}"),
    re.compile(rf"\b{_T}\s+(?:to|till|until)\s+{_T}"),
]
_AFTER_RE = re.compile(rf"\b(?:after|from|post)\s+{_T}")
_BEFORE_RE = re.compile(rf"\b(?:before|by|until|till)\s+{_T}")
_AROUND_RE = re.compile(rf"\b(?:around|about|approximately|roughly)\s+{_T}")
_SINGLE_RE = re.compile(rf"\b{_T}")
_ASAP_RE = re.compile(r"\b(asap|as soon as possible|earliest|right now|immediately|right away)\b")
_ANY_RE = re.compile(r"\b(anytime|any time|whenever|all day|no preference)\b")


def _normalize(phrase: str) -> str:
    t = phrase.lower()
    for a, b in (("p.m.", "pm"), ("a.m.", "am"), ("p.m", "pm"), ("a.m", "am")):
        t = t.replace(a, b)
    t = re.sub(r"[^a-z0-9: ]", " ", t)
    t = re.sub(r"\bo clock\b", " ", t)
    t = " ".join(t.split())
    t = _NUMBER_RE.sub(lambda m: _NUMBER_WORDS[m.group(1)], t)
    t = re.sub(r"\bhalf past (\d{1,2})\b", r"\1:30", t)
    t = re.sub(r"\bquarter past (\d{1,2})\b", r"\1:15", t)
    t = re.sub(r"\b(\d{1,2}) (15|30|45)\b", r"\1:\2", t)
    t = re.sub(r"\b(noon|midday)\b", "12 pm", t)
    return t


def _parse_day(t: str, now: datetime) -> tuple[date, bool]:
    today = now.date()
    if "day after tomorrow" in t:
        return today + timedelta(days=2), True
    if re.search(r"\btomorrow\b", t):
        return today + timedelta(days=1), True
    if re.search(r"\b(today|tonight)\b", t):
        return today, True
    for index, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", t):
            return today + timedelta(days=(index - today.weekday()) % 7), True
    return today, False


def _part_of_day(t: str) -> tuple[tuple[int, int] | None, str | None]:
    for word, hours, hint in _PARTS:
        if re.search(rf"\b{word}\b", t):
            return hours, hint
    return None, None


def _hour(h: str, m: str | None, ampm: str | None, hint: str | None) -> float | None:
    hour, minute = int(h), int(m or 0)
    if hour > 23 or minute > 59:
        return None
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    elif ampm is None and hour <= 12:
        if hint == "pm" and hour < 12:
            hour += 12
        elif hint is None and 1 <= hour <= 7:
            hour += 12
    return hour + minute / 60


def _parse_hours(t: str, hint: str | None) -> tuple[float, float] | None:
    for regex in _RANGE_RES:
        m = regex.search(t)
        if m:
            h1, m1, ap1, h2, m2, ap2 = m.groups()
            end = _hour(h2, m2, ap2, hint)
            start = _hour(h1, m1, ap1 or ap2, hint)
            if start is not None and end is not None and start > end and ap1 is None:
                start = _hour(h1, m1, None, hint)
            if start is not None and end is not None and end <= start and ap2 is None and end < 12:
                end += 12  # "seven to eight" -> 19:00-20:00
            return (start, end) if start is not None and end is not None else None
    m = _AFTER_RE.search(t)
    if m:
        x = _hour(*m.groups(), hint)
        return (x, CLOSE) if x is not None else None
    m = _BEFORE_RE.search(t)
    if m:
        x = _hour(*m.groups(), hint)
        return (OPEN, x) if x is not None else None
    m = _AROUND_RE.search(t)
    if m:
        x = _hour(*m.groups(), hint)
        return (x - 1, x + 1) if x is not None else None
    m = _SINGLE_RE.search(t)
    if m:
        x = _hour(*m.groups(), hint)
        return (x, x + 1) if x is not None else None
    return None


def _at(day: date, hours: float) -> datetime:
    return datetime.combine(day, time()) + timedelta(hours=hours)


def parse_time_window(phrase: str, now: datetime) -> TimeWindow | None:
    t = _normalize(phrase)
    if not t:
        return None
    if _ASAP_RE.search(t):
        close = _at(now.date(), CLOSE)
        if now >= close:
            tomorrow = now.date() + timedelta(days=1)
            return TimeWindow(_at(tomorrow, OPEN), _at(tomorrow, CLOSE))
        return TimeWindow(max(now, _at(now.date(), OPEN)), close)

    day, explicit_day = _parse_day(t, now)
    part, hint = _part_of_day(t)
    hours = _parse_hours(t, hint)
    if hours is None:
        if part is not None:
            hours = part
        elif explicit_day or _ANY_RE.search(t):
            hours = (OPEN, CLOSE)
        else:
            return None

    start_h, end_h = max(hours[0], OPEN), min(hours[1], CLOSE)
    if start_h >= end_h:
        return None
    if not explicit_day and _at(day, end_h) <= now:
        day += timedelta(days=1)
    return TimeWindow(_at(day, start_h), _at(day, end_h))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/domain/test_timeparse.py -q`
Expected: `27 passed`. If a case fails, fix the parser (not the expected values); the expected windows are the product behaviour.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/domain/timeparse.py tests/domain/test_timeparse.py
git commit -m "Add rule-based delivery time window parser"
```

---

### Task 5: Inventory service

**Files:**
- Create: `src/voiceagent/domain/inventory.py`, `tests/domain/test_inventory.py`

**Interfaces:**
- Consumes: `Repository` (Task 2).
- Produces:
  - `ProductMatch(sku, name, spoken_name, available: int, unit_price: float, score: float)`
  - `Shortage(sku, name, wanted: int, reserved: int, restock_eta: datetime | None, substitutes: list[ProductMatch])`
  - `AddItemResult(status: "added" | "insufficient", sku, name, qty: int, available: int)`
  - `InventoryService(repo)` with `find_products(warehouse_id, query, limit=3) -> list[ProductMatch]`, `shortages(order_id) -> list[Shortage]`, `add_item(order_id, sku, qty) -> AddItemResult`, `apply_substitution(order_id, sku, substitute_sku) -> bool`, `set_partial(order_id, sku) -> bool`, `wait_restock(order_id, sku) -> datetime | None`, `earliest_delivery(order_id) -> datetime | None`.

- [ ] **Step 1: Write the failing tests**

`tests/domain/test_inventory.py`:
```python
import pytest

from voiceagent.domain.inventory import InventoryService


@pytest.fixture
def inv(world, repo):
    return InventoryService(repo)


def test_find_products_ranks_best_match_first(inv):
    assert inv.find_products(1, "amul milk")[0].sku == "MILK1"
    assert inv.find_products(1, "nandini")[0].sku == "MILK2"
    assert inv.find_products(1, "Eggs")[0].sku == "EGGS12"


def test_find_products_reports_availability(inv):
    match = inv.find_products(1, "amul milk")[0]
    assert match.available == 8


def test_find_products_needs_token_overlap(inv):
    assert inv.find_products(1, "toothpaste") == []
    assert inv.find_products(1, "") == []


def test_shortages_lists_short_lines_with_substitutes(inv, now):
    [shortage] = inv.shortages(1)
    assert (shortage.sku, shortage.wanted, shortage.reserved) == ("BREAD1", 1, 0)
    assert shortage.restock_eta.day == 13 and shortage.restock_eta.hour == 9
    assert [s.sku for s in shortage.substitutes] == ["BREAD2"]


def test_add_new_item_reserves_stock(inv, repo):
    result = inv.add_item(1, "MILK2", 2)
    assert (result.status, result.available) == ("added", 3)
    assert repo.available_stock(1, "MILK2") == 3
    assert [l.sku for l in repo.get_order_lines(1)][-1] == "MILK2"


def test_add_existing_item_increases_quantity(inv, repo):
    inv.add_item(1, "MILK1", 1)
    line = next(l for l in repo.get_order_lines(1) if l.sku == "MILK1")
    assert (line.qty, line.reserved_qty) == (3, 3)


def test_add_item_insufficient_changes_nothing(inv, repo):
    result = inv.add_item(1, "RICE5", 5)
    assert (result.status, result.available) == ("insufficient", 2)
    assert all(l.sku != "RICE5" for l in repo.get_order_lines(1))
    assert repo.available_stock(1, "RICE5") == 2


def test_add_item_rejects_bad_input(inv):
    with pytest.raises(ValueError):
        inv.add_item(1, "MILK1", 0)
    with pytest.raises(ValueError):
        inv.add_item(1, "NOPE", 1)


def test_substitution_reserves_substitute(inv, repo):
    assert inv.apply_substitution(1, "BREAD1", "BREAD2") is True
    line = next(l for l in repo.get_order_lines(1) if l.sku == "BREAD1")
    assert (line.substitute_sku, line.substitute_qty, line.resolution) == ("BREAD2", 1, "substitute")
    assert repo.available_stock(1, "BREAD2") == 3
    assert inv.shortages(1) == []


def test_substitution_fails_without_stock_or_twice(inv, repo):
    assert inv.apply_substitution(1, "BREAD1", "RICE5") is True
    assert inv.apply_substitution(1, "BREAD1", "BREAD2") is False
    assert inv.apply_substitution(1, "MILK1", "MILK2") is False  # not short


def test_partial_marks_line(inv, repo):
    assert inv.set_partial(1, "BREAD1") is True
    assert next(l for l in repo.get_order_lines(1) if l.sku == "BREAD1").resolution == "partial"
    assert inv.shortages(1) == []


def test_wait_restock_sets_earliest_delivery(inv):
    eta = inv.wait_restock(1, "BREAD1")
    assert eta.day == 13 and eta.hour == 9
    assert inv.earliest_delivery(1) == eta


def test_wait_restock_on_line_that_is_not_short_returns_none(inv):
    assert inv.wait_restock(1, "MILK1") is None
    assert inv.earliest_delivery(1) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/domain/test_inventory.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.domain.inventory'`.

- [ ] **Step 3: Implement the service**

`src/voiceagent/domain/inventory.py`:
```python
from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from voiceagent.db.repository import Repository
from voiceagent.domain.models import OrderLine, Product


@dataclass(frozen=True)
class ProductMatch:
    sku: str
    name: str
    spoken_name: str
    available: int
    unit_price: float
    score: float


@dataclass(frozen=True)
class Shortage:
    sku: str
    name: str
    wanted: int
    reserved: int
    restock_eta: datetime | None
    substitutes: list[ProductMatch]


@dataclass(frozen=True)
class AddItemResult:
    status: Literal["added", "insufficient"]
    sku: str
    name: str
    qty: int
    available: int


def _tokens(text: str) -> list[str]:
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in text.lower().split()]


def _score(query: str, product: Product) -> float:
    q_tokens = _tokens(query)
    if not q_tokens:
        return 0.0
    hay = set(_tokens(f"{product.spoken_name} {product.name}"))
    overlap = sum(t in hay for t in q_tokens) / len(q_tokens)
    q = query.lower().strip()
    ratio = max(difflib.SequenceMatcher(None, q, product.spoken_name.lower()).ratio(),
                difflib.SequenceMatcher(None, q, product.name.lower()).ratio())
    return round(0.6 * overlap + 0.4 * ratio, 3)


class InventoryService:
    MATCH_THRESHOLD = 0.45

    def __init__(self, repo: Repository):
        self.repo = repo

    def _match(self, warehouse_id: int, product: Product, score: float) -> ProductMatch:
        return ProductMatch(product.sku, product.name, product.spoken_name,
                            self.repo.available_stock(warehouse_id, product.sku), product.unit_price, score)

    def _line(self, order_id: int, sku: str) -> OrderLine | None:
        return next((l for l in self.repo.get_order_lines(order_id) if l.sku == sku), None)

    def find_products(self, warehouse_id: int, query: str, limit: int = 3) -> list[ProductMatch]:
        matches = []
        for product in self.repo.list_products():
            score = _score(query, product)
            if score >= self.MATCH_THRESHOLD:
                matches.append(self._match(warehouse_id, product, score))
        matches.sort(key=lambda m: (-m.score, m.sku))
        return matches[:limit]

    def shortages(self, order_id: int) -> list[Shortage]:
        order = self.repo.get_order(order_id)
        if order is None:
            return []
        out = []
        for line in self.repo.get_order_lines(order_id):
            if line.reserved_qty >= line.qty or line.resolution is not None:
                continue
            product = self.repo.get_product(line.sku)
            missing = line.qty - line.reserved_qty
            substitutes = [
                self._match(order.warehouse_id, p, 1.0)
                for p in self.repo.list_products()
                if p.category == product.category and p.sku != product.sku
                and self.repo.available_stock(order.warehouse_id, p.sku) >= missing
            ][:2]
            out.append(Shortage(line.sku, product.spoken_name, line.qty, line.reserved_qty,
                                self.repo.restock_eta(order.warehouse_id, line.sku), substitutes))
        return out

    def add_item(self, order_id: int, sku: str, qty: int) -> AddItemResult:
        if qty <= 0:
            raise ValueError("qty must be positive")
        order, product = self.repo.get_order(order_id), self.repo.get_product(sku)
        if order is None or product is None:
            raise ValueError(f"unknown order {order_id} or sku {sku}")
        with self.repo.transaction():
            available = self.repo.available_stock(order.warehouse_id, sku)
            if available < qty:
                return AddItemResult("insufficient", sku, product.spoken_name, qty, available)
            self.repo.reserve_stock(order.warehouse_id, sku, qty)
            existing = self._line(order_id, sku)
            if existing:
                self.repo.update_order_line(order_id, sku, qty=existing.qty + qty,
                                            reserved_qty=existing.reserved_qty + qty)
            else:
                self.repo.insert_order_line(order_id, sku, qty, qty)
        return AddItemResult("added", sku, product.spoken_name, qty, available - qty)

    def _short_line(self, order_id: int, sku: str) -> OrderLine | None:
        line = self._line(order_id, sku)
        if line is None or line.reserved_qty >= line.qty or line.resolution is not None:
            return None
        return line

    def apply_substitution(self, order_id: int, sku: str, substitute_sku: str) -> bool:
        order, line = self.repo.get_order(order_id), self._short_line(order_id, sku)
        if order is None or line is None or substitute_sku == sku:
            return False
        missing = line.qty - line.reserved_qty
        with self.repo.transaction():
            if self.repo.available_stock(order.warehouse_id, substitute_sku) < missing:
                return False
            self.repo.reserve_stock(order.warehouse_id, substitute_sku, missing)
            self.repo.update_order_line(order_id, sku, substitute_sku=substitute_sku,
                                        substitute_qty=missing, resolution="substitute")
        return True

    def set_partial(self, order_id: int, sku: str) -> bool:
        if self._short_line(order_id, sku) is None:
            return False
        self.repo.update_order_line(order_id, sku, resolution="partial")
        return True

    def wait_restock(self, order_id: int, sku: str) -> datetime | None:
        order, line = self.repo.get_order(order_id), self._short_line(order_id, sku)
        if order is None or line is None:
            return None
        eta = self.repo.restock_eta(order.warehouse_id, sku)
        if eta is None:
            return None
        self.repo.update_order_line(order_id, sku, resolution="wait")
        return eta

    def earliest_delivery(self, order_id: int) -> datetime | None:
        order = self.repo.get_order(order_id)
        if order is None:
            return None
        etas = [self.repo.restock_eta(order.warehouse_id, l.sku)
                for l in self.repo.get_order_lines(order_id) if l.resolution == "wait"]
        etas = [e for e in etas if e is not None]
        return max(etas) if etas else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/domain/test_inventory.py -q`
Expected: `13 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/domain/inventory.py tests/domain/test_inventory.py
git commit -m "Add inventory service: lookup, shortages, add, substitute, partial, wait"
```

---

### Task 6: Scheduling service

**Files:**
- Create: `src/voiceagent/domain/scheduling.py`, `tests/domain/test_scheduling.py`

**Interfaces:**
- Consumes: `Repository` (Task 2), `InventoryService.earliest_delivery` (Task 5), `TimeWindow`, `Slot`.
- Produces:
  - `SlotOption(slot: Slot, available: int)`
  - `SlotCheck(status: "available" | "full" | "no_slots", option: SlotOption | None, alternatives: list[SlotOption])`
  - `SchedulingService(repo, inventory)` with `HOLD_TTL = timedelta(minutes=5)`, `HORIZON = timedelta(days=7)`, `free_slots(order_id, now, limit=6) -> list[SlotOption]`, `check_slot(order_id, window, now) -> SlotCheck`, `hold_slot(order_id, slot_id, now) -> bool`, `confirm_booking(order_id, now) -> Slot | None`.

- [ ] **Step 1: Write the failing tests**

`tests/domain/test_scheduling.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/domain/test_scheduling.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.domain.scheduling'`.

- [ ] **Step 3: Implement the service**

`src/voiceagent/domain/scheduling.py`:
```python
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
        if order is None or slot is None or slot.warehouse_id != order.warehouse_id:
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
            if slot_id is None or order is None or order.status not in ("pending_schedule", "callback"):
                return None
            self.repo.increment_booked(slot_id)
            self.repo.delete_hold(order_id)
            self.repo.update_order(order_id, status="scheduled", slot_id=slot_id, callback_at=None)
        return self.repo.get_slot(slot_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/domain/test_scheduling.py -q`
Expected: `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/domain/scheduling.py tests/domain/test_scheduling.py
git commit -m "Add scheduling service with slot holds and booking confirmation"
```

---

### Task 7: Order service

**Files:**
- Create: `src/voiceagent/domain/orders.py`, `tests/domain/test_orders.py`

**Interfaces:**
- Consumes: `Repository` (Task 2); tests also use `InventoryService` and `SchedulingService`.
- Produces: `OrderService(repo)` with `cancel_order(order_id) -> bool`, `update_address(order_id, address) -> None` (raises `ValueError` if under 5 characters after whitespace collapse), `add_note(order_id, note) -> None` (raises `ValueError` if empty; appends with `"; "`), `schedule_callback(order_id, when) -> None`, `mark_wrong_person(order_id) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/domain/test_orders.py`:
```python
from datetime import timedelta

import pytest

from voiceagent.domain.inventory import InventoryService
from voiceagent.domain.orders import OrderService
from voiceagent.domain.scheduling import SchedulingService


@pytest.fixture
def orders(world, repo):
    return OrderService(repo)


def test_cancel_releases_reservations(orders, repo):
    assert orders.cancel_order(1) is True
    assert repo.available_stock(1, "MILK1") == 10
    assert repo.available_stock(1, "EGGS12") == 20
    assert repo.get_order(1).status == "cancelled"
    assert orders.cancel_order(1) is False


def test_cancel_scheduled_order_frees_slot_and_substitute_stock(orders, repo, world, now):
    inv = InventoryService(repo)
    sched = SchedulingService(repo, inv)
    inv.apply_substitution(1, "BREAD1", "BREAD2")
    slot_id = world["slots"][(1, 18)]
    sched.hold_slot(1, slot_id, now)
    sched.confirm_booking(1, now)
    assert orders.cancel_order(1) is True
    assert repo.get_slot(slot_id).booked == 0
    assert repo.available_stock(1, "BREAD2") == 4
    assert repo.get_order(1).slot_id is None


def test_cancel_unknown_order(orders):
    assert orders.cancel_order(999) is False


def test_update_address_collapses_whitespace(orders, repo):
    orders.update_address(1, "  42   Church Street,  Bengaluru ")
    assert repo.get_order(1).address == "42 Church Street, Bengaluru"
    with pytest.raises(ValueError):
        orders.update_address(1, "  ")


def test_add_note_appends(orders, repo):
    orders.add_note(1, "leave with the security guard")
    orders.add_note(1, "call at the gate")
    assert repo.get_order(1).notes == "leave with the security guard; call at the gate"
    with pytest.raises(ValueError):
        orders.add_note(1, "   ")


def test_schedule_callback(orders, repo, now):
    orders.schedule_callback(1, now + timedelta(hours=1))
    order = repo.get_order(1)
    assert (order.status, order.callback_at) == ("callback", now + timedelta(hours=1))


def test_mark_wrong_person(orders, repo):
    orders.mark_wrong_person(1)
    order = repo.get_order(1)
    assert order.status == "callback" and "wrong person" in order.notes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/domain/test_orders.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.domain.orders'`.

- [ ] **Step 3: Implement the service**

`src/voiceagent/domain/orders.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/domain/test_orders.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/domain/orders.py tests/domain/test_orders.py
git commit -m "Add order service: cancel, address, notes, callback, wrong person"
```

---

### Task 8: Pre-call context builder

**Files:**
- Create: `src/voiceagent/domain/context.py`, `tests/domain/test_context.py`

**Interfaces:**
- Consumes: `Repository`, `InventoryService.shortages`, `SchedulingService.free_slots`, `SlotOption`, `Slot`.
- Produces: `spoken_time(dt) -> str` (`"5 PM"`, `"5:30 PM"`), `spoken_day(day: date, now) -> str` (`"today"`, `"tomorrow"`, weekday name), `spoken_slot(slot, now) -> str` (`"tomorrow, 5 to 6 PM"`, `"today, 11 AM to 12 PM"`), `CallContext(order_id, customer_name, prompt_text, slot_options)`, `build_call_context(repo, inventory, scheduling, order_id, now) -> CallContext` (raises `ValueError` for an unknown order). `prompt_text` is deterministic for the same DB state and `now`.

- [ ] **Step 1: Write the failing tests**

`tests/domain/test_context.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/domain/test_context.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.domain.context'`.

- [ ] **Step 3: Implement the builder**

`src/voiceagent/domain/context.py`:
```python
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
```

- [ ] **Step 4: Run the full domain suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all tests pass (smoke + repository + seed + timeparse + inventory + scheduling + orders + context).

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/domain/context.py tests/domain/test_context.py
git commit -m "Add pre-call context builder with spoken slot formatting"
```

---

### Task 9: Bench environment, shared helpers and model downloads

**Files:**
- Create: `bench/common.py`, `tests/bench/test_common.py` (no `__init__.py` — it would shadow the real `bench` package), `scripts/download_models.py`, `scripts/get_llama_cpp.sh`, `scripts/llama_server.sh`

**Interfaces:**
- Produces in `bench.common`: `ROOT`, `MODELS_DIR`, `CORPUS_DIR`, `RESULTS_DIR`, `MANIFEST_PATH`; `percentile(values, p) -> float` (nearest-rank, `nan` for empty); `summarize(values) -> dict` with keys `n, p50, p95, mean, min, max` (rounded to 1 decimal); `normalize_text(s) -> str`; `word_errors(ref, hyp) -> tuple[int, int]` (edits, reference words); `corpus_wer(pairs) -> float`; `GpuMemorySampler` context manager with `baseline_mb`, `peak_mb`, `delta_mb`; `write_result(name, payload, device) -> Path` (adds `name`, `device`, `recorded_at`, `machine`); `load_manifest(source="all") -> list[dict]`; `load_wav16k(path) -> np.ndarray`; `first_clause_end(text) -> int | None`; `first_sentence_end(text) -> int | None`.

- [ ] **Step 1: Install the bench dependencies**

```bash
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e ".[dev,bench]"
.venv/Scripts/python -m pip uninstall -y onnxruntime
.venv/Scripts/python -m pip install "onnxruntime-gpu[cuda,cudnn]>=1.21"
.venv/Scripts/python -c "import torch, onnxruntime as ort; print(torch.cuda.is_available(), ort.get_available_providers())"
```
Expected: `True ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']` (TensorRT may be absent; CUDA must be present). If the `cu128` index has no wheel for the installed torch version, use `--index-url https://download.pytorch.org/whl/cu126`.

- [ ] **Step 2: Write the failing helper tests**

`tests/bench/test_common.py`:
```python
import json
import math

from bench import common


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert common.percentile(values, 50) == 50
    assert common.percentile(values, 95) == 95
    assert common.percentile([7], 95) == 7
    assert math.isnan(common.percentile([], 50))


def test_summarize():
    s = common.summarize([10.0, 20.0, 30.0, 40.0])
    assert s == {"n": 4, "p50": 20.0, "p95": 40.0, "mean": 25.0, "min": 10.0, "max": 40.0}


def test_normalize_text():
    assert common.normalize_text("Tomorrow, after 5 P.M.!") == "tomorrow after 5 pm"
    assert common.normalize_text("Five thirty") == "5 30"
    assert common.normalize_text("5:30 PM") == "5 30 pm"
    assert common.normalize_text("I don't know") == "i dont know"
    assert common.normalize_text("Mm-hmm.") == "mm hmm"


def test_word_errors_and_corpus_wer():
    assert common.word_errors("deliver it at 5 pm", "deliver it at five p.m.") == (0, 5)
    assert common.word_errors("a b c", "a x c d") == (2, 3)
    assert common.word_errors("a b", "") == (2, 2)
    assert common.corpus_wer([("a b c", "a b c"), ("a b", "a")]) == 0.2


def test_clause_and_sentence_end():
    text = "Hi Priya, your order is ready. Shall I book it?"
    assert text[: common.first_clause_end(text)] == "Hi Priya,"
    assert text[: common.first_sentence_end(text)] == "Hi Priya, your order is ready."
    assert common.first_clause_end("Okay") is None
    assert common.first_clause_end("Okay,") is None  # needs at least two words


def test_write_result_includes_device(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    path = common.write_result("stt_demo", {"x": 1}, device="CUDAExecutionProvider")
    data = json.loads(path.read_text())
    assert data["name"] == "stt_demo"
    assert data["device"] == "CUDAExecutionProvider"
    assert data["x"] == 1 and "recorded_at" in data
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/bench/test_common.py -q`
Expected: FAIL with `ImportError` / `AttributeError` on `bench.common`.

- [ ] **Step 4: Implement the helpers**

`bench/common.py`:
```python
from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CORPUS_DIR = ROOT / "bench" / "corpus"
RESULTS_DIR = ROOT / "bench" / "results"
MANIFEST_PATH = CORPUS_DIR / "manifest.jsonl"

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50",
}


def percentile(values, p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(math.ceil(p / 100 * len(ordered)) - 1, 0)
    return ordered[min(index, len(ordered) - 1)]


def summarize(values) -> dict:
    values = list(values)
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    r = lambda v: round(float(v), 1)
    return {"n": len(values), "p50": r(percentile(values, 50)), "p95": r(percentile(values, 95)),
            "mean": r(sum(values) / len(values)), "min": r(min(values)), "max": r(max(values))}


def normalize_text(s: str) -> str:
    t = s.lower().replace("p.m.", "pm").replace("a.m.", "am")
    t = t.replace("o'clock", " ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t).replace("'", "")
    return " ".join(_NUM_WORDS.get(w, w) for w in t.split())


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1], len(r)


def corpus_wer(pairs) -> float:
    edits = words = 0
    for ref, hyp in pairs:
        e, n = word_errors(ref, hyp)
        edits, words = edits + e, words + n
    return edits / words if words else 0.0


def _end_after(text: str, pattern: str, min_words: int) -> int | None:
    for m in re.finditer(pattern, text):
        if len(text[: m.end()].split()) >= min_words:
            return m.end()
    return None


def first_clause_end(text: str) -> int | None:
    return _end_after(text, r"[,;:.!?]", 2)


def first_sentence_end(text: str) -> int | None:
    return _end_after(text, r"[.!?]", 1)


class GpuMemorySampler:
    """Samples total GPU memory used (MB) via nvidia-smi while the block runs."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.baseline_mb: int | None = None
        self.peak_mb: int | None = None

    @staticmethod
    def read_mb() -> int | None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            value = self.read_mb()
            if value is not None and (self.peak_mb is None or value > self.peak_mb):
                self.peak_mb = value

    def __enter__(self) -> "GpuMemorySampler":
        self.baseline_mb = self.peak_mb = self.read_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def delta_mb(self) -> int | None:
        if self.baseline_mb is None or self.peak_mb is None:
            return None
        return self.peak_mb - self.baseline_mb


def write_result(name: str, payload: dict, device: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"name": name, "device": device, "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "machine": platform.node(), **payload}
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_manifest(source: str = "all") -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise SystemExit("bench/corpus/manifest.jsonl missing: run `python -m bench.make_corpus` first")
    rows = [json.loads(l) for l in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows if source == "all" else [r for r in rows if r["source"] == source]


def load_wav16k(path):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(CORPUS_DIR / path, dtype="float32", always_2d=True)
    if sr != 16000:
        raise ValueError(f"{path}: expected 16 kHz, got {sr}")
    return np.ascontiguousarray(audio[:, 0])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/bench/test_common.py -q`
Expected: `6 passed`.

- [ ] **Step 6: Write the download scripts**

`scripts/download_models.py`:
```python
"""Download model files into models/. Usage: python scripts/download_models.py [turn] [tts] [stt] [llm]"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = Path(__file__).resolve().parents[1] / "models"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
LLMS = [
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf"),
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
]


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def main(groups: set[str]) -> None:
    if "turn" in groups:
        hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx", local_dir=MODELS / "smart-turn")
    if "tts" in groups:
        for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
            fetch(KOKORO_URL + name, MODELS / "kokoro" / name)
    if "stt" in groups:
        for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
            hf_hub_download("csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", name,
                            local_dir=MODELS / "parakeet-tdt-0.6b-v2-int8")
        snapshot_download("nvidia/nemotron-speech-streaming-en-0.6b",
                          ignore_patterns=["*.nemo", "*.gguf", "figures/*"])
    if "llm" in groups:
        for repo_id, filename in LLMS:
            hf_hub_download(repo_id, filename, local_dir=MODELS / "llm")
    print("done")


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"turn", "tts", "stt", "llm"})
```

`scripts/get_llama_cpp.sh`:
```bash
#!/usr/bin/env bash
# Download the llama.cpp Windows CUDA 12.4 build into tools/llama.cpp
set -euo pipefail
BUILD="${1:-b11514}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/tools/llama.cpp"
mkdir -p "$DEST"
cd "$DEST"
BASE="https://github.com/ggml-org/llama.cpp/releases/download/$BUILD"
for f in "llama-$BUILD-bin-win-cuda-12.4-x64.zip" "cudart-llama-bin-win-cuda-12.4-x64.zip"; do
  [ -f "$f" ] || curl -fL -o "$f" "$BASE/$f"
  unzip -o -q "$f"
done
"$(find "$DEST" -name llama-server.exe | head -1)" --version
```

`scripts/llama_server.sh`:
```bash
#!/usr/bin/env bash
# Usage: scripts/llama_server.sh models/llm/Qwen3.5-4B-Q4_K_M.gguf
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="$(find "$ROOT/tools/llama.cpp" -name llama-server.exe | head -1)"
exec "$SERVER" -m "$1" -ngl 99 -c 8192 --parallel 1 --host 127.0.0.1 --port 8080 --jinja --cache-reuse 256
```

- [ ] **Step 7: Run the downloads**

```bash
.venv/Scripts/python scripts/download_models.py
bash scripts/get_llama_cpp.sh
```
Expected: `done` from the Python script (≈ 9 GB total; GGUFs are the bulk) and a `version: ...` line from `llama-server.exe`.

- [ ] **Step 8: Commit**

```bash
git add bench/common.py tests/bench scripts
git commit -m "Add benchmark helpers and model download scripts"
```

---

### Task 10: Benchmark corpus

**Files:**
- Create: `bench/corpus/phrases.json`, `bench/tts_engine.py`, `bench/make_corpus.py`, `bench/record_corpus.py`

**Interfaces:**
- Consumes: `bench.common` paths (Task 9).
- Produces: `bench.tts_engine.load_kokoro() -> tuple[Kokoro, str]` (engine, actual execution provider); `bench/corpus/manifest.jsonl` rows `{"id", "path", "text", "label": "complete"|"incomplete", "voice", "source": "tts"|"human"}` with WAVs at 16 kHz mono under `bench/corpus/wav/`.

- [ ] **Step 1: Write the phrase list**

`bench/corpus/phrases.json`:
```json
[
  {"id": "c01", "label": "complete", "text": "Yes, this is Priya speaking."},
  {"id": "c02", "label": "complete", "text": "Yeah, that's me."},
  {"id": "c03", "label": "complete", "text": "No, you have the wrong number."},
  {"id": "c04", "label": "complete", "text": "She's not home right now, can you call back later?"},
  {"id": "c05", "label": "complete", "text": "Can you deliver it tomorrow after 5 p.m.?"},
  {"id": "c06", "label": "complete", "text": "Tomorrow evening works for me."},
  {"id": "c07", "label": "complete", "text": "Is it possible to get it between 2 and 4 tomorrow?"},
  {"id": "c08", "label": "complete", "text": "Any time after 6 is fine."},
  {"id": "c09", "label": "complete", "text": "I'd prefer the morning, around 10."},
  {"id": "c10", "label": "complete", "text": "Can you do Friday afternoon instead?"},
  {"id": "c11", "label": "complete", "text": "Not today, I'm at work until 7."},
  {"id": "c12", "label": "complete", "text": "Okay, 6 to 7 works."},
  {"id": "c13", "label": "complete", "text": "Yes, please confirm that."},
  {"id": "c14", "label": "complete", "text": "No, change the time please."},
  {"id": "c15", "label": "complete", "text": "Actually, deliver it to my office instead."},
  {"id": "c16", "label": "complete", "text": "My new address is 42 Church Street, Bengaluru."},
  {"id": "c17", "label": "complete", "text": "Please leave it with the security guard."},
  {"id": "c18", "label": "complete", "text": "Call me when you reach the gate."},
  {"id": "c19", "label": "complete", "text": "I want to cancel the order."},
  {"id": "c20", "label": "complete", "text": "Please cancel it, I don't need it anymore."},
  {"id": "c21", "label": "complete", "text": "Can you also add two packets of Amul milk?"},
  {"id": "c22", "label": "complete", "text": "Add one dozen bananas as well."},
  {"id": "c23", "label": "complete", "text": "Do you have Tata salt in stock?"},
  {"id": "c24", "label": "complete", "text": "If the bread is not available, send whole wheat instead."},
  {"id": "c25", "label": "complete", "text": "Just send whatever is available."},
  {"id": "c26", "label": "complete", "text": "I'll wait for the rice to come back in stock."},
  {"id": "c27", "label": "complete", "text": "Skip the eggs then."},
  {"id": "c28", "label": "complete", "text": "How much is the total?"},
  {"id": "c29", "label": "complete", "text": "Can you call me back in an hour?"},
  {"id": "c30", "label": "complete", "text": "I'm driving right now, call me later."},
  {"id": "c31", "label": "complete", "text": "Yes, that sounds good."},
  {"id": "c32", "label": "complete", "text": "That's all, thank you."},
  {"id": "c33", "label": "complete", "text": "Sorry, can you repeat that?"},
  {"id": "c34", "label": "complete", "text": "What time slots do you have?"},
  {"id": "c35", "label": "complete", "text": "Is there anything earlier?"},
  {"id": "c36", "label": "complete", "text": "The day after tomorrow at 11 would be better."},
  {"id": "c37", "label": "complete", "text": "Before noon, if possible."},
  {"id": "c38", "label": "complete", "text": "As soon as possible please."},
  {"id": "c39", "label": "complete", "text": "Half past five is perfect."},
  {"id": "c40", "label": "complete", "text": "Make it the 7 to 8 slot."},
  {"id": "c41", "label": "complete", "text": "No, the flat number is 304, not 403."},
  {"id": "c42", "label": "complete", "text": "It's B block, third floor."},
  {"id": "c43", "label": "complete", "text": "Can the delivery person bring change for 500?"},
  {"id": "c44", "label": "complete", "text": "Okay, go ahead."},
  {"id": "c45", "label": "complete", "text": "Mm-hmm."},
  {"id": "c46", "label": "complete", "text": "Sure."},
  {"id": "c47", "label": "complete", "text": "Nope."},
  {"id": "c48", "label": "complete", "text": "Wait, which order is this?"},
  {"id": "c49", "label": "complete", "text": "I already received it yesterday."},
  {"id": "c50", "label": "complete", "text": "Can you send it with the substitute instead?"},
  {"id": "i01", "label": "incomplete", "text": "Can you deliver it after"},
  {"id": "i02", "label": "incomplete", "text": "My address is"},
  {"id": "i03", "label": "incomplete", "text": "Actually I was thinking maybe"},
  {"id": "i04", "label": "incomplete", "text": "Tomorrow would be fine but"},
  {"id": "i05", "label": "incomplete", "text": "I want to add"},
  {"id": "i06", "label": "incomplete", "text": "Leave it with the"},
  {"id": "i07", "label": "incomplete", "text": "The flat number is"},
  {"id": "i08", "label": "incomplete", "text": "Can you make it around"},
  {"id": "i09", "label": "incomplete", "text": "No wait, I meant"},
  {"id": "i10", "label": "incomplete", "text": "Between four and"},
  {"id": "i11", "label": "incomplete", "text": "If the bread is not available then"},
  {"id": "i12", "label": "incomplete", "text": "I'll be home after"},
  {"id": "i13", "label": "incomplete", "text": "Could you also check if"},
  {"id": "i14", "label": "incomplete", "text": "So the time is"},
  {"id": "i15", "label": "incomplete", "text": "Yes but only if"}
]
```

- [ ] **Step 2: Write the Kokoro loader**

`bench/tts_engine.py`:
```python
from __future__ import annotations

from bench.common import MODELS_DIR


def load_kokoro():
    """Return (Kokoro, provider). Uses CUDA when onnxruntime-gpu can load it, else CPU."""
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    wanted = ("CUDAExecutionProvider", "CPUExecutionProvider")
    providers = [p for p in wanted if p in ort.get_available_providers()]
    session = ort.InferenceSession(str(MODELS_DIR / "kokoro" / "kokoro-v1.0.onnx"), providers=providers)
    provider = session.get_providers()[0]
    if provider != "CUDAExecutionProvider":
        print(f"WARNING: Kokoro running on {provider}, not CUDA")
    return Kokoro.from_session(session, str(MODELS_DIR / "kokoro" / "voices-v1.0.bin")), provider
```

- [ ] **Step 3: Write the corpus generator**

`bench/make_corpus.py`:
```python
"""Synthesize the benchmark corpus with Kokoro. Usage: python -m bench.make_corpus"""
from __future__ import annotations

import json
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from bench.common import CORPUS_DIR, MANIFEST_PATH
from bench.tts_engine import load_kokoro

VOICES = ["af_heart", "am_michael", "bf_emma", "bm_george"]
PAD = np.zeros(int(0.2 * 16000), dtype=np.float32)


def to16k(samples: np.ndarray, sr: int) -> np.ndarray:
    g = gcd(16000, sr)
    return resample_poly(samples, 16000 // g, sr // g).astype(np.float32)


def main() -> None:
    kokoro, provider = load_kokoro()
    phrases = json.loads((CORPUS_DIR / "phrases.json").read_text(encoding="utf-8"))
    wav_dir = CORPUS_DIR / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    if MANIFEST_PATH.exists():
        existing = [json.loads(l) for l in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in existing if r["source"] != "tts"]
    for i, phrase in enumerate(phrases):
        voice = VOICES[i % len(VOICES)]
        lang = "en-us" if voice.startswith("a") else "en-gb"
        samples, sr = kokoro.create(phrase["text"], voice=voice, lang=lang)
        audio = np.concatenate([PAD, to16k(samples, sr), PAD])
        rel = f"wav/{phrase['id']}_{voice}.wav"
        sf.write(CORPUS_DIR / rel, audio, 16000)
        rows.append({**phrase, "path": rel, "voice": voice, "source": "tts"})
    MANIFEST_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(phrases)} tts clips ({provider}); manifest has {len(rows)} rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the human recorder**

`bench/record_corpus.py`:
```python
"""Record yourself reading corpus phrases. Usage: python -m bench.record_corpus [--ids c01 c05 ...]

For each phrase: press Enter to start, read it naturally, press Enter to stop.
Incomplete phrases: stop mid-thought exactly where the text ends.
"""
from __future__ import annotations

import argparse
import json
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

from bench.common import CORPUS_DIR, MANIFEST_PATH


def record_until_enter() -> np.ndarray:
    chunks: list[np.ndarray] = []
    stop = threading.Event()

    def callback(indata, frames, time_info, status):
        chunks.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=callback):
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        stop.wait()
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", help="phrase ids to record (default: all)")
    args = parser.parse_args()
    phrases = json.loads((CORPUS_DIR / "phrases.json").read_text(encoding="utf-8"))
    if args.ids:
        phrases = [p for p in phrases if p["id"] in set(args.ids)]
    rows = []
    if MANIFEST_PATH.exists():
        rows = [json.loads(l) for l in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    recorded_ids = {p["id"] for p in phrases}
    rows = [r for r in rows if not (r["source"] == "human" and r["id"] in recorded_ids)]
    (CORPUS_DIR / "wav").mkdir(parents=True, exist_ok=True)
    for phrase in phrases:
        input(f"\n[{phrase['id']}] ({phrase['label']}) \"{phrase['text']}\"  -- Enter to start")
        print("recording... Enter to stop")
        audio = record_until_enter()
        rel = f"wav/{phrase['id']}_human.wav"
        sf.write(CORPUS_DIR / rel, audio, 16000)
        rows.append({**phrase, "path": rel, "voice": "human", "source": "human"})
        print(f"saved {rel} ({len(audio) / 16000:.1f}s)")
    MANIFEST_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Generate the corpus and check it**

```bash
.venv/Scripts/python -m bench.make_corpus
.venv/Scripts/python -c "from bench.common import load_manifest, load_wav16k; rows = load_manifest(); a = load_wav16k(rows[0]['path']); print(len(rows), a.dtype, round(len(a)/16000, 2))"
```
Expected: `wrote 65 tts clips (CUDAExecutionProvider); manifest has 65 rows`, then `65 float32 <seconds>`. Listen to two WAVs under `bench/corpus/wav/` to confirm they are intelligible.

Optional but recommended (real accent + real prosody): `.venv/Scripts/python -m bench.record_corpus` (about 10 minutes for all 65).

- [ ] **Step 6: Commit**

```bash
git add bench/corpus/phrases.json bench/tts_engine.py bench/make_corpus.py bench/record_corpus.py
git commit -m "Add benchmark corpus phrases, synthesizer and recorder"
```

---

### Task 11: STT benchmark

**Files:**
- Create: `bench/stt_bench.py`, `tests/bench/test_stt_bench.py`

**Interfaces:**
- Consumes: `bench.common` (Task 9), corpus manifest (Task 10).
- Produces: engines exposing `name: str`, `device: str`, `warmup() -> None`, `transcribe(audio: np.ndarray) -> tuple[str, float]` (text, post-speech ms). Engine names: `nemotron-80`, `nemotron-160`, `nemotron-560`, `parakeet-cpu`, `whisper-distil-small`. `run_engine(engine, rows) -> dict` returns `{"engine", "wer_all", "wer_by_source", "post_speech_ms", "rtf", "clips"}`. CLI `python -m bench.stt_bench --engine <name> [--source all|tts|human]` writes `bench/results/stt_<name>.json`.

Post-speech latency definition: offline engines (Parakeet, Whisper) re-decode the whole utterance at end of turn, so it is the full decode time. The streaming engine (Nemotron) has already processed every earlier chunk while the user spoke, so it is the time from yielding the final chunk to the final text.

- [ ] **Step 1: Write the failing test (engine-agnostic runner)**

`tests/bench/test_stt_bench.py`:
```python
import numpy as np

from bench import stt_bench


class FakeEngine:
    name = "fake"
    device = "cpu"

    def warmup(self):
        pass

    def transcribe(self, audio):
        return "deliver it at 5 pm", 42.0


def test_run_engine_aggregates(monkeypatch):
    rows = [
        {"id": "c1", "path": "x.wav", "text": "Deliver it at 5 p.m.", "source": "tts"},
        {"id": "c2", "path": "y.wav", "text": "Deliver it at six p.m.", "source": "human"},
    ]
    monkeypatch.setattr(stt_bench, "load_wav16k", lambda path: np.zeros(16000, dtype=np.float32))
    result = stt_bench.run_engine(FakeEngine(), rows)
    assert result["engine"] == "fake"
    assert result["wer_by_source"] == {"tts": 0.0, "human": 0.2}
    assert result["wer_all"] == 0.1
    assert result["post_speech_ms"]["p50"] == 42.0
    assert len(result["clips"]) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/bench/test_stt_bench.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.stt_bench'`.

- [ ] **Step 3: Implement engines and runner**

`bench/stt_bench.py`:
```python
"""STT benchmark. Usage: python -m bench.stt_bench --engine nemotron-160 [--source all]"""
from __future__ import annotations

import argparse
import time
from threading import Thread

import numpy as np

from bench.common import (MODELS_DIR, GpuMemorySampler, corpus_wer, load_manifest, load_wav16k, summarize,
                          write_result)

NEMOTRON_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
NEMOTRON_LOOKAHEAD = {"nemotron-80": 0, "nemotron-160": 1, "nemotron-560": 6}


class NemotronStreaming:
    """Cache-aware streaming RNNT via transformers (pattern from the model card)."""

    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        self.name, self.device = name, "cuda"
        self.processor = AutoProcessor.from_pretrained(NEMOTRON_ID)
        self.model = AutoModelForRNNT.from_pretrained(NEMOTRON_ID, device_map="cuda", dtype=torch.float32)
        self.processor.set_num_lookahead_tokens(NEMOTRON_LOOKAHEAD[name])
        self.sr = self.processor.feature_extractor.sampling_rate
        self.streaming_latency_ms = self.processor.streaming_latency_ms

    def warmup(self) -> None:
        self.transcribe(np.zeros(self.sr, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        from transformers import TextIteratorStreamer

        p, model = self.processor, self.model
        # trailing silence flushes the right context, as the VAD silence does in the live pipeline
        audio = np.concatenate([audio, np.zeros(2 * p.num_samples_per_audio_chunk, dtype=np.float32)])
        first = p(audio[: p.num_samples_first_audio_chunk], sampling_rate=self.sr, is_streaming=True,
                  is_first_audio_chunk=True, return_tensors="pt").to(model.device, dtype=model.dtype)
        marks: dict[str, float] = {}

        def features():
            yield first.input_features[:, : p.num_mel_frames_first_audio_chunk, :]
            mel_idx = p.num_mel_frames_first_audio_chunk
            hop, n_fft = p.feature_extractor.hop_length, p.feature_extractor.n_fft
            start = mel_idx * hop - n_fft // 2
            while (end := start + p.num_samples_per_audio_chunk) < audio.shape[0]:
                chunk = p(audio[start:end], sampling_rate=self.sr, is_streaming=True, is_first_audio_chunk=False,
                          return_tensors="pt").to(model.device, dtype=model.dtype)
                marks["last_yield"] = time.perf_counter()
                yield chunk.input_features
                mel_idx += p.num_mel_frames_per_audio_chunk
                start = mel_idx * hop - n_fft // 2

        streamer = TextIteratorStreamer(p.tokenizer, skip_special_tokens=True)
        thread = Thread(target=model.generate,
                        kwargs={**first, "input_features": features(), "streamer": streamer})
        thread.start()
        text = "".join(streamer)
        thread.join()
        done = time.perf_counter()
        return text.strip(), (done - marks.get("last_yield", done)) * 1000


class ParakeetSherpaCpu:
    def __init__(self, name: str):
        import sherpa_onnx

        d = MODELS_DIR / "parakeet-tdt-0.6b-v2-int8"
        self.name, self.device = name, "cpu"
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
            num_threads=4, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
            model_type="nemo_transducer",
        )

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip(), (time.perf_counter() - t0) * 1000


class FasterWhisper:
    def __init__(self, name: str):
        from faster_whisper import WhisperModel

        self.name, self.device = name, "cuda"
        self.model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, without_timestamps=True,
                                            vad_filter=False, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments)
        return text.strip(), (time.perf_counter() - t0) * 1000


def build_engine(name: str):
    if name in NEMOTRON_LOOKAHEAD:
        return NemotronStreaming(name)
    if name == "parakeet-cpu":
        return ParakeetSherpaCpu(name)
    if name == "whisper-distil-small":
        return FasterWhisper(name)
    raise SystemExit(f"unknown engine {name}")


def run_engine(engine, rows: list[dict]) -> dict:
    clips = []
    for row in rows:
        audio = load_wav16k(row["path"])
        t0 = time.perf_counter()
        text, post_ms = engine.transcribe(audio)
        total_s = time.perf_counter() - t0
        clips.append({"id": row["id"], "source": row["source"], "ref": row["text"], "hyp": text,
                      "post_speech_ms": round(post_ms, 1), "rtf": round(total_s / max(len(audio) / 16000, 1e-6), 3)})
    by_source = {}
    for source in dict.fromkeys(c["source"] for c in clips):
        by_source[source] = round(corpus_wer([(c["ref"], c["hyp"]) for c in clips if c["source"] == source]), 4)
    return {
        "engine": engine.name,
        "wer_all": round(corpus_wer([(c["ref"], c["hyp"]) for c in clips]), 4),
        "wer_by_source": by_source,
        "post_speech_ms": summarize([c["post_speech_ms"] for c in clips]),
        "rtf": summarize([c["rtf"] for c in clips]),
        "clips": clips,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True,
                        choices=[*NEMOTRON_LOOKAHEAD, "parakeet-cpu", "whisper-distil-small"])
    parser.add_argument("--source", default="all", choices=["all", "tts", "human"])
    args = parser.parse_args()
    rows = load_manifest(args.source)
    with GpuMemorySampler() as gpu:
        engine = build_engine(args.engine)
        engine.warmup()
        result = run_engine(engine, rows)
    result["vram_delta_mb"] = gpu.delta_mb
    result["streaming_latency_ms"] = getattr(engine, "streaming_latency_ms", None)
    path = write_result(f"stt_{args.engine}", result, device=engine.device)
    print(f"{args.engine}: WER {result['wer_all']:.3f}  post-speech p50 {result['post_speech_ms']['p50']} ms  "
          f"p95 {result['post_speech_ms']['p95']} ms  VRAM +{gpu.delta_mb} MB -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test**

Run: `.venv/Scripts/python -m pytest tests/bench/test_stt_bench.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Run every engine (one process each, so VRAM is measured cleanly)**

```bash
for e in nemotron-80 nemotron-160 nemotron-560 parakeet-cpu whisper-distil-small; do
  .venv/Scripts/python -m bench.stt_bench --engine "$e" || echo "FAILED: $e"
done
```
Expected: one summary line per engine and `bench/results/stt_<engine>.json` files. If a Nemotron run fails on the `transformers` streaming API, check the installed version (`pip show transformers`, must be ≥ 5.13) and compare `NemotronStreaming.transcribe` against the "Streaming transcription" snippet on the model card; record any failure in the decision doc rather than dropping the engine silently. Sanity-check a few `clips[].hyp` values by eye — a WER near 1.0 means the engine is misconfigured, not that it is bad.

- [ ] **Step 6: Commit**

```bash
git add bench/stt_bench.py tests/bench/test_stt_bench.py bench/results/stt_*.json
git commit -m "Add STT benchmark and first results"
```

---

### Task 12: End-of-turn and TTS benchmarks

**Files:**
- Create: `bench/turn_bench.py`, `bench/tts_bench.py`, `tests/bench/test_turn_bench.py`

**Interfaces:**
- Consumes: `bench.common`, `bench.tts_engine.load_kokoro`, corpus manifest.
- Produces: `bench.turn_bench.SmartTurn(path).predict(audio) -> float` (completion probability); `bench.turn_bench.score(rows_with_probs, threshold=0.5) -> dict` with `accuracy`, `complete_recall`, `incomplete_recall`; results `bench/results/turn_smart-turn-v3.2.json` and `bench/results/tts_kokoro.json`.

- [ ] **Step 1: Write the failing scoring test**

`tests/bench/test_turn_bench.py`:
```python
from bench.turn_bench import score


def test_score_counts_per_class():
    rows = [
        {"label": "complete", "prob": 0.9}, {"label": "complete", "prob": 0.4},
        {"label": "incomplete", "prob": 0.1}, {"label": "incomplete", "prob": 0.2},
    ]
    s = score(rows)
    assert s == {"accuracy": 0.75, "complete_recall": 0.5, "incomplete_recall": 1.0, "n": 4}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/bench/test_turn_bench.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.turn_bench'`.

- [ ] **Step 3: Implement the turn benchmark**

`bench/turn_bench.py`:
```python
"""Smart Turn v3.2 benchmark (CPU). Usage: python -m bench.turn_bench [--source all]"""
from __future__ import annotations

import argparse
import time

import numpy as np

from bench.common import MODELS_DIR, load_manifest, load_wav16k, summarize, write_result

MODEL_PATH = MODELS_DIR / "smart-turn" / "smart-turn-v3.2-cpu.onnx"
WINDOW_S = 8


class SmartTurn:
    """Inference adapted from pipecat-ai/smart-turn inference.py."""

    def __init__(self, path=MODEL_PATH):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.features = WhisperFeatureExtractor(chunk_length=WINDOW_S)

    def predict(self, audio: np.ndarray) -> float:
        audio = audio[-WINDOW_S * 16000:]
        inputs = self.features(audio, sampling_rate=16000, return_tensors="np", padding="max_length",
                               max_length=WINDOW_S * 16000, truncation=True, do_normalize=True)
        features = inputs.input_features.squeeze(0).astype(np.float32)[None, ...]
        return float(self.session.run(None, {"input_features": features})[0][0].item())


def score(rows: list[dict], threshold: float = 0.5) -> dict:
    def recall(label: str) -> float:
        subset = [r for r in rows if r["label"] == label]
        hits = [(r["prob"] > threshold) == (label == "complete") for r in subset]
        return round(sum(hits) / len(hits), 3) if hits else None

    correct = [(r["prob"] > threshold) == (r["label"] == "complete") for r in rows]
    return {"accuracy": round(sum(correct) / len(correct), 3), "complete_recall": recall("complete"),
            "incomplete_recall": recall("incomplete"), "n": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all", choices=["all", "tts", "human"])
    args = parser.parse_args()
    model = SmartTurn()
    model.predict(np.zeros(16000, dtype=np.float32))  # warmup
    rows, latencies = [], []
    for row in load_manifest(args.source):
        audio = load_wav16k(row["path"])
        t0 = time.perf_counter()
        prob = model.predict(audio)
        latencies.append((time.perf_counter() - t0) * 1000)
        rows.append({"id": row["id"], "source": row["source"], "label": row["label"], "prob": round(prob, 4)})
    result = {
        "model": "smart-turn-v3.2-cpu",
        "latency_ms": summarize(latencies),
        "overall": score(rows),
        "by_source": {s: score([r for r in rows if r["source"] == s]) for s in dict.fromkeys(r["source"] for r in rows)},
        "note": "TTS clips are read with finished-sentence prosody; trust the 'human' scores for incomplete turns.",
        "clips": rows,
    }
    path = write_result("turn_smart-turn-v3.2", result, device="CPUExecutionProvider")
    print(f"smart-turn: p50 {result['latency_ms']['p50']} ms  acc {result['overall']['accuracy']} -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement the TTS benchmark**

`bench/tts_bench.py`:
```python
"""Kokoro TTS benchmark. Usage: python -m bench.tts_bench"""
from __future__ import annotations

import time

from bench.common import GpuMemorySampler, first_clause_end, summarize, write_result
from bench.tts_engine import load_kokoro

REPLIES = [
    "Hi, this is Asha calling from QuickMart about your grocery order. Am I speaking with Priya?",
    "Great, thanks. Your order has two Amul milk packets, eggs, and brown bread.",
    "The brown bread is out of stock today, but I can send whole wheat bread instead.",
    "Sure, I have a slot tomorrow between five and six in the evening. Shall I book it?",
    "That slot is full, but six to seven or four to five tomorrow are open.",
    "Done, your delivery is booked for tomorrow, five to six PM.",
    "Got it, I'll ask them to leave it with the security guard.",
    "No problem, I'll call you back in an hour.",
    "Sorry, could you say that again?",
    "Thank you, have a nice day.",
]
VOICE = "af_heart"


def main() -> None:
    with GpuMemorySampler() as gpu:
        kokoro, provider = load_kokoro()
        kokoro.create("Warming up the voice.", voice=VOICE, lang="en-us")
        first_audio, rtfs = [], []
        for reply in REPLIES:
            clause = reply[: first_clause_end(reply) or len(reply)]
            t0 = time.perf_counter()
            kokoro.create(clause, voice=VOICE, lang="en-us")
            first_audio.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            samples, sr = kokoro.create(reply, voice=VOICE, lang="en-us")
            rtfs.append((time.perf_counter() - t0) / (len(samples) / sr))
    result = {"model": "kokoro-v1.0", "voice": VOICE, "first_clause_audio_ms": summarize(first_audio),
              "rtf": summarize(rtfs), "vram_delta_mb": gpu.delta_mb}
    path = write_result("tts_kokoro", result, device=provider)
    print(f"kokoro ({provider}): first clause p50 {result['first_clause_audio_ms']['p50']} ms  "
          f"p95 {result['first_clause_audio_ms']['p95']} ms -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and both benchmarks**

```bash
.venv/Scripts/python -m pytest tests/bench -q
.venv/Scripts/python -m bench.turn_bench
.venv/Scripts/python -m bench.tts_bench
```
Expected: tests pass; `smart-turn: p50 <~10-30> ms ...` and `kokoro (CUDAExecutionProvider): first clause p50 ... ms`. If Kokoro reports `CPUExecutionProvider`, fix the onnxruntime-gpu install (Task 9 Step 1) before trusting the numbers.

- [ ] **Step 6: Commit**

```bash
git add bench/turn_bench.py bench/tts_bench.py tests/bench/test_turn_bench.py bench/results/turn_*.json bench/results/tts_*.json
git commit -m "Add end-of-turn and TTS benchmarks with results"
```

---

### Task 13: LLM benchmark

**Files:**
- Create: `bench/llm_bench.py`, `tests/bench/test_llm_bench.py`

**Interfaces:**
- Consumes: `voiceagent` services and `build_call_context` (Tasks 2–8), seeded `data/warehouse.db` (Task 3), `bench.common` (Task 9), a running `llama-server` on `127.0.0.1:8080` (Task 9 scripts).
- Produces: `parse_sse_line(line) -> dict | None`; `strip_think(text) -> str`; `SYSTEM_TEMPLATE`; `USER_TURNS`; CLI `python -m bench.llm_bench --model-name <label> [--reps 3]` writing `bench/results/llm_<label>.json` with per-turn `ttft_ms`, `first_clause_ms`, `first_sentence_ms`, `total_ms`, `tokens_per_s`, `reply`, and summaries split into `cold` (turn 1 of rep 1) and `warm` (all other turns).

- [ ] **Step 1: Write the failing parser tests**

`tests/bench/test_llm_bench.py`:
```python
from bench.llm_bench import parse_sse_line, strip_think


def test_parse_sse_line():
    assert parse_sse_line('data: {"choices": [{"delta": {"content": "Hi"}}]}') == {
        "choices": [{"delta": {"content": "Hi"}}]
    }
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line("") is None
    assert parse_sse_line(": keep-alive") is None


def test_strip_think():
    assert strip_think("<think>\n\n</think>\n\nHello Priya.") == "Hello Priya."
    assert strip_think("Hello.") == "Hello."
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/bench/test_llm_bench.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.llm_bench'`.

- [ ] **Step 3: Implement the LLM benchmark**

`bench/llm_bench.py`:
```python
"""LLM latency benchmark against llama-server (OpenAI-compatible).

Start the server first:  bash scripts/llama_server.sh models/llm/Qwen3.5-4B-Q4_K_M.gguf
Then:                    python -m bench.llm_bench --model-name qwen3.5-4b-q4km
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime

import httpx

from bench.common import GpuMemorySampler, ROOT, first_clause_end, first_sentence_end, summarize, write_result
from voiceagent.db.repository import Repository
from voiceagent.domain.context import build_call_context
from voiceagent.domain.inventory import InventoryService
from voiceagent.domain.scheduling import SchedulingService

URL = "http://127.0.0.1:8080/v1/chat/completions"
NOW = datetime(2026, 10, 12, 10, 0)
SYSTEM_TEMPLATE = """You are Asha, a friendly delivery-scheduling assistant calling customers on behalf of QuickMart.
You are speaking on a phone call. Reply in one or two short spoken sentences.
Never use lists, markdown, symbols or emojis. Say times the way people speak them.
Only mention items, quantities and delivery slots that appear in the context below.

{context}"""
USER_TURNS = [
    "Yes, this is {name}.",
    "Okay. Is everything in stock?",
    "Fine, go with what you suggested.",
    "Can you deliver it tomorrow after 5?",
    "Yes, book that one.",
    "Please leave it with the security guard.",
    "No, that's all. Thank you.",
]


def parse_sse_line(line: str) -> dict | None:
    if not line.startswith("data: "):
        return None
    payload = line[len("data: "):].strip()
    if payload == "[DONE]":
        return None
    return json.loads(payload)


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def pick_order(repo: Repository, inventory: InventoryService) -> int:
    for (order_id,) in repo.conn.execute("SELECT id FROM orders ORDER BY id"):
        if inventory.shortages(order_id):
            return order_id
    raise SystemExit("no order with a shortage in data/warehouse.db")


def stream_turn(client: httpx.Client, messages: list[dict]) -> dict:
    body = {"model": "local", "messages": messages, "stream": True, "temperature": 0.3, "max_tokens": 80,
            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    text, ttft, clause, sentence, timings = "", None, None, None, {}
    with client.stream("POST", URL, json=body, timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            event = parse_sse_line(line)
            if event is None:
                continue
            timings = event.get("timings", timings)
            for choice in event.get("choices", []):
                piece = (choice.get("delta") or {}).get("content") or ""
                if not piece:
                    continue
                now_ms = (time.perf_counter() - t0) * 1000
                text += piece
                visible = strip_think(text)
                if ttft is None and visible:
                    ttft = now_ms
                if clause is None and first_clause_end(visible):
                    clause = now_ms
                if sentence is None and first_sentence_end(visible):
                    sentence = now_ms
    total = (time.perf_counter() - t0) * 1000
    reply = strip_think(text)
    return {"ttft_ms": round(ttft or total, 1), "first_clause_ms": round(clause or total, 1),
            "first_sentence_ms": round(sentence or total, 1), "total_ms": round(total, 1),
            "tokens_per_s": round(timings.get("predicted_per_second", 0.0), 1),
            "prompt_tokens": timings.get("prompt_n"), "reply": reply}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="label for the result file, e.g. qwen3.5-4b-q4km")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--db", default=str(ROOT / "data" / "warehouse.db"))
    args = parser.parse_args()

    repo = Repository(args.db)
    inventory = InventoryService(repo)
    scheduling = SchedulingService(repo, inventory)
    ctx = build_call_context(repo, inventory, scheduling, pick_order(repo, inventory), NOW)
    system = SYSTEM_TEMPLATE.format(context=ctx.prompt_text)
    first_name = ctx.customer_name.split()[0]

    turns = []
    with GpuMemorySampler() as gpu, httpx.Client() as client:
        for rep in range(args.reps):
            messages = [{"role": "system", "content": system},
                        {"role": "assistant", "content": f"Hi, this is Asha from QuickMart. Am I speaking with {first_name}?"}]
            for index, user in enumerate(USER_TURNS):
                messages.append({"role": "user", "content": user.format(name=first_name)})
                turn = stream_turn(client, messages)
                messages.append({"role": "assistant", "content": turn["reply"]})
                turns.append({"rep": rep, "turn": index, **turn})
                print(f"[{rep}.{index}] ttft {turn['ttft_ms']:>6} ms  clause {turn['first_clause_ms']:>6} ms  "
                      f"{turn['tokens_per_s']:>5} tok/s  | {turn['reply']}")

    warm = [t for t in turns if not (t["rep"] == 0 and t["turn"] == 0)]
    result = {
        "model": args.model_name,
        "order_context": ctx.prompt_text,
        "cold_ttft_ms": turns[0]["ttft_ms"],
        "warm": {k: summarize([t[k] for t in warm]) for k in ("ttft_ms", "first_clause_ms", "first_sentence_ms",
                                                             "total_ms", "tokens_per_s")},
        "gpu_used_mb_peak": gpu.peak_mb,
        "turns": turns,
    }
    path = write_result(f"llm_{args.model_name}", result, device="cuda (llama-server -ngl 99)")
    print(f"{args.model_name}: warm TTFT p50 {result['warm']['ttft_ms']['p50']} ms, "
          f"first clause p50 {result['warm']['first_clause_ms']['p50']} ms -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit tests**

Run: `.venv/Scripts/python -m pytest tests/bench/test_llm_bench.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Benchmark each model (server in a second terminal, one model at a time)**

For each `(gguf, label)` in
`(Qwen3.5-4B-Q4_K_M.gguf, qwen3.5-4b-q4km)`, `(Qwen3.5-2B-Q4_K_M.gguf, qwen3.5-2b-q4km)`, `(Qwen3-4B-Instruct-2507-Q4_K_M.gguf, qwen3-4b-2507-q4km)`:

```bash
# terminal 2
bash scripts/llama_server.sh models/llm/<gguf>
# terminal 1, once the server logs "server is listening"
.venv/Scripts/python -m bench.llm_bench --model-name <label>
# then stop the server (Ctrl+C) before the next model
```
Expected: 21 turn lines per model plus a summary line; `bench/results/llm_<label>.json`. Read the printed replies: note any reply that mentions an item or slot not in `order_context` — that matters as much as latency for the decision.

- [ ] **Step 6: Commit**

```bash
git add bench/llm_bench.py tests/bench/test_llm_bench.py bench/results/llm_*.json
git commit -m "Add LLM streaming latency benchmark with results"
```

---

### Task 14: Results report and model decision

**Files:**
- Create: `bench/report.py`, `tests/bench/test_report.py`, `docs/benchmarks/phase1-results.md` (generated), `docs/benchmarks/phase1-decision.md`

**Interfaces:**
- Consumes: `bench/results/*.json` from Tasks 11–13.
- Produces: `render(results: list[dict]) -> str` (markdown), CLI `python -m bench.report` writing `docs/benchmarks/phase1-results.md`.

- [ ] **Step 1: Write the failing test**

`tests/bench/test_report.py`:
```python
from bench.report import render


def test_render_has_a_row_per_result():
    results = [
        {"name": "stt_parakeet-cpu", "device": "cpu", "engine": "parakeet-cpu", "wer_all": 0.081,
         "wer_by_source": {"tts": 0.081}, "post_speech_ms": {"p50": 90.0, "p95": 140.0},
         "rtf": {"mean": 0.05}, "vram_delta_mb": 0},
        {"name": "tts_kokoro", "device": "CUDAExecutionProvider", "model": "kokoro-v1.0",
         "first_clause_audio_ms": {"p50": 60.0, "p95": 80.0}, "rtf": {"mean": 0.04}, "vram_delta_mb": 500},
        {"name": "turn_smart-turn-v3.2", "device": "CPUExecutionProvider", "model": "smart-turn-v3.2-cpu",
         "latency_ms": {"p50": 12.0, "p95": 15.0}, "overall": {"accuracy": 0.9},
         "by_source": {"tts": {"accuracy": 0.9, "incomplete_recall": 0.6}}},
        {"name": "llm_qwen3.5-4b-q4km", "device": "cuda", "model": "qwen3.5-4b-q4km", "cold_ttft_ms": 400.0,
         "warm": {"ttft_ms": {"p50": 80.0, "p95": 120.0}, "first_clause_ms": {"p50": 200.0, "p95": 300.0},
                  "tokens_per_s": {"p50": 70.0}}, "gpu_used_mb_peak": 5000},
    ]
    md = render(results)
    assert "| parakeet-cpu | cpu | 8.1% |" in md
    assert "| kokoro-v1.0 | CUDAExecutionProvider | 60.0 | 80.0 |" in md
    assert "| smart-turn-v3.2-cpu | 12.0 | 15.0 | 0.9 |" in md
    assert "| qwen3.5-4b-q4km | 400.0 | 80.0 | 120.0 | 200.0 | 300.0 | 70.0 |" in md
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/bench/test_report.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.report'`.

- [ ] **Step 3: Implement the report**

`bench/report.py`:
```python
"""Render bench/results/*.json to docs/benchmarks/phase1-results.md. Usage: python -m bench.report"""
from __future__ import annotations

import json

from bench.common import RESULTS_DIR, ROOT

OUT = ROOT / "docs" / "benchmarks" / "phase1-results.md"


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render(results: list[dict]) -> str:
    by_kind = lambda prefix: sorted((r for r in results if r["name"].startswith(prefix)), key=lambda r: r["name"])
    out = ["# Phase 1 benchmark results", "", "Generated by `python -m bench.report` from `bench/results/`.", ""]

    out += ["## STT", "", "| engine | device | WER all | WER human | post-speech p50 ms | p95 ms | RTF mean | VRAM +MB |",
            "|---|---|---|---|---|---|---|---|"]
    for r in by_kind("stt_"):
        out.append(f"| {r['engine']} | {r['device']} | {_pct(r['wer_all'])} | {_pct(r['wer_by_source'].get('human'))} "
                   f"| {r['post_speech_ms']['p50']} | {r['post_speech_ms']['p95']} | {r['rtf']['mean']} "
                   f"| {r.get('vram_delta_mb')} |")

    out += ["", "## End of turn", "", "| model | p50 ms | p95 ms | accuracy | human incomplete recall |",
            "|---|---|---|---|---|"]
    for r in by_kind("turn_"):
        human = r["by_source"].get("human", {}).get("incomplete_recall", "n/a")
        out.append(f"| {r['model']} | {r['latency_ms']['p50']} | {r['latency_ms']['p95']} "
                   f"| {r['overall']['accuracy']} | {human} |")

    out += ["", "## TTS", "", "| model | device | first clause audio p50 ms | p95 ms | RTF mean | VRAM +MB |",
            "|---|---|---|---|---|---|"]
    for r in by_kind("tts_"):
        out.append(f"| {r['model']} | {r['device']} | {r['first_clause_audio_ms']['p50']} "
                   f"| {r['first_clause_audio_ms']['p95']} | {r['rtf']['mean']} | {r.get('vram_delta_mb')} |")

    out += ["", "## LLM", "",
            "| model | cold TTFT ms | warm TTFT p50 | p95 | first clause p50 | p95 | tok/s p50 | GPU used peak MB |",
            "|---|---|---|---|---|---|---|---|"]
    for r in by_kind("llm_"):
        w = r["warm"]
        out.append(f"| {r['model']} | {r['cold_ttft_ms']} | {w['ttft_ms']['p50']} | {w['ttft_ms']['p95']} "
                   f"| {w['first_clause_ms']['p50']} | {w['first_clause_ms']['p95']} | {w['tokens_per_s']['p50']} "
                   f"| {r.get('gpu_used_mb_peak')} |")
    return "\n".join(out) + "\n"


def main() -> None:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("*.json"))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(results), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test and generate the report**

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m bench.report
```
Expected: all tests pass; `wrote .../docs/benchmarks/phase1-results.md`.

- [ ] **Step 5: Write the decision doc from the numbers**

Create `docs/benchmarks/phase1-decision.md` using these rules, quoting the numbers from `phase1-results.md`:

- **STT:** among engines whose overall WER is within 2 percentage points of the best (use human-source WER instead if human clips were recorded), choose the lowest post-speech p50. Engines whose RTF mean is ≥ 0.5 are rejected (they would not keep up during speech).
- **LLM:** among models whose replies contained no invented items or slots, choose the one with the lowest warm first-clause p50; prefer the 2B only if its replies were as correct as the 4B's.
- **End-of-turn:** keep Smart Turn v3.2 if p95 ≤ 50 ms; note the human incomplete-recall figure (or "not measured").
- **Budget check:** add the chosen p50s: 200 ms VAD silence + Smart Turn p50 + STT post-speech p50 + LLM warm first-clause p50 + Kokoro first-clause p50. State whether that sum is ≤ 700 ms, and the VRAM total of the chosen STT + LLM + TTS against 8 GB minus the display's usage.

Template:
```markdown
# Phase 1 model decision

Date: <YYYY-MM-DD>   Hardware: RTX 4060 Laptop 8 GB, Windows 11

| Stage | Choice | Key numbers | Rejected (why) |
|---|---|---|---|
| STT | | WER, post-speech p50/p95, VRAM | |
| End of turn | Smart Turn v3.2 (CPU) | p50/p95, accuracy | |
| LLM | | warm TTFT p50, first clause p50, tok/s | |
| TTS | Kokoro v1.0 | first clause p50/p95, provider | |

Estimated voice-to-voice p50: 200 + <turn> + <stt> + <llm clause> + <tts> = <sum> ms (target ≤ 700 ms).
Estimated VRAM: <stt> + <llm> + <tts> = <sum> MB of <available> MB.

Observations: <notable failures, hallucinated replies, surprises>
```

- [ ] **Step 6: Commit**

```bash
git add bench/report.py tests/bench/test_report.py docs/benchmarks
git commit -m "Add benchmark report and phase 1 model decision"
git push origin main
```

---

## Later plans (not part of this plan)

- **Plan 2 — Pipeline and demo:** Pipecat 1.x pipeline (SmallWebRTC, Silero VAD, Smart Turn v3.2, chosen STT, llama-server LLM, Kokoro), Flow nodes from the spec with tools wired to the services built here, latency observer writing `turn_metrics`, browser client, `calls` persistence.
- **Plan 3 — Evaluation:** text-only scenario harness (A+B+C) with DB-state assertions, then the audio end-to-end harness.
- **Plan 4 — Latency work:** preemptive generation, prompt-cache tuning, clause chunking, backchannel guard tuning.
- **Plan 5 — Fine-tuning:** synthetic dialogue generation, QLoRA, GGUF quantization, results table, cost model.
- **Branch `stack-jul2025`:** rebuild Plan 2 with the July-2025 stack after `main` works.
