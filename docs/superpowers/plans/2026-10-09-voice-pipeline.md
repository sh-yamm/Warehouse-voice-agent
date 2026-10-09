# Voice Pipeline (Live Browser Calls) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A caller agent you can talk to in the browser. It greets the customer, walks through their order, resolves shortages, books a delivery slot through the Plan 1 domain services, and handles interruptions. Every turn's latency and every call's outcome is recorded in SQLite and shown on a small dashboard.

**Architecture:** A Pipecat 1.12 pipeline (SmallWebRTC transport → Silero VAD + Smart Turn v3.2 → greedy faster-whisper → llama-server LLM → Kokoro PyTorch TTS) is driven by a Pipecat Flows conversation graph. Each node exposes 2–4 tools. The tools live in a framework-free `CallSession` that calls the Plan 1 services, so all business rules stay deterministic and unit-tested. Observers write per-turn latency breakdowns and barge-in latency to new `calls` / `turn_metrics` tables, which a FastAPI dashboard reads.

**Tech Stack:** Python 3.11, pipecat-ai 1.12.0 (`webrtc`, `runner`, `whisper` extras; Flows is built in as `pipecat.flows`), faster-whisper distil-small.en, `kokoro` 0.9.4 (PyTorch), llama.cpp `llama-server` with Qwen3.5-2B Q4_K_M, FastAPI + uvicorn, pytest.

**Spec:** `docs/specs/2026-10-09-voice-caller-agent-design.md`. Model choices: `docs/benchmarks/phase1-decision.md`.

## Global Constraints

- Python `>=3.11,<3.13`; venv Python is `.venv/Scripts/python`; commands run from the repo root in Git Bash.
- Zero spend, all inference local. The LLM is reached only at `http://127.0.0.1:8080/v1` (llama-server).
- English only. STT, LLM and TTS are constructed in one place (`voiceagent/agent/services.py`) so they can be swapped.
- Business rules (stock, slots, holds, cancellation) run only in the Plan 1 services. Agent code never writes SQL and never decides availability itself.
- The agent never states an item, quantity, price or slot that is not in the call facts or a tool result. Enforced by the role prompt and by tools returning facts.
- Responses are one or two short spoken sentences, with no markdown, lists or symbols.
- Slots are held for 5 minutes when offered and committed only by `confirm_booking`.
- Commit messages contain no co-author or tool-attribution trailers.
- Model files and `data/` stay git-ignored.

## Review Focus

1. **The LLM passes an invented `slot_id` or `sku`.** Tools must reject it with no database change. Pinned by `test_choose_slot_rejects_unknown_and_full_slots` and `test_resolve_shortage_unknown_sku_fails` (Task 2).
2. **The LLM passes `quantity` as `"2"` or `"two"`.** The first is coerced, the second returns an error rather than crashing the call. Pinned by `test_add_item_coerces_and_rejects_quantity` (Task 2).
3. **The customer takes longer than 5 minutes between hearing the slot and saying yes.** `confirm_booking` must fail and the flow must return to scheduling. Pinned by `test_confirm_after_hold_expiry_returns_to_schedule` (Task 3).
4. **The customer hangs up mid-call.** The call row is closed as `abandoned` with whatever transcript exists. Pinned by `test_finish_without_outcome_is_abandoned` (Task 5).
5. **The customer asks for a time twice.** Only one hold may exist for the order (the second check moves it). Pinned by `test_check_slot_twice_keeps_single_hold` (Task 2).

---

## File Structure

```
pyproject.toml                          + "agent" extra
src/voiceagent/db/schema.sql            + calls, turn_metrics tables
src/voiceagent/db/repository.py         + start_call, add_turn_metric, finish_call, get_call, list_calls, call_metrics
src/voiceagent/agent/__init__.py
src/voiceagent/agent/tools.py           CallSession: JSON-returning tools over Plan 1 services
src/voiceagent/agent/flow.py            DeliveryFlow: Pipecat Flows nodes (spec §3)
src/voiceagent/agent/services.py        GreedyWhisperSTTService, KokoroTorchTTSService, make_llm
src/voiceagent/agent/metrics.py         BargeInTimer, BargeInObserver, CallRecorder
src/voiceagent/agent/dashboard.py       FastAPI dashboard (calls, per-turn latency, transcript)
src/voiceagent/agent/bot.py             pipeline wiring + runner entry point
scripts/smoke_tts.py                    one-off GPU check of KokoroTorchTTSService
tests/domain/test_calls_repository.py
tests/agent/test_tools.py
tests/agent/test_flow.py
tests/agent/test_services.py
tests/agent/test_metrics.py
tests/agent/test_dashboard.py
tests/agent/test_bot.py
docs/benchmarks/phase2-live-calls.md    filled from real calls (Task 8)
```

---

### Task 1: Agent environment and call persistence

**Files:**
- Modify: `pyproject.toml`, `src/voiceagent/db/schema.sql`, `src/voiceagent/db/repository.py`
- Create: `tests/domain/test_calls_repository.py`

**Interfaces:**
- Consumes: `Repository`, `build_world` fixtures from Plan 1.
- Produces:
  - `Repository.start_call(order_id, started_at) -> int`
  - `Repository.add_turn_metric(call_id, kind, total_ms, breakdown: dict, recorded_at) -> None`, where `kind` is `"response" | "greeting" | "barge_in"`
  - `Repository.finish_call(call_id, ended_at, outcome, transcript: list[dict]) -> None`
  - `Repository.get_call(call_id) -> dict | None`, with keys `id, order_id, customer, started_at, ended_at, outcome, transcript`
  - `Repository.list_calls(limit=20) -> list[dict]`: newest first, same keys minus `transcript`
  - `Repository.call_metrics(call_id) -> list[dict]`, with keys `kind, total_ms, breakdown, recorded_at`, in insertion order

- [ ] **Step 1: Install the agent dependencies**

Add to `[project.optional-dependencies]` in `pyproject.toml`:
```toml
agent = [
  "pipecat-ai[webrtc,runner,whisper]==1.12.0",
  "kokoro>=0.9.4",
  "fastapi>=0.115",
  "uvicorn>=0.32",
]
```
Also change `pythonpath` under `[tool.pytest.ini_options]` to `["src", ".", "tests"]`, so tests in `tests/agent/` can `from conftest import NOW, build_world`.

Then:
```bash
.venv/Scripts/python -m pip install -e ".[dev,bench,agent]"
.venv/Scripts/python -m pip uninstall -y onnxruntime-gpu
.venv/Scripts/python -m pip install --force-reinstall --no-deps "onnxruntime~=1.24.3"
.venv/Scripts/python -c "import pipecat, onnxruntime, torch; from pipecat.flows import FlowManager; print(pipecat.__version__, onnxruntime.__version__, torch.cuda.is_available())"
```
Expected: `1.12.0 1.24.x True`. Pipecat pins CPU `onnxruntime`, and both packages provide the same `onnxruntime` module, so the GPU build is removed. Nothing in the live pipeline needs it: Smart Turn and Silero run on CPU, and Kokoro runs on PyTorch. `python -m bench.tts_bench --engine onnx` will now run on CPU.

- [ ] **Step 2: Write the failing tests**

`tests/domain/test_calls_repository.py`:
```python
from datetime import timedelta


def test_call_round_trip(world, repo, now):
    call_id = repo.start_call(1, now)
    repo.finish_call(call_id, now + timedelta(minutes=2), "scheduled",
                     [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Yes"}])
    call = repo.get_call(call_id)
    assert call["customer"] == "Priya Sharma"
    assert call["outcome"] == "scheduled"
    assert call["ended_at"] == now + timedelta(minutes=2)
    assert call["transcript"][1] == {"role": "user", "content": "Yes"}


def test_get_call_unknown(world, repo):
    assert repo.get_call(999) is None


def test_list_calls_newest_first(world, repo, now):
    first = repo.start_call(1, now)
    second = repo.start_call(2, now + timedelta(minutes=1))
    calls = repo.list_calls()
    assert [c["id"] for c in calls] == [second, first]
    assert calls[0]["customer"] == "Arjun Rao" and calls[0]["outcome"] is None
    assert "transcript" not in calls[0]


def test_turn_metrics_in_order_with_parsed_breakdown(world, repo, now):
    call_id = repo.start_call(1, now)
    repo.add_turn_metric(call_id, "greeting", 820.0, {"contributions": []}, now)
    repo.add_turn_metric(call_id, "response", 912.5, {"contributions": [["llm", "LLM inference", 300.0]]}, now)
    repo.add_turn_metric(call_id, "barge_in", 140.0, {}, now)
    metrics = repo.call_metrics(call_id)
    assert [m["kind"] for m in metrics] == ["greeting", "response", "barge_in"]
    assert metrics[1]["total_ms"] == 912.5
    assert metrics[1]["breakdown"]["contributions"][0][2] == 300.0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/domain/test_calls_repository.py -q`
Expected: FAIL with `AttributeError: 'Repository' object has no attribute 'start_call'`.

- [ ] **Step 4: Implement**

Append to `src/voiceagent/db/schema.sql`:
```sql
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    outcome TEXT,
    transcript_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS turn_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id INTEGER NOT NULL REFERENCES calls(id),
    kind TEXT NOT NULL,
    total_ms REAL NOT NULL,
    breakdown_json TEXT NOT NULL DEFAULT '{}',
    recorded_at TEXT NOT NULL
);
```

In `src/voiceagent/db/repository.py`, add `import json` to the imports and append these methods to `Repository`:
```python
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
```

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass (103 from Plan 1 + 4 new).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/voiceagent/db/schema.sql src/voiceagent/db/repository.py tests/domain/test_calls_repository.py
git commit -m "Add agent dependencies and call/turn-metric persistence"
```

---

### Task 2: Call tools (`CallSession`)

**Files:**
- Create: `src/voiceagent/agent/__init__.py` (empty), `src/voiceagent/agent/tools.py`, `tests/agent/test_tools.py`

**Interfaces:**
- Consumes: `InventoryService`, `SchedulingService`, `OrderService`, `build_call_context`, `spoken_slot`, `spoken_day`, `spoken_time`, `parse_time_window` (Plan 1).
- Produces: `CallSession(repo, order_id, clock=datetime.now)` with attributes `repo, order_id, clock, inventory, scheduling, orders, outcome: str | None, held_slot_id: int | None`. Every tool returns a JSON-serializable dict with a `status` key:
  - `context() -> CallContext`
  - `check_slot(preferred_time: str)`, with status `held` (`slot_id`, `slot`), or `full` / `no_slots` (`alternatives`: list of `{slot_id, slot}`), or `unclear` (`hint`)
  - `choose_slot(slot_id)`, with status `held` (`slot_id`, `slot`) or `unavailable` (`alternatives`)
  - `confirm_booking()`, with status `booked` (`slot`) or `failed` (`hint`). Sets `outcome = "scheduled"`.
  - `add_item(product_name, quantity)`, with status `added` / `insufficient` (`item`, `quantity`, `available`, `unit_price`), `not_found`, or `error` (`message`)
  - `resolve_shortage(sku, choice, substitute_sku="")`, with status `done` / `failed` (`remaining_shortages`, optional `deliver_after`) or `error`
  - `update_address(address)`, with status `updated` (`address`) or `error`; `add_note(note)`, with status `saved` or `error`
  - `cancel_order()`, with status `cancelled` (sets `outcome`) or `failed`
  - `schedule_callback(when)`, with status `scheduled` (`callback_at`). Sets `outcome = "callback"`.
  - `wrong_person()`, with status `noted`. Sets `outcome = "wrong_person"`.
  - `order_summary()`, returning `{items: list[str], address, notes, slot: str | None}`

- [ ] **Step 1: Write the failing tests**

`tests/agent/test_tools.py`:
```python
import pytest

from voiceagent.agent.tools import CallSession

from conftest import NOW


@pytest.fixture
def session(world, repo):
    return CallSession(repo, 1, clock=lambda: NOW)


def test_check_slot_holds_available_slot(session, repo):
    result = session.check_slot("tomorrow after 5")
    assert result == {"status": "held", "slot_id": 23, "slot": "tomorrow, 5 to 6 PM"}
    assert session.held_slot_id == 23
    assert repo.get_active_hold(1, NOW) == 23


def test_check_slot_full_offers_alternatives(session):
    result = session.check_slot("today at 5 pm")
    assert result["status"] == "full"
    assert [a["slot"] for a in result["alternatives"]] == [
        "today, 3 to 4 PM", "today, 4 to 5 PM", "today, 6 to 7 PM"]


def test_check_slot_unclear(session):
    assert session.check_slot("I don't know")["status"] == "unclear"


def test_check_slot_twice_keeps_single_hold(session, repo):
    session.check_slot("tomorrow after 5")
    session.check_slot("tomorrow morning")
    assert repo.active_holds(23, NOW) == 0
    assert repo.get_active_hold(1, NOW) == session.held_slot_id == 14


def test_choose_slot(session):
    assert session.choose_slot(9) == {"status": "held", "slot_id": 9, "slot": "today, 4 to 5 PM"}


def test_choose_slot_rejects_unknown_and_full_slots(session, repo):
    for slot_id in (10, 999):
        result = session.choose_slot(slot_id)
        assert result["status"] == "unavailable"
        assert len(result["alternatives"]) == 3
    assert repo.get_active_hold(1, NOW) is None


def test_confirm_booking(session, repo):
    assert session.confirm_booking()["status"] == "failed"
    session.check_slot("tomorrow after 5")
    assert session.confirm_booking() == {"status": "booked", "slot": "tomorrow, 5 to 6 PM"}
    assert session.outcome == "scheduled"
    assert repo.get_order(1).status == "scheduled"


def test_add_item(session, repo):
    added = session.add_item("nandini milk", 2)
    assert (added["status"], added["item"], added["available"]) == ("added", "Nandini milk", 3)
    short = session.add_item("basmati rice", 5)
    assert (short["status"], short["available"]) == ("insufficient", 2)
    assert session.add_item("toothpaste", 1) == {"status": "not_found"}


def test_add_item_coerces_and_rejects_quantity(session):
    assert session.add_item("amul milk", "2")["status"] == "added"
    assert session.add_item("amul milk", "two")["status"] == "error"
    assert session.add_item("amul milk", 0)["status"] == "error"


def test_resolve_shortage_substitute(session):
    assert session.resolve_shortage("BREAD1", "substitute", "BREAD2") == {
        "status": "done", "remaining_shortages": []}


def test_resolve_shortage_wait_reports_restock(session):
    result = session.resolve_shortage("BREAD1", "wait")
    assert result["status"] == "done" and result["deliver_after"] == "tomorrow 9 AM"


def test_resolve_shortage_bad_choice(session):
    assert session.resolve_shortage("BREAD1", "teleport")["status"] == "error"


def test_resolve_shortage_unknown_sku_fails(session, repo):
    assert session.resolve_shortage("NOPE", "partial")["status"] == "failed"
    assert [l.resolution for l in repo.get_order_lines(1)] == [None, None, None]


def test_update_address_and_note(session, repo):
    assert session.update_address("x")["status"] == "error"
    assert session.update_address("42 Church Street, Bengaluru") == {
        "status": "updated", "address": "42 Church Street, Bengaluru"}
    assert session.add_note("leave with the security guard") == {"status": "saved"}
    assert repo.get_order(1).notes == "leave with the security guard"
    assert session.add_note("  ")["status"] == "error"


def test_cancel_order(session):
    assert session.cancel_order() == {"status": "cancelled"}
    assert session.outcome == "cancelled"
    assert session.cancel_order() == {"status": "failed"}


def test_schedule_callback(session, repo):
    assert session.schedule_callback("tomorrow morning") == {"status": "scheduled", "callback_at": "tomorrow 8 AM"}
    assert session.outcome == "callback"
    assert session.schedule_callback("")["callback_at"] == "today 11 AM"


def test_wrong_person(session, repo):
    assert session.wrong_person() == {"status": "noted"}
    assert session.outcome == "wrong_person"


def test_order_summary(session):
    assert session.order_summary()["items"] == ["2 Amul milk", "1 brown bread, not available yet", "1 eggs"]
    session.resolve_shortage("BREAD1", "substitute", "BREAD2")
    session.check_slot("tomorrow after 5")
    summary = session.order_summary()
    assert summary["items"][1] == "1 whole wheat bread instead of brown bread"
    assert summary["slot"] == "tomorrow, 5 to 6 PM"
    assert summary["address"] == "12 MG Road, Bengaluru"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/agent/test_tools.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.agent'`.

- [ ] **Step 3: Implement**

`src/voiceagent/agent/__init__.py`: empty file.

`src/voiceagent/agent/tools.py`:
```python
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from voiceagent.db.repository import Repository
from voiceagent.domain.context import CallContext, build_call_context, spoken_day, spoken_slot, spoken_time
from voiceagent.domain.inventory import InventoryService
from voiceagent.domain.orders import OrderService
from voiceagent.domain.scheduling import SchedulingService, SlotOption
from voiceagent.domain.timeparse import parse_time_window

SHORTAGE_CHOICES = ("substitute", "partial", "wait")


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

    def choose_slot(self, slot_id: int) -> dict:
        now = self.clock()
        slot = self.repo.get_slot(slot_id)
        if slot is None or not self.scheduling.hold_slot(self.order_id, slot_id, now):
            return {"status": "unavailable",
                    "alternatives": self._alternatives(self.scheduling.free_slots(self.order_id, now, limit=3))}
        self.held_slot_id = slot_id
        return {"status": "held", "slot_id": slot_id, "slot": spoken_slot(slot, now)}

    def confirm_booking(self) -> dict:
        slot = self.scheduling.confirm_booking(self.order_id, self.clock())
        if slot is None:
            return {"status": "failed", "hint": "The slot hold expired or no slot was chosen. Check the time again."}
        self.outcome = "scheduled"
        self.held_slot_id = None
        return {"status": "booked", "slot": spoken_slot(slot, self.clock())}

    # ── items ──────────────────────────────────────────────────────────────
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
    def update_address(self, address: str) -> dict:
        try:
            self.orders.update_address(self.order_id, address)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "updated", "address": self.repo.get_order(self.order_id).address}

    def add_note(self, note: str) -> dict:
        try:
            self.orders.add_note(self.order_id, note)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "saved"}

    def cancel_order(self) -> dict:
        ok = self.orders.cancel_order(self.order_id)
        if ok:
            self.outcome = "cancelled"
            self.held_slot_id = None
        return {"status": "cancelled" if ok else "failed"}

    def schedule_callback(self, when: str) -> dict:
        now = self.clock()
        window = parse_time_window(when, now) if when and when.strip() else None
        at = window.start if window else now + timedelta(hours=1)
        self.orders.schedule_callback(self.order_id, at)
        self.outcome = "callback"
        return {"status": "scheduled", "callback_at": f"{spoken_day(at.date(), now)} {spoken_time(at)}"}

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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/agent/test_tools.py -q`
Expected: `18 passed`. If an expected slot id or spoken string differs, recheck it against the fixture (slot ids: today 08:00 = 1 … 20:00 = 13, tomorrow 08:00 = 14 … 20:00 = 26) before changing code.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/agent/__init__.py src/voiceagent/agent/tools.py tests/agent/test_tools.py
git commit -m "Add CallSession tools over the domain services"
```

---

### Task 3: Conversation graph (`DeliveryFlow`)

**Files:**
- Create: `src/voiceagent/agent/flow.py`, `tests/agent/test_flow.py`

**Interfaces:**
- Consumes: `CallSession` (Task 2); `pipecat.flows.FlowManager`, `NodeConfig` (a `TypedDict`, so nodes are dicts).
- Produces: `DeliveryFlow(session)` with attributes `session, customer_name, first_name, role, global_functions` (async functions `cancel_order`, `callback_later`, `add_item`) and methods `greeting() -> str`, `greet_node()`, `order_node()`, `schedule_node()`, `details_node()`, `confirm_node()`, `close_node(instruction)`, each returning a `NodeConfig`. Node names are `greet`, `present_order`, `schedule`, `details`, `confirm` and `close`. Node functions are Flows "direct functions": `async def f(flow_manager, ...) -> tuple[dict, NodeConfig | None]`, documented in Google style so Flows can build the tool schema.

- [ ] **Step 1: Write the failing tests**

`tests/agent/test_flow.py`:
```python
import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest

from voiceagent.agent.flow import DeliveryFlow
from voiceagent.agent.tools import CallSession

from conftest import NOW


@pytest.fixture
def clock():
    return {"now": NOW}


@pytest.fixture
def flow(world, repo, clock):
    return DeliveryFlow(CallSession(repo, 1, clock=lambda: clock["now"]))


def call(node_or_list, name, **kwargs):
    functions = node_or_list["functions"] if isinstance(node_or_list, dict) else node_or_list
    fn = next(f for f in functions if f.__name__ == name)
    return asyncio.run(fn(SimpleNamespace(state={}), **kwargs))


def names(node):
    return {f.__name__ for f in node["functions"]}


def test_role_carries_facts_and_rules(flow):
    assert "CALL FACTS" in flow.role
    assert "- 2 x Amul milk [MILK1]: ready" in flow.role
    assert "one or two short spoken sentences" in flow.role


def test_greet_node_speaks_fixed_greeting_then_waits(flow):
    node = flow.greet_node()
    assert node["name"] == "greet"
    assert node["respond_immediately"] is False
    assert node["pre_actions"][0]["type"] == "tts_say"
    assert "Priya" in node["pre_actions"][0]["text"]
    assert names(node) == {"confirm_identity", "wrong_person"}
    assert node["role_message"] == flow.role


def test_global_functions(flow):
    assert {f.__name__ for f in flow.global_functions} == {"cancel_order", "callback_later", "add_item"}


def test_confirm_identity_moves_to_order(flow):
    result, node = call(flow.greet_node(), "confirm_identity")
    assert node["name"] == "present_order"


def test_wrong_person_closes(flow):
    result, node = call(flow.greet_node(), "wrong_person")
    assert node["name"] == "close" and flow.session.outcome == "wrong_person"
    assert node["post_actions"] == [{"type": "end_conversation"}]


def test_items_confirmed_blocked_by_shortage(flow):
    result, node = call(flow.order_node(), "items_confirmed")
    assert result == {"status": "shortage_unresolved", "skus": ["BREAD1"]} and node is None


def test_resolving_last_shortage_moves_to_schedule(flow):
    result, node = call(flow.order_node(), "resolve_shortage", sku="BREAD1", choice="substitute",
                        substitute_sku="BREAD2")
    assert result["status"] == "done" and node["name"] == "schedule"


def test_schedule_then_details_then_confirm(flow):
    schedule = flow.schedule_node()
    result, node = call(schedule, "slot_agreed")
    assert result == {"status": "no_slot_held"} and node is None
    result, node = call(schedule, "check_slot", preferred_time="tomorrow after 5")
    assert result["status"] == "held" and node is None
    result, details = call(schedule, "slot_agreed")
    assert details["name"] == "details"
    assert "12 MG Road, Bengaluru" in details["task_messages"][0]["content"]
    result, confirm = call(details, "details_done")
    assert confirm["name"] == "confirm"
    assert "tomorrow, 5 to 6 PM" in confirm["task_messages"][0]["content"]


def test_confirm_booking_closes_call(flow, repo):
    call(flow.schedule_node(), "check_slot", preferred_time="tomorrow after 5")
    result, node = call(flow.confirm_node(), "confirm_booking")
    assert result["status"] == "booked" and node["name"] == "close"
    assert repo.get_order(1).status == "scheduled"


def test_confirm_after_hold_expiry_returns_to_schedule(flow, clock):
    call(flow.schedule_node(), "check_slot", preferred_time="tomorrow after 5")
    clock["now"] = NOW + timedelta(minutes=6)
    result, node = call(flow.confirm_node(), "confirm_booking")
    assert result["status"] == "failed" and node["name"] == "schedule"


def test_global_cancel_and_callback_close(flow):
    result, node = call(flow.global_functions, "callback_later", when="tomorrow morning")
    assert result["callback_at"] == "tomorrow 8 AM" and node["name"] == "close"
    result, node = call(flow.global_functions, "cancel_order")
    assert node["name"] == "close"


def test_global_add_item_stays_in_node(flow):
    result, node = call(flow.global_functions, "add_item", product_name="nandini milk", quantity=1)
    assert result["status"] == "added" and node is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/agent/test_flow.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.agent.flow'`.

- [ ] **Step 3: Implement**

`src/voiceagent/agent/flow.py`:
```python
from __future__ import annotations

from pipecat.flows import FlowManager, NodeConfig

from voiceagent.agent.tools import CallSession

AGENT_NAME = "Asha"
COMPANY = "QuickMart"

ROLE_TEMPLATE = """You are {agent}, calling customers on behalf of {company} to schedule delivery of their grocery order.
You are on a live phone call and everything you write is spoken aloud.
Rules:
- Reply in one or two short spoken sentences. No lists, markdown, symbols or emojis.
- Say times the way people speak them, for example "five to six PM".
- Only state items, quantities, prices, slots and dates that appear in the call facts below or in a function result. Never guess.
- Use the functions for every action: checking or booking slots, changing the order, cancelling, or arranging a callback.
- If the customer asks about something unrelated to this delivery, say you can only help with this delivery.

CALL FACTS
{facts}"""


class DeliveryFlow:
    """The call's conversation graph (spec section 3), built on a CallSession."""

    def __init__(self, session: CallSession):
        self.session = session
        ctx = session.context()
        self.customer_name = ctx.customer_name
        self.first_name = ctx.customer_name.split()[0]
        self.role = ROLE_TEMPLATE.format(agent=AGENT_NAME, company=COMPANY, facts=ctx.prompt_text)
        self.global_functions = self._global_functions()

    def greeting(self) -> str:
        return (f"Hi, this is {AGENT_NAME} calling from {COMPANY} about your grocery order. "
                f"Am I speaking with {self.first_name}?")

    def _node(self, name: str, task: str, functions: list, **extra) -> NodeConfig:
        return NodeConfig(name=name, role_message=self.role,
                          task_messages=[{"role": "developer", "content": task}], functions=functions, **extra)

    def close_node(self, instruction: str) -> NodeConfig:
        return self._node("close", instruction, [], post_actions=[{"type": "end_conversation"}])

    def _global_functions(self) -> list:
        session = self.session

        async def cancel_order(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """Cancel the whole order. Use only when the customer clearly asks to cancel."""
            return session.cancel_order(), self.close_node(
                "The order is cancelled. Confirm the cancellation and say goodbye in one sentence.")

        async def callback_later(flow_manager: FlowManager, when: str) -> tuple[dict, NodeConfig]:
            """Arrange a callback because the customer cannot talk now.

            Args:
                when (str): When to call back in the customer's words, for example "tomorrow morning". Use "" if they did not say.
            """
            return session.schedule_callback(when), self.close_node(
                "A callback is arranged. Tell the customer when you will call back and say goodbye.")

        async def add_item(flow_manager: FlowManager, product_name: str, quantity: int) -> tuple[dict, None]:
            """Add a product to the order when the customer asks for something extra.

            Args:
                product_name (str): The product the customer asked for, in their words.
                quantity (int): How many units to add.
            """
            return session.add_item(product_name, quantity), None

        return [cancel_order, callback_later, add_item]

    def greet_node(self) -> NodeConfig:
        session = self.session

        async def confirm_identity(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The person on the phone confirmed they are the customer."""
            return {"status": "confirmed"}, self.order_node()

        async def wrong_person(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The person on the phone is not the customer."""
            return session.wrong_person(), self.close_node(
                "You reached the wrong person. Apologize briefly, say you will try again later, and say goodbye.")

        task = (f'You already said: "{self.greeting()}" Wait for the answer. '
                f"If they confirm they are {self.customer_name}, call confirm_identity. "
                "If not, call wrong_person. If they cannot talk now, call callback_later.")
        return self._node("greet", task, [confirm_identity, wrong_person],
                          pre_actions=[{"type": "tts_say", "text": self.greeting()}], respond_immediately=False)

    def order_node(self) -> NodeConfig:
        session = self.session

        async def resolve_shortage(flow_manager: FlowManager, sku: str, choice: str,
                                   substitute_sku: str) -> tuple[dict, NodeConfig | None]:
            """Record how the customer wants a short item handled.

            Args:
                sku (str): SKU of the short item from the call facts, for example "BAK040".
                choice (str): One of "substitute", "partial" or "wait".
                substitute_sku (str): SKU of the substitute when choice is "substitute", otherwise "".
            """
            result = session.resolve_shortage(sku, choice, substitute_sku)
            done = result["status"] == "done" and not result["remaining_shortages"]
            return result, self.schedule_node() if done else None

        async def items_confirmed(flow_manager: FlowManager) -> tuple[dict, NodeConfig | None]:
            """The customer is happy with the items and no shortage is left unresolved."""
            remaining = [s.sku for s in session.inventory.shortages(session.order_id)]
            if remaining:
                return {"status": "shortage_unresolved", "skus": remaining}, None
            return {"status": "ok"}, self.schedule_node()

        task = ("Tell the customer in one or two sentences what their order contains. "
                "If an item is SHORT, explain it and offer the options: a listed substitute, sending what is "
                "available, or waiting for the restock; then call resolve_shortage with their choice. "
                "If nothing is short, call items_confirmed once they are happy.")
        return self._node("present_order", task, [resolve_shortage, items_confirmed])

    def schedule_node(self) -> NodeConfig:
        session = self.session

        async def check_slot(flow_manager: FlowManager, preferred_time: str) -> tuple[dict, None]:
            """Check the customer's preferred delivery time and hold the slot if it is free.

            Args:
                preferred_time (str): The customer's own words, for example "tomorrow after 5 PM".
            """
            return session.check_slot(preferred_time), None

        async def choose_slot(flow_manager: FlowManager, slot_id: int) -> tuple[dict, None]:
            """Hold an alternative slot the customer picked.

            Args:
                slot_id (int): The slot id from a check_slot result or the call facts.
            """
            return session.choose_slot(slot_id), None

        async def slot_agreed(flow_manager: FlowManager) -> tuple[dict, NodeConfig | None]:
            """The customer agreed to the slot that is currently held."""
            if session.held_slot_id is None:
                return {"status": "no_slot_held"}, None
            return {"status": "ok"}, self.details_node()

        task = ("Ask when they would like the delivery and call check_slot with their exact words. "
                "If a slot is held, ask them to confirm that time. If it is full, offer the alternatives and call "
                "choose_slot for the one they pick. When they agree to the held slot, call slot_agreed.")
        return self._node("schedule", task, [check_slot, choose_slot, slot_agreed])

    def details_node(self) -> NodeConfig:
        session = self.session
        address = session.repo.get_order(session.order_id).address

        async def update_address(flow_manager: FlowManager, address: str) -> tuple[dict, None]:
            """Change the delivery address.

            Args:
                address (str): The full new address as the customer said it.
            """
            return session.update_address(address), None

        async def add_note(flow_manager: FlowManager, note: str) -> tuple[dict, None]:
            """Save a delivery instruction, for example where to leave the order.

            Args:
                note (str): The instruction in a few words.
            """
            return session.add_note(note), None

        async def details_done(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The address is correct and any delivery instructions are saved."""
            return {"status": "ok"}, self.confirm_node()

        task = (f"Check that the delivery address is still {address}; if it changed, call update_address. "
                "Ask if there are any delivery instructions and save them with add_note. Then call details_done.")
        return self._node("details", task, [update_address, add_note, details_done])

    def confirm_node(self) -> NodeConfig:
        session = self.session
        summary = session.order_summary()

        async def confirm_booking(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The customer said yes to the read-back."""
            result = session.confirm_booking()
            if result["status"] != "booked":
                return result, self.schedule_node()
            return result, self.close_node("The delivery is booked. Thank the customer and say goodbye in one sentence.")

        async def change_time(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The customer wants a different delivery time."""
            return {"status": "ok"}, self.schedule_node()

        async def change_address(flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
            """The customer wants to change the address or delivery instructions."""
            return {"status": "ok"}, self.details_node()

        task = ("Read this back in one or two sentences and ask the customer to confirm: "
                f"items {'; '.join(summary['items'])}; delivery {summary['slot']}; address {summary['address']}. "
                "If they say yes, call confirm_booking.")
        return self._node("confirm", task, [confirm_booking, change_time, change_address])
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/agent/test_flow.py -q`
Expected: `13 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/agent/flow.py tests/agent/test_flow.py
git commit -m "Add delivery conversation graph on Pipecat Flows"
```

---

### Task 4: Speech and LLM services

**Files:**
- Create: `src/voiceagent/agent/services.py`, `tests/agent/test_services.py`, `scripts/smoke_tts.py`

**Interfaces:**
- Produces:
  - `float_to_pcm16(samples: np.ndarray) -> bytes`
  - `GreedyWhisperSTTService(WhisperSTTService)`: same constructor as the parent; decodes greedily (`beam_size=1, best_of=1, without_timestamps=True, condition_on_previous_text=False`)
  - `KokoroTorchTTSService(TTSService)`: `__init__(*, voice="af_heart", device="cuda", **kwargs)`
  - `make_llm(base_url="http://127.0.0.1:8080/v1") -> OpenAILLMService`

- [ ] **Step 1: Write the failing tests**

`tests/agent/test_services.py`:
```python
import asyncio

import numpy as np

from voiceagent.agent import services


def test_float_to_pcm16_clips_and_scales():
    pcm = np.frombuffer(services.float_to_pcm16(np.array([0.0, 0.5, 2.0, -2.0], dtype=np.float32)), dtype=np.int16)
    assert pcm.tolist() == [0, 16383, 32767, -32767]


class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return [], None


def test_greedy_whisper_forces_greedy_decoding(monkeypatch):
    monkeypatch.setattr(services.whisper_stt, "WhisperModel", FakeWhisperModel)

    async def build():
        return services.GreedyWhisperSTTService(device="cpu")

    stt = asyncio.run(build())
    stt._model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
    assert stt._model.calls[0] == {"beam_size": 1, "best_of": 1, "without_timestamps": True,
                                   "condition_on_previous_text": False, "language": "en"}


def test_make_llm_targets_local_server():
    llm = services.make_llm("http://127.0.0.1:8080/v1")
    assert str(llm._client.base_url).startswith("http://127.0.0.1:8080/v1")
    extra = llm._settings.extra["extra_body"]
    assert extra["chat_template_kwargs"] == {"enable_thinking": False} and extra["cache_prompt"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/agent/test_services.py -q`
Expected: FAIL with `ImportError: cannot import name 'services'`.

- [ ] **Step 3: Implement**

`src/voiceagent/agent/services.py`:
```python
"""The only place STT, TTS and LLM services are constructed (swap models here)."""
from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.services.whisper import stt as whisper_stt
from pipecat.transcriptions.language import Language

KOKORO_SAMPLE_RATE = 24000


def float_to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class GreedyWhisperSTTService(whisper_stt.WhisperSTTService):
    """Whisper with greedy decoding, as benchmarked in Phase 1 (Pipecat's default beam search is slower)."""

    def _load(self):
        import torch  # noqa: F401  (torch/lib ships cublas64_12.dll, which CTranslate2 needs on Windows)

        super()._load()
        self._model.transcribe = functools.partial(
            self._model.transcribe, beam_size=1, best_of=1, without_timestamps=True,
            condition_on_previous_text=False,
        )


class KokoroTorchTTSService(TTSService):
    """Kokoro-82M on PyTorch/CUDA: 395 ms first clause vs 612 ms for kokoro-onnx in Phase 1."""

    def __init__(self, *, voice: str = "af_heart", device: str = "cuda", **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True,
                         settings=TTSSettings(model="kokoro-82m", voice=voice, language=Language.EN), **kwargs)
        from kokoro import KModel, KPipeline

        self._voice = voice
        model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
        self._pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
        self._resampler = create_stream_resampler()

    def can_generate_metrics(self) -> bool:
        return True

    def _synthesize(self, text: str) -> np.ndarray:
        chunks = [r.audio.cpu().numpy() for r in self._pipeline(text, voice=self._voice) if r.audio is not None]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_tts_usage_metrics(text)
            samples = await asyncio.to_thread(self._synthesize, text)
            await self.stop_ttfb_metrics()
            if samples.size:
                audio = await self._resampler.resample(float_to_pcm16(samples), KOKORO_SAMPLE_RATE, self.sample_rate)
                yield TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1,
                                       context_id=context_id)
        except Exception as exc:
            logger.exception("Kokoro synthesis failed")
            yield ErrorFrame(error=f"Kokoro synthesis failed: {exc}")
        finally:
            await self.stop_ttfb_metrics()


def make_llm(base_url: str = "http://127.0.0.1:8080/v1") -> OpenAILLMService:
    """Qwen3.5-2B (or whatever GGUF llama-server was started with), non-thinking, prompt cache on."""
    return OpenAILLMService(
        base_url=base_url,
        api_key="local",
        settings=OpenAILLMService.Settings(
            model="local",
            temperature=0.3,
            max_tokens=120,
            extra={"extra_body": {"cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}},
        ),
    )
```

`scripts/smoke_tts.py`:
```python
"""GPU smoke test: synthesize one clause with KokoroTorchTTSService. Usage: python scripts/smoke_tts.py"""
import time

from voiceagent.agent.services import KokoroTorchTTSService

tts = KokoroTorchTTSService()
tts._synthesize("Warming up.")
t0 = time.perf_counter()
samples = tts._synthesize("Sure, I have a slot tomorrow between five and six.")
print(f"{len(samples) / 24000:.2f}s of audio in {(time.perf_counter() - t0) * 1000:.0f} ms")
```

- [ ] **Step 4: Run the tests and the smoke script**

```bash
.venv/Scripts/python -m pytest tests/agent/test_services.py -q
.venv/Scripts/python scripts/smoke_tts.py
```
Expected: `3 passed`, then `~2.5s of audio in ~300-500 ms`. If constructing `GreedyWhisperSTTService` outside a running pipeline fails, the test already builds it inside `asyncio.run`; inspect the traceback before changing the service.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/agent/services.py tests/agent/test_services.py scripts/smoke_tts.py
git commit -m "Add greedy Whisper, Kokoro PyTorch TTS and local LLM services"
```

---

### Task 5: Latency and call recording

**Files:**
- Create: `src/voiceagent/agent/metrics.py`, `tests/agent/test_metrics.py`

**Interfaces:**
- Consumes: the `Repository` call methods (Task 1); Pipecat `UserBotLatencyObserver`, `LatencyBreakdown`, `BaseObserver`.
- Produces:
  - `BargeInTimer(now=time.perf_counter)` with `bot_started()`, `user_started()` and `bot_stopped() -> float | None` (ms)
  - `BargeInObserver(on_barge_in)`
  - `breakdown_to_dict(breakdown) -> dict` (`contributions`: `[[key, label, ms], ...]`, `ttfb`: `[[processor, ms], ...]`, `user_turn_ms`)
  - `CallRecorder(repo, order_id, clock=datetime.now)` with `call_id`, `record_breakdown(breakdown)`, `record_barge_in(ms)`, `finish(outcome, messages)` and `observers() -> list[BaseObserver]`

- [ ] **Step 1: Write the failing tests**

`tests/agent/test_metrics.py`:
```python
from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown, LatencyContribution, LatencyOwnerKind, MeasuredFrom, TTFBBreakdownMetrics,
    UserBotLatencyObserver)

from voiceagent.agent.metrics import BargeInObserver, BargeInTimer, CallRecorder

from conftest import NOW


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_barge_in_timer_measures_interruptions_only():
    clock = FakeClock()
    timer = BargeInTimer(now=clock)
    timer.user_started()  # bot silent: not an interruption
    timer.bot_started()
    assert timer.bot_stopped() is None
    timer.bot_started()
    clock.t = 1.0
    timer.user_started()
    clock.t = 1.15
    assert round(timer.bot_stopped(), 1) == 150.0


def breakdown(measured_from=MeasuredFrom.USER_SILENCE):
    return LatencyBreakdown(
        contributions=[
            LatencyContribution(key="vad", label="endpointing wait", owner="config: VAD stop_secs",
                                owner_kind=LatencyOwnerKind.SETTING, start_time=0.0, duration_secs=0.2),
            LatencyContribution(key="llm", label="LLM inference", owner="OpenAILLMService#0",
                                owner_kind=LatencyOwnerKind.SERVICE, start_time=0.2, duration_secs=0.55),
        ],
        ttfb=[TTFBBreakdownMetrics(processor="OpenAILLMService#0", start_time=0.2, duration_secs=0.3)],
        measured_from=measured_from, total_secs=0.75, user_turn_secs=0.26)


def test_record_breakdown(world, repo):
    recorder = CallRecorder(repo, 1, clock=lambda: NOW)
    recorder.record_breakdown(breakdown())
    recorder.record_breakdown(breakdown(MeasuredFrom.CLIENT_CONNECTED))
    recorder.record_barge_in(142.0)
    metrics = repo.call_metrics(recorder.call_id)
    assert [m["kind"] for m in metrics] == ["response", "greeting", "barge_in"]
    assert metrics[0]["total_ms"] == 750.0
    assert metrics[0]["breakdown"]["contributions"] == [["vad", "endpointing wait", 200.0],
                                                       ["llm", "LLM inference", 550.0]]
    assert metrics[0]["breakdown"]["ttfb"] == [["OpenAILLMService#0", 300.0]]
    assert metrics[0]["breakdown"]["user_turn_ms"] == 260.0


def test_finish_keeps_spoken_turns_only(world, repo):
    recorder = CallRecorder(repo, 1, clock=lambda: NOW)
    recorder.finish("scheduled", [
        {"role": "system", "content": "rules"},
        {"role": "assistant", "content": "Hi, is this Priya?"},
        {"role": "user", "content": "Yes."},
        {"role": "assistant", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "{}"},
    ])
    call = repo.get_call(recorder.call_id)
    assert call["outcome"] == "scheduled"
    assert call["transcript"] == [{"role": "assistant", "content": "Hi, is this Priya?"},
                                  {"role": "user", "content": "Yes."}]


def test_finish_without_outcome_is_abandoned(world, repo):
    recorder = CallRecorder(repo, 1, clock=lambda: NOW)
    recorder.finish(None, [])
    assert repo.get_call(recorder.call_id)["outcome"] == "abandoned"


def test_observers(world, repo):
    observers = CallRecorder(repo, 1, clock=lambda: NOW).observers()
    assert [type(o) for o in observers] == [UserBotLatencyObserver, BargeInObserver]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/agent/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.agent.metrics'`.

- [ ] **Step 3: Implement**

`src/voiceagent/agent/metrics.py`:
```python
"""Per-turn latency, barge-in latency and call outcome, persisted to SQLite."""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import datetime

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, VADUserStartedSpeakingFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import LatencyBreakdown, MeasuredFrom, UserBotLatencyObserver

from voiceagent.db.repository import Repository


class BargeInTimer:
    """User speech onset while the agent is talking -> agent audio stopped (ms)."""

    def __init__(self, now: Callable[[], float] = time.perf_counter):
        self._now = now
        self._bot_speaking = False
        self._interrupted_at: float | None = None

    def bot_started(self) -> None:
        self._bot_speaking = True
        self._interrupted_at = None

    def user_started(self) -> None:
        if self._bot_speaking and self._interrupted_at is None:
            self._interrupted_at = self._now()

    def bot_stopped(self) -> float | None:
        self._bot_speaking = False
        if self._interrupted_at is None:
            return None
        elapsed_ms = (self._now() - self._interrupted_at) * 1000
        self._interrupted_at = None
        return elapsed_ms


class BargeInObserver(BaseObserver):
    def __init__(self, on_barge_in: Callable[[float], Awaitable[None]], **kwargs):
        super().__init__(observe_every_push=False, **kwargs)
        self._timer = BargeInTimer()
        self._on_barge_in = on_barge_in

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, BotStartedSpeakingFrame):
            self._timer.bot_started()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._timer.user_started()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            elapsed_ms = self._timer.bot_stopped()
            if elapsed_ms is not None:
                await self._on_barge_in(elapsed_ms)


def breakdown_to_dict(breakdown: LatencyBreakdown) -> dict:
    ms = lambda secs: round(secs * 1000, 1)
    return {
        "contributions": [[c.key, c.label, ms(c.duration_secs)] for c in breakdown.contributions],
        "ttfb": [[t.processor, ms(t.duration_secs)] for t in breakdown.ttfb],
        "user_turn_ms": ms(breakdown.user_turn_secs) if breakdown.user_turn_secs is not None else None,
    }


class CallRecorder:
    """Persists one call: a calls row, a turn_metrics row per measured turn or barge-in, outcome and transcript."""

    def __init__(self, repo: Repository, order_id: int, clock: Callable[[], datetime] = datetime.now):
        self.repo = repo
        self.clock = clock
        self.call_id = repo.start_call(order_id, clock())

    def record_breakdown(self, breakdown: LatencyBreakdown) -> None:
        kind = "greeting" if breakdown.measured_from == MeasuredFrom.CLIENT_CONNECTED else "response"
        self.repo.add_turn_metric(self.call_id, kind, breakdown.total_secs * 1000, breakdown_to_dict(breakdown),
                                  self.clock())

    def record_barge_in(self, elapsed_ms: float) -> None:
        self.repo.add_turn_metric(self.call_id, "barge_in", elapsed_ms, {}, self.clock())

    def finish(self, outcome: str | None, messages: list) -> None:
        transcript = [{"role": m["role"], "content": m["content"]} for m in messages
                      if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                      and isinstance(m.get("content"), str)]
        self.repo.finish_call(self.call_id, self.clock(), outcome or "abandoned", transcript)

    def observers(self) -> list[BaseObserver]:
        latency = UserBotLatencyObserver()

        @latency.event_handler("on_latency_breakdown")
        async def _on_breakdown(observer, breakdown):
            self.record_breakdown(breakdown)

        async def _on_barge_in(elapsed_ms: float):
            self.record_barge_in(elapsed_ms)

        return [latency, BargeInObserver(_on_barge_in)]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/agent/test_metrics.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/agent/metrics.py tests/agent/test_metrics.py
git commit -m "Record per-turn latency, barge-in latency and call outcomes"
```

---

### Task 6: Dashboard

**Files:**
- Create: `src/voiceagent/agent/dashboard.py`, `tests/agent/test_dashboard.py`

**Interfaces:**
- Consumes: `Repository.list_calls`, `get_call` and `call_metrics` (Task 1).
- Produces: `create_app(db_path) -> FastAPI` with these routes:
  - `GET /api/calls`: each call plus `responses`, `p50_ms`, `p95_ms`, `greeting_ms` and `barge_in_p50_ms`
  - `GET /api/calls/{id}`: the call plus `metrics` and `transcript`, or 404
  - `GET /`: the HTML page

  CLI: `python -m voiceagent.agent.dashboard [--db data/warehouse.db] [--port 7861]`. Also `nearest_rank(values, p)`.

- [ ] **Step 1: Write the failing tests**

`tests/agent/test_dashboard.py`:
```python
from fastapi.testclient import TestClient

from voiceagent.agent.dashboard import create_app, nearest_rank
from voiceagent.db.repository import Repository

from conftest import NOW, build_world


def make_db(tmp_path):
    path = tmp_path / "w.db"
    repo = Repository(path)
    repo.init_schema()
    build_world(repo, NOW)
    call_id = repo.start_call(1, NOW)
    for kind, ms in [("greeting", 900.0), ("response", 700.0), ("response", 1100.0), ("barge_in", 150.0)]:
        repo.add_turn_metric(call_id, kind, ms, {"contributions": [["llm", "LLM inference", 300.0]]}, NOW)
    repo.finish_call(call_id, NOW, "scheduled", [{"role": "user", "content": "Yes."}])
    repo.close()
    return path


def test_nearest_rank():
    assert nearest_rank([700.0, 1100.0], 50) == 700.0
    assert nearest_rank([700.0, 1100.0], 95) == 1100.0
    assert nearest_rank([], 50) is None


def test_api_calls_summary(tmp_path):
    client = TestClient(create_app(make_db(tmp_path)))
    [call] = client.get("/api/calls").json()
    assert call["customer"] == "Priya Sharma" and call["outcome"] == "scheduled"
    assert (call["responses"], call["p50_ms"], call["p95_ms"]) == (2, 700.0, 1100.0)
    assert call["greeting_ms"] == 900.0 and call["barge_in_p50_ms"] == 150.0


def test_api_call_detail_and_404(tmp_path):
    client = TestClient(create_app(make_db(tmp_path)))
    detail = client.get("/api/calls/1").json()
    assert len(detail["metrics"]) == 4 and detail["transcript"] == [{"role": "user", "content": "Yes."}]
    assert client.get("/api/calls/99").status_code == 404


def test_index_page(tmp_path):
    response = TestClient(create_app(make_db(tmp_path))).get("/")
    assert response.status_code == 200 and "Live calls" in response.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/agent/test_dashboard.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.agent.dashboard'`.

- [ ] **Step 3: Implement**

`src/voiceagent/agent/dashboard.py`:
```python
"""Live call dashboard. Usage: python -m voiceagent.agent.dashboard [--db data/warehouse.db] [--port 7861]"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from voiceagent.db.repository import Repository

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Live calls</title>
<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:4px 10px;
border-bottom:1px solid #ddd;text-align:left}pre{background:#f6f6f6;padding:8px;white-space:pre-wrap}</style></head>
<body><h1>Live calls</h1><table id="calls"></table><h2 id="title"></h2><pre id="detail"></pre>
<script>
const fmt = v => v == null ? "-" : v;
async function load() {
  const calls = await (await fetch("/api/calls")).json();
  document.getElementById("calls").innerHTML = "<tr><th>call</th><th>customer</th><th>outcome</th><th>turns</th>"
    + "<th>p50 ms</th><th>p95 ms</th><th>greeting ms</th><th>barge-in p50 ms</th></tr>"
    + calls.map(c => `<tr onclick="show(${c.id})" style="cursor:pointer"><td>${c.id}</td><td>${c.customer}</td>`
      + `<td>${fmt(c.outcome)}</td><td>${c.responses}</td><td>${fmt(c.p50_ms)}</td><td>${fmt(c.p95_ms)}</td>`
      + `<td>${fmt(c.greeting_ms)}</td><td>${fmt(c.barge_in_p50_ms)}</td></tr>`).join("");
}
async function show(id) {
  const d = await (await fetch(`/api/calls/${id}`)).json();
  document.getElementById("title").textContent = `Call ${id}: ${d.customer} (${fmt(d.outcome)})`;
  document.getElementById("detail").textContent =
    d.metrics.map(m => `${m.kind.padEnd(9)} ${String(m.total_ms).padStart(7)} ms  `
      + (m.breakdown.contributions || []).map(c => `${c[1]} ${c[2]}`).join(" | ")).join("\\n")
    + "\\n\\n" + d.transcript.map(t => `${t.role}: ${t.content}`).join("\\n");
}
load(); setInterval(load, 2000);
</script></body></html>"""


def nearest_rank(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(max(math.ceil(p / 100 * len(ordered)) - 1, 0), len(ordered) - 1)]


def _summary(repo: Repository, call: dict) -> dict:
    metrics = repo.call_metrics(call["id"])
    responses = [m["total_ms"] for m in metrics if m["kind"] == "response"]
    greeting = next((m["total_ms"] for m in metrics if m["kind"] == "greeting"), None)
    barge = [m["total_ms"] for m in metrics if m["kind"] == "barge_in"]
    return {**call, "started_at": call["started_at"].isoformat(),
            "ended_at": call["ended_at"].isoformat() if call["ended_at"] else None,
            "responses": len(responses), "p50_ms": nearest_rank(responses, 50), "p95_ms": nearest_rank(responses, 95),
            "greeting_ms": greeting, "barge_in_p50_ms": nearest_rank(barge, 50)}


def create_app(db_path: str | Path) -> FastAPI:
    app = FastAPI(title="Warehouse voice agent calls")

    def repo() -> Repository:
        return Repository(db_path)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/calls")
    def calls() -> list[dict]:
        r = repo()
        try:
            return [_summary(r, c) for c in r.list_calls()]
        finally:
            r.close()

    @app.get("/api/calls/{call_id}")
    def call_detail(call_id: int) -> dict:
        r = repo()
        try:
            call = r.get_call(call_id)
            if call is None:
                raise HTTPException(status_code=404, detail="call not found")
            metrics = [{**m, "recorded_at": m["recorded_at"].isoformat()} for m in r.call_metrics(call_id)]
            return {**_summary(r, call), "metrics": metrics}
        finally:
            r.close()

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/warehouse.db")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    uvicorn.run(create_app(args.db), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/agent/test_dashboard.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/agent/dashboard.py tests/agent/test_dashboard.py
git commit -m "Add live call dashboard"
```

---

### Task 7: Pipeline wiring and runner entry point

**Files:**
- Create: `src/voiceagent/agent/bot.py`, `tests/agent/test_bot.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 2–5; Pipecat runner (`main(parser)` passes custom args through as `runner_args.cli_args`).
- Produces: `build_parser() -> argparse.ArgumentParser` (`--order-id`, `--db`, `--llm-url`), `run_bot(transport, runner_args)`, `bot(runner_args)`, and the CLI `python -m voiceagent.agent.bot -t webrtc --order-id N`.

- [ ] **Step 1: Write the failing test**

`tests/agent/test_bot.py`:
```python
from voiceagent.agent.bot import build_parser


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert (args.order_id, args.db, args.llm_url) == (1, "data/warehouse.db", "http://127.0.0.1:8080/v1")
    assert build_parser().parse_args(["--order-id", "42"]).order_id == 42
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/agent/test_bot.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'voiceagent.agent.bot'`.

- [ ] **Step 3: Implement**

`src/voiceagent/agent/bot.py`:
```python
"""Delivery caller agent.

    llama-server:  bash scripts/llama_server.sh models/llm/Qwen3.5-2B-Q4_K_M.gguf
    agent:         python -m voiceagent.agent.bot -t webrtc --order-id 1
    browser:       http://localhost:7860/client  (click Connect; allow the microphone)
    dashboard:     python -m voiceagent.agent.dashboard  ->  http://localhost:7861
"""
from __future__ import annotations

import argparse

from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.flows import FlowManager
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from voiceagent.agent.flow import DeliveryFlow
from voiceagent.agent.metrics import CallRecorder
from voiceagent.agent.services import GreedyWhisperSTTService, KokoroTorchTTSService, make_llm
from voiceagent.agent.tools import CallSession
from voiceagent.db.repository import Repository

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warehouse delivery caller agent")
    parser.add_argument("--order-id", type=int, default=1, help="order to call the customer about")
    parser.add_argument("--db", default="data/warehouse.db")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080/v1")
    return parser


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    args = runner_args.cli_args
    repo = Repository(args.db)
    session = CallSession(repo, args.order_id)
    flow = DeliveryFlow(session)
    recorder = CallRecorder(repo, args.order_id)

    stt = GreedyWhisperSTTService(device="cuda", compute_type="float16",
                                  settings=GreedyWhisperSTTService.Settings(model="distil-small.en"))
    llm = make_llm(args.llm_url)
    tts = KokoroTorchTTSService(voice="af_heart")

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3(cpu_count=4))],
            ),
        ),
    )
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=recorder.observers(),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    flow_manager = FlowManager(worker=worker, llm=llm, context_aggregator=context_aggregator, transport=transport,
                               global_functions=flow.global_functions)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Calling about order {args.order_id} ({flow.customer_name})")
        await flow_manager.initialize(flow.greet_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        recorder.finish(session.outcome, context.get_messages())
        logger.info(f"Call {recorder.call_id} ended: {session.outcome or 'abandoned'}")
        await runner.cancel()

    await runner.run()
    repo.close()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main(build_parser())
```

Append to `README.md`:
```markdown

## Talk to the agent (local, browser)

    .venv/Scripts/python -m pip install -e ".[dev,bench,agent]"
    .venv/Scripts/python -m voiceagent.db.seed --db data/warehouse.db
    bash scripts/llama_server.sh models/llm/Qwen3.5-2B-Q4_K_M.gguf        # terminal 1
    .venv/Scripts/python -m voiceagent.agent.bot -t webrtc --order-id 1   # terminal 2
    .venv/Scripts/python -m voiceagent.agent.dashboard                    # terminal 3

Open http://localhost:7860/client, click Connect and answer the call. Per-turn latency: http://localhost:7861.
```

- [ ] **Step 4: Run the test, then smoke-start the server**

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m voiceagent.agent.bot -t webrtc --order-id 1 --port 7870 > /tmp/bot.log 2>&1 &
BOT=$!; for i in $(seq 1 60); do curl -s -o /dev/null -w "%{http_code}" http://localhost:7870/client/ | grep -q 200 && break; sleep 1; done
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7870/client/; kill $BOT
```
Expected: the full suite passes, then `200` (the runner serves the prebuilt client; models load only when a browser connects).

- [ ] **Step 5: Commit**

```bash
git add src/voiceagent/agent/bot.py tests/agent/test_bot.py README.md
git commit -m "Wire the live voice pipeline and runner entry point"
```

---

### Task 8: Live call acceptance (human in the loop)

**Files:**
- Create: `docs/benchmarks/phase2-live-calls.md`

This task needs a person speaking into the microphone. Run it with headphones, so the agent's voice doesn't loop back into the mic.

- [ ] **Step 1: Start everything**

```bash
.venv/Scripts/python -m voiceagent.db.seed --db data/warehouse.db          # "now" = current time
bash scripts/llama_server.sh models/llm/Qwen3.5-2B-Q4_K_M.gguf             # terminal 1
.venv/Scripts/python -m voiceagent.agent.bot -t webrtc --order-id 1        # terminal 2
.venv/Scripts/python -m voiceagent.agent.dashboard                         # terminal 3
```
Pick order ids from the seed for each scenario. A shortage order is any order whose context shows `SHORT`. One way to list them:
`.venv/Scripts/python -c "from voiceagent.db.repository import Repository; from voiceagent.domain.inventory import InventoryService as I; r=Repository('data/warehouse.db'); i=I(r); print([o for (o,) in r.conn.execute('select id from orders order by id limit 40') if i.shortages(o)])"`

- [ ] **Step 2: Make these calls** (restart the bot with the matching `--order-id` between calls)

| # | Scenario | What to say | Pass if |
|---|---|---|---|
| 1 | Happy path | confirm identity, accept items, "tomorrow after five", "yes", no instructions, "yes" | order `scheduled`, the agent read back the right slot |
| 2 | Slot full | ask for a full hour (use the dashboard or the context's missing hours) | agent offers alternatives from the tool result; booking succeeds |
| 3 | Shortage | on a SHORT order, choose the substitute | read-back names the substitute; order `scheduled` |
| 4 | Add item | "can you also add two Amul milk" | agent reports added or insufficient, matching stock |
| 5 | Callback | "I'm busy, call me tomorrow morning" | call ends; order `callback` |
| 6 | Wrong person | "no, wrong number" | call ends; order `callback` with note |
| 7 | Barge-in | start talking while the agent is mid-sentence | agent stops within about 0.2 s; dashboard shows a `barge_in` row |

- [ ] **Step 3: Record results**

Create `docs/benchmarks/phase2-live-calls.md` with:
- a table of the 7 scenarios (pass/fail plus one line on what happened)
- the dashboard's per-call p50/p95 response latency, greeting latency and barge-in latency
- one per-turn breakdown copied from the dashboard detail view (endpointing / STT / LLM / TTS)
- any invented fact the agent said, quoted, plus the node it happened in

Then commit:
```bash
git add docs/benchmarks/phase2-live-calls.md
git commit -m "Record phase 2 live call results"
```

---

## Later plans (not part of this plan)

- **Plan 3, evaluation:**
  - A text-only scenario harness that drives `DeliveryFlow` tools through the LLM, with database-state assertions.
  - An audio end-to-end harness using Pipecat's `eval` transport and synthesized customer audio.
- **Plan 4, latency:**
  - Load models once per process instead of per call.
  - Pre-synthesized openers.
  - Clause-level TTS aggregation.
  - Preemptive LLM generation.
  - Smart Turn thread and GPU tuning.
  - A WSL2/Linux comparison.
  - A backchannel guard. Interruptions are currently VAD-based, so "mm-hmm" interrupts. A words-based guard needs streaming STT.
- **Plan 5:** fine-tuning and quantization.
