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
