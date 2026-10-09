"""Intent reader accuracy against llama-server.

    bash scripts/llama_server.sh models/llm/Qwen3.5-2B-Q4_K_M.gguf
    python -m bench.intent_eval --model-name qwen3.5-2b-q4km
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter

from bench.common import CORPUS_DIR, summarize, write_result
from voiceagent.agent.dialog import STATE_INTENTS
from voiceagent.agent.intent import IntentClassifier
from voiceagent.agent.nlu import grounded


def score(rows: list[dict]) -> dict:
    per_intent: dict[str, dict] = {}
    for row in rows:
        stats = per_intent.setdefault(row["expected"], {"n": 0, "correct": 0})
        stats["n"] += 1
        stats["correct"] += row["got"] == row["expected"]
    confusions = Counter((r["expected"], r["got"]) for r in rows if r["got"] != r["expected"])
    return {
        "accuracy": round(sum(r["got"] == r["expected"] for r in rows) / len(rows), 3),
        "value_grounded_rate": round(sum(r["value_ok"] for r in rows) / len(rows), 3),
        "per_intent": per_intent,
        "confusions": [[e, g, n] for (e, g), n in confusions.most_common()],
    }


async def run(model_name: str, url: str) -> dict:
    cases = json.loads((CORPUS_DIR / "intent_cases.json").read_text(encoding="utf-8"))
    classifier = IntentClassifier(url)
    rows, latencies = [], []
    try:
        await classifier.classify("warm up", "warm up", ["unclear"])
        for case in cases:
            started = time.perf_counter()
            intent = await classifier.classify(case["agent"], case["customer"], STATE_INTENTS[case["state"]])
            latencies.append((time.perf_counter() - started) * 1000)
            value_ok = (not case["value"]) or (case["value"].lower() in intent.value.lower()
                                               and grounded(intent.value, case["customer"]))
            rows.append({**case, "expected": case["intent"], "got": intent.name, "got_value": intent.value,
                         "value_ok": value_ok})
    finally:
        await classifier.aclose()
    result = {"model": model_name, "latency_ms": summarize(latencies), **score(rows), "rows": rows}
    path = write_result(f"intent_{model_name}", result, device="cuda (llama-server)")
    for row in rows:
        if row["got"] != row["expected"] or not row["value_ok"]:
            print(f"MISS [{row['state']}] {row['customer']!r}: expected {row['expected']} got {row['got']} "
                  f"value={row['got_value']!r}")
    print(f"{model_name}: accuracy {result['accuracy']}, value ok {result['value_grounded_rate']}, "
          f"latency p50 {result['latency_ms']['p50']} ms p95 {result['latency_ms']['p95']} ms -> {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1")
    args = parser.parse_args()
    asyncio.run(run(args.model_name, args.url))


if __name__ == "__main__":
    main()
