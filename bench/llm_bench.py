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
SYSTEM_TEMPLATE = """You are Avio, a friendly delivery-scheduling assistant calling customers on behalf of AvioStack.
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
                        {"role": "assistant", "content": f"Hi, this is Avio from AvioStack. Am I speaking with {first_name}?"}]
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
