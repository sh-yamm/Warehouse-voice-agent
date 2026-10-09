"""The one LLM call per customer turn: name the intent and keep the customer's own words."""
from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
from loguru import logger

INTENT_DESCRIPTIONS = {
    "confirm": "yes, agrees, that's me, that's right, that's all, go ahead",
    "deny": "no, disagrees, that's not right",
    "give_time": "names a delivery day or time",
    "pick_option": "picks one of the options the agent just listed (first, second, last, or by naming it)",
    "substitute": "accepts the suggested substitute product",
    "send_available": "send the order without the missing item, or only what is available",
    "wait_restock": "wait until the missing item is back in stock",
    "add_item": "asks to add a product to the order",
    "change_address": "gives a different delivery address",
    "add_note": "gives delivery instructions for the driver",
    "callback": "busy now, or asks to be called back later",
    "cancel": "wants to cancel the whole order",
    "wrong_person": "is not the person the agent asked for",
    "question": "asks a question",
    "unclear": "anything else, or unintelligible",
}

SYSTEM_PROMPT = (
    "Classify the customer's last utterance on a grocery delivery phone call. "
    'Reply with JSON {"intent": ..., "value": ...}.\n'
    "value = the customer's own words for the time, product with quantity, address, instruction, chosen option "
    "or question; \"\" if none. Never copy the agent's words.\n"
    "Intents:\n" + "\n".join(f"- {name}: {meaning}" for name, meaning in INTENT_DESCRIPTIONS.items())
)


@dataclass(frozen=True)
class Intent:
    name: str
    value: str = ""


class IntentClassifier:
    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", client: httpx.AsyncClient | None = None,
                 timeout_s: float = 10.0):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    @staticmethod
    def request_body(agent_said: str, customer_said: str, allowed: list[str]) -> dict:
        schema = {"type": "object",
                  "properties": {"intent": {"type": "string", "enum": allowed},
                                 "value": {"type": "string", "maxLength": 80}},
                  "required": ["intent", "value"]}
        return {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": f"AGENT: {agent_said}\nCUSTOMER: {customer_said}"}],
            "temperature": 0,
            "max_tokens": 48,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "turn", "schema": schema}},
        }

    async def classify(self, agent_said: str, customer_said: str, allowed: list[str]) -> Intent:
        try:
            response = await self._client.post(self._url, json=self.request_body(agent_said, customer_said, allowed))
            response.raise_for_status()
            data = json.loads(response.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning(f"intent reader failed, treating the turn as unclear: {exc}")
            return Intent("unclear")
        name = data.get("intent") if isinstance(data, dict) else None
        if name not in allowed:
            return Intent("unclear")
        return Intent(name, str(data.get("value") or "").strip())

    async def aclose(self) -> None:
        await self._client.aclose()
