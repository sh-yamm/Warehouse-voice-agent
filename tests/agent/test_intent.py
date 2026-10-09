import asyncio
import json

import httpx

from voiceagent.agent.intent import Intent, IntentClassifier

ALLOWED = ["confirm", "give_time", "unclear"]


def classifier(handler):
    return IntentClassifier("http://llm/v1", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def answering(content, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    return handler


def test_classify_sends_allowed_enum_and_parses():
    seen = []
    c = classifier(answering('{"intent": "give_time", "value": "tomorrow evening"}', seen))
    result = asyncio.run(c.classify("When would you like it?", "Tomorrow evening works.", ALLOWED))
    assert result == Intent("give_time", "tomorrow evening")
    body = seen[0]
    assert body["response_format"]["json_schema"]["schema"]["properties"]["intent"]["enum"] == ALLOWED
    assert body["messages"][1]["content"] == "AGENT: When would you like it?\nCUSTOMER: Tomorrow evening works."
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0


def test_intent_outside_allowed_is_unclear():
    c = classifier(answering('{"intent": "cancel", "value": ""}'))
    assert asyncio.run(c.classify("a", "b", ALLOWED)) == Intent("unclear")


def test_failures_are_unclear():
    assert asyncio.run(classifier(answering("not json")).classify("a", "b", ALLOWED)) == Intent("unclear")
    down = classifier(lambda request: httpx.Response(500, json={"error": "boom"}))
    assert asyncio.run(down.classify("a", "b", ALLOWED)) == Intent("unclear")


def test_system_prompt_is_static():
    a = IntentClassifier.request_body("x", "y", ["confirm"])["messages"][0]
    b = IntentClassifier.request_body("p", "q", ["deny", "unclear"])["messages"][0]
    assert a == b and "Never copy the agent's words" in a["content"]
