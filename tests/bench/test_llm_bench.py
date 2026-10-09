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
