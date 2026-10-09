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
