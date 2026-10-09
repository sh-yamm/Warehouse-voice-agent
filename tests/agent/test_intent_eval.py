from bench.intent_eval import score


def test_score():
    rows = [
        {"expected": "confirm", "got": "confirm", "value_ok": True},
        {"expected": "give_time", "got": "give_time", "value_ok": True},
        {"expected": "add_item", "got": "confirm", "value_ok": False},
        {"expected": "add_item", "got": "add_item", "value_ok": False},
    ]
    s = score(rows)
    assert s["accuracy"] == 0.75 and s["value_grounded_rate"] == 0.5
    assert s["per_intent"]["add_item"] == {"n": 2, "correct": 1}
    assert s["confusions"] == [["add_item", "confirm", 1]]
