from bench.turn_bench import score


def test_score_counts_per_class():
    rows = [
        {"label": "complete", "prob": 0.9}, {"label": "complete", "prob": 0.4},
        {"label": "incomplete", "prob": 0.1}, {"label": "incomplete", "prob": 0.2},
    ]
    s = score(rows)
    assert s == {"accuracy": 0.75, "complete_recall": 0.5, "incomplete_recall": 1.0, "n": 4}
