import json
import math

from bench import common


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert common.percentile(values, 50) == 50
    assert common.percentile(values, 95) == 95
    assert common.percentile([7], 95) == 7
    assert math.isnan(common.percentile([], 50))


def test_summarize():
    s = common.summarize([10.0, 20.0, 30.0, 40.0])
    assert s == {"n": 4, "p50": 20.0, "p95": 40.0, "mean": 25.0, "min": 10.0, "max": 40.0}


def test_normalize_text():
    assert common.normalize_text("Tomorrow, after 5 P.M.!") == "tomorrow after 5 pm"
    assert common.normalize_text("Five thirty") == "5 30"
    assert common.normalize_text("5:30 PM") == "5 30 pm"
    assert common.normalize_text("I don't know") == "i dont know"
    assert common.normalize_text("Mm-hmm.") == "mm hmm"


def test_word_errors_and_corpus_wer():
    assert common.word_errors("deliver it at 5 pm", "deliver it at five p.m.") == (0, 5)
    assert common.word_errors("a b c", "a x c d") == (2, 3)
    assert common.word_errors("a b", "") == (2, 2)
    assert common.corpus_wer([("a b c", "a b c"), ("a b", "a")]) == 0.2


def test_clause_and_sentence_end():
    text = "Hi Priya, your order is ready. Shall I book it?"
    assert text[: common.first_clause_end(text)] == "Hi Priya,"
    assert text[: common.first_sentence_end(text)] == "Hi Priya, your order is ready."
    assert common.first_clause_end("Okay") is None
    assert common.first_clause_end("Okay,") is None  # needs at least two words


def test_write_result_includes_device(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    path = common.write_result("stt_demo", {"x": 1}, device="CUDAExecutionProvider")
    data = json.loads(path.read_text())
    assert data["name"] == "stt_demo"
    assert data["device"] == "CUDAExecutionProvider"
    assert data["x"] == 1 and "recorded_at" in data
