from datetime import datetime

import pytest

from voiceagent.domain.timeparse import parse_time_window

NOW = datetime(2026, 10, 12, 10, 0)  # Monday


def d(day, h, m=0):
    return datetime(2026, 10, day, h, m)


@pytest.mark.parametrize("phrase, start, end", [
    ("tomorrow after 5", d(13, 17), d(13, 21)),
    ("Tomorrow after 5 p.m.", d(13, 17), d(13, 21)),
    ("tomorrow after 5 PM.", d(13, 17), d(13, 21)),
    ("today evening", d(12, 17), d(12, 21)),
    ("between 2 and 4 tomorrow", d(13, 14), d(13, 16)),
    ("5 to 7 pm", d(12, 17), d(12, 19)),
    ("around 6", d(12, 17), d(12, 19)),
    ("in the morning", d(12, 8), d(12, 12)),
    ("Friday afternoon", d(16, 12), d(16, 17)),
    ("day after tomorrow at 11", d(14, 11), d(14, 12)),
    ("half past five", d(12, 17, 30), d(12, 18, 30)),
    ("five thirty pm", d(12, 17, 30), d(12, 18, 30)),
    ("5:30 PM", d(12, 17, 30), d(12, 18, 30)),
    ("as soon as possible", d(12, 10), d(12, 21)),
    ("anytime tomorrow", d(13, 8), d(13, 21)),
    ("tomorrow", d(13, 8), d(13, 21)),
    ("before noon", d(12, 8), d(12, 12)),
    ("by 6", d(12, 8), d(12, 18)),
    ("after 7 o'clock tonight", d(12, 19), d(12, 21)),
    ("seven to eight", d(12, 19), d(12, 20)),
])
def test_parses(phrase, start, end):
    window = parse_time_window(phrase, NOW)
    assert window is not None, phrase
    assert (window.start, window.end) == (start, end)


@pytest.mark.parametrize("phrase", [
    "I don't know",
    "",
    "11 at night",
    "after 9 pm",
    "on the 15th",
])
def test_unparseable_or_outside_hours(phrase):
    assert parse_time_window(phrase, NOW) is None


def test_rolls_to_tomorrow_when_window_passed():
    late = datetime(2026, 10, 12, 15, 0)
    window = parse_time_window("in the morning", late)
    assert (window.start, window.end) == (d(13, 8), d(13, 12))


def test_asap_after_close_is_tomorrow():
    late = datetime(2026, 10, 12, 21, 30)
    window = parse_time_window("asap", late)
    assert (window.start, window.end) == (d(13, 8), d(13, 21))


def test_next_weekday_on_same_weekday_is_next_week():
    window = parse_time_window("next Monday", NOW)
    assert (window.start, window.end) == (d(19, 8), d(19, 21))


def test_same_weekday_after_close_is_next_week():
    late = datetime(2026, 10, 12, 22, 0)
    window = parse_time_window("Monday morning", late)
    assert (window.start, window.end) == (d(19, 8), d(19, 12))


def test_today_after_window_passed_is_none():
    late = datetime(2026, 10, 12, 15, 0)
    assert parse_time_window("today morning", late) is None
