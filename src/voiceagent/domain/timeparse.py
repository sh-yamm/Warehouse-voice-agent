from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

from voiceagent.domain.models import DELIVERY_DAY_END_HOUR as CLOSE
from voiceagent.domain.models import DELIVERY_DAY_START_HOUR as OPEN
from voiceagent.domain.models import TimeWindow

_NUMBER_WORDS = {
    "forty five": "45", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "fifteen": "15", "thirty": "30",
}
_NUMBER_RE = re.compile(r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b")
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_PARTS = [  # (word, (start_hour, end_hour), am/pm hint)
    ("morning", (8, 12), "am"), ("afternoon", (12, 17), "pm"), ("evening", (17, 21), "pm"),
    ("tonight", (17, 21), "pm"), ("night", (19, 21), "pm"),
]
_T = r"(\d{1,2})(?::(\d{2}))?(?:\s*(am|pm))?(?![a-z0-9])"
_RANGE_RES = [
    re.compile(rf"\b(?:between|from)\s+{_T}\s+(?:and|to|till|until)\s+{_T}"),
    re.compile(rf"\b{_T}\s+(?:to|till|until)\s+{_T}"),
]
_AFTER_RE = re.compile(rf"\b(?:after|from|post)\s+{_T}")
_BEFORE_RE = re.compile(rf"\b(?:before|by|until|till)\s+{_T}")
_AROUND_RE = re.compile(rf"\b(?:around|about|approximately|roughly)\s+{_T}")
_SINGLE_RE = re.compile(rf"\b{_T}")
_ASAP_RE = re.compile(r"\b(asap|as soon as possible|earliest|right now|immediately|right away)\b")
_ANY_RE = re.compile(r"\b(anytime|any time|whenever|all day|no preference)\b")


def _normalize(phrase: str) -> str:
    t = phrase.lower()
    for a, b in (("p.m.", "pm"), ("a.m.", "am"), ("p.m", "pm"), ("a.m", "am")):
        t = t.replace(a, b)
    t = re.sub(r"[^a-z0-9: ]", " ", t)
    t = re.sub(r"\bo clock\b", " ", t)
    t = " ".join(t.split())
    # "the second one" / "that one" pick an option; their "one" is not 1 PM
    t = re.sub(r"\b(the|that|this)\s+(?:(?:first|second|third|last|later|earlier|other|next|same)\s+)?one\b", " ", t)
    t = _NUMBER_RE.sub(lambda m: _NUMBER_WORDS[m.group(1)], t)
    t = re.sub(r"\bhalf past (\d{1,2})\b", r"\1:30", t)
    t = re.sub(r"\bquarter past (\d{1,2})\b", r"\1:15", t)
    t = re.sub(r"\b(\d{1,2}) (15|30|45)\b", r"\1:\2", t)
    t = re.sub(r"\b(noon|midday)\b", "12 pm", t)
    return t


def _parse_day(t: str, now: datetime) -> tuple[date, bool]:
    today = now.date()
    if "day after tomorrow" in t:
        return today + timedelta(days=2), True
    if re.search(r"\btomorrow\b", t):
        return today + timedelta(days=1), True
    if re.search(r"\b(today|tonight)\b", t):
        return today, True
    for index, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", t):
            delta = (index - today.weekday()) % 7
            if delta == 0 and (re.search(rf"\bnext {name}\b", t) or now.hour >= CLOSE):
                delta = 7
            return today + timedelta(days=delta), True
    return today, False


def _part_of_day(t: str) -> tuple[tuple[int, int] | None, str | None]:
    for word, hours, hint in _PARTS:
        if re.search(rf"\b{word}\b", t):
            return hours, hint
    return None, None


def _hour(h: str, m: str | None, ampm: str | None, hint: str | None) -> float | None:
    hour, minute = int(h), int(m or 0)
    if hour > 23 or minute > 59:
        return None
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    elif ampm is None and hour <= 12:
        if hint == "pm" and hour < 12:
            hour += 12
        elif hint is None and 1 <= hour <= 7:
            hour += 12
    return hour + minute / 60


def _parse_hours(t: str, hint: str | None) -> tuple[float, float] | None:
    for regex in _RANGE_RES:
        m = regex.search(t)
        if m:
            h1, m1, ap1, h2, m2, ap2 = m.groups()
            end = _hour(h2, m2, ap2, hint)
            start = _hour(h1, m1, ap1 or ap2, hint)
            if start is not None and end is not None and start > end and ap1 is None:
                start = _hour(h1, m1, None, hint)
            if start is not None and end is not None and end <= start and ap2 is None and end < 12:
                end += 12  # "seven to eight" -> 19:00-20:00
            return (start, end) if start is not None and end is not None else None
    m = _AFTER_RE.search(t)
    if m:
        x = _hour(*m.groups(), hint)
        return (x, CLOSE) if x is not None else None
    m = _BEFORE_RE.search(t)
    if m:
        x = _hour(*m.groups(), hint)
        return (OPEN, x) if x is not None else None
    m = _AROUND_RE.search(t)
    if m:
        x = _hour(*m.groups(), hint)
        return (x - 1, x + 1) if x is not None else None
    m = _SINGLE_RE.search(t)
    if m:
        x = _hour(*m.groups(), hint)
        return (x, x + 1) if x is not None else None
    return None


def _at(day: date, hours: float) -> datetime:
    return datetime.combine(day, time()) + timedelta(hours=hours)


def parse_time_window(phrase: str, now: datetime) -> TimeWindow | None:
    t = _normalize(phrase)
    if not t:
        return None
    if _ASAP_RE.search(t):
        close = _at(now.date(), CLOSE)
        if now >= close:
            tomorrow = now.date() + timedelta(days=1)
            return TimeWindow(_at(tomorrow, OPEN), _at(tomorrow, CLOSE))
        return TimeWindow(max(now, _at(now.date(), OPEN)), close)

    day, explicit_day = _parse_day(t, now)
    part, hint = _part_of_day(t)
    hours = _parse_hours(t, hint)
    if hours is None:
        if part is not None:
            hours = part
        elif explicit_day or _ANY_RE.search(t):
            hours = (OPEN, CLOSE)
        else:
            return None

    start_h, end_h = max(hours[0], OPEN), min(hours[1], CLOSE)
    if start_h >= end_h:
        return None
    if _at(day, end_h) <= now:
        if explicit_day:
            return None  # the caller named a day whose window has already passed
        day += timedelta(days=1)
    return TimeWindow(_at(day, start_h), _at(day, end_h))
