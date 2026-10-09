"""Deterministic checks on values the intent reader extracted."""
from __future__ import annotations

import re

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "a": 1, "an": 1, "another": 1}
_FILLERS = {"packet", "packets", "pack", "packs", "of", "unit", "units", "piece", "pieces", "more", "also",
            "please", "some", "extra"}
_STOPWORDS = {"the", "of", "to", "and", "or", "please", "can", "could", "you", "also", "add", "some", "my",
              "it", "is", "its", "for", "me", "want", "like", "would", "packet", "packets", "pack", "packs",
              "more", "with", "at", "in", "on", "an", "extra"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def grounded(value: str, utterance: str, min_share: float = 0.6) -> bool:
    """True when most content words of `value` were actually said by the customer."""
    words = [w for w in _words(value) if len(w) > 1 and w not in _STOPWORDS]
    if not words:
        return False
    said = set(_words(utterance))
    return sum(w in said for w in words) / len(words) >= min_share


def said_number(qty: int, utterance: str) -> bool:
    """True when the customer actually said this quantity (as digits or a number word)."""
    if qty == 1:
        return True
    words = _words(utterance)
    return str(qty) in words or any(w in words for w, n in _NUMBER_WORDS.items() if n == qty)


def split_quantity(value: str) -> tuple[int, str]:
    """'two packets of Amul milk' -> (2, 'Amul milk'). The quantity defaults to 1."""
    qty, found, rest = 1, False, []
    for token in value.split():
        word = re.sub(r"[^a-z0-9]", "", token.lower())
        if not found and (word.isdigit() or word in _NUMBER_WORDS):
            qty, found = (int(word) if word.isdigit() else _NUMBER_WORDS[word]), True
            continue
        if not word or word in _FILLERS:
            continue
        rest.append(token.strip(".,?!"))
    return max(qty, 1), " ".join(rest)
