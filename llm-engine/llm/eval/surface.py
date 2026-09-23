"""Surface properties of generated text: is it words, and is it looping?

The lexicon is derived from the reference corpus rather than shipped as a
dictionary, which is worth being precise about: `real_word_rate` measures
agreement with *this* corpus's vocabulary. That is the right instrument for
separating byte soup and invented morphology from actual vocabulary, but it
cannot detect a real English word the reference text happens not to contain.
"""

import math
import re
from dataclasses import dataclass

# Markers reach word extraction spelled out, so `<|par|>` would yield "par" and
# every marker the model emits would count against real_word_rate. Markup
# correctness is the markup suite's job; vocabulary is this one's.
MARKER = re.compile(r"<\|/?[a-zA-Z0-9:_]+\|>")


@dataclass
class SurfaceMetrics:
    real_word_rate: float
    distinct_1: float
    distinct_2: float
    distinct_3: float
    repeat_3: float
    words: int
    mean_sample_words: float

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in vars(self).items()}


class Lexicon:
    """The vocabulary of the reference corpus."""

    def __init__(self, words: set[str]):
        self.words = words

    @classmethod
    def build(cls, text: str) -> "Lexicon":
        return cls(set(words_in(text)))

    def contains(self, word: str) -> bool:
        return word in self.words

    def __len__(self) -> int:
        return len(self.words)


def words_in(text: str) -> list[str]:
    """Lowercased alphabetic words, keeping mid-word apostrophes.

    Deliberately marker-unaware — callers strip markers first.
    """
    words: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch.isalpha():
            current.append(ch.lower())
        elif ch in ("'", "’") and current:
            # Only mid-word; a leading quote starts no word and a trailing one is
            # trimmed on flush.
            current.append("'")
        elif current:
            words.append("".join(current).rstrip("'"))
            current = []
    if current:
        words.append("".join(current).rstrip("'"))
    return [w for w in words if w]


def strip_markers(text: str) -> str:
    return MARKER.sub(" ", text)


def distinct_n(words: list[str], n: int) -> float:
    """Unique n-grams over total n-grams. 1.0 means nothing repeated; low values
    mean the model is cycling."""
    if n == 0 or len(words) < n:
        return math.nan
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def repeat_rate(words: list[str], n: int) -> float:
    """Fraction of n-grams appearing more than once anywhere in the pooled text."""
    if n == 0 or len(words) < n:
        return math.nan
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts: dict[tuple, int] = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    return sum(1 for g in grams if counts[g] > 1) / len(grams)


def analyze(samples: list[str], lexicon: Lexicon) -> SurfaceMetrics:
    per_sample = [words_in(strip_markers(s)) for s in samples]
    pooled = [w for sample in per_sample for w in sample]

    if not pooled:
        return SurfaceMetrics(math.nan, math.nan, math.nan, math.nan, math.nan, 0, math.nan)

    real = sum(1 for w in pooled if lexicon.contains(w))
    return SurfaceMetrics(
        real_word_rate=real / len(pooled),
        distinct_1=distinct_n(pooled, 1),
        distinct_2=distinct_n(pooled, 2),
        distinct_3=distinct_n(pooled, 3),
        repeat_3=repeat_rate(pooled, 3),
        words=len(pooled),
        mean_sample_words=len(pooled) / len(per_sample),
    )
