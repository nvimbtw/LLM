"""Entity prelexing: the small static lexicon of names that earn a vocab slot.

A byte-level BPE spends four pieces on "Australia" every time it appears, and
this corpus mentions such phrases ~77 times per 1000 tokens. Reserving a handful
of single-token ids for the most expensive recurring names buys a few percent off
the corpus with no change to the model at all — the trunk simply sees shorter,
less noisy id sequences.

The detector is deliberately dumb: a Titlecase-run regex plus stopword and
calendar filters. It is a *token accounting* heuristic, not named-entity
recognition — a phrase earns its slot by costing pieces and recurring, and it
does not matter whether a linguist would call it an entity.

Derived from engine/explore/entity (E7), with one correction: collapsing an
n-piece phrase into one token saves (n-1) pieces per mention, not n. E7's budget
table summed the latter and overstated the corpus reduction by about a third.
"""

import re
from collections import Counter
from pathlib import Path

from llm.eval.surface import strip_markers

# Runs of Titlecase words. Deliberately not matching ALLCAPS: those are usually
# acronyms, which are already cheap, or shouting, which is not an entity.
PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")

# Calendar words are Titlecase and frequent but carry no identity, so a slot
# spent on one buys nothing a merge would not already have found.
CALENDAR = frozenset(
    "january february march april may june july august september october "
    "november december monday tuesday wednesday thursday friday saturday "
    "sunday summer winter spring autumn today yesterday tomorrow".split()
)

# Sentence-initial capitalisation makes any word look like a name. These are the
# ones that actually start sentences often enough to matter.
LEADING_STOPWORDS = frozenset(
    "the of and to in is on for with by are was be at that as from or have an "
    "also were has been which they do more than it its these so one two he she "
    "we you what when where how those their not into over all this after but if "
    "there here then now while because although however".split()
)

# A phrase must recur to be worth a permanent id; a hapax name would burn a slot
# to save a couple of pieces exactly once.
MIN_MENTIONS = 3


def _is_name_like(phrase: str) -> bool:
    words = phrase.lower().split()
    if not words:
        return False
    if words[0] in LEADING_STOPWORDS:
        return False
    if all(w in CALENDAR for w in words):
        return False
    return True


def candidates(files: list[Path], min_mentions: int = MIN_MENTIONS) -> Counter:
    """Count name-like phrases across raw source text, keyed by surface form.

    Runs on the `.txt` sources rather than the packed corpus on purpose: the
    lexicon has to be chosen *before* the tokenizer that would encode it exists.
    """
    counts: Counter = Counter()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for phrase in PROPER.findall(strip_markers(text)):
            if _is_name_like(phrase):
                counts[phrase] += 1

    return Counter({p: n for p, n in counts.items() if n >= min_mentions})


def rank(counts: Counter, piece_cost, slots: int) -> list[str]:
    """The `slots` phrases whose single-token ids save the most pieces.

    `piece_cost` maps a phrase to how many pieces it currently costs. Savings are
    `mentions * (pieces - 1)`, so a phrase the tokenizer already emits as one
    piece scores zero and is dropped rather than wasting a slot.
    """
    if slots <= 0:
        return []

    savings = {}
    for phrase, mentions in counts.items():
        pieces = piece_cost(phrase)
        if pieces >= 2:
            savings[phrase] = mentions * (pieces - 1)

    ranked = sorted(savings, key=lambda p: (-savings[p], p))
    return ranked[:slots]
