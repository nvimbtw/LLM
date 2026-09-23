"""Seeded ordering puzzles checked by exhaustive enumeration."""

import hashlib
import itertools
import random

from gyms.contracts import Generation, Task

VERSION = 1
LEVELS = ("easy", "medium", "hard")
SYSTEM = "Use the clues to find the order. End with Answer: followed by one name. No tools."
NAMES = {
    "train": ("Ava", "Bo", "Cy", "Dee"),
    "validation": ("Mia", "Leo", "Uma", "Ned"),
    "test": ("Nia", "Sam", "Tia", "Max"),
}
COUNT = {"easy": 2, "medium": 3, "hard": 4}
POSITIONS = ("first", "second", "third", "fourth")


def _rng(request: Generation) -> random.Random:
    digest = hashlib.sha256(
        f"logic:{VERSION}:{request.split}:{request.seed}".encode()
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _clues(order: tuple[str, ...]) -> list[tuple[str, str, str | None]]:
    facts = []
    for left, right in itertools.permutations(order, 2):
        if order.index(left) < order.index(right):
            facts.append(("before", left, right))
        if order.index(left) + 1 == order.index(right):
            facts.append(("next", left, right))
    if len(order) > 2:
        for name in order:
            if order[0] != name:
                facts.append(("not_first", name, None))
            if order[-1] != name:
                facts.append(("not_last", name, None))
    return facts


def _holds(order: tuple[str, ...], clue: tuple[str, str, str | None]) -> bool:
    kind, left, right = clue
    if kind == "not_first":
        return order[0] != left
    if kind == "not_last":
        return order[-1] != left
    if kind == "before":
        return order.index(left) < order.index(right)
    return order.index(left) + 1 == order.index(right)


def _describe(clue: tuple[str, str, str | None]) -> str:
    kind, left, right = clue
    if kind == "not_first":
        return f"{left} is not first."
    if kind == "not_last":
        return f"{left} is not last."
    if kind == "before":
        return f"{left} is before {right}."
    return f"{left} is immediately before {right}."


def _select(order: tuple[str, ...], rng: random.Random) -> list[tuple[str, str, str | None]]:
    remaining = _clues(order)
    rng.shuffle(remaining)
    possible = list(itertools.permutations(order))
    chosen = []
    while len(possible) > 1:
        useful = [(clue, [candidate for candidate in possible if _holds(candidate, clue)])
                  for clue in remaining]
        useful = [(clue, subset) for clue, subset in useful if len(subset) < len(possible)]
        clue, possible = min(useful, key=lambda item: len(item[1]))
        chosen.append(clue)
        remaining.remove(clue)
    rng.shuffle(chosen)
    return chosen


def generate(level: str, request: Generation) -> Task:
    if request.version != VERSION:
        raise ValueError(f"unsupported constraint logic generator version {request.version}")
    if request.split not in NAMES:
        raise ValueError(f"unknown constraint logic split {request.split!r}")
    if level not in COUNT:
        raise ValueError(f"unknown constraint logic level {level!r}")
    rng = _rng(request)
    order = tuple(rng.sample(NAMES[request.split], COUNT[level]))
    clues = _select(order, rng)
    queried = rng.randrange(len(order))
    people = ", ".join(sorted(order))
    sentences = [_describe(clue) for clue in clues]
    prompt = (f"{people} stand in a line, facing forward. "
              + " ".join(sentences)
              + f" Who is {POSITIONS[queried]}?")
    answer = order[queried]
    solution = ("The only order from first to last is " + ", ".join(order)
                + f".\nAnswer: {answer}")
    key = ":".join([*order, str(queried), *sentences])
    return Task("constraint_logic", f"constraint_logic_{level}", request.seed,
                VERSION, prompt, SYSTEM, (), request.split, key,
                {"answer": answer, "solution": solution,
                 "people": list(order), "clues": [list(clue) for clue in clues],
                 "queried": queried})
