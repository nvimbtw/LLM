"""Seeded transfer histories whose final holder is computed by simulation."""

import hashlib
import random

from gyms.contracts import Generation, Task

VERSION = 1
LEVELS = ("easy", "medium", "hard")
SYSTEM = "Track who has the named object. End with Answer: followed by one name. No tools."
NAMES = {
    "train": ("Ava", "Bo", "Cy", "Dee", "Eli"),
    "validation": ("Mia", "Leo", "Uma", "Ned", "Ira"),
    "test": ("Nia", "Sam", "Tia", "Max", "Zoe"),
}
OBJECTS = {
    "train": ("key", "coin", "ring"),
    "validation": ("badge", "card", "map"),
    "test": ("token", "note", "gem"),
}
EVENTS = {
    "easy": ("target",),
    "medium": ("target", "other", "swap"),
    "hard": ("target", "other", "swap", "target", "other", "swap"),
}


def _rng(request: Generation) -> random.Random:
    digest = hashlib.sha256(
        f"state:{VERSION}:{request.split}:{request.seed}".encode()
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


def generate(level: str, request: Generation) -> Task:
    if request.version != VERSION:
        raise ValueError(f"unsupported state tracking generator version {request.version}")
    if request.split not in NAMES:
        raise ValueError(f"unknown state tracking split {request.split!r}")
    if level not in EVENTS:
        raise ValueError(f"unknown state tracking level {level!r}")
    rng = _rng(request)
    names = rng.sample(NAMES[request.split], 3 if level == "easy" else 4)
    objects = rng.sample(OBJECTS[request.split], 1 if level == "easy" else 2)
    starting_holders = rng.sample(names, len(objects))
    holders = dict(zip(objects, starting_holders))
    initial = holders.copy()
    target = objects[0]
    other = objects[1] if len(objects) > 1 else None
    events = []
    trace = [", ".join(f"{item}: {holders[item]}" for item in objects)]
    for kind in EVENTS[level]:
        if kind == "swap":
            holders[target], holders[other] = holders[other], holders[target]
            events.append(f"The holder of the {target} swaps objects with the holder of the {other}.")
        else:
            item = target if kind == "target" else other
            previous = holders[item]
            # Keep holders distinct so a swap always changes both objects.
            blocked = {previous, holders[other if item == target else target]} if other else {previous}
            next_holder = rng.choice([name for name in names if name not in blocked])
            holders[item] = next_holder
            events.append(f"{previous} gives the {item} to {next_holder}.")
        trace.append(", ".join(f"{item}: {holders[item]}" for item in objects))
    facts = [f"{initial[item]} has the {item}." for item in objects]
    prompt = " ".join([*facts, *events, f"Who has the {target} now?"])
    answer = holders[target]
    solution = "\n".join([*trace, f"Answer: {answer}"])
    key = ":".join([*objects, *[initial[item] for item in objects], *events])
    return Task("state_tracking", f"state_tracking_{level}", request.seed, VERSION,
                prompt, SYSTEM, (), request.split, key,
                {"answer": answer, "solution": solution, "initial": initial,
                 "events": events, "target": target})
