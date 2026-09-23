"""Versioned, seeded problems. The solution is private until grading.

Splits use separate wording and random streams. Elementary facts can overlap
between splits: passing them measures fluency, not unseen-fact generalization.
"""

import hashlib
import random
from dataclasses import asdict, dataclass

VERSION = 1
SYSTEM = "Solve the question. End with Answer: followed by one number. No tools."
LEVELS = ("easy", "medium", "hard")
FAMILIES = ("addition", "subtraction", "multiplication", "division", "expressions", "word")
WORDING = {
    "train": ("Calculate {expr}.", "What is {expr}?", "Find {expr}."),
    "validation": ("Compute: {expr}", "Give the value of {expr}."),
    "test": ("Evaluate {expr}.", "Work out {expr}."),
}


@dataclass(frozen=True)
class Problem:
    task_type: str
    seed: int
    prompt: str
    answer: str
    solution: str
    problem_key: str
    split: str = "train"
    generator_version: int = VERSION
    system: str = SYSTEM

    def stored(self) -> dict:
        return asdict(self)

    def public(self) -> dict:
        return {key: getattr(self, key) for key in ("task_type", "seed", "prompt", "system")}


def prerequisites(family: str, level: str) -> tuple[str, ...]:
    rung = LEVELS.index(level)
    previous = (f"math_{family}_{LEVELS[rung - 1]}",) if rung else ()
    if family == "expressions":
        return previous + ("math_addition_easy", "math_multiplication_easy")
    if family == "word":
        return previous + ("math_addition_easy", "math_subtraction_easy")
    return previous


def _arithmetic(rng, family, rung):
    bound = (19, 99, 999)[rung]
    a, b = rng.randint(0, bound), rng.randint(0, bound)
    if family == "addition":
        if rung == 1:
            a, b = rng.randint(1, 8) * 10 + rng.randint(5, 9), rng.randint(1, 8) * 10 + rng.randint(5, 9)
        return f"{a} + {b}", a + b
    if family == "subtraction":
        if rung == 1:
            a = rng.randint(2, 9) * 10 + rng.randint(0, 3)
            b = rng.randint(0, a // 10 - 1) * 10 + rng.randint(5, 9)
        elif rung == 0:
            a, b = max(a, b), min(a, b)
        return f"{a} - {b}", a - b
    a, b = rng.randint(0, (12, 99, 999)[rung]), rng.randint(1, (12, 12, 99)[rung])
    if family == "multiplication":
        return f"{a} * {b}", a * b
    return f"{a * b} / {b}", a


def _expression(rng, rung):
    a, b, c = (rng.randint(1, (12, 20, 40)[rung]) for _ in range(3))
    if rung == 0:
        return f"{a} + {b} + {c}", a + b + c, f"{a} + {b} = {a+b}; {a+b} + {c} = {a+b+c}."
    if rung == 1:
        return f"{a} + {b} * {c}", a + b * c, f"{b} * {c} = {b*c}; {a} + {b*c} = {a+b*c}."
    return f"({a} + {b}) * {c}", (a+b)*c, f"{a} + {b} = {a+b}; {a+b} * {c} = {(a+b)*c}."


def _word(rng, rung, split):
    a, b, c = (rng.randint(1, 19) for _ in range(3))
    name = rng.choice({"train": ("Ada", "Sam"), "validation": ("Mia", "Leo"), "test": ("Ira", "Nia")}[split])
    item = {"train": "beads", "validation": "buttons", "test": "coins"}[split]
    if rung == 0:
        return f"{name} has {a} {item} and gets {b} more. How many now?", a+b, f"{a} + {b} = {a+b}.", f"add:{a}:{b}"
    if rung == 1:
        total = a + b + c
        return f"{name} has {total} {item}, gives away {b}, then gets {c}. How many now?", total-b+c, f"{total} - {b} = {total-b}; {total-b} + {c} = {total-b+c}.", f"change:{total}:{b}:{c}"
    total = a * b + c
    return f"{name} has {a} bags with {b} {item} each and {c} loose {item}. The bags are blue. How many {item} altogether?", total, f"{a} * {b} = {a*b}; {a*b} + {c} = {total}.", f"bags:{a}:{b}:{c}"


def generate(family: str, level: str, seed: int, split: str = "train") -> Problem:
    if split not in WORDING:
        raise ValueError(f"unknown math split {split!r}")
    digest = hashlib.sha256(f"{VERSION}:{family}:{level}:{split}:{seed}".encode()).digest()
    rng, rung = random.Random(int.from_bytes(digest, "big")), LEVELS.index(level)
    if family == "word":
        prompt, value, steps, key = _word(rng, rung, split)
    else:
        if family == "expressions":
            expr, value, steps = _expression(rng, rung)
        else:
            expr, value = _arithmetic(rng, family, rung)
            steps = f"{expr} = {value}."
        prompt, key = rng.choice(WORDING[split]).format(expr=expr), expr
    # Simple facts teach answering and stopping; composition teaches short steps.
    answer = str(value)
    solution = f"Answer: {answer}" if family not in ("expressions", "word") else f"{steps}\nAnswer: {answer}"
    return Problem(f"math_{family}_{level}", seed, prompt, answer, solution, key, split)
