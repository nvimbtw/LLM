#!/usr/bin/env python3
"""
Generate synthetic supervised fine-tuning (SFT) chat data as JSONL.

Emits one conversation per line into the backend's SFT input dir
(`backend/data/raw/sft/`), the directory `build_sft_dataset` reads. Each line
matches the schema the Rust side deserializes (see `SftConversation` /
`SftTurn` in backend/src/training.rs):

    {"system": "<optional system prompt>",
     "turns": [{"role": "user", "text": "..."},
               {"role": "assistant", "text": "..."}, ...]}

Rules the backend enforces (keep them true if you edit the banks below):
  - roles are exactly "user" or "assistant",
  - turns alternate user/assistant, starting with user,
  - every conversation ends on an assistant turn,
  - "system" is optional; omit the key entirely when there's no system prompt.

The content is deliberately plain, short, and templated. The point is to teach
the *chat turn structure and the assistant reply shape* to a tiny model, not to
teach math or world knowledge. Each category is a small generator function that
returns (user_text, assistant_text) pairs; add categories or expand the banks
(e.g. with a cheaper model) and the mix picks them up automatically.

Usage:
    python3 generate_sft.py --count 500
    python3 generate_sft.py --count 2000 --seed 7 --out /path/to/synthetic.jsonl
    python3 generate_sft.py --count 500 --append     # add to existing file
"""

import argparse
import json
import os
import random

DEFAULT_OUT = "/home/x_x/Programming/LLM/backend/data/raw/sft/synthetic.jsonl"

# Roughly one in SYSTEM_PROMPT_ODDS conversations gets a system prompt, so the
# model sees the system scaffold on a fraction of examples (and, via the loss
# mask, learns to condition on it without predicting it).
SYSTEM_PROMPT_ODDS = 3

SYSTEM_PROMPTS = [
    "You are a helpful assistant. Answer concisely.",
    "You are a friendly assistant. Be clear and brief.",
    "You are a precise assistant. Give short, direct answers.",
]

# --- Category generators -----------------------------------------------------
# Each returns a list of (role, text) turns, alternating user/assistant and
# ending on assistant. Most are single Q/A; some emit a follow-up turn so the
# model sees multi-turn structure too.


def arithmetic(rng):
    a, b = rng.randint(1, 50), rng.randint(1, 50)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        q, ans = f"What is {a} plus {b}?", a + b
    elif op == "-":
        q, ans = f"What is {a} minus {b}?", a - b
    else:
        q, ans = f"What is {a} times {b}?", a * b
    return [("user", q), ("assistant", f"{a} {op} {b} is {ans}.")]


def sort_numbers(rng):
    nums = rng.sample(range(1, 100), rng.randint(3, 6))
    ordered = sorted(nums)
    q = "Sort these numbers in ascending order: " + ", ".join(map(str, nums)) + "."
    a = "In ascending order: " + ", ".join(map(str, ordered)) + "."
    return [("user", q), ("assistant", a)]


CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Brazil", "Brasilia"),
    ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Norway", "Oslo"),
    ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Italy", "Rome"),
    ("Australia", "Canberra"), ("Spain", "Madrid"), ("Greece", "Athens"),
]


def capital(rng):
    country, city = rng.choice(CAPITALS)
    q = f"What is the capital of {country}?"
    a = f"The capital of {country} is {city}."
    return [("user", q), ("assistant", a)]


def echo(rng):
    words = ["apple", "river", "mountain", "coffee", "window", "planet",
             "garden", "pencil", "harbor", "lantern"]
    word = rng.choice(words)
    return [("user", f"Please repeat the word {word!r} back to me."),
            ("assistant", f"{word}")]


def define(rng):
    terms = {
        "gravity": "the force that pulls objects toward each other, like things toward the Earth",
        "photosynthesis": "the process plants use to turn sunlight into energy",
        "algorithm": "a step-by-step set of instructions for solving a problem",
        "democracy": "a system of government where people vote to make decisions",
        "evaporation": "when a liquid turns into a gas, usually from heat",
    }
    term, meaning = rng.choice(list(terms.items()))
    return [("user", f"What does {term} mean?"),
            ("assistant", f"{term.capitalize()} is {meaning}.")]


def count_letters(rng):
    words = ["banana", "strawberry", "hello", "mississippi", "balloon", "assistant"]
    word = rng.choice(words)
    letter = rng.choice(list(set(word)))
    n = word.count(letter)
    return [("user", f"How many times does the letter '{letter}' appear in {word!r}?"),
            ("assistant", f"The letter '{letter}' appears {n} time{'s' if n != 1 else ''} in {word!r}.")]


def greeting_followup(rng):
    # A short two-exchange conversation so the model sees multi-turn structure.
    name = rng.choice(["Alex", "Sam", "Priya", "Kenji", "Maria"])
    return [
        ("user", f"Hi, my name is {name}."),
        ("assistant", f"Hello {name}, nice to meet you. How can I help?"),
        ("user", "Can you remind me what my name is?"),
        ("assistant", f"Your name is {name}."),
    ]


CATEGORIES = [
    arithmetic,
    sort_numbers,
    capital,
    echo,
    define,
    count_letters,
    greeting_followup,
]


def make_conversation(rng):
    turns = rng.choice(CATEGORIES)(rng)
    conv = {}
    if rng.randint(1, SYSTEM_PROMPT_ODDS) == 1:
        conv["system"] = rng.choice(SYSTEM_PROMPTS)
    conv["turns"] = [{"role": r, "text": t} for r, t in turns]
    return conv


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic SFT chat JSONL.")
    ap.add_argument("--count", type=int, default=500,
                    help="number of conversations to generate (default 500)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output .jsonl path (default {DEFAULT_OUT})")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducibility (default 0)")
    ap.add_argument("--append", action="store_true",
                    help="append to the output file instead of overwriting")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    mode = "a" if args.append else "w"
    with open(args.out, mode, encoding="utf-8") as f:
        for _ in range(args.count):
            conv = make_conversation(rng)
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    verb = "Appended" if args.append else "Wrote"
    print(f"{verb} {args.count} conversations to {args.out}")


if __name__ == "__main__":
    main()
