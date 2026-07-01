#!/usr/bin/env python3
"""
Inject paired thinking blocks into already-marked-up training files.

A thinking block looks like `<|thinking|>...reasoning...<|/thinking|>` — both
delimiters are protected special tokens (see SPECIAL_TOKENS in pretokenize.rs),
the reasoning between them is ordinary generated text. The frontend hides the
content and shows a themed status label, so the model only learns the two clean
markers.

Blocks are inserted on their own line at random gaps BETWEEN top-level blocks,
never inside a <|par|>…<|/par|> (or any other element). Any pre-existing thinking
blocks (in either the paired or the older parametric form) are stripped first, so
the script is safe to re-run and to switch formats.

Usage:
    python3 inject_thinking.py <file_or_dir> [<file_or_dir> ...]
"""

import os
import random
import re
import sys

# Roughly one thinking block per this many eligible gaps.
INJECT_EVERY = 6
SEED = 1234

THOUGHTS = [
    "weighing the options here",
    "considering what comes next",
    "tracing the logic of this",
    "piecing the details together",
    "reasoning about the cause",
    "mapping out the structure",
    "checking the assumptions",
    "thinking through the implications",
    "recalling what was established",
    "deciding how to frame this",
]

PAR_OPEN = "<|par|>"
PAR_CLOSE = "<|/par|>"

# Matches a whole thinking element in either form: paired
# <|thinking|>...<|/thinking|> or the older parametric <|thinking | '...'|>.
THINKING_ANY_RE = re.compile(r"<\|thinking\|>.*?<\|/thinking\|>|<\|thinking\b.*?\|>", re.S)


def split_blocks(text: str):
    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", text) if b.strip()]
    # Drop any existing thinking blocks so we start from a clean slate.
    return [b for b in blocks if not b.startswith("<|thinking")]


def inject(text: str, rng: random.Random) -> str:
    # Remove inline remnants first (in case a thinking element shares a line).
    text = THINKING_ANY_RE.sub("", text)
    blocks = split_blocks(text)
    out = []
    depth = 0  # open paragraph depth; only inject at a balanced boundary
    for i, block in enumerate(blocks):
        out.append(block)
        depth += block.count(PAR_OPEN) - block.count(PAR_CLOSE)
        last = i == len(blocks) - 1
        if not last and depth == 0 and rng.randrange(INJECT_EVERY) == 0:
            thought = rng.choice(THOUGHTS)
            out.append(f"<|thinking|>{thought}<|/thinking|>")
    return "\n\n".join(out) + "\n"


def iter_files(targets):
    for t in targets:
        if os.path.isdir(t):
            for name in sorted(os.listdir(t)):
                if name.endswith(".txt"):
                    yield os.path.join(t, name)
        elif os.path.isfile(t) and t.endswith(".txt"):
            yield t


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    rng = random.Random(SEED)
    changed = total_tokens = 0
    for path in iter_files(sys.argv[1:]):
        original = open(path, encoding="utf-8").read()
        injected = inject(original, rng)
        with open(path, "w", encoding="utf-8") as f:
            f.write(injected)
        changed += 1
        total_tokens += injected.count("<|thinking|>")

    print(f"processed {changed} file(s)")
    print(f"total thinking blocks: {total_tokens}")


if __name__ == "__main__":
    main()
