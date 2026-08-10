#!/usr/bin/env python3
"""
Populate a Books dataset folder from PG19, a local Project Gutenberg corpus.

Source: the PG19 corpus already on disk at ../pg19 — 28,602 books, 11 GB, in
`train/`, `validation/` and `test/` splits of one .txt per book. Nothing is
downloaded; this is the cheapest source in the corpus.

It is also the only *long-form narrative* text in it. Everything else fetched
here is web pages, encyclopedia entries, Q&A or code — documents of a few
thousand characters with no structure past the paragraph. A book is the one
thing in the corpus that rewards a context window, and the only place the model
meets a voice sustained over more than a page.

## Boilerplate, and why it is stripped

Measured over 3,000 books of the train split:

    *** START/END OF THIS PROJECT GUTENBERG ***    0.0%   (already stripped)
    "End of the Project Gutenberg EBook of ..."   78.5%
    "Produced by ..." transcriber credit          74.3%

The standard markers are gone, but the trailer and the credit are not. Repeated
across 28,602 books, "End of the Project Gutenberg EBook of" is a sentence the
model would learn better than anything in the actual prose. Both are cut, and
only when they appear in the outer edges of the file — a novel that mentions
Project Gutenberg in its middle keeps its text. Measured over 800 books, this
removes 100% of both.

What it does not remove is the "SMALL PRINT" licence block some older texts
carry (3.9%), which sits above the trailer and has no consistent opening. At
that frequency it is not worth the false positives a looser pattern would cost.

## Splits

Only `train/` is written to input/. `validation/` and `test/` are held out by
construction, and `--heldout-mb` draws from `validation/` rather than from the
tail of train — the same reasoning that keeps eval/ out of input/ at all.

## Book-sized documents

A book is 400 KB on average and can be several megabytes — far larger than one
shard. Documents are never split, so each book effectively becomes its own shard
file. That is the right trade: the pretrain loop shuffles whole files, so one
book per file is *better* shuffle granularity than a book cut across three, and
`--max-chars` is there if a specific run wants a ceiling.

Usage:
    python3 prepare_pg19.py                    # 11 GB, the whole train split
    python3 prepare_pg19.py --mb 2000          # a smaller slice
    python3 prepare_pg19.py --heldout-mb 8     # plus held-out books
    python3 prepare_pg19.py --src /path/to/pg19
"""

import os
import re
import sys

import corpus_shards as cs

DATASET = "Books"

# The whole train split. Local, so the only cost of the default is disk.
DEFAULT_MB = 11_000

# Where PG19 already lives, relative to the repo root.
DEFAULT_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pg19"
)

# A book is long; a file this short is a fragment rather than a volume.
MIN_CHARS = 2000

# How far into each end of a file the boilerplate patterns are honoured. Outside
# these windows a match is treated as the book's own words, not as a wrapper:
# `Produced by` is an ordinary English phrase and cutting on it mid-novel would
# silently discard chapters.
HEAD_WINDOW = 5000
TAIL_WINDOW = 10_000

# The transcriber credit, and the trailer. Anchored to a line start so a
# sentence that merely contains the words does not match.
HEAD_CREDIT = re.compile(r"^(Produced by|Transcribed (from|by)|E-?text prepared by)\b.*?(\n\s*\n)",
                         re.M | re.S | re.I)
TAIL_TRAILER = re.compile(r"\n[^\n]*End of (the |The )?Project Gutenberg.*$", re.S | re.I)


def strip_boilerplate(text: str) -> str:
    """Remove the Gutenberg wrapper from the edges of a book."""
    head, rest = text[:HEAD_WINDOW], text[HEAD_WINDOW:]
    match = HEAD_CREDIT.search(head)
    if match:
        head = head[match.end():]
    text = head + rest

    cut = len(text) - TAIL_WINDOW
    tail_match = TAIL_TRAILER.search(text, max(cut, 0))
    if tail_match:
        text = text[: tail_match.start()]
    return text.strip()


def iter_books(directory: str, min_chars: int, max_chars: int):
    """Every book in one split, sorted by name so a capped run is reproducible."""
    if not os.path.isdir(directory):
        sys.exit(
            f"No PG19 split at {directory}.\n"
            f"Pass --src pointing at the pg19 directory (it should contain "
            f"train/ and validation/)."
        )
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(directory, name), encoding="utf-8", errors="replace") as f:
            body = strip_boilerplate(f.read())
        if len(body) < min_chars:
            continue
        if max_chars and len(body) > max_chars:
            body = body[:max_chars].rstrip()
        # Neutralized like every other source: these are transcribed texts and a
        # book about typography can contain anything.
        yield cs.neutralize(body)


def main():
    ap = cs.base_parser(__doc__.split("\n")[1], default_mb=DEFAULT_MB, dataset=DATASET)
    ap.set_defaults(min_chars=MIN_CHARS)
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help=f"the PG19 directory (default: {DEFAULT_SRC})")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="truncate books longer than this (0 = no cap)")
    args = ap.parse_args()

    cs.projection(f"local PG19 at {args.src}", args.mb, {"download": "none (already on disk)"})
    if args.dry_run:
        return

    if not os.path.isdir(args.src):
        sys.exit(f"No PG19 directory at {args.src}. Pass --src.")

    cs.run_fetch(
        args, "pg19",
        # Held-out books come from the validation split, never from train — so
        # "held out" needs no bookkeeping and cannot drift.
        lambda reverse: iter_books(
            os.path.join(args.src, "validation" if reverse else "train"),
            args.min_chars, args.max_chars,
        ),
        prefix="pg19",
        manifest={"source": args.src, "split": "train", "min_chars": args.min_chars},
    )


if __name__ == "__main__":
    main()
