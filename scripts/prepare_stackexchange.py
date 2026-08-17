#!/usr/bin/env python3
"""
Populate a StackExchange dataset folder with technical question-and-answer prose.

Source: donfu/oa-stackexchange — StackExchange dumps already paired into
question/answer rows, 14 parquet shards, ~452k rows each. Verified schema:

    INSTRUCTION  the question, title line followed by its body
    RESPONSE     one answer
    SOURCE       which site, e.g. "stackexchange-mythology"
    METADATA     JSON string: {"answer_score", "question_score", "tags"}

Chosen over the RedPajama and Dolma StackExchange slices because the pairing is
already done and the scores are already extracted; the alternative is parsing raw
post dumps to rebuild threads that this repo has rebuilt correctly.

## Why this source at all

The SFT sets this model is heading toward are written in a technical register —
questions, answers, code spans, step-by-step explanation. Nothing else in the
pretrain corpus looks like that: TinyStories is children's fiction, Wikipedia is
encyclopedic, FineWeb-Edu is filtered toward textbook prose, PG19 is novels.
StackExchange is the closest large source to the shape of an actual assistant
turn while still being ordinary pretrain text rather than chat data.

## One document per exchange

The question and its answer are written as a single document. An answer without
its question is unanchored — the model would learn to produce explanations with
no notion of what prompts one. The layout is fixed across every document, so it
is a learnable pattern rather than noise:

    <title/question body>

    Answer:

    <answer body>

Kept deliberately plain. This is pretrain text, so it is **not** converted to the
marker vocabulary — that conversion belongs to the SFT fetchers, and doing it
here would teach the base model to emit chat markup in the middle of prose.

`--min-score` drops low-rated answers. The default keeps anything non-negative:
a score of 0 is usually a correct answer to an unpopular question, while
negative ones are wrong often enough to be worth losing.

Usage:
    python3 prepare_stackexchange.py                 # 10 GB
    python3 prepare_stackexchange.py --mb 2000
    python3 prepare_stackexchange.py --min-score 3   # only well-rated answers
    python3 prepare_stackexchange.py --dry-run
"""

import json

import corpus_shards as cs

REPO_ID = "donfu/oa-stackexchange"
DATASET = "StackExchange"
DEFAULT_MB = 10_000

# Answers below this are kept out. 0 rather than 1: an unvoted answer to a quiet
# question is usually fine, a downvoted one usually is not.
MIN_SCORE = 0

# Shorter than this and the pair is a one-line question with a one-line answer —
# true, but not enough prose to learn the register from.
MIN_CHARS = 400

ANSWER_SEPARATOR = "\n\nAnswer:\n\n"


def _score(metadata) -> int:
    """The answer's score, or 0 when the metadata is missing or malformed.

    METADATA is a JSON *string*, not a struct, so it has to be parsed per row and
    can fail per row.
    """
    if not metadata:
        return 0
    try:
        return int(json.loads(metadata).get("answer_score", 0) or 0)
    except (ValueError, TypeError, AttributeError):
        return 0


def iter_exchanges(shards, min_score, min_chars):
    for row in cs.iter_parquet_rows(
        REPO_ID, shards, ["INSTRUCTION", "RESPONSE", "METADATA"]
    ):
        question = (row["INSTRUCTION"] or "").strip()
        answer = (row["RESPONSE"] or "").strip()
        if not question or not answer:
            continue
        if _score(row["METADATA"]) < min_score:
            continue
        body = cs.neutralize(question + ANSWER_SEPARATOR + answer)
        if len(body) < min_chars:
            continue
        yield body


def main():
    ap = cs.base_parser(__doc__.split("\n")[1], default_mb=DEFAULT_MB, dataset=DATASET)
    ap.set_defaults(min_chars=MIN_CHARS)
    ap.add_argument("--min-score", type=int, default=MIN_SCORE,
                    help=f"drop answers scored below this (default: {MIN_SCORE})")
    args = ap.parse_args()

    cs.projection(REPO_ID, args.mb)
    if args.dry_run:
        return

    shards = cs.list_parquet_shards(REPO_ID, prefix="data/")
    print(f"{len(shards)} shards available; fetching from the front as needed.")

    cs.run_fetch(
        args, "stackexchange",
        lambda reverse: iter_exchanges(
            list(reversed(shards)) if reverse else shards, args.min_score, args.min_chars
        ),
        prefix="stackexchange",
        manifest={"repo": REPO_ID, "min_score": args.min_score, "min_chars": args.min_chars},
    )


if __name__ == "__main__":
    main()
