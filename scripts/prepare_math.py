#!/usr/bin/env python3
"""
Populate a Math dataset folder with math and reasoning-heavy text.

Source: HuggingFaceTB/finemath — CommonCrawl filtered for mathematical content,
in four quality-graded subsets. Verified schema (the columns used here):

    text         the page's extracted text
    score        the classifier's rating
    int_score    the same, bucketed
    language     ISO code, with language_score beside it

Chosen over open-web-math because it is the newer and more heavily filtered of
the two, and because its subsets are already graded — `finemath-4plus` is the
high-confidence slice and is the default here. That grading is the same idiom
prepare_fineweb.py's `--min-score` uses, so the two scripts read alike.

## Why math is in the corpus

The thinking-mode SFT stage trains the model to produce a `<|thinking|>` block
before answering. A base model whose pretrain corpus contains no worked
reasoning has nothing to put in one — it learns the *shape* of a reasoning trace
without the substance. Math text is the cheapest large source of explicit
step-by-step derivation.

## Deliberately smaller than its siblings

The default budget is well under the other sources'. A model at this scale gets
less out of math than out of general prose, and this is the most specialized
text in the corpus: it earns its place by giving the thinking block something to
imitate, not by volume. Raise `--mb` if a run shows it paying.

## LaTeX is kept; pages that are mostly LaTeX are not

`$…$` and `\\begin{align}` are legitimate structured content and the model may as
well see them. But a page that is 90% backslashes is markup with prose sprinkled
on top — it teaches the tokenizer far more than it teaches the model, exactly the
way a minified bundle does in prepare_code.py. Filtered on the same principle:
the fraction of non-alphanumeric, non-whitespace characters.

Usage:
    python3 prepare_math.py                       # 3 GB from finemath-4plus
    python3 prepare_math.py --mb 8000
    python3 prepare_math.py --subset finemath-3plus   # larger, lower precision
    python3 prepare_math.py --dry-run
"""

import corpus_shards as cs

REPO_ID = "HuggingFaceTB/finemath"
DATASET = "Math"

# Under the other sources on purpose — see the docstring. 3 GB is ~1.4B tokens,
# enough for the reasoning register to register without crowding out prose.
DEFAULT_MB = 3_000

# The high-confidence subset. `finemath-3plus` is roughly twice the size at lower
# precision, and `infiwebmath-*` are the companion web-math sets.
DEFAULT_SUBSET = "finemath-4plus"
SUBSETS = ("finemath-4plus", "finemath-3plus", "infiwebmath-4plus", "infiwebmath-3plus")

MIN_CHARS = 1000

# Above this fraction of symbols the page is a formula dump rather than an
# explanation. Higher than prepare_code.py's threshold because mathematical
# notation is legitimately symbol-dense — this only catches the extreme tail.
MAX_SYMBOL_RATIO = 0.6


def mostly_markup(text: str) -> bool:
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return symbols / max(len(text), 1) > MAX_SYMBOL_RATIO


def iter_pages(shards, min_chars):
    for row in cs.iter_parquet_rows(REPO_ID, shards, ["text"]):
        body = cs.neutralize((row["text"] or "").strip())
        if len(body) < min_chars or mostly_markup(body):
            continue
        yield body


def main():
    ap = cs.base_parser(__doc__.split("\n")[1], default_mb=DEFAULT_MB, dataset=DATASET)
    ap.set_defaults(min_chars=MIN_CHARS)
    ap.add_argument("--subset", default=DEFAULT_SUBSET, choices=SUBSETS,
                    help=f"which graded subset to draw from (default: {DEFAULT_SUBSET})")
    args = ap.parse_args()

    cs.projection(f"{REPO_ID} [{args.subset}]", args.mb)
    if args.dry_run:
        return

    shards = cs.list_parquet_shards(REPO_ID, prefix=f"{args.subset}/")
    print(f"{len(shards)} shards in {args.subset}; fetching from the front as needed.")

    cs.run_fetch(
        args, "math",
        lambda reverse: iter_pages(
            list(reversed(shards)) if reverse else shards, args.min_chars
        ),
        prefix="math",
        manifest={"repo": REPO_ID, "subset": args.subset, "min_chars": args.min_chars},
    )


if __name__ == "__main__":
    main()
