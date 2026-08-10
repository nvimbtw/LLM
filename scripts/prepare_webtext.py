#!/usr/bin/env python3
"""
Populate a WebText dataset folder with a second, differently-filtered web pool.

Source: allenai/c4, the `en` config — 1,024 gzipped JSON-lines shards. Schema is
`{"text", "timestamp", "url"}`. Note this is **not** parquet like the other
fetchers; corpus_shards.iter_jsonl_rows handles it.

## Why a second web source

FineWeb-Edu is CommonCrawl filtered by an educational-quality classifier. That
filter is why it was chosen — the cheapest source of varied real English at
volume — but it is also a single model's opinion about what good text looks like,
applied to every token. At 1.5 GB out of 1.5 GB it was the only option. At ~50 GB
it should not still be the only view of the web.

**C4 over DCLM-baseline**, which was the other candidate and is the stronger
corpus by benchmark. C4 is cleaned heuristically — deduplication, boilerplate and
bad-word filters, no learned quality model — which makes it maximally *different*
from FineWeb-Edu. DCLM is itself model-filtered, and its filtering rhymes with
FineWeb-Edu's, so it would buy less diversity per gigabyte. Diversity is the
entire job of this dataset; if raw quality were the goal the right move would be
to fetch more FineWeb instead. (DCLM is also `.jsonl.zst`, which needs a
`zstandard` dependency this repo does not have.)

## Overlap with FineWeb is real and accepted

Both derive from CommonCrawl, so some documents appear in both and the packed
corpus will see them twice. Cross-dataset deduplication at this scale is its own
project — it needs a minhash index over tens of millions of documents — and is
out of scope here. Stated rather than left for the next reader to wonder about.

Usage:
    python3 prepare_webtext.py                 # 10 GB
    python3 prepare_webtext.py --mb 2000
    python3 prepare_webtext.py --dry-run
"""

import corpus_shards as cs

REPO_ID = "allenai/c4"
DATASET = "WebText"
DEFAULT_MB = 10_000

# The blocklist-filtered English config. `en.noblocklist` is the same crawl with
# the bad-word filter lifted; `en` is the one to want.
CONFIG = "en"

MIN_CHARS = 1000


def iter_pages(shards, min_chars):
    for row in cs.iter_jsonl_rows(REPO_ID, shards, ["text"]):
        body = cs.neutralize((row["text"] or "").strip())
        if len(body) < min_chars:
            continue
        yield body


def main():
    ap = cs.base_parser(__doc__.split("\n")[1], default_mb=DEFAULT_MB, dataset=DATASET)
    ap.set_defaults(min_chars=MIN_CHARS)
    args = ap.parse_args()

    cs.projection(f"{REPO_ID} [{CONFIG}]", args.mb)
    if args.dry_run:
        return

    # Only the train shards. The validation split is small and is the sort of
    # thing that belongs in eval/, not in training text.
    shards = [
        f for f in cs.list_parquet_shards(REPO_ID, prefix=f"{CONFIG}/", suffix=".json.gz")
        if "c4-train" in f
    ]
    print(f"{len(shards)} shards in {CONFIG}; fetching from the front as needed.")

    cs.run_fetch(
        args, "webtext",
        lambda reverse: iter_pages(
            list(reversed(shards)) if reverse else shards, args.min_chars
        ),
        prefix="webtext",
        manifest={"repo": REPO_ID, "config": CONFIG, "min_chars": args.min_chars},
    )


if __name__ == "__main__":
    main()
