#!/usr/bin/env python3
"""
Populate a Wikipedia dataset folder with plain-text article documents.

Source: the Wikimedia-maintained, pre-cleaned dump
(https://huggingface.co/datasets/wikimedia/wikipedia). Its `text` field is
already stripped of wiki markup, references, and tables — clean prose — so it
needs no further cleaning. The dataset is sharded parquet, keyed by a config of
the form "<date>.<lang>" (e.g. "20231101.en", "20231101.simple").

Articles are batched into shard files under a `--mb` budget, like every other
fetcher here. **This used to write one article per .txt, named by uuid4**, and
both halves of that were wrong at the scale full English Wikipedia needs:

- ~6.9M articles is ~6.9M files. `sources.collect()` sorts that whole glob on
  every pack, and no filesystem enjoys the directory.
- uuid names are not stable across fetches. `sources.collect` caps a dataset by
  taking the first N of the *sorted* list and promises the same request selects
  the same text every time; with random names a re-fetch reshuffled which
  articles sorted first, so a capped Wikipedia selection silently trained on
  different text than the run it was meant to reproduce. Sequential names fix
  that.

English Wikipedia is ~20 GB of text across 41 shards, which is the anchor source
for a bigger model. Shards are downloaded one at a time and the run stops as
soon as the budget is spent, so a small fetch only ever pulls the first. For a
smaller, self-contained corpus use the single-shard Simple English config:

    python3 prepare_wikipedia.py --lang simple

Marker neutralization matters here now. At 2000 hand-limited articles it did
not; across millions, the encyclopedia certainly contains an article quoting
`<|endoftext|>` verbatim, which would inject a false document boundary
mid-article.

Usage:
    python3 prepare_wikipedia.py                  # 20 GB, full English
    python3 prepare_wikipedia.py --mb 2000        # a smaller slice
    python3 prepare_wikipedia.py --lang simple    # Simple English (1 shard)
    python3 prepare_wikipedia.py --lang de --date 20231101
    python3 prepare_wikipedia.py --dry-run
"""

import corpus_shards as cs

REPO_ID = "wikimedia/wikipedia"
DATASET = "Wikipedia"

# Full English is ~20 GB of text; the default fetches it. Every other source in
# the corpus has its own budget, so this one being the largest is a choice made
# here rather than a limit of the script.
DEFAULT_MB = 20_000

# Articles below this are stubs — a title, an infobox and one sentence.
MIN_CHARS = 500


def iter_articles(config, shards, min_chars, max_chars):
    """Wikipedia rows as document bodies, each prefixed with its title.

    The title is kept as a heading line because it is the only structural signal
    in an article once the markup has been stripped.
    """
    for row in cs.iter_parquet_rows(REPO_ID, shards, ["title", "text"]):
        body = (row["text"] or "").strip()
        if len(body) < min_chars:
            continue
        if max_chars and len(body) > max_chars:
            body = body[:max_chars].rstrip()
        yield cs.neutralize(f"{row['title']}\n\n{body}")


def main():
    ap = cs.base_parser(__doc__.split("\n")[1], default_mb=DEFAULT_MB, dataset=DATASET)
    ap.set_defaults(min_chars=MIN_CHARS)
    ap.add_argument("--lang", default="en",
                    help="Wikipedia language code, e.g. en, simple, de (default: en)")
    ap.add_argument("--date", default="20231101",
                    help="dump snapshot date, YYYYMMDD (default: 20231101)")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="truncate articles longer than this (0 = no cap)")
    args = ap.parse_args()

    config = f"{args.date}.{args.lang}"
    cs.projection(f"{REPO_ID} [{config}]", args.mb)
    if args.dry_run:
        return

    shards = cs.list_parquet_shards(REPO_ID, prefix=f"{config}/")
    print(f"{len(shards)} shards in {config}; fetching from the front as needed.")

    cs.run_fetch(
        args, "wikipedia",
        lambda reverse: iter_articles(
            config, list(reversed(shards)) if reverse else shards,
            args.min_chars, args.max_chars,
        ),
        prefix="wikipedia",
        manifest={"repo": REPO_ID, "config": config, "min_chars": args.min_chars},
    )


if __name__ == "__main__":
    main()
