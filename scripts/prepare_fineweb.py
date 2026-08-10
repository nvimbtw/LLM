#!/usr/bin/env python3
"""
Populate a FineWeb-Edu dataset folder with general-purpose English web text.

Source: EleutherAI/fineweb-edu-dedup-10b — a deduplicated ~10B-token slice of
HuggingFace's FineWeb-Edu, which is CommonCrawl filtered by an educational-quality
classifier. 97 parquet shards, 9.5M documents, ~48 GB of text uncompressed. Each
row carries `text`, `id`, and a `metadata` struct whose `score`/`int_score` are the
classifier's rating (higher = more textbook-like).

Shards are fetched only as far as the byte budget needs, which matters: one shard
is 278 MB on disk and yields ~496 MB of text, so the whole dataset is ~27 GB of
download. (The dataset card's `download_size` says 2.8 GB; that is wrong by ~10x —
the figure above is measured.) A default 1500 MB run pulls ~3 shards.

Why this dataset and not more Wikipedia: it is the cheapest source of *varied,
real* English at volume, which is what the model needs before an SFT set written
in a technical register is within reach at all. It is not the only general-web
source any more — see prepare_webtext.py, which exists because this one's
educational-quality classifier is a single model's opinion applied to every
token.

The sharding, marker neutralization, manifest and held-out machinery live in
corpus_shards.py; what stays here is FineWeb's schema and its quality filter.

The held-out slice is worth taking. The benchmark's reference corpus
(data/eval/corpus/) was 100% TinyStories, so once pretraining shifted to web text
the headline bits-per-byte measured a domain the model was deliberately spending
less capacity on — it can get worse while the model gets better. Passing
--heldout-mb writes a FineWeb slice there too, drawn from the *last* shards while
training data comes from the first, so the two are disjoint by construction.

Usage:
    python3 prepare_fineweb.py                        # 1500 MB, ~680M tokens
    python3 prepare_fineweb.py --mb 10000             # ~10 GB
    python3 prepare_fineweb.py --mb 1500 --heldout-mb 4
    python3 prepare_fineweb.py --min-score 3.5        # stricter quality filter
    python3 prepare_fineweb.py --dry-run              # just show the projections
"""

import corpus_shards as cs
import paths  # noqa: F401  (imported for its side-effect-free path resolution)

REPO_ID = "EleutherAI/fineweb-edu-dedup-10b"
DATASET = "FineWeb"

# Text a shard yields, and what it costs to fetch. Measured, not taken from the
# dataset card: the card's `download_size` (2.8 GB for 97 shards) understates it by
# ~10x — shard 0 is 278 MB on disk. Text-per-shard is the card's `num_bytes`
# (48.1 GB) over 97, which matches what a shard actually decodes to.
TEXT_PER_SHARD = 48_116_092_268 / 97
DOWNLOAD_PER_SHARD = 291_943_086


def iter_documents(shards, min_score, min_chars):
    """FineWeb rows as cleaned document bodies."""
    for row in cs.iter_parquet_rows(REPO_ID, shards, ["text", "metadata"]):
        if min_score and (row["metadata"] or {}).get("score", 0.0) < min_score:
            continue
        body = cs.neutralize((row["text"] or "").strip())
        if len(body) < min_chars:
            continue
        yield body


def main():
    ap = cs.base_parser(__doc__.split("\n")[1], default_mb=1500, dataset=DATASET)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop documents below this educational-quality score (0 = keep all)")
    args = ap.parse_args()

    # Filtering discards documents, so more shards are read than the raw text
    # ratio implies; the download figure is a floor and rises with --min-score.
    needed = (args.mb * 1e6 + args.heldout_mb * 1e6) / TEXT_PER_SHARD
    cs.projection(REPO_ID, args.mb, {
        "download": f">={needed * DOWNLOAD_PER_SHARD / 1e9:.1f} GB "
                    f"(~{max(1, round(needed))} shards, cached under ~/.cache/huggingface)",
    })
    if args.dry_run:
        return

    shards = cs.list_parquet_shards(REPO_ID)
    print(f"{len(shards)} shards available; fetching from the front as needed.")

    cs.run_fetch(
        args, "fineweb",
        lambda reverse: iter_documents(
            list(reversed(shards)) if reverse else shards, args.min_score, args.min_chars
        ),
        prefix="fineweb",
        manifest={"repo": REPO_ID, "min_score": args.min_score, "min_chars": args.min_chars},
    )


if __name__ == "__main__":
    main()
