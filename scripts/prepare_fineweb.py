#!/usr/bin/env python3
"""
Populate a FineWeb-Edu dataset folder under the backend uploads dir with
general-purpose English web text.

Source: EleutherAI/fineweb-edu-dedup-10b — a deduplicated ~10B-token slice of
HuggingFace's FineWeb-Edu, which is CommonCrawl filtered by an educational-quality
classifier. 97 parquet shards, 9.5M documents, ~48 GB of text uncompressed. Each
row carries `text`, `id`, and a `metadata` struct whose `score`/`int_score` are the
classifier's rating (higher = more textbook-like).

Shards are fetched only as far as the byte budget needs, which matters: one shard
is 278 MB on disk and yields ~496 MB of text, so the whole dataset is ~27 GB of
download. (The dataset card's `download_size` says 2.8 GB; that is wrong by ~10x —
the figure above is measured.) A default 1500 MB run pulls ~3 shards.

Why this dataset and not more Wikipedia: the models here have been trained on
TinyStories plus a few thousand Wikipedia articles, which is a few million tokens
against a parameter count that wants billions. FineWeb-Edu is the cheapest source
of *varied, real* English at that volume — which is what the model needs before an
SFT set written in a technical register is within reach at all.

Unlike prepare_wikipedia.py, which writes one article per .txt, this batches many
documents into each output file. At these volumes one-file-per-document would mean
hundreds of thousands of files; batching to a byte target keeps the directory sane
while staying fine-grained enough for the pretrain loop, which shuffles at file
granularity (each file becomes one DocSpan in segments.json).

Documents are separated by `<|endoftext|>`, the marker the Rust pretrain builder
already treats as a document boundary. Any literal marker-looking text in the
source is neutralized first — this is scraped web text and can genuinely contain
`<|endoftext|>` in, say, a page about tokenizers.

The held-out slice is worth taking. The benchmark's reference corpus
(data/eval/corpus/) is currently 100% TinyStories, so once pretraining shifts to
web text the headline bits-per-byte measures a domain the model is deliberately
spending less capacity on — it can get worse while the model gets better. Passing
--heldout-mb writes a FineWeb slice there too, drawn from the *last* shards while
training data comes from the first, so the two are disjoint by construction.

Usage:
    python3 prepare_fineweb.py                        # 1500 MB, ~680M tokens
    python3 prepare_fineweb.py --mb 400               # a smaller first run
    python3 prepare_fineweb.py --mb 1500 --heldout-mb 4
    python3 prepare_fineweb.py --min-score 3.5        # stricter quality filter
    python3 prepare_fineweb.py --dry-run              # just show the projections
"""

import paths
import argparse
import glob
import json
import os
import re

from huggingface_hub import HfApi, hf_hub_download

try:
    import pyarrow.parquet as pq
except ImportError:
    raise SystemExit(
        "This script needs pyarrow to read the dataset's parquet shards.\n"
        "Install it with:  pip install pyarrow"
    )

REPO_ID = "EleutherAI/fineweb-edu-dedup-10b"
OUT_DIR = paths.dataset_dir("FineWeb")
EVAL_DIR = paths.eval_corpus_dir()
EOT = "<|endoftext|>"

# Bytes of source text per token under this project's BPE, measured over real
# builds (10 MB of mixed uploads -> 4.75M tokens; 6 MB of this dataset -> 2.74M
# tokens). Only used for the projections printed below — nothing depends on it.
BYTES_PER_TOKEN = 2.2

# Text a shard yields, and what it costs to fetch. Measured, not taken from the
# dataset card: the card's `download_size` (2.8 GB for 97 shards) understates it by
# ~10x — shard 0 is 278 MB on disk. Text-per-shard is the card's `num_bytes`
# (48.1 GB) over 97, which matches what a shard actually decodes to.
TEXT_PER_SHARD = 48_116_092_268 / 97
DOWNLOAD_PER_SHARD = 291_943_086

# Bytes per output .txt. The pretrain loop shuffles whole files, so this trades
# directory size against shuffle granularity: 256 KiB is ~50 documents per file
# and ~6k files per 1.5 GB, which ext4 handles without complaint.
SHARD_BYTES = 256 * 1024

# Documents shorter than this are navigation chrome, cookie banners and stubs
# rather than prose worth spending context on.
MIN_CHARS = 1000

# Anything matching the tokenizer's marker grammar, so scraped text that happens to
# contain `<|endoftext|>` cannot inject a spurious document boundary. The zero-width
# break keeps the words readable while stopping `replace_special_tokens` matching.
MARKER = re.compile(r"<\|(/?[a-zA-Z0-9]+)\|>")


def clear_previous(out_dir: str, pattern: str = "*.txt") -> int:
    """Delete previously-prepared files from this dataset's folder so it holds
    exactly this run's text. Scoped to one folder — sibling datasets under
    uploads/ (TinyStories, Wikipedia, Chat, …) are never touched."""
    removed = 0
    for path in glob.glob(os.path.join(out_dir, pattern)):
        os.remove(path)
        removed += 1
    for path in glob.glob(os.path.join(out_dir, "_*manifest.json")):
        os.remove(path)
    return removed


def shard_files() -> list[str]:
    """Every parquet shard, sorted, so shard 0 comes first. Training data is taken
    from the front of this list and held-out text from the back."""
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    shards = sorted(f for f in files if f.endswith(".parquet"))
    if not shards:
        raise SystemExit(f"No parquet shards found in {REPO_ID}.")
    return shards


def iter_documents(shards: list[str], min_score: float, min_chars: int):
    """Yield cleaned document bodies, downloading one shard at a time.

    Shards are fetched lazily and rows streamed out of the parquet row groups, so
    memory stays flat and a small run only ever downloads the shards it reaches —
    at 278 MB per shard there is no reason to pull all 97 to write 400 MB of
    text."""
    for shard in shards:
        local = hf_hub_download(REPO_ID, filename=shard, repo_type="dataset")
        parquet = pq.ParquetFile(local)
        for batch in parquet.iter_batches(batch_size=256, columns=["text", "metadata"]):
            cols = batch.to_pydict()
            for text, meta in zip(cols["text"], cols["metadata"]):
                if min_score and (meta or {}).get("score", 0.0) < min_score:
                    continue
                body = MARKER.sub(r"<|\1|​>", (text or "").strip())
                if len(body) < min_chars:
                    continue
                yield body


def write_shards(documents, out_dir: str, budget: int, shard_bytes: int, prefix: str):
    """Batch documents into `shard_bytes`-sized .txt files until `budget` bytes of
    text have been written. Returns (bytes, documents, files)."""
    os.makedirs(out_dir, exist_ok=True)
    written = total_docs = files = 0
    buf: list[str] = []
    buf_bytes = 0

    def flush():
        nonlocal buf, buf_bytes, files
        if not buf:
            return
        path = os.path.join(out_dir, f"{prefix}{files:06d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(buf))
        buf, buf_bytes = [], 0
        files += 1

    for body in documents:
        # The trailing marker is what the Rust builder reads as a document
        # boundary; the blank line keeps the raw file readable.
        block = f"{body}\n\n{EOT}\n"
        buf.append(block)
        size = len(block.encode("utf-8"))
        buf_bytes += size
        written += size
        total_docs += 1
        if buf_bytes >= shard_bytes:
            flush()
        if written >= budget:
            break
    flush()
    return written, total_docs, files


def report(label: str, written: int, docs: int, files: int) -> None:
    tokens = written / BYTES_PER_TOKEN
    print(
        f"  {label}: {written / 1e6:.0f} MB in {files} files, {docs} documents"
        f"  (~{tokens / 1e6:.0f}M tokens)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=1500,
                    help="megabytes of training text to write (default: 1500)")
    ap.add_argument("--heldout-mb", type=float, default=0,
                    help="megabytes of held-out text for data/eval/corpus (0 = none)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop documents below this educational-quality score (0 = keep all)")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS,
                    help=f"drop documents shorter than this (default: {MIN_CHARS})")
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES,
                    help=f"bytes per output .txt (default: {SHARD_BYTES})")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--eval-out", default=EVAL_DIR)
    ap.add_argument("--keep-existing", action="store_true",
                    help="do NOT clear previously-prepared files first")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the projections and exit without downloading")
    args = ap.parse_args()

    budget = int(args.mb * 1e6)
    heldout_budget = int(args.heldout_mb * 1e6)
    tokens = budget / BYTES_PER_TOKEN

    # Filtering discards documents, so more shards are read than the raw text
    # ratio implies; the download figure is a floor and rises with --min-score.
    shards_needed = (budget + heldout_budget) / TEXT_PER_SHARD
    print(f"Preparing {args.mb:.0f} MB from {REPO_ID}")
    print(f"  projected      ~{tokens / 1e6:.0f}M tokens at ~{BYTES_PER_TOKEN} bytes/token")
    print(f"  corpus.bin     ~{tokens * 2 / 1e9:.2f} GB on disk (u16 ids)")
    print(f"  resident       ~{tokens * 2 / 1e9:.2f} GB while training")
    print(f"  ingestion peak ~1.1 GB, independent of corpus size")
    print(f"  download       >={shards_needed * DOWNLOAD_PER_SHARD / 1e9:.1f} GB "
          f"(~{max(1, round(shards_needed))} shards, cached under ~/.cache/huggingface)")
    # Chinchilla-optimal is ~20 tokens per parameter. Printed as a sanity check
    # against the model you are about to configure, not as a rule.
    print(f"  Chinchilla-optimal for this much data: ~{tokens / 20 / 1e6:.0f}M parameters")
    if args.dry_run:
        return

    shards = shard_files()
    print(f"{len(shards)} shards available; fetching from the front as needed.")

    if not args.keep_existing:
        print(f"Cleared {clear_previous(args.out)} previously-prepared file(s).")

    written, docs, files = write_shards(
        iter_documents(shards, args.min_score, args.min_chars),
        args.out, budget, args.shard_bytes, prefix="fineweb",
    )
    print("Wrote training text:")
    report("uploads/FineWeb", written, docs, files)

    with open(os.path.join(args.out, "_fineweb_manifest.json"), "w") as f:
        json.dump(
            {"repo": REPO_ID, "bytes": written, "documents": docs, "files": files,
             "min_score": args.min_score, "min_chars": args.min_chars},
            f, indent=2,
        )

    if heldout_budget:
        # Drawn from the far end of the shard list. A training run would have to
        # consume ~48 GB before reaching these, so "held out" needs no bookkeeping.
        if not args.keep_existing:
            clear_previous(args.eval_out, "fineweb*.txt")
        held_written, held_docs, held_files = write_shards(
            iter_documents(list(reversed(shards)), args.min_score, args.min_chars),
            args.eval_out, heldout_budget, args.shard_bytes * 4, prefix="fineweb-heldout",
        )
        print("Wrote held-out text (never selectable as training data):")
        report("eval/corpus", held_written, held_docs, held_files)

    print()
    print("Next: start a pretrain with the curriculum "
          "TinyStories -> Wikipedia -> FineWeb -> Chat (1 epoch).")
    print("The Chat stage needs no preparation — a run renders its own share of "
          "raw/sft/ when it starts.")


if __name__ == "__main__":
    main()
