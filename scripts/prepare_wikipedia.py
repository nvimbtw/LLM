#!/usr/bin/env python3
"""
Populate a Wikipedia dataset folder under the backend uploads dir with
plain-text article "documents", one article per .txt file.

Source: the Wikimedia-maintained, pre-cleaned dump
(https://huggingface.co/datasets/wikimedia/wikipedia). Its `text` field is
already stripped of wiki markup, references, and tables — clean prose — so, like
TinyStories, it needs no further cleaning. The dataset is sharded parquet, keyed
by a config of the form "<date>.<lang>" (e.g. "20231101.en", "20231101.simple").

This mirrors prepare_tinystories.py: it downloads shard(s) for one config, reads
articles out of the parquet row groups, and writes one article per .txt file —
each re-terminated with a single `<|endoftext|>`, the document-boundary marker
the Rust pretrain builder (backend/src/training.rs: build_dataset) concatenates
on. Each file is prefixed with the article title as a heading line.

Unlike the old TinyStories layout, datasets now live in their OWN subfolder of
uploads/ (each subfolder is a checkbox in the training UI), so this writes to
uploads/Wikipedia and only ever clears that folder — every other dataset
(TinyStories, Synthetic, …) is left untouched.

English Wikipedia is tens of GB across 41 shards; to keep this cheap we download
shards one at a time and stop as soon as --limit articles have been written, so a
few-thousand-article run only ever fetches the first shard. For a smaller, self-
contained fetch, use the single-shard Simple English config:

    python3 prepare_wikipedia.py --lang simple

Usage:
    python3 prepare_wikipedia.py                       # 2000 English articles
    python3 prepare_wikipedia.py --lang simple         # Simple English (1 shard)
    python3 prepare_wikipedia.py --limit 0             # every article in the config
    python3 prepare_wikipedia.py --lang de --date 20231101
"""

import argparse
import glob
import json
import os
import uuid

from huggingface_hub import HfApi, hf_hub_download

try:
    import pyarrow.parquet as pq
except ImportError:
    raise SystemExit(
        "This script needs pyarrow to read the dataset's parquet shards.\n"
        "Install it with:  pip install pyarrow"
    )

REPO_ID = "wikimedia/wikipedia"
OUT_DIR = "/home/x_x/Programming/LLM/backend/data/raw/uploads/Wikipedia"
EOT = "<|endoftext|>"


def clear_previous(out_dir: str) -> int:
    """Delete previously-prepared articles (.txt files + their manifest) from this
    dataset's folder so it holds exactly this run's articles. Scoped to the
    Wikipedia folder — sibling datasets under uploads/ are never touched."""
    removed = 0
    for path in glob.glob(os.path.join(out_dir, "*.txt")):
        os.remove(path)
        removed += 1
    for path in glob.glob(os.path.join(out_dir, "_*manifest.json")):
        os.remove(path)
    return removed


def shard_files(config: str) -> list[str]:
    """List the parquet shard paths for a config (e.g. '20231101.en'), sorted so
    shard 0 comes first (we fetch them in order and usually stop early)."""
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    prefix = config + "/"
    shards = sorted(
        f for f in files if f.startswith(prefix) and f.endswith(".parquet")
    )
    if not shards:
        raise SystemExit(
            f"No parquet shards found for config '{config}'.\n"
            f"Check the language/date — browse configs at "
            f"https://huggingface.co/datasets/{REPO_ID}"
        )
    return shards


def iter_articles(config: str):
    """Yield (title, text) for every article in a config, downloading its shards
    one at a time and streaming rows out of the parquet row groups (so memory
    stays flat and English's later shards are only fetched if actually reached)."""
    for shard in shard_files(config):
        local = hf_hub_download(REPO_ID, filename=shard, repo_type="dataset")
        parquet = pq.ParquetFile(local)
        for batch in parquet.iter_batches(batch_size=256, columns=["title", "text"]):
            cols = batch.to_pydict()
            for title, text in zip(cols["title"], cols["text"]):
                yield title, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en",
                    help="Wikipedia language code, e.g. en, simple, de (default: en)")
    ap.add_argument("--date", default="20231101",
                    help="dump snapshot date, YYYYMMDD (default: 20231101)")
    ap.add_argument("--limit", type=int, default=2000,
                    help="max articles to write (0 = every article in the config)")
    ap.add_argument("--min-chars", type=int, default=500,
                    help="skip articles shorter than this (drops stubs)")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="truncate articles longer than this (0 = no cap)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--keep-existing", action="store_true",
                    help="do NOT clear previously-prepared articles first")
    args = ap.parse_args()

    config = f"{args.date}.{args.lang}"
    print(f"Preparing {config} from {REPO_ID} …")

    os.makedirs(args.out, exist_ok=True)
    if not args.keep_existing:
        removed = clear_previous(args.out)
        print(f"Cleared {removed} previously-prepared article file(s).")

    manifest = []
    for title, text in iter_articles(config):
        body = (text or "").strip()
        if len(body) < args.min_chars:
            continue
        if args.max_chars and len(body) > args.max_chars:
            body = body[:args.max_chars].rstrip()
        document = f"{title}\n\n{body}"
        out_name = f"{uuid.uuid4()}.txt"
        with open(os.path.join(args.out, out_name), "w", encoding="utf-8") as f:
            f.write(document + "\n\n" + EOT + "\n")
        manifest.append({"file": out_name, "title": title, "chars": len(document)})
        if args.limit and len(manifest) >= args.limit:
            break

    manifest_path = os.path.join(args.out, f"_wikipedia_{config}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(manifest)} articles -> {args.out}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
