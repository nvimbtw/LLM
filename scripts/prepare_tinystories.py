#!/usr/bin/env python3
"""
Populate the backend uploads dir with TinyStories as plain-text "documents",
replacing the old PG-19 book pipeline.

TinyStories (https://huggingface.co/datasets/roneneldan/TinyStories) is a
curated set of short, simple synthetic stories — clean prose with no OCR/scan
noise — so unlike PG-19 it needs no cleaning. The dataset ships as raw .txt
files whose stories are already separated by `<|endoftext|>`, which is exactly
the document-boundary convention the Rust pretrain builder
(llm-engine/llm/corpus/pack.py) expects: it concatenates the
.txt files, and a single trailing `<|endoftext|>` per file marks where one
document ends and the next begins.

So all this does is: download the split, split it on `<|endoftext|>`, and write
one story per .txt file (each re-terminated with a single `<|endoftext|>`) into
the uploads dir. By default it first clears any previously-prepared uploads
(the old PG-19 books) so the corpus is TinyStories only. The SFT dataset lives
elsewhere (data/raw/sft) and is never touched.

Usage:
    python3 prepare_tinystories.py                    # 2000 stories, V2 valid split
    python3 prepare_tinystories.py --limit 0          # every story in the split
    python3 prepare_tinystories.py --split train      # the (much larger) train split
"""

import paths
import argparse
import glob
import json
import os
import uuid

from huggingface_hub import hf_hub_download

REPO_ID = "roneneldan/TinyStories"
# Datasets now live in their own subfolder of uploads/ (each is a checkbox in the
# training UI); write to uploads/TinyStories so clear_previous only ever touches
# this dataset and leaves sibling datasets (Synthetic, Wikipedia, …) alone.
OUT_DIR = paths.dataset_dir("TinyStories")
EOT = "<|endoftext|>"

# The dataset's raw text files. V2 is the GPT-4-only regeneration (higher quality
# than the original V1 mix); the valid split is small (fast to fetch) yet holds
# tens of thousands of stories — plenty for the tiny models this repo trains.
FILENAMES = {
    (2, "train"): "TinyStoriesV2-GPT4-train.txt",
    (2, "valid"): "TinyStoriesV2-GPT4-valid.txt",
    (1, "train"): "TinyStories-train.txt",
    (1, "valid"): "TinyStories-valid.txt",
}


def clear_previous(out_dir: str) -> int:
    """Delete previously-prepared uploads (.txt files + their manifest) so the
    corpus is exactly this run's stories. Only touches loose files in the uploads
    dir itself — the SFT data lives under a sibling dir and is left alone."""
    removed = 0
    for path in glob.glob(os.path.join(out_dir, "*.txt")):
        os.remove(path)
        removed += 1
    for path in glob.glob(os.path.join(out_dir, "_*manifest.json")):
        os.remove(path)
    return removed


def iter_stories(path: str):
    """Yield each non-empty story from a TinyStories .txt file (stories are
    delimited by `<|endoftext|>` in the source)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for chunk in text.split(EOT):
        story = chunk.strip()
        if story:
            yield story


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["valid", "train"])
    ap.add_argument("--version", type=int, default=2, choices=[1, 2],
                    help="TinyStories dataset version (2 = GPT-4-only, default)")
    ap.add_argument("--limit", type=int, default=2000,
                    help="max stories to write (0 = all in the split)")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="skip stories shorter than this (drops truncated tails)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--keep-existing", action="store_true",
                    help="do NOT clear previously-prepared uploads first")
    args = ap.parse_args()

    filename = FILENAMES[(args.version, args.split)]
    print(f"Downloading {filename} from {REPO_ID} …")
    src = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")

    os.makedirs(args.out, exist_ok=True)
    if not args.keep_existing:
        removed = clear_previous(args.out)
        print(f"Cleared {removed} previously-prepared upload file(s).")

    manifest = []
    for story in iter_stories(src):
        if len(story) < args.min_chars:
            continue
        out_name = f"{uuid.uuid4()}.txt"
        with open(os.path.join(args.out, out_name), "w", encoding="utf-8") as f:
            f.write(story + "\n\n" + EOT + "\n")
        manifest.append({"file": out_name, "chars": len(story)})
        if args.limit and len(manifest) >= args.limit:
            break

    manifest_path = os.path.join(args.out, f"_tinystories_{args.split}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(manifest)} stories -> {args.out}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
