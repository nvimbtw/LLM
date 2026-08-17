#!/usr/bin/env python3
"""
Populate the benchmark's reference corpus at engine/data/eval/corpus/.

The benchmark's primary number is held-out bits-per-byte: how well the model
predicts English it was never trained on. That only means anything if the text
really is held out, so this writes to `data/eval/`, which is deliberately outside
`engine/data/input/` — anything under input is selectable as training data in the
UI, and a held-out set that can be accidentally trained on is not a held-out set.

By default it pulls the TinyStories **train** split, because
`prepare_tinystories.py` populates uploads from the **valid** split. The two are
disjoint by construction, so the eval corpus is genuinely unseen without you
having to track which files went into which run.

Output is a handful of large .txt files rather than thousands of small ones. The
Rust loader (backend/src/llm/src/eval/corpus.rs) splits each file 90/10 — the head
fits the n-gram baseline, the tail is scored — and that split is only meaningful
on files with enough text in them to be worth splitting.

Usage:
    python3 prepare_eval_corpus.py                  # ~2 MB of TinyStories train
    python3 prepare_eval_corpus.py --mb 8           # a larger reference corpus
    python3 prepare_eval_corpus.py --source wikipedia
"""

import paths
import argparse
import glob
import os

from huggingface_hub import hf_hub_download

OUT_DIR = paths.eval_corpus_dir()
EOT = "<|endoftext|>"

# Held-out sources. TinyStories train is the default because the uploads dir is
# built from the valid split, making the two disjoint without any bookkeeping.
SOURCES = {
    "tinystories": ("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt"),
}

# How many files to spread the corpus across. Each is split 90/10 independently,
# so more files means the held-out slice samples more places in the source rather
# than being one contiguous run of text.
NUM_FILES = 8


def clear_previous(out_dir: str) -> int:
    removed = 0
    for path in glob.glob(os.path.join(out_dir, "*.txt")):
        os.remove(path)
        removed += 1
    return removed


def iter_documents(path: str, chunk_chars: int = 1 << 20):
    """Yield each non-empty document from an <|endoftext|>-delimited file.

    Streamed deliberately. The sibling prepare_tinystories.py reads the whole file
    and splits it in one go, which is fine there because it only ever opens the
    *valid* split (~22 MB) — but this script defaults to the *train* split, which
    is 2.1 GB on disk and needs ~5 GB of RAM to hold the text plus the list of
    every document. That OOMs an 8 GB box, and since the process is killed rather
    than raising, it takes the terminal down with it.

    Reading a window at a time and holding only the trailing partial document
    keeps this at a couple of MB. The caller stops after a few megabytes, and
    generator laziness means the rest of the file is never touched.
    """
    buffer = ""
    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            buffer += chunk
            documents = buffer.split(EOT)
            # The last piece may be a document the next read completes, so it
            # stays in the buffer rather than being yielded truncated.
            buffer = documents.pop()
            for document in documents:
                text = document.strip()
                if text:
                    yield text
    tail = buffer.strip()
    if tail:
        yield tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tinystories", choices=sorted(SOURCES))
    ap.add_argument("--mb", type=float, default=2.0,
                    help="total megabytes of reference text to write")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--keep-existing", action="store_true",
                    help="do NOT clear the existing eval corpus first")
    args = ap.parse_args()

    repo_id, filename = SOURCES[args.source]
    print(f"Downloading {filename} from {repo_id} …")
    src = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

    os.makedirs(args.out, exist_ok=True)
    if not args.keep_existing:
        print(f"Cleared {clear_previous(args.out)} previous eval file(s).")

    budget = int(args.mb * 1024 * 1024)
    per_file = max(budget // NUM_FILES, 1)

    written = 0
    index = 0
    buffer = []
    buffered = 0

    def flush():
        nonlocal buffer, buffered, index, written
        if not buffer:
            return
        path = os.path.join(args.out, f"heldout-{index:02d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(buffer))
        written += buffered
        print(f"  {path}  ({buffered:,} bytes)")
        index += 1
        buffer = []
        buffered = 0

    for doc in iter_documents(src):
        buffer.append(doc)
        buffered += len(doc.encode("utf-8"))
        if buffered >= per_file:
            flush()
        if written >= budget:
            break
    flush()

    print(f"\nWrote {written:,} bytes across {index} file(s) -> {args.out}")
    print("The benchmark splits each file 90/10: head fits the n-gram baseline,")
    print("tail is the held-out slice the model is scored on.")


if __name__ == "__main__":
    main()
