"""Shared machinery for the pretrain fetch scripts.

Every fetcher does the same thing to a different source: pull documents, drop the
ones too short to be prose, batch them into fixed-size `.txt` shards under a byte
budget, and record what it wrote. Only the middle — which repo, which columns,
what counts as a document — is per-source. That half stays in the fetcher; this
module is everything else.

Factored when the second fetcher was written rather than the sixth, because the
parts most worth sharing are the ones most dangerous to fork: the marker regex
has to track the tokenizer's grammar, and the held-out slice has to stay disjoint
from training text. Six hand-maintained copies of either is six chances to be
subtly wrong in a way no test catches and only a bad model reveals.

Output layout, matching what `engine/llm/corpus/sources.py` selects:

    engine/data/input/<Dataset>/<prefix>000000.txt

Documents inside a shard are separated by `<|endoftext|>`, which the corpus
packer reads as a document boundary.
"""

import argparse
import glob
import json
import os
import re
import sys
import time

# Bytes per output .txt. The pretrain loop shuffles whole files, so this trades
# directory size against shuffle granularity: 256 KiB is ~50 documents per file
# and ~6k files per 1.5 GB, which ext4 handles without complaint.
SHARD_BYTES = 256 * 1024

# Documents shorter than this are navigation chrome, cookie banners and stubs
# rather than prose worth spending context on.
MIN_CHARS = 1000

# Bytes of source text per token under this project's BPE, measured over real
# builds (10 MB of mixed uploads -> 4.75M tokens; 6 MB of FineWeb -> 2.74M
# tokens). Only used for the projections printed below — nothing depends on it.
BYTES_PER_TOKEN = 2.2

# Chinchilla-optimal training tokens per non-embedding parameter. Printed as a
# sanity check against the model you are about to configure, not as a rule.
TOKENS_PER_PARAM = 20

# How often `write_shards` prints a progress line. These fetches run for hours
# against multi-gigabyte budgets, and a silent process is indistinguishable from
# a hung one.
PROGRESS_EVERY = 250 * 1024 * 1024

EOT = "<|endoftext|>"

# Anything matching the tokenizer's marker grammar, so scraped text that happens
# to contain `<|endoftext|>` cannot inject a spurious document boundary. The
# zero-width break keeps the words readable while stopping the tokenizer's
# special-token matching. Includes `:` so `<|think:low|>` is caught too.
MARKER = re.compile(r"<\|(/?[a-zA-Z0-9:]+)\|>")


def neutralize(text: str) -> str:
    """Break any literal marker so it cannot be tokenized as a real one."""
    return MARKER.sub("<|\\1|​>", text)


def clear_previous(out_dir: str, pattern: str = "*.txt") -> int:
    """Delete previously-prepared files from one dataset folder, so it holds
    exactly this run's text.

    Scoped to a single folder on purpose — sibling datasets under input/ are
    never touched, because re-fetching one source must not cost the other five.
    """
    removed = 0
    for path in glob.glob(os.path.join(out_dir, pattern)):
        os.remove(path)
        removed += 1
    for path in glob.glob(os.path.join(out_dir, "_*manifest.json")):
        os.remove(path)
    return removed


def write_shards(documents, out_dir, budget, shard_bytes=SHARD_BYTES, prefix="part",
                 label=None, quiet=False):
    """Batch documents into `shard_bytes`-sized files until `budget` bytes of
    text have been written. Returns `(bytes, documents, files)`.

    `documents` is an iterator of cleaned bodies, so a fetcher can stream out of
    parquet without materializing a shard's worth of source rows.

    A shard is closed once it *passes* the size, and the budget is checked after
    the document is appended — so both overrun by up to one document. That is
    deliberate: splitting a document across shards would put half a page in one
    training example and half in another, which is worse than a slightly large
    file. Sources with very large documents (books) overrun by more, which is
    why `prepare_pg19.py` says so explicitly.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = total_docs = files = 0
    next_report = PROGRESS_EVERY
    started = time.time()
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
        # The trailing marker is what the packer reads as a document boundary;
        # the blank line keeps the raw file readable.
        block = f"{body}\n\n{EOT}\n"
        buf.append(block)
        size = len(block.encode("utf-8"))
        buf_bytes += size
        written += size
        total_docs += 1
        if buf_bytes >= shard_bytes:
            flush()
        if not quiet and written >= next_report:
            rate = written / max(time.time() - started, 1e-6) / 1e6
            print(f"    {label or prefix}: {written / 1e9:.2f} GB, {total_docs:,} documents"
                  f"  ({rate:.1f} MB/s)", flush=True)
            next_report += PROGRESS_EVERY
        if written >= budget:
            break
    flush()
    return written, total_docs, files


def write_manifest(out_dir: str, name: str, payload: dict) -> str:
    """Record what a fetch wrote, beside the text it wrote.

    Named `_<name>_manifest.json` so `clear_previous` finds it and the `*.txt`
    glob that selects training data does not.
    """
    path = os.path.join(out_dir, f"_{name}_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def report(label: str, written: int, docs: int, files: int) -> None:
    tokens = written / BYTES_PER_TOKEN
    print(f"  {label}: {written / 1e6:.0f} MB in {files} files, {docs:,} documents"
          f"  (~{tokens / 1e6:.0f}M tokens)")


def projection(repo: str, mb: float, extra: dict | None = None) -> None:
    """What a budget costs, printed before anything is downloaded.

    The Chinchilla line is the one that matters when sizing a run: it says which
    model this much text can actually train, which is the whole reason for
    fetching more of it.
    """
    budget = int(mb * 1e6)
    tokens = budget / BYTES_PER_TOKEN
    print(f"Preparing {mb:.0f} MB from {repo}")
    print(f"  projected      ~{tokens / 1e6:.0f}M tokens at ~{BYTES_PER_TOKEN} bytes/token")
    print(f"  corpus.bin     ~{tokens * 2 / 1e9:.2f} GB on disk (u16 ids)")
    for key, value in (extra or {}).items():
        print(f"  {key:<14} {value}")
    print(f"  Chinchilla-optimal for this much data: "
          f"~{tokens / TOKENS_PER_PARAM / 1e6:.0f}M non-embedding parameters")


def base_parser(description: str, default_mb: float, dataset: str) -> argparse.ArgumentParser:
    """The flags every fetcher shares, so they are spelled the same everywhere.

    `--mb` is the control that matters: it is what lets one source be dialled
    down without editing the script, and it is why none of these take a document
    count instead.
    """
    import paths

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--mb", type=float, default=default_mb,
                    help=f"megabytes of training text to write (default: {default_mb:g})")
    ap.add_argument("--heldout-mb", type=float, default=0,
                    help="megabytes of held-out text for data/eval/corpus (0 = none)")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS,
                    help=f"drop documents shorter than this (default: {MIN_CHARS})")
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES,
                    help=f"bytes per output .txt (default: {SHARD_BYTES})")
    ap.add_argument("--out", default=paths.dataset_dir(dataset),
                    help=f"where to write (default: input/{dataset})")
    ap.add_argument("--eval-out", default=paths.eval_corpus_dir())
    ap.add_argument("--keep-existing", action="store_true",
                    help="do NOT clear previously-prepared files first")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the projections and exit without downloading")
    return ap


def parquet():
    """`pyarrow.parquet`, with an actionable message when it is missing."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("This script needs pyarrow to read parquet shards.\n"
                 "Install it with:  pip install pyarrow")
    return pq


def list_parquet_shards(repo: str, prefix: str = "", suffix: str = ".parquet") -> list[str]:
    """Every shard in a Hub dataset, sorted.

    Sorted so shard 0 comes first and the order is stable: training text is taken
    from the front of this list and held-out text from the back, which is what
    makes the two disjoint without any bookkeeping.

    `suffix` because not every source is parquet — C4 ships gzipped JSON lines,
    and the listing logic is otherwise identical.
    """
    from huggingface_hub import HfApi

    try:
        files = HfApi().list_repo_files(repo, repo_type="dataset")
    except Exception as e:  # gated repos, network, renamed datasets
        sys.exit(f"Could not list {repo}: {e}\n"
                 f"If it is gated, accept the terms at "
                 f"https://huggingface.co/datasets/{repo} and run "
                 f"`huggingface-cli login`.")

    shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(suffix))
    if not shards:
        sys.exit(f"No {suffix} shards found in {repo}"
                 + (f" under {prefix!r}" if prefix else "")
                 + f".\nBrowse it at https://huggingface.co/datasets/{repo}")
    return shards


def iter_jsonl_rows(repo: str, shards: list[str], columns: list[str] | None = None):
    """Yield row dicts from gzipped JSON-lines shards, one shard at a time.

    The JSON-lines counterpart of `iter_parquet_rows`, and lazy for the same
    reason. A malformed line is skipped rather than fatal: these are
    multi-gigabyte crawl dumps, and one bad row should not cost the fetch.
    """
    import gzip
    import json as _json

    from huggingface_hub import hf_hub_download

    for shard in shards:
        local = hf_hub_download(repo, filename=shard, repo_type="dataset")
        opener = gzip.open if local.endswith(".gz") else open
        with opener(local, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = _json.loads(line)
                except ValueError:
                    continue
                yield {c: row.get(c) for c in columns} if columns else row


def iter_parquet_rows(repo: str, shards: list[str], columns: list[str], batch_size: int = 256):
    """Yield row dicts, downloading one shard at a time.

    Lazy on purpose: a shard is hundreds of megabytes, and a small `--mb` run has
    no reason to pull the whole repo to write a fraction of one shard. The caller
    stops iterating when its budget is spent and the rest are never fetched.
    """
    from huggingface_hub import hf_hub_download

    pq = parquet()
    for shard in shards:
        local = hf_hub_download(repo, filename=shard, repo_type="dataset")
        for batch in pq.ParquetFile(local).iter_batches(batch_size=batch_size, columns=columns):
            cols = batch.to_pydict()
            for values in zip(*(cols[c] for c in columns)):
                yield dict(zip(columns, values))


def run_fetch(args, name: str, documents_for, prefix: str, manifest: dict | None = None) -> None:
    """The tail every fetcher shares: clear, write training text, write the
    held-out slice, record a manifest.

    `documents_for(reverse)` returns a fresh document iterator — taken from the
    far end of the source when `reverse` is true. Held-out text is drawn from
    there rather than from the tail of what training consumed, so a run would
    have to exhaust the entire dataset before the two could overlap.
    """
    if not args.keep_existing:
        print(f"Cleared {clear_previous(args.out)} previously-prepared file(s).")

    written, docs, files = write_shards(
        documents_for(False), args.out, int(args.mb * 1e6),
        args.shard_bytes, prefix=prefix, label=name,
    )
    print("Wrote training text:")
    report(f"input/{os.path.basename(args.out)}", written, docs, files)

    write_manifest(args.out, name, {
        "bytes": written, "documents": docs, "files": files, **(manifest or {}),
    })

    if args.heldout_mb:
        if not args.keep_existing:
            clear_previous(args.eval_out, f"{prefix}-heldout*.txt")
        held = write_shards(
            documents_for(True), args.eval_out, int(args.heldout_mb * 1e6),
            args.shard_bytes * 4, prefix=f"{prefix}-heldout", label=f"{name} held-out",
        )
        print("Wrote held-out text (never selectable as training data):")
        report("eval/corpus", *held)
