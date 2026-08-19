"""Packing tokenized text into the training-ready corpus.

Two artifacts, mirroring the Rust `build_dataset`:

    corpus.bin     flat little-endian uint16 token ids, documents separated by
                   a single <|endoftext|>
    spans.npy      where each document starts and how long it is, tagged with
                   the dataset it came from — see `llm.corpus.spans`
    segments.json  what the build was made from, and the name table the spans
                   index into

The sidecar is what makes per-epoch shuffling possible: the corpus is one flat
stream, and without recorded boundaries there is no way to reorder documents
without also teaching the model to predict across a document boundary.

Both live under a key, because a packed corpus is a function of two things: the
text that was selected, and the vocabulary it was encoded with. Sharing one path
across either is how a run came to train on data nobody asked for.
"""

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field

import numpy as np
from tokenizers import Tokenizer

from llm import storage
from llm.corpus import bpe, remote, spans as spans_mod, special
from llm.corpus.spans import SpanBuilder, Spans

# uint16 halves the corpus on disk and in memory versus uint32, and caps the
# vocabulary at 65536 — far above anything this project trains.
CORPUS_DTYPE = np.uint16

# How many tokens to hold before writing a block out.
#
# The corpus is streamed to disk rather than assembled whole because a Python list
# boxes every element: 28 bytes for the int object plus an 8-byte list slot, against
# the 2 bytes the same token occupies as uint16. Buffering a whole corpus therefore
# costs about 18x its packed size — a 250 MB corpus.bin needs ~4.5 GiB of RAM to
# build — and the OOM killer takes the process before the file is ever written. The
# Rust `build_dataset` this was ported from streamed, so the ceiling arrived with
# the port rather than with any change in the data.
#
# 2M tokens is ~72 MiB of buffer and a 4 MiB write. Peak memory no longer scales
# with the corpus.
FLUSH_TOKENS = 1 << 21

# How much text to hold before handing it to the encoder as one batch.
#
# The library encodes a batch across every core; one text at a time uses one.
# Measured over 327 MB of Wikipedia shards: 2.3 MB/s file by file, 7.0 MB/s
# batched — six hours against two on the full input tree, and packing is the step
# a rebuild spends nearly all of its time in.
#
# Sized in characters rather than files because a batch is bounded by what it
# holds, not by how many things it holds: 4 MiB is ~16 shards at the shape on
# disk here (the measured optimum — 64 shards was slower and 330 MB heavier) and
# is still 4 MiB when one file is 28 MB. The texts and their encodings are the
# whole of the addition: ~40 MB of boxed ids on top of the flush buffer.
ENCODE_BATCH_CHARS = 1 << 22

#: How a key reads in a directory listing: a couple of dataset names, then the
#: vocabulary it was packed with, then the digest that actually identifies it.
KEY_SLUG_DATASETS = 2
KEY_FINGERPRINT_CHARS = 8
KEY_DIGEST_CHARS = 10

#: One curriculum stage as the corpus knows it: what it was called, and which
#: datasets it drew from, each resolved against the data engine. Epochs are
#: absent on purpose — see `corpus_key`.
StageSelection = tuple[str | None, list[remote.Selected]]


@dataclass(frozen=True)
class SourceDoc:
    """One document on its way into the corpus, tagged with the dataset it came
    from so a span can record which dataset it belongs to.

    Text, not a path. The engine owns no corpora — this arrives a page at a time
    from the data engine, through the backend, and is dropped as soon as it has
    been encoded.
    """

    dataset: str
    text: str
    stage: int = 0


@dataclass
class PackStats:
    datasets: int
    chars: int
    vocab_size: int
    corpus_len: int
    documents: int
    # Which reserved marker ids this corpus actually contained, sorted.
    #
    # Recorded because it is the only durable answer to "which marker embeddings
    # did this model never train?", which a later SFT needs in order to
    # warm-start them. It cannot be recovered afterwards: not from the weights (a
    # small trained embedding is indistinguishable from an untrained one) and not
    # from corpus.bin, which the next pretrain overwrites.
    special_ids_seen: list[int] = field(default_factory=list)


def resolve(client, stages) -> list[StageSelection]:
    """What an ordered curriculum selects right now, stage by stage.

    Resolved against the data engine before anything is keyed, exactly as the
    directory expansion it replaces was: a dataset that has been pulled again
    since the last build is a different corpus under the same id, and a key that
    could not see that would reuse the smaller one and report success.
    """
    return [
        (getattr(stage, "name", None), remote.resolve(client, getattr(stage, "datasets", None)))
        for stage in stages
    ]


def corpus_key(fingerprint: str, stages: list[StageSelection]) -> str:
    """The directory a corpus packed from this vocabulary and this selection
    lives in.

    Order is preserved, never sorted: concatenation order is a real training
    difference under a seeded shuffle, and an extra build is the cheap mistake
    here. Stage *epochs* and stage *names* are deliberately absent — neither
    changes a single token, and including them would re-pack a byte-identical
    corpus because someone renamed a stage. Both are still recorded in the
    sidecar, and `train` takes both from the request rather than from there.

    A dataset's *name* is absent from the digest for the same reason, but it is
    still in the slug, and the slug is part of the path — so renaming a dataset
    in the data engine does re-pack, even though the identity is unchanged. That
    is the price of a directory listing a person can read, paid knowingly: a
    rename is rare, and the alternative is a tree of bare digests.
    """
    payload = json.dumps(
        {
            "tokenizer": fingerprint,
            "stages": [
                [selected.identity() for selected in datasets] for _, datasets in stages
            ],
        }
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:KEY_DIGEST_CHARS]
    names = [s.name for _, datasets in stages for s in datasets][:KEY_SLUG_DATASETS]
    slug = "-".join(re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") for name in names)
    return f"{slug or 'empty'}-{fingerprint[:KEY_FINGERPRINT_CHARS]}-{digest}"


def pack(
    documents_in,
    tok: Tokenizer,
    *,
    key: str,
    stages: list[dict] | None = None,
    tokenizer_id: str | None = None,
    fingerprint: str = "",
) -> PackStats:
    """Encode every document into one flat corpus, recording its boundaries.

    `documents_in` is an iterable of `SourceDoc`, consumed once and never held:
    it is a paged read from the data engine, and materializing it would be the
    engine keeping a copy of a corpus it does not own.
    """
    vocab = bpe.vocab_size(tok)
    if vocab > np.iinfo(CORPUS_DTYPE).max + 1:
        raise ValueError(
            f"vocab {vocab} exceeds what {CORPUS_DTYPE.__name__} can hold; "
            f"widen CORPUS_DTYPE to uint32"
        )

    eot_id = bpe.token_id(tok, special.END_OF_TEXT)

    marker_ids = {
        tok.token_to_id(name)
        for name in special.token_list()
        if tok.token_to_id(name) is not None
    }

    documents = SpanBuilder()
    seen: set[int] = set()
    chars = 0
    corpus_len = 0

    storage.ensure_dir(storage.pretrain_dataset_dir(key))

    # Written through a .partial and renamed, so an interrupted pack leaves the
    # previous corpus intact rather than a truncated file that still loads.
    corpus = storage.pretrain_corpus_path(key)
    partial = corpus.with_suffix(".bin.partial")

    buffer: list[int] = []
    # How often each token id occurs, accumulated over the blocks on their way to
    # disk. Free here — every id is already in hand, as a numpy array, at the one
    # moment the whole corpus passes through a single point.
    #
    # Recorded because the alternative is reading it back: the training loop's
    # diagnostics need the most frequent ids, and computing them from corpus.bin
    # cost 64 s and pulled all 8.3 GB of the mapping resident before the first
    # step of every run. See `llm.train.diagnostics.histogram`.
    counts = np.zeros(vocab, dtype=np.int64)

    def flush() -> None:
        if buffer:
            block = np.asarray(buffer, dtype=CORPUS_DTYPE)
            counts[:] += np.bincount(block, minlength=vocab)[:vocab]
            out.write(block.tobytes())
            buffer.clear()

    seen_datasets: set[str] = set()

    with partial.open("wb") as out:
        batch: list[SourceDoc] = []
        texts: list[str] = []
        pending = 0

        def drain() -> None:
            """Encode the files held so far, in order, and append them."""
            nonlocal corpus_len, pending
            if not texts:
                return

            for source, ids in zip(batch, bpe.encode_batch(tok, texts)):
                if not ids:
                    continue

                documents.append(
                    source.dataset,
                    start=corpus_len,
                    length=len(ids),
                    stage=source.stage,
                )
                buffer.extend(ids)
                buffer.append(eot_id)
                corpus_len += len(ids) + 1
                seen.update(marker_ids.intersection(ids))

                if len(buffer) >= FLUSH_TOKENS:
                    flush()

            batch.clear()
            texts.clear()
            pending = 0

        for source in documents_in:
            chars += len(source.text)
            seen_datasets.add(source.dataset)
            batch.append(source)
            texts.append(source.text)
            pending += len(source.text)
            # One batch is one page: the documents are encoded and released
            # before the next page is asked for, so peak memory is a page and
            # not a corpus.
            if pending >= ENCODE_BATCH_CHARS:
                drain()

        drain()
        flush()

    partial.replace(corpus)

    spans = documents.build()
    spans.save(storage.pretrain_spans_path(key))

    # The selection and the vocabulary are written beside the documents so a
    # packed corpus says what it is. `is_built` compares against this rather than
    # trusting the directory name, which is the check that makes reuse safe.
    storage.pretrain_segments_path(key).write_text(
        json.dumps(
            {
                "eot_id": eot_id,
                "tokenizer_id": tokenizer_id,
                "tokenizer_fingerprint": fingerprint,
                "vocab_size": vocab,
                "stages": stages or [],
                "special_ids_seen": sorted(seen),
                "token_counts": counts.tolist(),
                # The rows themselves are in spans.npy; what stays here is the
                # count a person reads and the name table those rows index into.
                "documents": len(spans),
                "datasets": spans.names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return PackStats(
        datasets=len(seen_datasets),
        chars=chars,
        vocab_size=vocab,
        corpus_len=corpus_len,
        documents=len(spans),
        special_ids_seen=sorted(seen),
    )


def load_corpus(key: str) -> np.ndarray:
    path = storage.pretrain_corpus_path(key)
    if not path.is_file():
        raise FileNotFoundError(f"No corpus at {path}; build the dataset first.")
    # Mapped, not read. `fromfile` makes the whole corpus resident and private —
    # 745 MiB of anonymous memory on the 390M-token corpus, on a machine with
    # 7 GiB — and training reads it in small spans it can page in on demand. A
    # mapping is backed by the file, so the kernel can reclaim it under pressure
    # instead of the process being killed.
    return np.memmap(path, dtype=CORPUS_DTYPE, mode="r")


def load_segments(key: str) -> tuple[int, Spans, dict]:
    """Document boundaries plus what the build was made from.

    Returns the same `(eot_id, documents, info)` shape `sft.load_segments` does,
    so the two sidecars are read the same way. A missing sidecar is an empty
    table — the caller falls back to one whole-corpus document.
    """
    path = storage.pretrain_segments_path(key)
    if not path.is_file():
        return 0, Spans.empty(), {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    info = {
        "stages": raw.get("stages", []),
        "tokenizer_id": raw.get("tokenizer_id"),
        "tokenizer_fingerprint": raw.get("tokenizer_fingerprint"),
        "vocab_size": raw.get("vocab_size"),
        # Absent on a build made before it was recorded; None means "unknown", so
        # a later SFT scans rather than concluding the corpus used no markers.
        "special_ids_seen": raw.get("special_ids_seen"),
        # Per-token frequencies, for the training loop's diagnostics. None on a
        # build that predates them, and then the corpus is sampled instead.
        "token_counts": raw.get("token_counts"),
    }
    return (
        raw["eot_id"],
        spans_mod.from_sidecar(storage.pretrain_spans_path(key), raw),
        info,
    )


def is_built(key: str, stages: list[StageSelection], fingerprint: str) -> bool:
    """Whether the corpus on disk is the one this run asked for.

    Existence is not enough, and never was: the old check asked only whether a
    corpus and a tokenizer were present anywhere, so a run with different
    datasets skipped the build and trained on whatever the last run left behind.
    What is compared is what a corpus is a function of — the vocabulary that
    encoded it and the per-stage selection — not the epochs or the stage names,
    which change no tokens.
    """
    if not (
        storage.pretrain_corpus_path(key).is_file()
        and storage.pretrain_segments_path(key).is_file()
    ):
        return False

    _, _, info = load_segments(key)
    if not info or info.get("tokenizer_fingerprint") != fingerprint:
        return False
    recorded = [stage.get("datasets", []) for stage in info.get("stages", [])]
    return recorded == [
        [selected.record() for selected in datasets] for _, datasets in stages
    ]


def builds() -> list[dict]:
    """Every packed corpus on disk: its key, the vocabulary it belongs to, and
    what it occupies. The keying trades disk for correctness, so something has to
    be able to say how much."""
    root = storage.datasets_dir() / "pretrain"
    if not root.is_dir():
        return []
    packed = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        _, documents, info = load_segments(child.name)
        packed.append(
            {
                "key": child.name,
                "tokenizer_id": info.get("tokenizer_id"),
                "tokenizer_fingerprint": info.get("tokenizer_fingerprint"),
                "documents": len(documents),
                "bytes": sum(f.stat().st_size for f in child.rglob("*") if f.is_file()),
            }
        )
    return packed


def latest_key() -> str | None:
    """The key of the most recently packed corpus, or None.

    For the ad-hoc tools — `bench.py` and everything under `explore/` — which
    want "whatever was packed last" rather than a particular build. A training
    run always names its own key; nothing in the pipeline calls this.
    """
    root = storage.datasets_dir() / "pretrain"
    if not root.is_dir():
        return None
    packed = [p for p in root.iterdir() if (p / "corpus.bin").is_file()]
    if not packed:
        return None
    return max(packed, key=lambda p: (p / "corpus.bin").stat().st_mtime).name


def delete(key: str) -> bool:
    directory = storage.pretrain_dataset_dir(key)
    if not directory.is_dir():
        return False
    shutil.rmtree(directory)
    return True
