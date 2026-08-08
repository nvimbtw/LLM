"""Packing tokenized text into the training-ready corpus.

Two artifacts, mirroring the Rust `build_dataset`:

    corpus.bin     flat little-endian uint16 token ids, documents separated by
                   a single <|endoftext|>
    segments.json  where each document starts and how long it is, tagged with
                   the dataset it came from

The sidecar is what makes per-epoch shuffling possible: the corpus is one flat
stream, and without recorded boundaries there is no way to reorder documents
without also teaching the model to predict across a document boundary.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from llm import storage
from llm.corpus import bpe, special
from llm.corpus.sources import SourceFile

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


@dataclass
class DocSpan:
    """A contiguous span of the flat corpus forming one document. `start`/`len`
    are exclusive of the <|endoftext|> separators baked between documents."""

    dataset: str
    start: int
    length: int
    stage: int = 0


@dataclass
class PackStats:
    files: int
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


def pack(files: list[SourceFile], tok: Tokenizer, stages: list[dict] | None = None) -> PackStats:
    """Encode every file into one flat corpus, recording document boundaries."""
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

    documents: list[DocSpan] = []
    seen: set[int] = set()
    chars = 0
    corpus_len = 0

    storage.ensure_dir(storage.datasets_dir())

    # Written through a .partial and renamed, so an interrupted pack leaves the
    # previous corpus intact rather than a truncated file that still loads.
    corpus = storage.corpus_path()
    partial = corpus.with_suffix(".bin.partial")

    buffer: list[int] = []

    def flush() -> None:
        if buffer:
            out.write(np.asarray(buffer, dtype=CORPUS_DTYPE).tobytes())
            buffer.clear()

    with partial.open("wb") as out:
        for source in files:
            text = source.path.read_text(encoding="utf-8", errors="replace")
            chars += len(text)
            ids = bpe.encode(tok, text)
            if not ids:
                continue

            documents.append(
                DocSpan(
                    dataset=source.dataset,
                    start=corpus_len,
                    length=len(ids),
                    stage=source.stage,
                )
            )
            buffer.extend(ids)
            buffer.append(eot_id)
            corpus_len += len(ids) + 1
            seen.update(marker_ids.intersection(ids))

            if len(buffer) >= FLUSH_TOKENS:
                flush()

        flush()

    partial.replace(corpus)

    storage.segments_path().write_text(
        json.dumps(
            {
                "eot_id": eot_id,
                "stages": stages or [],
                "documents": [asdict(d) for d in documents],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return PackStats(
        files=len(files),
        chars=chars,
        vocab_size=vocab,
        corpus_len=corpus_len,
        documents=len(documents),
        special_ids_seen=sorted(seen),
    )


def load_corpus() -> np.ndarray:
    path = storage.corpus_path()
    if not path.is_file():
        raise FileNotFoundError(f"No corpus at {path}; build the dataset first.")
    return np.fromfile(path, dtype=CORPUS_DTYPE)


def load_segments() -> tuple[int, list[DocSpan], list[dict]]:
    """The document boundaries for the packed corpus, or an empty list when the
    sidecar is missing — the caller falls back to one whole-corpus document."""
    path = storage.segments_path()
    if not path.is_file():
        return 0, [], []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["eot_id"], [DocSpan(**d) for d in raw["documents"]], raw.get("stages", [])


def is_built() -> bool:
    return storage.corpus_path().is_file() and storage.tokenizer_file().is_file()
