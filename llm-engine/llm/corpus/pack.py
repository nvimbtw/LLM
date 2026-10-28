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
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from typing import BinaryIO

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
# Sized in characters, and — unlike the version of this that counted documents
# into it — actually bounded by them: a batch is built out of `bpe.CHUNK_CHARS`
# pieces, so it is 4 MiB whether the corpus is web snippets or a 27 MB book.
# Filled with whole documents it was 4 MiB *plus the last one*, and on
# `gutenberg_eng-clean` that last one is up to 27.6 MB — a 4.5 GiB peak inside
# the encoder, on a machine that has 11 GiB in total. 4 MiB was measured the
# optimum at the shape on disk here; 64 shards' worth was slower and 330 MB
# heavier.
ENCODE_BATCH_CHARS = 1 << 22

#: How many encode batches the reader runs ahead of the encoder.
#:
#: Packing alternates between two things that need nothing from each other: a
#: page is fetched — a Postgres query, a JSON encode, two service hops — while
#: every core sits idle, and then encoded while the socket sits idle. Measured on
#: the run this was written for, the six encoding threads were busy 37% of the
#: time and the thread doing everything else spent 83% of its own blocked on the
#: network. Overlapping them is worth more than anything else here.
#:
#: Depth is what it costs in memory: this many batches are queued while one more
#: is being built and one more is being encoded. One is enough to cover a page,
#: and counting in batches rather than pages or documents is the same argument as
#: sizing the batch in characters — neither of those is an amount of text.
READ_AHEAD_BATCHES = 1

#: How long the reader waits on a full queue before checking whether the pack it
#: is feeding is still there. Only a pack that ended early ever reaches it.
READ_AHEAD_POLL_SECS = 0.5

#: How a key reads in a directory listing: a couple of dataset names, then the
#: vocabulary it was packed with, then the digest that actually identifies it.
KEY_SLUG_DATASETS = 2
KEY_FINGERPRINT_CHARS = 8
KEY_DIGEST_CHARS = 10

#: How many datasets the reported fertility names, most tokens first. A
#: selection is a handful, and the cap is what keeps a fifty-stage curriculum
#: from putting a list nobody reads on every phase-1 frame.
FERTILITY_DATASETS = 12

#: How many sources one *live* frame names, and the same argument for the same
#: reason — except that this list goes out every second rather than once, so the
#: cap is what bounds the whole pack's reporting rather than one frame's.
#:
#: The window is centred on the source being read rather than taken from the
#: front, which is the difference between showing where a build has got to and
#: showing where it started. `Tokenize.sources_total` carries the real count, so
#: a window is never mistaken for the whole selection.
BUILD_SOURCES = 16

#: One curriculum stage as the corpus knows it: what it was called, and which
#: datasets it drew from, each resolved against the data engine. Epochs are
#: absent on purpose — see `corpus_key`.
StageSelection = tuple[str | None, list[remote.Selected]]


@dataclass(frozen=True)
class PackBudget:
    """Bound streamed token bytes and retained document rows before either grows past admission."""

    max_corpus_bytes: int
    max_documents: int

    def __post_init__(self) -> None:
        """Require positive integer byte/row ceilings; booleans cannot stand in for resource budgets."""
        if any(type(value) is not int or value < 1 for value in (self.max_corpus_bytes, self.max_documents)):
            raise ValueError("packing budgets must be positive integer byte/row ceilings")


@contextmanager
def _corpus_output(corpus: Path) -> Iterator[BinaryIO]:
    """Replace token bytes after successful streaming and remove incomplete output on every failure."""
    partial = corpus.with_suffix(".bin.partial")
    try:
        with partial.open("wb") as output:
            yield output
        partial.replace(corpus)
    finally:
        partial.unlink(missing_ok=True)


@dataclass(frozen=True)
class SourceDoc:
    """One document on its way into the corpus, tagged with the dataset it came
    from so a span can record which dataset it belongs to.

    Text, not a path. The LLM engine owns no corpora — this arrives a page at a time
    from the data engine, through the backend, and is dropped as soon as it has
    been encoded.
    """

    dataset: str
    text: str
    stage: int = 0


@dataclass(frozen=True)
class _Piece:
    """One item of an encode batch: a slice of one document, and where in that
    document it falls.

    Documents are cut up (`bpe.split_for_encoding`) for two reasons that arrive
    together. The encoder parallelizes across the *items* of a batch, so a batch
    of nine books is nine work units for six cores and the largest book is the
    batch; and a batch built out of whole documents is as big as the biggest one
    in it, which on `gutenberg_eng-clean` is 27.6 MB of text and 4.5 GiB inside
    the encoder.

    The two flags are what puts the document back together: a span cannot be
    recorded until the last of its pieces has been encoded, and a batch boundary
    is allowed to fall between two pieces of the same document.
    """

    source: SourceDoc
    text: str
    opens: bool
    closes: bool


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
    # Characters read and tokens written, per dataset: `[{dataset, chars, tokens}]`.
    #
    # Their ratio is fertility, and it is per source because it is uneven across
    # sources — code encodes at ~2 chars a token where prose runs at ~4, so a
    # selection's mix decides how much text a context window actually holds, and
    # how large a vocabulary is worth paying for. Neither number survives the
    # pack: the text is dropped as it is encoded, and the tokens are a flat
    # stream with only the span table to say which document they came from.
    fertility: list[dict] = field(default_factory=list)


def _blank_row(dataset: str, stage: int) -> dict:
    """One source's counters before it has been read. What the plan knows on top
    of this — how much there is going to be — is added by whoever seeds it."""
    return {"dataset": dataset, "stage": stage, "documents": 0, "chars": 0, "tokens": 0}


class _Tally:
    """What the pack has counted so far, and the live frame built off it.

    One object rather than the three parallel dicts this used to keep, because
    the live report needs a fourth counter keyed exactly like the others and
    four dicts walked in step is a fifth one waiting to disagree with them.

    Rows are keyed by `(stage, dataset)` and not by the dataset alone: a dataset
    named by two stages is read twice on purpose, and a bar that summed the two
    passes would sit at half through the whole of the first. `fertility` sums
    them back down to the name, which is the granularity the sidecar records and
    the only one that survives the build.
    """

    def __init__(self, plan: list[dict] | None) -> None:
        # Seeded in reading order from the plan, so a source that has not been
        # reached yet is still on the strip — at zero, which is a fact about the
        # build rather than a gap in it. A source absent from the plan is
        # appended as it arrives; nothing is dropped for being unexpected.
        self._rows: dict[tuple[int, str], dict] = {}
        for one in plan or []:
            row = _blank_row(one["dataset"], one["stage"])
            row["expected"] = one["expected"]
            # Absent rather than zero where the plan does not state it — the
            # convention `Tokenize` follows, and here it also decides which unit
            # the ETA is projected in. See `expected_chars`.
            if one.get("expected_chars"):
                row["expected_chars"] = one["expected_chars"]
            self._rows[(one["stage"], one["dataset"])] = row
        self._active: tuple[int, str] | None = None
        self.documents = 0
        self.chars = 0
        self.tokens = 0

    def read(self, source: SourceDoc) -> None:
        row = self._row(source)
        row["documents"] += 1
        row["chars"] += len(source.text)
        self.documents += 1
        self.chars += len(source.text)

    def wrote(self, source: SourceDoc, tokens: int) -> None:
        self._row(source)["tokens"] += tokens
        self.tokens += tokens

    def _row(self, source: SourceDoc) -> dict:
        key = (source.stage, source.dataset)
        self._active = key
        if key not in self._rows:
            self._rows[key] = _blank_row(source.dataset, source.stage)
        return self._rows[key]

    @property
    def datasets(self) -> int:
        return len({row["dataset"] for row in self._rows.values() if row["documents"]})

    @property
    def expected(self) -> int:
        """Documents the plan says will be read."""
        return sum(row.get("expected", 0) for row in self._rows.values())

    @property
    def expected_chars(self) -> int:
        """Characters the plan says will be read, and the unit the bar and the
        ETA are measured in wherever it says.

        Documents are what a person counts and are the wrong thing to project
        from, because a document is not an amount of text.
        `gutenberg_eng-clean` is 86% of a typical selection's bytes and 5.6% of
        its documents, so a pack a quarter of the way through the books reported
        fourteen hours left and would have collapsed to under one the moment it
        reached the snippets. A dataset's recorded `byte_size` is within 1.5% of
        its character count on every corpus here, which is far inside what an ETA
        is worth.
        """
        return sum(row.get("expected_chars", 0) for row in self._rows.values())

    def fertility(self) -> list[dict]:
        """Characters read and tokens written per dataset, for the sidecar.

        Summed across stages, and over every source that was actually read —
        a planned source the selection turned out to hold nothing for
        contributed no text and has no ratio to record.
        """
        per_dataset: dict[str, dict] = {}
        for row in self._rows.values():
            if not row["documents"]:
                continue
            one = per_dataset.setdefault(
                row["dataset"], {"dataset": row["dataset"], "chars": 0, "tokens": 0}
            )
            one["chars"] += row["chars"]
            one["tokens"] += row["tokens"]
        return list(per_dataset.values())

    def report(self, report, *, force: bool = False) -> None:
        """Say where the pack is, if anyone is listening and a frame is due.

        The `due` check comes first so that a build with nowhere to report — a
        script, a test — never pays for the window and the row copies it is
        about to throw away.
        """
        if report is None or not (force or report.due):
            return
        expected = self.expected
        # Off characters where the plan states them, and off documents where it
        # does not — a script or a test hands over a plan with counts alone, and
        # a fraction measured against a total nobody stated would be a fraction
        # of nothing. See `expected_chars` for why the unit matters this much.
        wanted = self.expected_chars
        report.say(
            **(
                report.pace(self.chars, wanted, self.tokens)
                if wanted
                else report.pace(self.documents, expected, self.tokens)
            ),
            documents=expected or None,
            documents_done=self.documents,
            chars=self.chars,
            total_tokens=self.tokens,
            sources=self._window(),
            sources_total=len(self._rows),
        )

    def _window(self) -> list[dict]:
        """`BUILD_SOURCES` rows centred on the one being read."""
        rows = list(self._rows.values())
        if len(rows) <= BUILD_SOURCES:
            return rows
        keys = list(self._rows)
        centre = keys.index(self._active) if self._active in self._rows else 0
        start = min(max(0, centre - BUILD_SOURCES // 2), len(rows) - BUILD_SOURCES)
        return rows[start : start + BUILD_SOURCES]


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


def _batches(documents_in, limit: int | None = None, chunk: int | None = None):
    """`documents_in`, cut into pieces and grouped into one encode's worth each.

    The one place a stream of documents becomes a stream of work, and a
    generator so that `_read_ahead` can run the whole of it — the page fetch,
    the JSON, the cutting — on a thread of its own.

    A batch overshoots `limit` by at most one piece, which is what makes it
    bounded: `bpe.CHUNK_CHARS` and not "however long the next document is".

    Both sizes are read here rather than bound into the signature, so that they
    stay knobs a test can turn down without needing a corpus scaled to them.
    """
    limit = limit or ENCODE_BATCH_CHARS
    chunk = chunk or bpe.CHUNK_CHARS
    batch: list[_Piece] = []
    pending = 0
    for source in documents_in:
        pieces = bpe.split_for_encoding(source.text, chunk)
        last = len(pieces) - 1
        for index, text in enumerate(pieces):
            batch.append(_Piece(source, text, opens=index == 0, closes=index == last))
            pending += len(text)
            if pending >= limit:
                yield batch
                batch, pending = [], 0
    if batch:
        yield batch


def _read_ahead(batches, depth: int = READ_AHEAD_BATCHES):
    """`batches`, filled by a thread of its own.

    So that waiting for a page overlaps encoding the page before it. Nothing
    else moves: the same batches arrive in the same order, and every counter,
    every span and the corpus file itself are still touched only by the thread
    that consumes this.

    What the thread is for is also what makes it safe — it does the reading and
    nothing else, so `documents_in` has exactly one consumer, as it did when the
    packer iterated it directly.
    """
    queue: Queue = Queue(maxsize=depth)
    stop = Event()
    ended = object()

    def offer(item) -> bool:
        """Hand one item over, unless the pack has gone away in the meantime."""
        while not stop.is_set():
            try:
                queue.put(item, timeout=READ_AHEAD_POLL_SECS)
                return True
            except Full:
                continue
        return False

    def fill() -> None:
        try:
            for batch in batches:
                if not offer(batch):
                    return
        except Exception as failure:  # noqa: BLE001 — re-raised below, in the caller
            offer(failure)
        else:
            offer(ended)

    reader = Thread(target=fill, name="corpus-read-ahead", daemon=True)
    reader.start()
    try:
        while True:
            item = queue.get()
            if item is ended:
                return
            if isinstance(item, Exception):
                # A dropped page belongs to the pack that asked for it, not to a
                # thread nobody is watching. `remote.Client` has already retried.
                raise item
            yield item
    finally:
        # A pack that ended early — its own failure, or a caller that stopped
        # consuming — leaves the reader blocked on a queue nobody will drain. It
        # checks between attempts; a fetch already in flight has its own timeout
        # and the thread is a daemon, so nothing here waits on the network.
        stop.set()
        reader.join(timeout=READ_AHEAD_POLL_SECS * 2)


def pack(
    documents_in,
    tok: Tokenizer,
    *,
    key: str,
    stages: list[dict] | None = None,
    tokenizer_id: str | None = None,
    fingerprint: str = "",
    report=None,
    plan: list[dict] | None = None,
    budget: PackBudget | None = None,
) -> PackStats:
    """Encode every document into one flat corpus, recording its boundaries.

    `documents_in` is an iterable of `SourceDoc`, consumed once and never held:
    it is a paged read from the data engine, and materializing it would be the
    engine keeping a copy of a corpus it does not own.

    `report` is the caller's `llm.train.progress.BuildReport`, passed in rather
    than imported — `corpus` sits left of `train` in the layering, and a packer
    that built its own frames would be reaching right. `plan` is what it is
    reporting against: `[{dataset, stage, expected, expected_chars}]` in reading
    order, which is the one thing a stream of `SourceDoc` cannot say about
    itself. Without it the pack still reports what it has done, and cannot say
    how much is left. `expected_chars` is optional and is what an ETA is
    projected from where it is there — see `_Tally.expected_chars`.
    `budget` bounds token bytes, including separators, and document rows before
    buffering/writing them. A refusal removes partial token output and retains
    a previous complete build. Encoding one text batch precedes this
    output check; the budget is not a tokenizer allocation estimate.
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
    corpus_len = 0

    tally = _Tally(plan)

    storage.ensure_dir(storage.pretrain_dataset_dir(key))

    corpus = storage.pretrain_corpus_path(key)

    buffer: list[int] = []

    counts = np.zeros(vocab, dtype=np.int64)

    open_start = 0
    open_length = 0
    document_count = 0

    def flush() -> None:
        """Write the current bounded token buffer and accumulate its exact vocabulary histogram."""
        if buffer:
            block = np.asarray(buffer, dtype=CORPUS_DTYPE)
            counts[:] += np.bincount(block, minlength=vocab)[:vocab]
            out.write(block.tobytes())
            buffer.clear()

    with _corpus_output(corpus) as out:

        def write(batch: list[_Piece]) -> None:
            """Encode one batch and append what it holds, in order."""
            nonlocal corpus_len, open_start, open_length, document_count

            encoded = bpe.encode_batch(tok, [piece.text for piece in batch])
            for piece, ids in zip(batch, encoded):
                source = piece.source
                closes_document = piece.closes and (bool(ids) or (not piece.opens and open_length > 0))
                next_tokens = corpus_len + len(ids) + int(closes_document)
                if budget is not None:
                    if next_tokens * np.dtype(CORPUS_DTYPE).itemsize > budget.max_corpus_bytes:
                        raise ValueError("packed corpus exceeds training input byte budget")
                    if document_count + int(closes_document) > budget.max_documents:
                        raise ValueError("packed corpus exceeds training document row budget")
                if piece.opens:

                    tally.read(source)
                    open_start, open_length = corpus_len, 0

                if ids:
                    buffer.extend(ids)
                    corpus_len += len(ids)
                    open_length += len(ids)
                    seen.update(marker_ids.intersection(ids))
                    if len(buffer) >= FLUSH_TOKENS:
                        flush()

                if piece.closes and open_length:
                    document_count += 1
                    documents.append(
                        source.dataset,
                        start=open_start,
                        length=open_length,
                        stage=source.stage,
                    )

                    buffer.append(eot_id)
                    corpus_len += 1
                    tally.wrote(source, open_length)

            tally.report(report)

        for batch in _read_ahead(_batches(documents_in)):
            write(batch)
        flush()

    tally.report(report, force=True)

    per_dataset = tally.fertility()
    spans = documents.build()
    spans.save(storage.pretrain_spans_path(key))

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

                "fertility": per_dataset,

                "documents": len(spans),
                "datasets": spans.names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return PackStats(
        datasets=tally.datasets,
        chars=tally.chars,
        vocab_size=vocab,
        corpus_len=corpus_len,
        documents=len(spans),
        special_ids_seen=sorted(seen),
        fertility=per_dataset,
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
        # Characters read and tokens written per dataset. Absent on a build made
        # before it was recorded — see `fertility`.
        "fertility": raw.get("fertility"),
    }
    return (
        raw["eot_id"],
        spans_mod.from_sidecar(storage.pretrain_spans_path(key), raw),
        info,
    )


def fertility(info: dict) -> list[dict] | None:
    """Chars per token by source, from what `load_segments` read.

    None for a build packed before the counters existed, which is the difference
    between "this corpus has no uneven sources" and "nobody measured" — the same
    distinction `special_ids_seen` draws, and for the same reason.

    A dataset that encoded to nothing is dropped rather than divided by zero: it
    contributed no context budget, so it has no fertility to report.
    """
    recorded = info.get("fertility")
    if not recorded:
        return None
    measured = [one for one in recorded if one.get("tokens")]
    measured.sort(key=lambda one: one["tokens"], reverse=True)
    return [
        {
            "dataset": one["dataset"],
            "chars_per_token": one["chars"] / one["tokens"],
            "tokens": one["tokens"],
        }
        for one in measured[:FERTILITY_DATASETS]
    ] or None


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
