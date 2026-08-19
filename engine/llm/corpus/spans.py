"""Document boundaries, held as columns rather than as objects.

A packed corpus is one flat token stream; this is the table that says where each
document in it starts and ends. Both builds write one — `pack` for pretraining,
`sft` for conversations — and `llm.train.plan` reads it to order an epoch.

Columnar because the table is the one structure here that scales with the
*number* of documents rather than with their size. A `DocSpan` object costs 145
bytes resident; a row costs 22. At the granularity a corpus of files has (one
span per shard, ~40k tokens) either would do. At the granularity a corpus of
documents has — hundreds of thousands of rows for the same tokens — that 6x, the
JSON encoding of the same table, and the per-epoch copy of it in `EpochStream`
are together enough to lose the machine, which is the same failure the corpus
memmap and the epoch index arrays were introduced to prevent.

So the rows go to a `.npy` beside `corpus.bin`, loaded by mapping, and the
sidecar keeps only what a person reads: what the build was made from, and the
table of dataset names these rows index into.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: One row. `dataset` indexes the name table rather than repeating the string:
#: a corpus has a handful of datasets and millions of documents.
SPAN_DTYPE = np.dtype(
    [("start", np.int64), ("length", np.int64), ("dataset", np.int32), ("stage", np.int16)]
)

#: How many rows a builder holds as Python ints before folding them into a numpy
#: block. The buffer is the only part of a build whose cost is per-row rather
#: than per-block, so it is kept small; the blocks are what memory is actually
#: spent on, and they are `SPAN_DTYPE`-sized.
BUILD_BLOCK = 1 << 16


@dataclass
class DocSpan:
    """A contiguous span of the flat corpus forming one document.

    `start`/`len` are exclusive of the <|endoftext|> separators baked between
    documents. This is the *view* of a row — one is materialized when something
    iterates the table or asks for a single document. Nothing holds a list of
    them for a whole corpus any more.
    """

    dataset: str
    start: int
    length: int
    stage: int = 0


class Spans:
    """The document table of one packed corpus.

    Iterating yields `DocSpan`s, so a script that wants documents one at a time
    reads the same as it always did; the training path takes `starts`/`lengths`
    as arrays and never materializes a row at all.
    """

    __slots__ = ("names", "rows")

    def __init__(self, names: list[str], rows: np.ndarray):
        self.names = names
        self.rows = rows

    @classmethod
    def empty(cls) -> "Spans":
        return cls([], np.empty(0, dtype=SPAN_DTYPE))

    @classmethod
    def of(cls, documents) -> "Spans":
        """Build a table from an iterable of `DocSpan` — for a caller that has a
        handful of them in hand (a test, a benchmark fixture, a legacy sidecar)
        rather than a corpus being packed."""
        builder = SpanBuilder()
        for document in documents:
            builder.append(document.dataset, document.start, document.length, document.stage)
        return builder.build()

    @classmethod
    def coerce(cls, documents) -> "Spans":
        """Accept either a table or a plain list of `DocSpan`.

        The training path is given spans by two callers that differ in kind: a
        run reads them off disk as a table, while `bench.py` and the tests
        synthesize a few by hand. Neither should have to know which the other
        does.
        """
        return documents if isinstance(documents, Spans) else cls.of(documents)

    def __len__(self) -> int:
        return len(self.rows)

    def __bool__(self) -> bool:
        return len(self.rows) > 0

    def __getitem__(self, index: int) -> DocSpan:
        row = self.rows[index]
        return DocSpan(
            dataset=self.names[int(row["dataset"])],
            start=int(row["start"]),
            length=int(row["length"]),
            stage=int(row["stage"]),
        )

    def __iter__(self):
        for index in range(len(self.rows)):
            yield self[index]

    @property
    def starts(self) -> np.ndarray:
        return self.rows["start"]

    @property
    def lengths(self) -> np.ndarray:
        return self.rows["length"]

    def in_stage(self, stage: int) -> "Spans":
        """The documents packed into one curriculum stage.

        The name table is carried over whole rather than rebuilt: it is a few
        strings, and keeping the indices valid means a stage's rows are a view of
        the parent's rather than a rewrite of them.
        """
        return Spans(self.names, self.rows[self.rows["stage"] == stage])

    def save(self, path: Path) -> None:
        """Write the rows. The name table is the sidecar's — see `names_of`."""
        with path.open("wb") as out:
            np.save(out, self.rows, allow_pickle=False)

    @classmethod
    def load(cls, path: Path, names: list[str]) -> "Spans":
        """Map the rows back. Mapped rather than read for the reason the corpus
        is: a run touches the table by column and by slice, and the kernel can
        reclaim a mapping under pressure."""
        return cls(names, np.load(path, mmap_mode="r", allow_pickle=False))


def from_sidecar(spans_path: Path, raw: dict) -> Spans:
    """The document table of a build, from either shape of sidecar.

    A corpus packed before the rows moved out of `segments.json` carries them
    inline, as a list of objects with a `dataset` string each. Those builds are
    hours of packing apiece and are otherwise perfectly valid corpora, so they
    are converted on the way in rather than invalidated — and rewritten in the
    new shape only when something re-packs them.
    """
    if spans_path.is_file():
        return Spans.load(spans_path, list(raw.get("datasets", [])))

    documents = raw.get("documents")
    if isinstance(documents, list):
        return Spans.of(DocSpan(**document) for document in documents)
    return Spans.empty()


class SpanBuilder:
    """Accumulates rows during a build, in blocks.

    Appending straight into a numpy array means resizing it, and resizing means
    holding the old and the new at once. Appending into a Python list means
    paying the interpreter's per-int overhead across the whole corpus. So rows
    are buffered as tuples only up to `BUILD_BLOCK` and folded into a typed block
    from there — peak overhead is one block, whatever the corpus turns out to be.
    """

    def __init__(self):
        self._names: list[str] = []
        self._index: dict[str, int] = {}
        self._buffer: list[tuple[int, int, int, int]] = []
        self._blocks: list[np.ndarray] = []

    def append(self, dataset: str, start: int, length: int, stage: int = 0) -> None:
        code = self._index.get(dataset)
        if code is None:
            code = self._index[dataset] = len(self._names)
            self._names.append(dataset)
        self._buffer.append((start, length, code, stage))
        if len(self._buffer) >= BUILD_BLOCK:
            self._fold()

    def _fold(self) -> None:
        if self._buffer:
            self._blocks.append(np.array(self._buffer, dtype=SPAN_DTYPE))
            self._buffer.clear()

    def build(self) -> Spans:
        self._fold()
        rows = (
            np.concatenate(self._blocks) if self._blocks else np.empty(0, dtype=SPAN_DTYPE)
        )
        return Spans(self._names, rows)

    def __len__(self) -> int:
        return sum(len(block) for block in self._blocks) + len(self._buffer)
