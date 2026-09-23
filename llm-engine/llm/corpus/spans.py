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

from dataclasses import asdict, dataclass
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

#: Bins in the document-length histogram. Fixed and small for the reason
#: `llm.train.diagnostics.ACT_HIST_BINS` is: it rides a phase-1 frame from here
#: to a panel, and a histogram whose width is a property of the corpus is a
#: frame whose size is too.
LENGTH_HIST_BINS = 24

#: How far the histogram reaches, in multiples of `ctx_window`; the last bin
#: absorbs everything beyond it. Two windows rather than one because the
#: interesting question is *how far* past the window a corpus of books runs, and
#: a histogram that stopped at the window would answer only "past it".
LENGTH_HIST_SPAN = 2.0

#: How many rows the measurements read at a time. The table is mapped and a real
#: corpus has millions of rows, so a whole-column temporary is the one
#: allocation in a measurement that would scale with the build.
MEASURE_BLOCK = 1 << 20

#: How many training windows `span_ribbon` draws. Fixed for the reason
#: `LENGTH_HIST_BINS` is, and the reason `llm.train.diagnostics.ACT_HIST_BINS`
#: is: a corpus has millions of windows, so a sample whose *count* were a
#: property of the corpus would be a frame whose size is too. Forty-eight is what
#: a strip the width of a figure can draw as separate bars and a reader can still
#: take in at once — the ribbon is a picture of the packing, not a census of it.
RIBBON_WINDOWS = 48

#: Documents drawn per sampled window before the remainder is counted rather than
#: listed. This is the other half of the frame bound: `RIBBON_WINDOWS *
#: RIBBON_SEGMENTS` is the hard ceiling on how many segments a frame can carry,
#: whatever the build. A window is only bounded by the corpus otherwise — nothing
#: stops a window of tweets holding `ctx_window // 2` documents. Sixteen is where
#: a segment stops being wide enough to read as a bar; past it the honest drawing
#: is a texture, which is what `RibbonWindow.overflow` lets the renderer make.
RIBBON_SEGMENTS = 16


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


@dataclass
class WindowFit:
    """How a packed corpus sits against the context window it will be trained at.

    `straddle` is the fraction of training windows holding an `<|endoftext|>`
    anywhere but their last slot — the windows across which the model attends
    from one document into another. It is a property of the *packing* rather than
    of the model, which is why it is measured here and not in the loop.

    `length_hist` is the document-length distribution in `LENGTH_HIST_BINS` bins
    reaching `LENGTH_HIST_SPAN` windows, as fractions; `over_ctx` is the share of
    documents no single window can hold, which is the pretrain counterpart of the
    fine-tune path's `skipped_too_long` — except that nothing is skipped, the
    document is simply split across windows.
    """

    straddle: float
    length_hist: list[float]
    over_ctx: float


def window_fit(spans: Spans, ctx_window: int, corpus_len: int) -> WindowFit | None:
    """Measure `spans` against `ctx_window`, or None when there is nothing to say.

    Read in the table's own order, which is corpus order — what `EpochStream`
    yields for a run that does not shuffle, and what the distinct-window count
    below relies on, since sorted boundaries make "how many windows" a count of
    changes rather than a set. A shuffled epoch re-tiles the same multiset of
    lengths, so it moves the straddle by the arrangement of at most one window
    and not by the shape of the corpus.
    """
    windows = max(0, (corpus_len - 1) // ctx_window) if ctx_window > 0 else 0
    if not spans or windows == 0:
        return None

    edges = np.linspace(0.0, LENGTH_HIST_SPAN * ctx_window, LENGTH_HIST_BINS + 1)
    counts = np.zeros(LENGTH_HIST_BINS, dtype=np.int64)
    over = 0
    crossings = 0
    # A window index no window can have, so the first boundary always counts.
    previous = -1
    limit = windows * ctx_window

    for start in range(0, len(spans), MEASURE_BLOCK):
        rows = spans.rows[start : start + MEASURE_BLOCK]
        lengths = np.asarray(rows["length"], dtype=np.int64)
        counts += np.histogram(np.clip(lengths, 0, edges[-1]), bins=edges)[0]
        over += int(np.count_nonzero(lengths > ctx_window))

        # The separator that closes each document. Windows past `limit` are the
        # trailing partial batch `iter_batches` drops, so they are not trained on.
        boundaries = np.asarray(rows["start"], dtype=np.int64) + lengths
        inside = boundaries[
            (boundaries < limit) & (boundaries % ctx_window != ctx_window - 1)
        ]
        if inside.size:
            index = inside // ctx_window
            crossings += int(np.count_nonzero(np.diff(index, prepend=previous)))
            previous = int(index[-1])

    return WindowFit(
        straddle=crossings / windows,
        length_hist=(counts / max(1, int(counts.sum()))).tolist(),
        over_ctx=over / len(spans),
    )


@dataclass
class RibbonSegment:
    """One document's presence inside one training window.

    `start` and `length` are in tokens, measured from the window's first slot and
    clipped to it, so a renderer needs nothing but the window's width to place
    them. `opens`/`closes` are what that clipping throws away and the figure
    exists to show: whether the document's own first and last token are in this
    window. An edge that is neither is the window's edge rather than a boundary
    the model attends across, and drawing the two alike is what turns one book
    into nine unrelated documents.
    """

    dataset: int
    start: int
    length: int
    opens: bool
    closes: bool


@dataclass
class RibbonWindow:
    """One sampled window: `index` in corpus order, and what is inside it.

    `overflow` is how many documents past `RIBBON_SEGMENTS` this window held.
    Non-zero means `segments` stops short of the window's width on purpose — the
    tail is real and is not drawn, and a renderer that fills it with the last
    segment would be claiming a long document where there are dozens of short
    ones.
    """

    index: int
    segments: list[RibbonSegment]
    overflow: int


@dataclass
class SpanRibbon:
    """A sample of training windows, each broken into the documents inside it.

    The concrete picture behind `WindowFit.straddle`. "0.63" says the model
    attends across a document boundary in most of its windows; this says what
    that looks like, which is the thing that changes a decision — a corpus of
    books is windows of one full-width segment and a straddle that means "the
    window is mid-chapter", a corpus of tweets is confetti and the same number
    means "every window is forty unrelated things".

    `datasets` is the name table `RibbonSegment.dataset` indexes, carried whole
    rather than narrowed to the sampled windows so the codes stay the table's own.

    Offsets are in tokens against `ctx_window`, which is deliberately *not*
    repeated here: it is not part of the corpus key — the same build is
    legitimately trained at two window sizes — so it belongs to the run that is
    reporting, and the caller already has it. Same reason `doc_length_hist` ships
    bare fractions.
    """

    datasets: list[str]
    windows: list[RibbonWindow]

    def as_dict(self) -> dict:
        return asdict(self)


def _sampled_windows(windows: int) -> np.ndarray:
    """Which windows the ribbon draws, as ascending indices.

    Evenly spaced across the trained region rather than the first
    `RIBBON_WINDOWS`. The head of a corpus is the cheapest sample and the least
    representative one: `orchestrate.train._documents` yields stage by stage and
    dataset by dataset, so a build's opening windows are all one dataset, and a
    ribbon of them would show every multi-dataset corpus as homogeneous — not
    because it is, but because it is sorted. Spreading the sample gives each
    dataset a share of the strip in proportion to the tokens it contributed, and
    keeps the strip's own order the corpus's, so the bands read as the layout.

    Deterministic rather than random, and that outlives the figure: the run that
    packs a build and every later run that reuses it both report this frame, and
    two ribbons of one corpus that disagree read as the corpus having changed
    when only the sample did. The cost of a systematic sample is aliasing against
    a periodic corpus; packing is blocked by dataset rather than interleaved, so
    there is no period here for the stride to lock onto.
    """
    count = min(RIBBON_WINDOWS, windows)
    return np.unique(np.linspace(0, windows - 1, count).round().astype(np.int64))


def _overlapping(
    starts: np.ndarray, lengths: np.ndarray, lows: np.ndarray, highs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per window, the `[begin, end)` rows of this block that touch it.

    Both searches are binary rather than a mask over the block, because a mask is
    one pass over every row per window and there are `RIBBON_WINDOWS` of them.
    They are legitimate because documents are packed in ascending, non-overlapping
    order, which makes `starts` sorted and — since a document is followed by the
    separator that closes it — `starts + lengths` sorted too.

    A window whose opening token falls *inside* a document is the case the
    searches miss: that document starts before `low` and so sorts before the
    range. It is at most one row back, again because documents do not overlap, so
    it is recovered by stepping `begin` back one and keeping it only if it really
    does reach into the window. When it is the previous block's last row this
    block's `begin` clamps to 0 instead, which is the same answer — the block it
    lives in has already drawn it.
    """
    begins = np.searchsorted(starts, lows, side="right") - 1
    ends = np.searchsorted(starts, highs, side="left")
    reaches = (begins >= 0) & (starts[begins] + lengths[begins] > lows)
    return np.maximum(np.where(reaches, begins, begins + 1), 0), ends


def _segment(code: int, start: int, length: int, low: int, high: int) -> RibbonSegment:
    end = start + length
    visible = max(start, low)
    return RibbonSegment(
        dataset=int(code),
        start=int(visible - low),
        length=int(min(end, high) - visible),
        opens=bool(start >= low),
        closes=bool(end <= high),
    )


def span_ribbon(spans: Spans, ctx_window: int, corpus_len: int) -> SpanRibbon | None:
    """Lay out a sample of `ctx_window` windows, or None when there are none.

    Windows are tiled from the corpus's first token exactly as `window_fit`
    counts them, so the two figures describe the same tiling and a segment
    crossing a bar's edge is one of the crossings the straddle counted.
    """
    windows = max(0, (corpus_len - 1) // ctx_window) if ctx_window > 0 else 0
    if not spans or windows == 0:
        return None

    sampled = _sampled_windows(windows)
    lows = sampled * ctx_window
    highs = lows + ctx_window
    drawn: list[list[RibbonSegment]] = [[] for _ in sampled]
    overflow = [0] * len(sampled)

    for block in range(0, len(spans), MEASURE_BLOCK):
        rows = spans.rows[block : block + MEASURE_BLOCK]
        # Blocked and copied out for the reason `window_fit` is: the table is
        # mapped, and a whole-column temporary is the one allocation here that
        # would scale with the build.
        starts = np.asarray(rows["start"], dtype=np.int64)
        lengths = np.asarray(rows["length"], dtype=np.int64)
        codes = np.asarray(rows["dataset"], dtype=np.int64)
        begins, ends = _overlapping(starts, lengths, lows, highs)

        for at in range(len(sampled)):
            # Bounded by the window: a document is at least one token and carries
            # a separator, so no window holds more than `ctx_window // 2` of them.
            for row in range(int(begins[at]), int(ends[at])):
                if len(drawn[at]) >= RIBBON_SEGMENTS:
                    overflow[at] += 1
                    continue
                drawn[at].append(
                    _segment(
                        codes[row], starts[row], lengths[row], lows[at], highs[at]
                    )
                )

    return SpanRibbon(
        datasets=list(spans.names),
        windows=[
            RibbonWindow(index=int(sampled[at]), segments=drawn[at], overflow=overflow[at])
            for at in range(len(sampled))
        ],
    )


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
