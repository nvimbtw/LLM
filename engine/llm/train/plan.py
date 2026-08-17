"""Turning the packed corpus into the order the model actually sees it.

The corpus is one flat stream. Training straight through it means a fixed data
order every epoch, and — once documents are reordered — predicting across a
document boundary. Both are fixed here: documents are shuffled as units and
rejoined with an explicit separator.
"""

from dataclasses import dataclass, field

import numpy as np

from llm.corpus.pack import DocSpan


@dataclass
class Stage:
    """One curriculum stage: documents trained for `epochs` epochs before the run
    moves on to the next."""

    name: str | None
    epochs: int
    documents: list[DocSpan]


@dataclass
class DataPlan:
    """How to order and segment the corpus for a run."""

    documents: list[DocSpan]
    eot_id: int
    shuffle: bool
    seed: int
    # Ordered curriculum. Empty means a single implicit stage over `documents`.
    stages: list[Stage] = field(default_factory=list)

    def ordered_stages(self, default_epochs: int) -> list[Stage]:
        """The stages to train, or the one implicit stage a flat run describes."""
        if self.stages:
            return self.stages
        return [Stage(name=None, epochs=default_epochs, documents=self.documents)]

    @classmethod
    def whole_corpus(cls, corpus_len: int) -> "DataPlan":
        """The fallback when no segments sidecar exists: one document covering
        everything, in fixed order."""
        return cls(
            documents=[DocSpan(dataset="corpus", start=0, length=corpus_len)],
            eot_id=0,
            shuffle=False,
            seed=0,
        )


class EpochStream:
    """One epoch's token order, addressable without ever being built.

    The epoch is the documents shuffled and rejoined with a separator between
    them. Materializing that as an array — which is what this class replaced —
    costs a second full copy of the corpus while the first is still held: 745 MiB
    on the 390M-token corpus, per epoch, on a machine with 7 GiB of RAM. It is
    what OOM-killed the engine, and it scales with the corpus, so the bigger the
    run the more certainly it fails.

    Nothing needs the whole thing at once. A batch is a contiguous span of the
    epoch, so this keeps the shuffled order as two small index arrays and copies
    out spans on demand. Documents here average ~40k tokens against a 4k-token
    batch, so a span usually touches one document and never many.

    Reshuffled per epoch off `seed + epoch`, so a run is reproducible while no
    two epochs see the same order. `mask` (SFT) is reordered by the same
    permutation and gathered alongside the tokens rather than separately — doing
    the two independently would misalign every target in the run.
    """

    def __init__(
        self,
        corpus: np.ndarray,
        plan: DataPlan,
        epoch: int,
        mask: np.ndarray | None = None,
        documents: list[DocSpan] | None = None,
    ):
        documents = list(plan.documents if documents is None else documents)
        if plan.shuffle:
            np.random.default_rng(plan.seed + epoch).shuffle(documents)

        self.corpus = corpus
        self.mask = mask
        self.eot_id = plan.eot_id
        # No documents means no segmentation to honour: the epoch is the corpus
        # as it lies, with nothing joining anything, so no separator either.
        self.gap = 1 if documents else 0

        if documents:
            self.starts = np.fromiter((d.start for d in documents), np.int64, len(documents))
            self.lengths = np.fromiter((d.length for d in documents), np.int64, len(documents))
        else:
            self.starts = np.zeros(1, np.int64)
            self.lengths = np.array([len(corpus)], np.int64)

        # Where each document begins in the epoch. The last entry is the epoch's
        # length, which is why the step count no longer needs the stream built.
        self.offsets = np.zeros(len(self.lengths) + 1, np.int64)
        np.cumsum(self.lengths + self.gap, out=self.offsets[1:])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def gather(self, lo: int, hi: int, tokens: np.ndarray, weights: np.ndarray | None = None):
        """Copy the epoch's [lo, hi) into `tokens`, and its mask into `weights`.

        Writes into caller-owned buffers rather than allocating: the caller reuses
        one pinned pair for the whole run, and an allocation per batch would undo
        the point of pinning it.
        """
        doc = int(np.searchsorted(self.offsets, lo, side="right")) - 1
        at = 0
        while at < hi - lo:
            within = lo + at - self.offsets[doc]
            length = self.lengths[doc]
            if within < length:
                # Inside a document: copy as much of it as still fits.
                take = int(min(length - within, hi - lo - at))
                src = int(self.starts[doc] + within)
                tokens[at : at + take] = self.corpus[src : src + take]
                if weights is not None:
                    weights[at : at + take] = (
                        self.mask[src : src + take] if self.mask is not None else 1
                    )
            else:
                # The separator that joins this document to the next. Scaffold,
                # so it is never a trainable target.
                take = 1
                tokens[at] = self.eot_id
                if weights is not None:
                    weights[at] = 0
                doc += 1
            at += take
        return tokens


def batch_count(stream_len: int, ctx_window: int, batch_size: int) -> int:
    """How many full batches a stream of this length yields."""
    windows = max(0, (stream_len - 1) // ctx_window)
    return windows // batch_size


def iter_batches(
    stream: EpochStream,
    ctx_window: int,
    batch_size: int,
    weighted: bool = False,
    allocate=None,
):
    """Yield (inputs, targets, weights) as arrays of [batch, ctx_window].

    Targets are inputs shifted one token left — the next-token objective. The
    trailing partial batch is dropped rather than padded: padding would train the
    model on filler it never sees at inference.

    A batch's rows are consecutive windows of the epoch, so the whole batch is
    one contiguous span of it, gathered once. Inputs and targets are then two
    offset views of that one span rather than two gathers of their own.

    `weights` is None for pretraining (every position counts) and the SFT loss
    mask otherwise. It aligns with *input* positions, matching the convention
    `build_training_example` sets: position i is trained iff the token it
    predicts, i+1, belongs to an assistant reply.

    The buffers are allocated once and rewritten per batch, and the arrays
    yielded are views into them — valid until the next batch, which is exactly
    as long as the training loop needs them. `allocate(n, dtype)` lets the caller
    place them in pinned memory, where the copy to the device is faster.
    """
    allocate = allocate or (lambda n, dtype: np.empty(n, dtype))
    windows = (len(stream) - 1) // ctx_window
    usable = windows - (windows % batch_size)
    span = batch_size * ctx_window

    # One token longer than the batch: the targets are the same span shifted by
    # one, so the two share a buffer instead of being gathered twice.
    tokens = allocate(span + 1, np.int64)
    weights = allocate(span + 1, np.float32) if weighted else None

    for start in range(0, usable, batch_size):
        offset = start * ctx_window
        stream.gather(offset, offset + span + 1, tokens, weights)
        yield (
            tokens[:span].reshape(batch_size, ctx_window),
            tokens[1:].reshape(batch_size, ctx_window),
            weights[:span].reshape(batch_size, ctx_window) if weights is not None else None,
        )
