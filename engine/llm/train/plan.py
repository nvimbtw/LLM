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


def epoch_stream(
    corpus: np.ndarray,
    plan: DataPlan,
    epoch: int,
    mask: np.ndarray | None = None,
    documents: list[DocSpan] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """The token stream for one epoch, documents reordered and rejoined.

    Reshuffled per epoch off `seed + epoch`, so a run is reproducible while no
    two epochs see the same order.

    `mask` (SFT) is reordered by the same permutation and returned alongside.
    Reordering the two independently would misalign every target in the run —
    they are gathered here together so they cannot get out of step.
    """
    documents = list(plan.documents if documents is None else documents)
    if plan.shuffle:
        np.random.default_rng(plan.seed + epoch).shuffle(documents)

    separator = np.array([plan.eot_id], dtype=corpus.dtype)
    # The separator is scaffold, never a target.
    separator_mask = np.zeros(1, dtype=np.uint8)

    tokens: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for doc in documents:
        end = doc.start + doc.length
        tokens.append(corpus[doc.start : end])
        tokens.append(separator)
        if mask is not None:
            masks.append(mask[doc.start : end])
            masks.append(separator_mask)

    if not tokens:
        return corpus, mask
    return np.concatenate(tokens), (np.concatenate(masks) if mask is not None else None)


def stream_length(
    corpus_len: int, plan: DataPlan, documents: list[DocSpan] | None = None
) -> int:
    """How long `epoch_stream` would be, without building it.

    Mirrors that function exactly: one separator per document, and the whole
    corpus when there are no documents. Shuffling reorders, so the length does
    not depend on the epoch.

    Exists because the step count is measured before training starts, and
    materializing the stream just to read its length costs a full copy of the
    corpus — 745 MiB on a 390M-token run, once per stage, immediately discarded.
    """
    documents = plan.documents if documents is None else documents
    if not documents:
        return corpus_len
    return sum(doc.length + 1 for doc in documents)


def batch_count(stream_len: int, ctx_window: int, batch_size: int) -> int:
    """How many full batches a stream of this length yields."""
    windows = max(0, (stream_len - 1) // ctx_window)
    return windows // batch_size


def iter_batches(
    stream: np.ndarray, ctx_window: int, batch_size: int, mask: np.ndarray | None = None
):
    """Yield (inputs, targets, weights) as arrays of [batch, ctx_window].

    Targets are inputs shifted one token left — the next-token objective. The
    trailing partial batch is dropped rather than padded: padding would train
    the model on filler it never sees at inference.

    `weights` is None for pretraining (every position counts) and the SFT loss
    mask otherwise. It aligns with *input* positions, matching the convention
    `build_training_example` sets: position i is trained iff the token it
    predicts, i+1, belongs to an assistant reply.
    """
    windows = (len(stream) - 1) // ctx_window
    usable = windows - (windows % batch_size)
    # A batch's rows are consecutive windows of one flat stream, so the batch is
    # a single contiguous slice reshaped — not `batch_size` separate gathers. Row
    # r covers [(start+r)*ctx, (start+r+1)*ctx), and those intervals tile the
    # slice exactly, which is as true of the one-token-shifted targets as of the
    # inputs.
    span = batch_size * ctx_window

    for start in range(0, usable, batch_size):
        offset = start * ctx_window
        # The cast is the copy. `stream` is uint16 and torch has no such dtype,
        # and cross-entropy wants int64 targets, so one astype per array does the
        # widening and the copy at once.
        inputs = stream[offset : offset + span].astype(np.int64).reshape(batch_size, ctx_window)
        targets = (
            stream[offset + 1 : offset + span + 1]
            .astype(np.int64)
            .reshape(batch_size, ctx_window)
        )
        weights = (
            mask[offset : offset + span].astype(np.float32).reshape(batch_size, ctx_window)
            if mask is not None
            else None
        )

        yield inputs, targets, weights
