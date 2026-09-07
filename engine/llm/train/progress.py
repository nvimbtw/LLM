"""Progress reporting.

A callback rather than prints, so the orchestrator decides where frames go —
today a log line, later a WebSocket. The seam exists now because retrofitting it
means changing every signature between here and the route.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

# Roughly how many report frames a run should produce, whatever its length.
#
# A fixed step interval does not work across both ends of this project: the Rust
# loop reported every 100 chunks because an epoch was thousands of them, but a
# batched PyTorch epoch here is often under a hundred steps, so the same constant
# yields two or three points for an entire run — a "curve" that is a straight
# line between its endpoints, with no window ever carrying batch spread or
# diagnostics.
#
# Targeting a point count instead keeps the chart the same shape at any scale,
# and bounds the frame rate a long run can reach.
TARGET_REPORTS = 150

# Diagnostics cost an explicit attention softmax and an SVD, so they are sampled
# across the reports rather than taken on every one.
TARGET_DIAG_SAMPLES = 30

# Roughly how many checkpoints a run should leave behind, whatever its length.
#
# A checkpoint is a whole model — weights, vocabulary, and the optimizer moments
# that let it be trained further — so it costs about three times a `model.pt` and
# ten of them cost thirty. That is the trade this number is: enough of the run
# kept to go back to, few enough that a model list is still a list.
TARGET_CHECKPOINTS = 10

# The shortest gap between two phase-1 frames, and roughly how many of them a
# build should produce however long it turns out to be.
#
# Phase 1 cannot be budgeted the way `TARGET_REPORTS` budgets a training run. A
# run knows its step count before it takes a step; a build does not know whether
# packing is four minutes or four hours until it is over, and both are ordinary.
# So the interval widens with the build's own age — `elapsed /
# TARGET_BUILD_REPORTS`, floored at `BUILD_REPORT_SECS` — which spends frames
# where they are worth the most: a second apart while a watcher is still
# deciding whether anything is happening at all, a minute apart an hour in when
# the only question left is the ETA. The count then grows with the *logarithm*
# of the build's length rather than with the build: ~240 frames over four
# minutes, ~2,300 over four hours, and a run's frame history is kept in memory
# for as long as somebody is attached to it.
BUILD_REPORT_SECS = 1.0
TARGET_BUILD_REPORTS = 240

# How much of the recent past `tokens_per_sec` is measured over. Long enough that
# one slow page does not read as a stall, short enough that a real one does — the
# average the ETA is projected from would take minutes to show either, which is
# why the frame carries both numbers rather than one.
RATE_WINDOW_SECS = 10.0

# Where each of phase 1's steps sits on the one bar they share.
#
# The steps are fixed and their costs are wildly uneven — packing is hours where
# learning merges is seconds — so equal shares would leave the bar at a third
# for all but the last minute of a build. These are cumulative spans and a build
# enters them in order, which is what makes `progress` monotone without anything
# having to clamp it.
#
# A step absent from this table keeps the bar where it is and moves it no
# further, which is the honest reading of a step nobody has sized.
BUILD_SPANS: dict[str, tuple[float, float]] = {
    # Pretrain, in order: draw the vocabulary's sample, learn its merges, encode
    # the corpus, measure what was packed.
    "sampling": (0.00, 0.06),
    "merges": (0.06, 0.10),
    "packing": (0.10, 0.97),
    # A fine-tune's one long step, and the corpus one either mode may find
    # already on disk — both land where packing does, because what follows them
    # is the same measuring pass.
    "rendering": (0.02, 0.97),
    "reusing": (0.90, 0.97),
    "measuring": (0.97, 1.00),
}


def report_every(total_steps: int) -> int:
    """Steps between report frames for a run of this length."""
    return max(1, total_steps // TARGET_REPORTS)


def diagnose_every(total_reports: int) -> int:
    """Reports between diagnostics samples."""
    return max(1, total_reports // TARGET_DIAG_SAMPLES)


def checkpoint_every(total_steps: int, total_epochs: int, target: int | None = None) -> int:
    """Steps between checkpoints for a run of this length and this shape.

    Two numbers decide it, because "ten per run" alone is wrong at both ends. A
    one-epoch run wants roughly `TARGET_CHECKPOINTS` of them spread across it. A
    forty-epoch run wants at least one per epoch — an epoch boundary is where a
    person reaches back to, and a run that kept ten of forty would have thrown
    away three quarters of the points anybody asks for by name.

    Snapped up to a multiple of `report_every`, so a checkpoint always lands on a
    step that is already publishing a frame rather than on one that has to
    invent one.

    `target` of 0 means no intermediate checkpoints at all; the run's own final
    save is not this function's business.
    """
    if target == 0:
        return 0
    wanted = max(target or TARGET_CHECKPOINTS, total_epochs, 1)
    interval = report_every(total_steps)
    return max(interval, -(-max(1, total_steps // wanted) // interval) * interval)


@dataclass
class Tokenize:
    """One phase-1 report: building the data a run trains on.

    The field names are the frontend's `TokenizeStats`, for the same reason
    `Progress`'s are its `TrainStats`. A field left `None` is one this phase does
    not measure — a pretrain counts files and merges, a fine-tune counts
    conversations — and it is dropped from the frame rather than sent as a zero,
    so the panel shows "—" for what was never measured instead of a number that
    reads as a real count of nothing.
    """

    # What the run is doing, as a fraction of phase 1, and whether it is over.
    progress: float = 0.0
    done: bool = False
    # Which of phase 1's steps is running, named rather than inferred. The panel
    # used to read "the step before the first one that has not reported yet is
    # what is happening now", which is only true while every step reports
    # something: packing is the longest step of the run and reported nothing at
    # all, so the whole of it read as a tokenizer that had already finished.
    # `llm.train.progress.BUILD_SPANS` is the list and the order.
    step: str | None = None
    # Wall clock since phase 1 began, and what is left of the step in flight.
    # Both are the build's own — a run's `Progress.eta_secs` covers training and
    # begins at zero, and phase 1 is the part nobody could put a number on.
    elapsed_secs: float | None = None
    eta_secs: float | None = None
    # Documents encoded so far, against `documents`, which a build reports as
    # the count it set out to read before it reads any of them. On a fine-tune
    # these are conversation rows read, which `conversations` then counts again
    # after rendering — the difference between the two is what was skipped.
    documents_done: int | None = None
    # Tokens a second over the last report window rather than over the build.
    # The average is what the ETA is projected from and is deliberately smooth;
    # this is the one that says whether the thing is still moving *now*.
    tokens_per_sec: float | None = None
    # Bytes of text drawn for the vocabulary's merges, against the budget they
    # are drawn to (`llm.corpus.bpe.SAMPLE_BYTES`). Only the sampling step
    # measures these; learning the merges from them reports neither, because the
    # library exposes no callback — see `bpe.merge_sequence`.
    sampled_bytes: int | None = None
    sample_budget: int | None = None
    # Per-source progress while the corpus is packed:
    # `[{dataset, stage, documents, expected, chars, tokens}]`, in the order they
    # are read. This is what the aggregate counters cannot say — which dataset a
    # stalled build is stalled on, and what each one is costing in tokens before
    # the pack is over and `fertility` can answer properly.
    #
    # Bounded by `llm.corpus.pack.BUILD_SOURCES`, and `sources_total` says how
    # many there really are: a long curriculum shows a window around the source
    # in flight rather than its own first sixteen entries forever.
    sources: list[dict] | None = None
    sources_total: int | None = None
    # Pretrain: the BPE tokenizer and the packed corpus.
    #
    # `documents` rather than `files`: the text arrives from the data engine a
    # page at a time and there are no files to count. `files` is kept for a run
    # queued before that, whose frames the panel still has to render.
    documents: int | None = None
    files: int | None = None
    chars: int | None = None
    vocab: int | None = None
    merges: int | None = None
    # What the tokenization and the packing actually produced, as opposed to how
    # much of it there was. Measured off the packed corpus and its document
    # table, so a reused build reports them exactly as a fresh one does.
    #
    # `[{dataset, chars_per_token, tokens}]`, most tokens first. Fertility is per
    # source because it is uneven across sources, and that unevenness is how the
    # context budget is really being spent.
    fertility: list[dict] | None = None
    # Token ids the packed corpus never uses. Tied embeddings make each one a row
    # paid for in every forward pass — see `llm.train.diagnostics.dead_ids`.
    dead_vocab: int | None = None
    # `[{token, count}]` — the head of the merge list in the order BPE learned
    # it, with how often each merged token occurs in the corpus. See
    # `llm.corpus.bpe.merge_sequence` for what that count is and is not.
    merge_sequence: list[dict] | None = None
    # The window every field below was measured against. Sent with them rather
    # than left for the reader to find: `ctx_window` is a property of the run and
    # not of the corpus — deliberately, since one build is legitimately trained
    # at two of them — so the only place it and these numbers are known to agree
    # is the frame that carries both. The panel needs it to label an axis and to
    # size a window, and the run row it used to read has no config field for a
    # training job.
    ctx_window: int | None = None
    # The fraction of `ctx_window` training windows that hold an <|endoftext|>,
    # i.e. how often the model attends across a document boundary.
    boundary_straddle: float | None = None
    # Document lengths against `ctx_window`: fixed-width bins reaching
    # `llm.corpus.spans.LENGTH_HIST_SPAN` windows, as fractions, plus the share
    # of documents no single window can hold.
    doc_length_hist: list[float] | None = None
    docs_over_ctx: float | None = None
    # `llm.corpus.spans.SpanRibbon` as a dict: a sample of training windows, each
    # broken into the documents inside it. The straddle above says how often a
    # window crosses a boundary; this is what those windows look like.
    span_ribbon: dict | None = None
    # Fine-tune: chat conversations rendered into a masked dataset.
    conversations: int | None = None
    total_tokens: int | None = None
    trainable_tokens: int | None = None
    skipped_too_long: int | None = None


class BuildReport:
    """Phase 1 saying where it is, on an interval that widens as it runs.

    Made by the orchestrator and handed *down* into `llm.corpus`, which sits
    left of `llm.train` in the layering and may not import a frame type of its
    own accord. A packer knows the names of the things it is counting; what
    those become on the wire is this object's business, and passing it down is
    what keeps that true without a translation layer in between renaming
    anything.

    Three calls, and what separates them is the throttle rather than the
    payload:

    - `at` opens a step and moves the bar to where that step starts.
    - `tick` is a live update, and is dropped when one went out recently.
    - `say` is a live update that may not be dropped.

    Every frame carries the current step and the elapsed clock whether the
    caller mentioned them or not, because a frame that omitted either would
    merge into the panel's state as the *previous* step still running.
    """

    def __init__(self, on_progress: "ProgressCallback | None") -> None:
        self._on_progress = on_progress
        self._started = time.monotonic()
        self._entered = self._started
        self._last = 0.0
        self._step: str | None = None
        self._span = (0.0, 1.0)
        self._progress = 0.0
        self._marks: deque[tuple[float, int]] = deque([(self._started, 0)])

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    @property
    def due(self) -> bool:
        return time.monotonic() - self._last >= max(
            BUILD_REPORT_SECS, self.elapsed / TARGET_BUILD_REPORTS
        )

    def at(self, step: str, **fields) -> None:
        self._step = step
        lo, hi = BUILD_SPANS.get(step, (self._progress, self._progress))
        self._span = (lo, hi)
        # The clock the ETA is projected from is the *step's*, not the phase's:
        # a pack that inherited the minute the vocabulary took would read as
        # slower than it is for as long as it took to catch up.
        self._entered = time.monotonic()
        self._marks = deque([(self._entered, 0)])
        self.say(**fields)

    def tick(self, **fields) -> bool:
        """A live update, if one is due. Returns whether it went out, so a
        caller can skip assembling anything expensive for a frame nobody will
        see — `sources` is a list per report and the packer builds it only when
        this says yes."""
        if not self.due:
            return False
        self.say(**fields)
        return True

    def finish(self, **fields) -> None:
        """The frame that closes phase 1: the bar goes to full whatever step it
        was on, because the step it was on is the last one there was.

        Separate from `say` rather than a flag on it, because "this step is
        finished" and "the phase is finished" are different claims and the same
        call used to make both — a vocabulary build reporting itself complete
        sent the panel to the training view for the whole of the pack that
        followed it."""
        self._span = (1.0, 1.0)
        self.say(done=True, **fields)

    def say(self, *, fraction: float | None = None, **fields) -> None:
        """One frame, now. `fraction` is how far through *this step* the build
        is; where that lands on the bar is `BUILD_SPANS`' business, and the bar
        never goes backwards even when a step reports one that does."""
        self._last = time.monotonic()
        lo, hi = self._span
        reached = lo if fraction is None else lo + (hi - lo) * min(1.0, max(0.0, fraction))
        self._progress = max(self._progress, reached)
        if self._on_progress is None:
            return
        self._on_progress(
            Tokenize(
                step=self._step,
                progress=self._progress,
                elapsed_secs=round(self.elapsed, 1),
                **fields,
            )
        )

    def pace(self, done: int, total: int, tokens: int | None = None) -> dict:
        """The three fields every step of a build works out the same way: how far
        through it is, how long that leaves, and how fast it is going right now.

        Called once per frame, and it moves the rate window as it goes — so it is
        part of reporting rather than a query about it. `eta_secs` is projected
        from the rate over the whole step, which is the smooth one and the same
        estimator `train.loop` gives a run; `tokens_per_sec` is over the last
        `RATE_WINDOW_SECS`, which is the one that answers "is it still moving".

        Anything it cannot measure is left out rather than sent as a zero: a rate
        of nothing is not slow, it is unmeasured.
        """
        now = time.monotonic()
        elapsed = now - self._entered
        paced: dict = {}
        if total > 0:
            paced["fraction"] = done / total
            if done > 0 and elapsed > 0:
                paced["eta_secs"] = round(elapsed / done * max(0, total - done), 1)
        if tokens is not None:
            while len(self._marks) > 1 and now - self._marks[0][0] > RATE_WINDOW_SECS:
                self._marks.popleft()
            since, then = self._marks[0]
            self._marks.append((now, tokens))
            if now > since:
                paced["tokens_per_sec"] = round((tokens - then) / (now - since), 1)
        return paced


@dataclass
class Progress:
    """One report frame.

    The field names are the frontend's `TrainStats` contract verbatim, so a frame
    travels engine -> queue -> WebSocket -> chart without anything in between
    having to rename it. A renaming layer is where these drift apart silently.
    """

    step: int
    total_steps: int
    epoch: int
    total_epochs: int
    # Position within the current epoch, which is what the per-epoch charts plot.
    chunk: int
    total_chunks: int
    loss: float
    lr: float
    elapsed_secs: float
    # Cumulative over the epoch so far.
    avg_loss: float = 0.0
    min_loss: float = 0.0
    max_loss: float = 0.0
    # Spread over just this report window — the last REPORT_INTERVAL steps — as
    # opposed to the epoch-cumulative figures above.
    batch_min: float = 0.0
    batch_avg: float = 0.0
    batch_max: float = 0.0
    # Seconds remaining, projected from the rate achieved so far. Whole-run, not
    # per-epoch: it is the number a person is actually waiting on.
    eta_secs: float = 0.0
    epoch_done: bool = False
    epoch_duration_secs: float = 0.0
    checkpoint_path: str | None = None
    # Curriculum stage this frame belongs to (0-based) and its label, so a
    # consumer can mark the transitions.
    stage: int = 0
    stage_name: str | None = None
    # Per-report telemetry (grad norms, update:weight ratios, activation RMS,
    # attention entropy, embedding anisotropy). Present only on measured frames.
    #
    # Named `diag` because that is what the frontend's `TrainStats` calls it, and
    # a frame travels from here to the chart untranslated — a rename anywhere in
    # between is a field that silently arrives as undefined.
    diag: dict | None = None

    @property
    def fraction(self) -> float:
        return self.step / self.total_steps if self.total_steps else 0.0


@dataclass
class Checkpoint:
    """One saved training state, announced as it lands.

    A checkpoint is a *model*: its own id, its own directory laid out exactly
    like a finished model's, and its own row. This frame is how the row gets
    written — `llm/` has never heard of the database, so the id is minted here
    and the queue turns it into a row when the frame reaches it.

    It travels the progress callback rather than a seam of its own because it is
    the same thing that seam already carries: something the run has to say while
    it is still running. What it must not inherit is the throttle — a progress
    bar can miss a frame, a model cannot.
    """

    #: uuid4, minted by whoever wrote the directory. It *is* the directory name.
    model_id: str
    #: 1-based ordinal within the run — the numbered suffix a person reads.
    index: int
    #: Global step, 1-based, matching what `Progress.step` reported.
    step: int
    total_steps: int
    epoch: int
    stage: int
    stage_name: str | None
    #: The training state, which is the file a later run resumes from.
    path: str
    loss: float | None = None
    #: The shape the weights were saved at, read off the model rather than off
    #: the config that asked for one: a resumed run's request states no
    #: architecture — inheriting the checkpoint's is what saying nothing means —
    #: so the row this frame writes has nowhere else to learn it from, and
    #: `context` on it would otherwise be empty or a default nobody chose.
    #: Empty from a producer that does not measure it, never zeroes.
    architecture: dict[str, int] = field(default_factory=dict)
    #: The run's own finished model rather than an intermediate. Recorded by the
    #: handler that was going to write that row anyway, so it carries no index.
    final: bool = False


ProgressCallback = Callable[[Progress | Tokenize | Checkpoint], None]


def _reading(value) -> str:
    """One measurement on a log line. A distribution is summarised by its width:
    the merge sequence is hundreds of entries and the frame it rides is where it
    is meant to be read."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return f"[{len(value):,}]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} {_reading(v)}" for k, v in value.items()) + "}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return f"{value:,}"


def log_progress(progress: Progress | Tokenize) -> None:
    """The default sink: one line per report."""
    if isinstance(progress, Tokenize):
        measured = {k: v for k, v in vars(progress).items() if v is not None}
        # The bar and the terminal want opposite things out of the same frame:
        # the fraction is what a bar is made of and is noise beside the counts
        # it was computed from, and the step is the first thing a reader wants
        # rather than the twenty-third field of a dataclass.
        measured.pop("progress")
        measured.pop("done")
        step = measured.pop("step", None)
        line = ", ".join(f"{k} {_reading(v)}" for k, v in measured.items())
        print("  " + (f"{step}: {line}" if step else line), flush=True)
        return

    print(
        f"  step {progress.step}/{progress.total_steps} "
        f"(epoch {progress.epoch + 1}/{progress.total_epochs})  "
        f"loss {progress.loss:.4f}  lr {progress.lr:.2e}  "
        f"{progress.elapsed_secs:.0f}s",
        flush=True,
    )
