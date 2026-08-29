"""Progress reporting.

A callback rather than prints, so the orchestrator decides where frames go —
today a log line, later a WebSocket. The seam exists now because retrofitting it
means changing every signature between here and the route.
"""

from dataclasses import dataclass
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


def report_every(total_steps: int) -> int:
    """Steps between report frames for a run of this length."""
    return max(1, total_steps // TARGET_REPORTS)


def diagnose_every(total_reports: int) -> int:
    """Reports between diagnostics samples."""
    return max(1, total_reports // TARGET_DIAG_SAMPLES)


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


ProgressCallback = Callable[[Progress | Tokenize], None]


def _reading(value) -> str:
    """One measurement on a log line. A distribution is summarised by its width:
    the merge sequence is hundreds of entries and the frame it rides is where it
    is meant to be read."""
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
        measured.pop("progress")
        measured.pop("done")
        print("  " + ", ".join(f"{k} {_reading(v)}" for k, v in measured.items()), flush=True)
        return

    print(
        f"  step {progress.step}/{progress.total_steps} "
        f"(epoch {progress.epoch + 1}/{progress.total_epochs})  "
        f"loss {progress.loss:.4f}  lr {progress.lr:.2e}  "
        f"{progress.elapsed_secs:.0f}s",
        flush=True,
    )
