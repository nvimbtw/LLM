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


def log_progress(progress: Progress | Tokenize) -> None:
    """The default sink: one line per report."""
    if isinstance(progress, Tokenize):
        measured = {k: v for k, v in vars(progress).items() if v is not None}
        measured.pop("progress")
        measured.pop("done")
        print("  " + ", ".join(f"{k} {v:,}" for k, v in measured.items()), flush=True)
        return

    print(
        f"  step {progress.step}/{progress.total_steps} "
        f"(epoch {progress.epoch + 1}/{progress.total_epochs})  "
        f"loss {progress.loss:.4f}  lr {progress.lr:.2e}  "
        f"{progress.elapsed_secs:.0f}s",
        flush=True,
    )
