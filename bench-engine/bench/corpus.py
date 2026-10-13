"""Held-out compression: the headline language metric."""

import math
from dataclasses import dataclass
from pathlib import Path


from bench.ngram import ByteNgram

BASELINE_ORDER = 5
# Cost of predicting nothing at all: a uniform byte.
UNMODELLED_BITS_PER_BYTE = 8.0

DEFAULT_MAX_HELDOUT_BYTES = 256 * 1024
DEFAULT_MAX_BASELINE_BYTES = 4 * 1024 * 1024


@dataclass
class CorpusSplit:
    """Reference prose, split into a slice the baseline is fitted on and a slice
    both it and the model are scored against."""

    train: str
    heldout: str


@dataclass
class CorpusReport:
    bits_per_byte: float
    bits_per_token: float
    perplexity: float
    tokens_scored: int
    baseline_bits_per_byte: float
    # How much better than the counting table, in [0,1]. NaN when there is no
    # usable baseline.
    advantage: float
    # Where the model sits between predicting nothing and the baseline's floor.
    floor_progress: float

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in vars(self).items()}


def load(
    directory: Path,
    max_train_bytes: int = DEFAULT_MAX_BASELINE_BYTES,
    max_heldout_bytes: int = DEFAULT_MAX_HELDOUT_BYTES,
) -> CorpusSplit:
    """Read reference `.txt` and split 90/10.

    Point this at text the model was **not** trained on: scoring a model on its
    own training data measures memorization, not language.
    """
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in sorted(directory.glob("*.txt"))
    )
    if not text.strip():
        return CorpusSplit(train="", heldout="")

    cut = int(len(text) * 0.9)
    return CorpusSplit(
        train=text[:cut][-max_train_bytes:],
        heldout=text[cut:][:max_heldout_bytes],
    )


def evaluate(raw: dict, split: CorpusSplit) -> CorpusReport | None:
    count = raw["count"]
    if not count:
        return None
    # Natural log -> bits.
    total_bits = -raw["total"] / math.log(2)
    bits_per_token = total_bits / count

    # The first token is context, not a prediction, so its bytes were never
    # scored — subtract them or the per-byte figure is optimistic by that much.
    unscored = raw["unscored_bytes"]
    bytes_scored = max(1, len(split.heldout.encode("utf-8")) - unscored)
    bits_per_byte = total_bits / bytes_scored

    baseline = ByteNgram.fit(split.train.encode("utf-8"), BASELINE_ORDER).bits_per_byte(
        split.heldout.encode("utf-8")
    )

    # With no training slice the baseline degenerates to the uniform byte
    # distribution, which any model beats trivially — report the comparison as
    # unavailable rather than as a flattering advantage.
    measurable = math.isfinite(baseline) and baseline > 0.0 and bool(split.train)
    advantage = _clamp(1.0 - bits_per_byte / baseline, 0.0, 1.0) if measurable else math.nan

    # Guarded against a floor at or above the unmodelled cost, which would leave
    # no span to place the model on. That means an unusable baseline, not a
    # perfect model.
    span = UNMODELLED_BITS_PER_BYTE - baseline if measurable else 0.0
    floor_progress = (
        _clamp((UNMODELLED_BITS_PER_BYTE - bits_per_byte) / span, 0.0, 1.0)
        if measurable and span > 0
        else math.nan
    )

    return CorpusReport(
        bits_per_byte=bits_per_byte,
        bits_per_token=bits_per_token,
        # The effective number of tokens the model is choosing between per step.
        perplexity=math.exp(bits_per_token * math.log(2)),
        tokens_scored=count,
        baseline_bits_per_byte=baseline,
        advantage=advantage,
        floor_progress=floor_progress,
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
