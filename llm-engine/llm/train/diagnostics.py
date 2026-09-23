"""Per-report training telemetry.

Loss alone says a run is not working; it does not say why. These say where:
whether gradients are reaching the deep layers, whether the update is a sensible
fraction of the weight it is applied to, whether attention has collapsed onto one
token, whether the embedding table is folding in on itself.

Everything here is measured on a single step, once per report interval, so the
cost sits next to a forward pass that already happened rather than on every step.

Those questions are the *pretraining* ones. A fine-tune is asked something else —
is it forgetting the prose it came from, is it drifting on the prompt it is only
supposed to condition on, which conversations are actually costing it — and
`Probes` plus the fine-tuning fields on `Diagnostics` are the answers. They ride
the pass `observe` already runs, and they are absent on a pretrain run rather
than zero.

Ported from backend/crates/llm/src/training/diagnostics.rs. The gradient and
optimizer plumbing that file needs — reading raw device buffers, reconstructing
Adam moments — is `p.grad` and a parameter snapshot here.
"""

from dataclasses import dataclass, field
from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F

from llm.model.transformer import Transformer

# How many of the most-frequent tokens the embedding metrics track. Fixed and
# computed once per run, so the measurement follows the same tokens over training
# rather than drifting with whatever is common in the current batch.
ANISOTROPY_TOKENS = 32

# Tokens per block when counting corpus frequencies. Bounds the int64 temporary
# `np.bincount` makes internally: 8M tokens is a 64 MiB cast, against the 3.1 GiB
# a whole 390M-token corpus would need in one pass.
_COUNT_BLOCK = 1 << 23

# How much of a corpus the fallback histogram reads when the build did not record
# one: this many evenly spaced blocks of `_COUNT_BLOCK` tokens.
#
# Blocks, not a stride. `corpus[::16]` looks like it reads a sixteenth and reads
# all of it — tokens are 2 bytes and a page is 4096, so every 16th token still
# touches every page, which is the entire cost being avoided. Contiguous blocks
# spread across the file skip pages, and the mapping stays that size resident.
#
# Spread across the whole corpus rather than taken from the front, because
# `corpus.bin` is laid out in dataset and stage order: measured on the 4.17B-token
# build, the first 200M tokens contain only 16 of the true top 32 tokens, while
# 32 blocks spread across the file reproduce all 32.
_SAMPLE_BLOCKS = 32

ACT_HIST_BINS = 24
ACT_HIST_RANGE = 4.0

# How many rows of the training batch the recording forward runs over.
#
# The attention readout is a mean over heads and query rows, and one row already
# averages `ctx_window` of them per head — so the batch dimension buys precision
# the chart cannot show. It costs a great deal: the recorded attention matrix is
# [rows, heads, seq, seq] in fp32 (softmax is on autocast's fp32 list), so every
# row is another `heads * seq^2 * 4` bytes per layer, and seq^2 is what makes a
# longer context unaffordable rather than merely expensive.
#
# Two rather than one because the second row is nearly free now that the pass is
# no_grad, and it measurably tightens the reading: against the eight-row version
# this replaced, the largest per-head deviation at 512d/8h/6L/ctx512 was 0.032
# at one row and 0.019 at two, on a 0-1 scale. Raise it if a future reading needs
# to be tighter still — the cost is linear in it.
DIAGNOSTIC_ROWS = 2


def diagnostic_rows(batch_size: int) -> int:
    """How many rows of a batch of this size the recording pass covers.

    A batch smaller than `DIAGNOSTIC_ROWS` is legal — `bench.py` runs one — and
    the caller assembling a window's per-token readings has to slice the same
    rows the forward did, or the two arrays are different lengths.
    """
    return min(DIAGNOSTIC_ROWS, batch_size)


# Slices of the context window the per-position loss is bucketed into. Fixed
# rather than derived from `ctx_window`, so the series is the same width for
# every run and a consumer can stack reports into one heatmap.
POSITION_BUCKETS = 16

# Most conversations one diagnosed step reports, hardest first. The window covers
# whatever conversations fall in `DIAGNOSTIC_ROWS x ctx_window` tokens, and a
# corpus of short chats puts far more than a chart can carry there — this is the
# wire budget, not a property of the data.
CONVERSATION_SAMPLES = 16


@dataclass
class Probes:
    """Fixed batches a fine-tune is measured against, built once before the run.

    All three are optional and each is independently absent: a base model whose
    pretrain corpus is no longer on disk still gets the mask splits, and a run
    that did not ask for the KL anchor never pays for the second model.

    Built by `llm.orchestrate.sft`, which is the layer that knows what a base
    model is and where held-out chat lives. This module only knows they are token
    arrays it may run a forward over.
    """

    # [rows, ctx_window + 1] of the base model's own pretraining text. Held out
    # from *this* run — an SFT run never trains on prose — so a rise in its loss
    # is the fine-tune forgetting, which is the number that says when to stop.
    pretrain: np.ndarray | None = None
    # [rows, ctx_window] of held-out conversations, for the KL anchor.
    chat: np.ndarray | None = None
    # The unmodified base model, resident alongside the one being trained. This
    # is the field that doubles a run's model memory, which is why it is opt-in.
    reference: Transformer | None = None


@dataclass
class Diagnostics:
    # The global step this sample was taken at. Every time-series chart plots
    # against it, so a sample without one has no x position and draws nothing —
    # which looks exactly like "diagnostics are not arriving" even while the
    # latest-sample visuals (the histogram, the embedding projection) render.
    step: int = 0
    grad_global: float = 0.0
    grad_layers: list[float] = field(default_factory=list)
    grad_embedding: float = 0.0
    # Tied embeddings: the output projection *is* the embedding table, so both
    # paths accumulate into one gradient and this is the same number. Reported
    # separately because the consumer plots them as two series, and because a
    # future untied model would make them genuinely differ.
    grad_unembed: float = 0.0
    # Per-layer AdamW update:weight ratio ||dW|| / ||W||.
    update_ratio_layers: list[float] = field(default_factory=list)
    # RMS of the residual stream entering each block.
    act_rms_layers: list[float] = field(default_factory=list)
    # Per-head attention entropy, normalized to [0,1], grouped by layer.
    attn_entropy: list[list[float]] = field(default_factory=list)
    # Per-head learned softmax temperature, grouped by layer. Empty unless the run
    # enabled it. Entropy alone cannot separate a head that has nothing to say from
    # one whose temperature is holding it flat, so the two are read together: a
    # diffuse head whose temperature is climbing is still specializing.
    attn_temperature: list[list[float]] = field(default_factory=list)
    # Pairwise cosine similarity among the most-frequent tokens' embeddings, after
    # subtracting the table's mean row. A high mean means the table is collapsing
    # toward one direction.
    emb_cos_mean: float = 0.0
    emb_cos_p5: float = 0.0
    emb_cos_p95: float = 0.0
    act_hist: list[float] = field(default_factory=list)
    emb_pca: list[list[float]] = field(default_factory=list)
    emb_pca3: list[list[float]] = field(default_factory=list)

    # ── Fine-tuning ─────────────────────────────────────────────────────────
    # None means "not measured", and `as_dict` drops it rather than sending a
    # zero — same convention, and the same reason, as `Tokenize`: a zero here
    # reads as a real measurement of nothing, and a pretrain run would draw a
    # flat line along the bottom of four charts that do not apply to it.
    #
    # Cross-entropy on the base model's pretraining text, which this run never
    # trains on. Flat means the fine-tune is holding its language; climbing is
    # catastrophic forgetting, in the unit the loss chart is already in.
    pretrain_loss: float | None = None
    # The same window's loss split by the SFT mask: `masked` is the assistant
    # tokens the run takes gradient on, `unmasked` the prompt it is only
    # conditioning on. The second is not optimized, so it drifting is the model
    # spending capacity on text nobody asked it to model.
    loss_masked: float | None = None
    loss_unmasked: float | None = None
    # Mean trainable-token loss per `POSITION_BUCKETS` slice of the context
    # window, front to back. A bucket with nothing trainable in it is None.
    loss_by_position: list[float | None] | None = None
    # The hardest conversations in the diagnosed window, loss-descending, with
    # the corpus offset that identifies each — the same conversation is the same
    # offset for the whole run, so a consumer can accumulate a ranking.
    conversation_loss: list[float] | None = None
    conversation_start: list[int] | None = None
    # Mean KL(base || tuned) in nats per token over a fixed held-out chat batch:
    # how much of what the base model believed the fine-tune has moved away from.
    base_kl: float | None = None

    def as_dict(self) -> dict:
        """The frame, with anything unmeasured absent rather than zero.

        A renaming layer between here and the chart is what makes these drift
        apart (see `progress.Progress`), and a *silently defaulted* one is the
        same failure wearing a number: the panel cannot tell a pretrain run from
        a fine-tune whose base corpus was unreachable if both send 0.0.
        """
        return {k: v for k, v in vars(self).items() if v is not None}


def histogram(corpus: np.ndarray, width: int) -> np.ndarray:
    """How often each of the `width` token ids occurs in `corpus`.

    Counted with `bincount` over blocks rather than `Counter(corpus.tolist())`.
    `tolist()` boxes every token into a Python int — 36 bytes against the 2 it
    occupies as uint16 — so on a real corpus it asks for ~18x the corpus size in
    RAM (390M tokens wanted ~13 GiB) and the OOM killer arrives before the first
    training step. Blocking the bincount also bounds its internal intp cast,
    which would otherwise copy the whole corpus to int64.

    `width` is the vocabulary, passed in rather than derived from `corpus.max()`.
    That max is a second full pass over the mapping — the same 8.3 GB of page
    faults again, for a number the caller already knows.
    """
    flat = corpus.reshape(-1)
    counts = np.zeros(width, dtype=np.int64)
    for start in range(0, flat.size, _COUNT_BLOCK):
        block = flat[start : start + _COUNT_BLOCK]
        counts += np.bincount(block, minlength=width)[:width]
    return counts


def sampled_histogram(corpus: np.ndarray, width: int) -> np.ndarray:
    """`histogram` over `_SAMPLE_BLOCKS` blocks spread across the corpus.

    The fallback for a corpus packed before the build recorded its own histogram.
    It reads a bounded slice — 32 blocks is 512 MiB touched whatever the corpus
    weighs — where the exact count reads all of it: measured at 64.1 s and 8.0 GB
    resident on the 4.17B-token build, before the first training step, for 32
    token ids.
    """
    flat = corpus.reshape(-1)
    if flat.size <= _SAMPLE_BLOCKS * _COUNT_BLOCK:
        return histogram(flat, width)

    counts = np.zeros(width, dtype=np.int64)
    # The last block ends at the corpus's end rather than starting there, so the
    # tail is sampled like everything else instead of half a block of it.
    stride = (flat.size - _COUNT_BLOCK) // (_SAMPLE_BLOCKS - 1)
    for i in range(_SAMPLE_BLOCKS):
        start = i * stride
        counts += np.bincount(flat[start : start + _COUNT_BLOCK], minlength=width)[:width]
    return counts


def dead_ids(counts) -> int:
    """How many token ids never occur in the corpus `counts` was taken over.

    The zero end of the same distribution `histogram` computes for the frequent
    end, and it is a cost rather than a curiosity: the embedding table is tied
    (see `Diagnostics.grad_unembed`), so an id that never appears is a row that
    is read in every forward pass, carried in every checkpoint, and trained by
    nothing.

    Only meaningful against an exact histogram. A sampled one has zeros that are
    merely unseen — see `sampled_histogram`.
    """
    return int(np.count_nonzero(np.asarray(counts, dtype=np.int64) == 0))


def rank_tokens(counts, count: int = ANISOTROPY_TOKENS) -> list[int]:
    """The `count` most-frequent token ids, high to low, ties broken by id.

    At most one entry per token id survives — the vocabulary, not the corpus — so
    the exact sort is cheap however long the corpus was.
    """
    counts = np.asarray(counts, dtype=np.int64)
    present = np.flatnonzero(counts)
    ranked = sorted(present.tolist(), key=lambda t: (-int(counts[t]), int(t)))
    return ranked[:count]


def frequent_tokens(
    corpus: np.ndarray,
    count: int = ANISOTROPY_TOKENS,
    width: int | None = None,
) -> list[int]:
    """The most-frequent token ids, from a bounded sample of the corpus.

    Kept for the ad-hoc tools under `explore/`, which have a corpus and no
    sidecar. A training run does not call this: `pack` records the histogram
    while the tokens are already passing through it, and `Collector` ranks that.
    """
    flat = corpus.reshape(-1)
    if flat.size == 0:
        return []
    width = width or int(flat.max()) + 1
    return rank_tokens(sampled_histogram(flat, width), count)


def snapshot_weights(model: Transformer) -> list[torch.Tensor]:
    """CPU copies of per-block parameters, for the update:weight ratio.

    Taken before the optimizer steps and diffed after. Reconstructing the step
    from Adam's moments is the alternative and gets the bias correction, epsilon
    and weight decay subtly wrong; measuring the actual change cannot. Copies
    must not compete with AdamW for VRAM. Transfers happen only at the diagnostic
    cadence, never on ordinary training steps.
    """
    return [
        torch.cat([p.detach().cpu().reshape(-1) for p in block.parameters()])
        for block in model.blocks
    ]


def gradient_norms(model: Transformer) -> tuple[float, list[float], float]:
    """Global, per-block, and embedding gradient L2 norms.

    Must be called *before* the optimizer zeroes the gradients.
    """

    def norm(params) -> float:
        squares = [p.grad.detach().pow(2).sum() for p in params if p.grad is not None]
        return float(torch.sqrt(torch.stack(squares).sum())) if squares else 0.0

    layers = [norm(block.parameters()) for block in model.blocks]
    embedding = norm([model.embedding.weight])
    total = float(np.sqrt(sum(n**2 for n in layers) + embedding**2))
    return total, layers, embedding


def _entropy(weights: torch.Tensor) -> list[float]:
    """Mean per-head attention entropy, normalized to [0,1].

    Each query row is normalized by the entropy of a uniform distribution over
    the keys it may actually see — under a causal mask that is the row index plus
    one, so early rows are not scored against a spread they cannot have. A row
    with a single key has no spread to measure and contributes 0.
    """
    weights = weights.float()
    _, heads, seq, keys = weights.shape

    safe = weights.clamp_min(1e-9)
    row_entropy = -(weights * safe.log()).sum(-1)  # [batch, heads, seq]

    visible = torch.arange(keys - seq + 1, keys + 1, device=weights.device).float()
    uniform = visible.log().clamp_min(1e-9)  # entropy of uniform over the visible keys
    normalized = (row_entropy / uniform).clamp(0.0, 1.0)
    # Rows with one visible key: no spread is representable, so report 0 rather
    # than the 0/0 the division above would otherwise produce.
    normalized = torch.where(visible <= 1, torch.zeros_like(normalized), normalized)

    return [float(normalized[:, h].mean()) for h in range(heads)]


def _embedding_metrics(model: Transformer, tokens: list[int]) -> dict:
    """Anisotropy and PCA over the tracked tokens' current embeddings."""
    if len(tokens) < 2:
        return {}

    with torch.no_grad():
        weight = model.embedding.weight
        rows = weight[torch.tensor(tokens, device=weight.device)]
        # Centred, because uncentred this measured almost entirely a quantity the
        # architecture is invariant to. Tied models grow a large shared offset —
        # every row is `mu + delta_i` for one common `mu` — and that offset cannot
        # affect the output distribution: the logit for token i is `h·(mu + delta_i)`,
        # and `h·mu` is the same number for every token, a constant shift across the
        # row that softmax is exactly invariant to. On the input side the same `mu`
        # lands at every position and the first Pre-LN subtracts its mean back out.
        # An uncentred reading climbs all run precisely because nothing in the loss
        # opposes a component that never reaches the loss.
        #
        # The mean row is taken over the whole table, not just the chosen tokens':
        # the offset is a property of the table, and estimating it from 32 rows would
        # subtract part of the very spread this is trying to measure.
        mu = weight.float().mean(dim=0, keepdim=True)
        unit = torch.nn.functional.normalize(rows.float() - mu, dim=1)
        cosine = unit @ unit.T
        # Self-similarity is 1 by construction and would drag the mean up with
        # a number that says nothing about the geometry.
        off_diagonal = cosine[~torch.eye(len(tokens), dtype=torch.bool, device=cosine.device)]

        centered = rows.float() - rows.float().mean(dim=0, keepdim=True)
        # Full SVD rather than pca_lowrank: 32 rows is tiny, and the low-rank
        # approximation is randomized, which would make the scatter plot jitter
        # between reports for reasons that are not the model changing.
        _, _, components = torch.linalg.svd(centered, full_matrices=False)
        projected = centered @ components[:3].T

    percentiles = np.percentile(off_diagonal.cpu().numpy(), [5, 95])
    return {
        "emb_cos_mean": float(off_diagonal.mean()),
        "emb_cos_p5": float(percentiles[0]),
        "emb_cos_p95": float(percentiles[1]),
        "emb_pca": projected[:, :2].cpu().tolist(),
        "emb_pca3": projected[:, :3].cpu().tolist(),
    }


def _activation_histogram(residual: torch.Tensor) -> list[float]:
    """Z-scored distribution of the deepest block's residual stream.

    One per report, so a consumer can stack them into a distribution-over-time
    heatmap and see the activations drift.
    """
    flat = residual.float().reshape(-1)
    centered = (flat - flat.mean()) / flat.std().clamp_min(1e-6)
    counts = torch.histc(
        centered.clamp(-ACT_HIST_RANGE, ACT_HIST_RANGE),
        bins=ACT_HIST_BINS,
        min=-ACT_HIST_RANGE,
        max=ACT_HIST_RANGE,
    )
    return (counts / counts.sum().clamp_min(1)).cpu().tolist()


def _token_losses(logits: torch.Tensor, targets: torch.Tensor, vocab: int) -> torch.Tensor:
    """Per-position cross-entropy, [rows, seq].

    The reduced form is what `loop._loss` computes for the training step; this is
    the same arithmetic left unreduced, so every fine-tuning split below is a
    different weighting of one softmax rather than another forward pass.
    """
    per_token = F.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
    )
    return per_token.reshape(targets.shape)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float | None:
    """None when nothing is selected — a mean of no tokens is not zero."""
    total = float(weights.sum())
    if total <= 0.0:
        return None
    return float((values * weights).sum() / total)


def _bucket_means(values: torch.Tensor, weights: torch.Tensor) -> list[float | None]:
    """Weighted mean per `POSITION_BUCKETS` slice of the last axis.

    Bucketed rather than sent per position: the wire carries a fixed 16 numbers
    whatever `ctx_window` a run trains at, and a per-position series would be the
    one thing here that scales with the context.
    """
    seq = values.shape[-1]
    bucket = torch.arange(seq, device=values.device) * POSITION_BUCKETS // seq
    index = bucket.expand(values.shape).reshape(-1)

    weighted = torch.zeros(POSITION_BUCKETS, device=values.device, dtype=torch.float32)
    counts = torch.zeros_like(weighted)
    weighted.index_add_(0, index, (values * weights).reshape(-1).float())
    counts.index_add_(0, index, weights.reshape(-1).float())

    totals = counts.tolist()
    means = (weighted / counts.clamp_min(1e-12)).tolist()
    return [mean if total > 0 else None for mean, total in zip(means, totals)]


def _conversation_losses(
    values: torch.Tensor, weights: torch.Tensor, documents: np.ndarray
) -> dict:
    """The hardest conversations in the window, by mean trainable-token loss.

    Aggregated on the host: the window is `DIAGNOSTIC_ROWS x ctx_window` floats,
    the step is already synchronizing to read every other metric off the device,
    and a scatter-add keyed by corpus offset would need the key table uploaded
    per step to save a copy of four kilobytes.
    """
    loss = (values * weights).reshape(-1).float().cpu().numpy()
    weight = weights.reshape(-1).float().cpu().numpy()
    # Both sides slice `diagnostic_rows(batch_size)` rows, so a mismatch is a
    # caller that walked a different window than it scored — silently attributing
    # every loss to the wrong conversation. Reported as unmeasured instead.
    if documents.shape[0] != loss.shape[0]:
        return {}

    keys, inverse = np.unique(documents, return_inverse=True)
    sums = np.bincount(inverse, weights=loss, minlength=len(keys))
    counts = np.bincount(inverse, weights=weight, minlength=len(keys))

    # A separator (-1) belongs to no conversation, and a conversation whose share
    # of this window is all prompt has no trainable token to average.
    keep = np.flatnonzero((keys >= 0) & (counts > 0))
    if keep.size == 0:
        return {}

    means = sums[keep] / counts[keep]
    hardest = keep[np.argsort(-means)][:CONVERSATION_SAMPLES]
    return {
        "conversation_loss": (sums[hardest] / counts[hardest]).tolist(),
        "conversation_start": keys[hardest].tolist(),
    }


def _optional_measurement(method):
    """A diagnostic OOM must not discard an otherwise valid training run.

    Disable subsequent samples, say why, and release captures before training
    continues. Only CUDA OOM is recoverable here; other errors remain errors.
    """
    @wraps(method)
    def measure(self, *args, **kwargs):
        if self.disabled:
            return None
        try:
            return method(self, *args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            self.disabled = True
            self._clear()
            self.snapshot = None
            self.measured = {}
            self._entropy = []
        # Outside the except: the failed frame must no longer hold its tensors.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[memory] diagnostics disabled after CUDA OOM; training continues without samples")
        return None

    return measure


class Collector:
    """Turns diagnostics on for one forward, then reads them back off.

    Recording changes the attention path (an explicit softmax instead of the
    fused kernel) and holds activations alive, so it is scoped tightly: enabled
    for exactly the forward pass `observe` runs and disabled straight after.

    That forward is **not** the training step. See `observe` — it is a separate
    no-grad pass over a slice of the batch, and the separation is what keeps a
    diagnosed step from setting the memory ceiling for the whole run.

    Constructing one is not free — the tracked tokens have to come from
    somewhere — so `loop.run` builds it on the first measured step rather than at
    the top of the run. See `_LazyCollector` there.
    """

    def __init__(
        self,
        model: Transformer,
        corpus: np.ndarray,
        counts=None,
        probes: Probes | None = None,
    ):
        self.model = model
        self.disabled = False
        # `counts` is the histogram the build recorded beside the corpus. Absent
        # only for a corpus packed before it was recorded, and then the corpus is
        # sampled rather than read: this is the step that spent 64 s and pulled
        # 8.3 GB resident before step 1 of every run.
        self.tokens = (
            rank_tokens(counts) if counts is not None else frequent_tokens(corpus, width=model.vocab)
        )
        self.snapshot: list[torch.Tensor] | None = None
        self.probes = probes or Probes()
        # Uploaded once and kept: a few thousand ids against a model already
        # resident, and re-uploading them per diagnosed step would be a host copy
        # inside the one window this class exists to keep cheap.
        device = model.embedding.weight.device
        self._pretrain = _as_batch(self.probes.pretrain, device)
        self._chat = _as_batch(self.probes.chat, device)
        self.measured: dict = {}
        # Per-head attention entropy, one entry per layer, filled by the hook
        # `__enter__` installs. Here as well so `after_step` on a collector that
        # never recorded reads an empty list rather than raising.
        self._entropy: list[list[float]] = []
        self._handles: list = []

    def _set_recording(self, on: bool) -> None:
        for block in self.model.blocks:
            block.record_activations = on
            block.attn.record_attention = on

    def _clear(self) -> None:
        for block in self.model.blocks:
            block.last_input = None
            block.attn.last_attention = None

    def _reduce_attention(self, module, _args, _output) -> None:
        """Turn one layer's attention matrix into its per-head entropy, and drop it.

        A forward hook, so this runs as each layer finishes rather than after the
        whole stack has. That is the difference between one `[rows, heads, seq,
        seq]` fp32 matrix being alive and all `layers` of them, and at a long
        context it is most of what a diagnosed step costs: the matrix is
        quadratic in sequence length, so the retention measured 302 MiB at
        ctx512, 1224 MiB at ctx1024 and 4832 MiB at ctx2048 on an 18-layer model
        — held across the whole training step, because `after_step` is where the
        walk used to be and that runs after the optimizer.

        The reduction stays here in `Collector` rather than moving into
        `Attention`, which is the boundary
        `docs/decisions/gpu-memory-is-handed-back.md` drew and this does not
        cross: the module still hands back raw weights and still knows nothing
        about what is measured off them.
        """
        if module.last_attention is not None:
            self._entropy.append(_entropy(module.last_attention))
            module.last_attention = None

    def __enter__(self) -> "Collector":
        # Cleared on the way IN, not out. The captures are read by `after_step`,
        # which necessarily runs after the optimizer has stepped and so after
        # this context has closed — dropping them at exit left every activation
        # metric silently empty while the gradient ones still worked.
        #
        # The attention matrices are the exception, and they are why `_entropy`
        # is a list built here: they are far too large to hold that long, so they
        # are reduced to their per-head entropy by `_reduce_attention` as the
        # forward walks past and never survive the pass that made them.
        self._clear()
        self._entropy = []
        self._handles = [
            block.attn.register_forward_hook(self._reduce_attention)
            for block in self.model.blocks
        ]
        self._set_recording(True)
        return self

    def __exit__(self, *_) -> None:
        self._set_recording(False)
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @_optional_measurement
    def observe(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        documents: np.ndarray | None = None,
    ) -> None:
        """Record activations and attention for `inputs`, off the training path.

        A no-grad forward over the first `DIAGNOSTIC_ROWS` rows, run immediately
        before the training step it describes — same weights, same batch, so it
        reports the step it is filed under.

        `targets` and `weights` turn the same logits into the fine-tuning splits
        — masked against unmasked, by position, by conversation — which is why
        they are read here and not in a pass of their own: a second forward would
        roughly double the cost of a diagnosed step for arithmetic that is a
        weighting of a softmax this one already computed. `documents` is the
        corpus offset per position, from `plan.EpochStream.document_starts`.
        Passing none of the three is a pretrain run, and then nothing below the
        forward runs at all.

        It used to be the training forward itself, with recording flipped on. That
        read the same numbers and cost far more than it looks:

        - **Autograd retained every attention matrix.** The explicit softmax
          materializes `[batch, heads, seq, seq]` in fp32 where the fused kernel
          materializes nothing, and each one was kept alive for a backward pass
          that did not need it. No-grad frees them as the forward walks past.
        - **It ran the whole batch**, for a mean over rows. See `DIAGNOSTIC_ROWS`.
        - **It forced the uncompiled callable**, which put cross-entropy back in
          eager and with it the second fp32 copy of the logits that `loop.step_fn`
          exists to avoid.

        Measured at 512d/8h/6L/ctx512/bs8, the diagnosed step's peak fell from
        1704 to 1110 MiB against a 1003 MiB ordinary step; at ctx1024, from 4169
        to 2020 against 1612. The old peak was not transient — nothing hands a
        CUDA pool back, so the first diagnosed step raised the run's floor for
        good. `loop.release_device` is the other half of that story.

        A quieter consequence: the training step is now the *same* step on every
        iteration. A diagnosed step used to train through the explicit-softmax
        path, so one step in every `interval * diag_interval` differed
        numerically from the rest — on exactly the steps being measured.
        """
        self.measured = {}
        rows = diagnostic_rows(inputs.shape[0])
        with torch.no_grad():
            with self:
                logits, _ = self.model(inputs[:rows])
                if targets is not None and weights is not None:
                    self.measured.update(
                        self._split_losses(
                            logits, targets[:rows], weights[:rows], documents
                        )
                    )
            # Outside the recording context on purpose. These are further
            # forwards, and recorded each would materialize a
            # `[rows, heads, seq, seq]` fp32 attention matrix per layer for a
            # reading nothing takes off them — the exact cost `Collector` exists
            # to keep scoped.
            del logits
            self.measured.update(self._probe_losses())

    def _split_losses(self, logits, targets, weights, documents) -> dict:
        """The SFT mask's readings, from the logits the recording forward made."""
        per_token = _token_losses(logits, targets, self.model.vocab)
        trained = weights.float()
        # The prompt: everything the mask excludes. That includes the handful of
        # separators joining conversations in the window, which is a rounding
        # error against a prompt region and not worth a second mask to remove.
        conditioned = 1.0 - trained

        measured = {
            "loss_masked": _weighted_mean(per_token, trained),
            "loss_unmasked": _weighted_mean(per_token, conditioned),
            "loss_by_position": _bucket_means(per_token, trained),
        }
        if documents is not None:
            measured.update(_conversation_losses(per_token, trained, documents))
        return measured

    def _probe_losses(self) -> dict:
        """Loss on the base model's prose, and KL from the base model itself."""
        measured: dict = {}
        if self._pretrain is not None:
            inputs, targets = self._pretrain[:, :-1], self._pretrain[:, 1:]
            logits, _ = self.model(inputs)
            measured["pretrain_loss"] = float(
                F.cross_entropy(logits.reshape(-1, self.model.vocab), targets.reshape(-1))
            )
            del logits
        if self._chat is not None and self.probes.reference is not None:
            measured["base_kl"] = self._reference_kl()
        return measured

    def _reference_kl(self) -> float:
        """Mean KL(base || tuned) in nats per token over the held-out chat batch.

        Row at a time. Both log-softmaxes are `[seq, vocab]` in fp32 and the whole
        batch at once is that twice over — the one place in this file where the
        vocabulary, not the sequence, is what would set the peak.

        The direction is base-first: it weights the divergence by what the *base*
        believed, so mass the fine-tune has abandoned counts and mass it has
        newly invented does not. That is the quantity a KL anchor penalizes, and
        reading it is the point of measuring it.
        """
        total = 0.0
        for row in self._chat:
            batch = row.unsqueeze(0)
            tuned = self.model(batch)[0].float().log_softmax(-1)
            base = self.probes.reference(batch)[0].float().log_softmax(-1)
            total += float((base.exp() * (base - tuned)).sum(-1).mean())
        return total / len(self._chat)

    @_optional_measurement
    def before_step(self) -> tuple[float, list[float], float] | None:
        """Read the gradients and snapshot the weights, between backward and step."""
        self.snapshot = snapshot_weights(self.model)
        return gradient_norms(self.model)

    @_optional_measurement
    def after_step(self, grad: tuple[float, list[float], float], step: int = 0) -> Diagnostics | None:
        """Assemble the report, after the optimizer has stepped."""
        total, layers, embedding = grad

        ratios: list[float] = []
        if self.snapshot is not None:
            for before, block in zip(self.snapshot, self.model.blocks):
                after = torch.cat([p.detach().cpu().reshape(-1) for p in block.parameters()])
                weight_norm = after.norm().clamp_min(1e-12)
                ratios.append(float((after - before).norm() / weight_norm))

        activations = [b.last_input for b in self.model.blocks if b.last_input is not None]

        # Already reduced, layer by layer, by the hook `__enter__` installed —
        # nothing the size of an attention matrix survives to be walked here.
        entropy = self._entropy

        diagnostics = Diagnostics(
            step=step,
            grad_global=total,
            grad_layers=layers,
            grad_embedding=embedding,
            grad_unembed=embedding,
            update_ratio_layers=ratios,
            act_rms_layers=[float(a.float().pow(2).mean().sqrt()) for a in activations],
            attn_entropy=entropy,
            attn_temperature=[
                block.attn.log_temperature.detach().exp().float().cpu().tolist()
                for block in self.model.blocks
                if block.attn.log_temperature is not None
            ],
            act_hist=_activation_histogram(activations[-1]) if activations else [],
        )
        for key, value in _embedding_metrics(self.model, self.tokens).items():
            setattr(diagnostics, key, value)
        # `observe` measured these against the weights of the step being
        # reported; a key it left out stays None, which is how it reaches the
        # frame as absent rather than as a zero.
        for key, value in self.measured.items():
            setattr(diagnostics, key, value)

        # Released once read: an activation tensor per block is the largest thing
        # here, and holding it until the next report would double the resident
        # activation memory for no reason.
        self._clear()
        self.snapshot = None
        self.measured = {}
        return diagnostics


def _as_batch(probe: np.ndarray | None, device) -> torch.Tensor | None:
    if probe is None:
        return None
    return torch.from_numpy(np.ascontiguousarray(probe, dtype=np.int64)).to(device)
