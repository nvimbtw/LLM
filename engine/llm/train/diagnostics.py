"""Per-report training telemetry.

Loss alone says a run is not working; it does not say why. These say where:
whether gradients are reaching the deep layers, whether the update is a sensible
fraction of the weight it is applied to, whether attention has collapsed onto one
token, whether the embedding table is folding in on itself.

Everything here is measured on a single step, once per report interval, so the
cost sits next to a forward pass that already happened rather than on every step.

Ported from backend/crates/llm/src/training/diagnostics.rs. The gradient and
optimizer plumbing that file needs — reading raw device buffers, reconstructing
Adam moments — is `p.grad` and a parameter snapshot here.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from llm.model.transformer import Transformer

# How many of the most-frequent tokens the embedding metrics track. Fixed and
# computed once per run, so the measurement follows the same tokens over training
# rather than drifting with whatever is common in the current batch.
ANISOTROPY_TOKENS = 32

# Tokens per block when counting corpus frequencies. Bounds the int64 temporary
# `np.bincount` makes internally: 8M tokens is a 64 MiB cast, against the 3.1 GiB
# a whole 390M-token corpus would need in one pass.
_COUNT_BLOCK = 1 << 23

ACT_HIST_BINS = 24
ACT_HIST_RANGE = 4.0


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

    def as_dict(self) -> dict:
        return dict(vars(self))


def frequent_tokens(corpus: np.ndarray, count: int = ANISOTROPY_TOKENS) -> list[int]:
    """The `count` most-frequent token ids, high to low, ties broken by id.

    Computed once per run — the corpus is fixed — so the embedding metrics track
    the same tokens for the whole of training.

    Counted with `bincount` over blocks rather than `Counter(corpus.tolist())`.
    `tolist()` boxes every token into a Python int — 36 bytes against the 2 it
    occupies as uint16 — so on a real corpus it asks for ~18x the corpus size in
    RAM (390M tokens wanted ~13 GiB) and the OOM killer arrives before the first
    training step. Blocking the bincount also bounds its internal intp cast,
    which would otherwise copy the whole corpus to int64.
    """
    flat = corpus.reshape(-1)
    if flat.size == 0:
        return []

    width = int(flat.max()) + 1
    counts = np.zeros(width, dtype=np.int64)
    for start in range(0, flat.size, _COUNT_BLOCK):
        counts += np.bincount(flat[start : start + _COUNT_BLOCK], minlength=width)

    # At most `width` entries survive (the vocabulary, not the corpus), so the
    # exact ordering below is cheap: highest count first, ties broken by id.
    present = np.flatnonzero(counts)
    ranked = sorted(present.tolist(), key=lambda t: (-int(counts[t]), int(t)))
    return ranked[:count]


def snapshot_weights(model: Transformer) -> list[torch.Tensor]:
    """Flattened per-block parameters, for the update:weight ratio.

    Taken before the optimizer steps and diffed after. Reconstructing the step
    from Adam's moments is the alternative and gets the bias correction, epsilon
    and weight decay subtly wrong; measuring the actual change cannot.
    """
    return [
        torch.cat([p.detach().reshape(-1) for p in block.parameters()]).clone()
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


class Collector:
    """Turns diagnostics on for one step, then reads them back off.

    Recording changes the attention path (an explicit softmax instead of the
    fused kernel) and holds activations alive, so it is scoped tightly: enabled
    immediately before the measured step and disabled straight after.
    """

    def __init__(self, model: Transformer, corpus: np.ndarray):
        self.model = model
        self.tokens = frequent_tokens(corpus)
        self.snapshot: list[torch.Tensor] | None = None

    def _set_recording(self, on: bool) -> None:
        for block in self.model.blocks:
            block.record_activations = on
            block.attn.record_attention = on

    def _clear(self) -> None:
        for block in self.model.blocks:
            block.last_input = None
            block.attn.last_attention = None

    def __enter__(self) -> "Collector":
        # Cleared on the way IN, not out. The captures are read by `after_step`,
        # which necessarily runs after the optimizer has stepped and so after
        # this context has closed — dropping them at exit left every activation
        # metric silently empty while the gradient ones still worked.
        self._clear()
        self._set_recording(True)
        return self

    def __exit__(self, *_) -> None:
        self._set_recording(False)

    def before_step(self) -> tuple[float, list[float], float]:
        """Read the gradients and snapshot the weights, between backward and step."""
        self.snapshot = snapshot_weights(self.model)
        return gradient_norms(self.model)

    def after_step(self, grad: tuple[float, list[float], float], step: int = 0) -> Diagnostics:
        """Assemble the report, after the optimizer has stepped."""
        total, layers, embedding = grad

        ratios: list[float] = []
        if self.snapshot is not None:
            for before, block in zip(self.snapshot, self.model.blocks):
                after = torch.cat([p.detach().reshape(-1) for p in block.parameters()])
                weight_norm = after.norm().clamp_min(1e-12)
                ratios.append(float((after - before).norm() / weight_norm))

        activations = [b.last_input for b in self.model.blocks if b.last_input is not None]
        attention = [b.attn.last_attention for b in self.model.blocks]

        diagnostics = Diagnostics(
            step=step,
            grad_global=total,
            grad_layers=layers,
            grad_embedding=embedding,
            grad_unembed=embedding,
            update_ratio_layers=ratios,
            act_rms_layers=[float(a.float().pow(2).mean().sqrt()) for a in activations],
            attn_entropy=[_entropy(w) for w in attention if w is not None],
            attn_temperature=[
                block.attn.log_temperature.detach().exp().float().cpu().tolist()
                for block in self.model.blocks
                if block.attn.log_temperature is not None
            ],
            act_hist=_activation_histogram(activations[-1]) if activations else [],
        )
        for key, value in _embedding_metrics(self.model, self.tokens).items():
            setattr(diagnostics, key, value)

        # Released once read: an activation tensor per block is the largest thing
        # here, and holding it until the next report would double the resident
        # activation memory for no reason.
        self._clear()
        self.snapshot = None
        return diagnostics
