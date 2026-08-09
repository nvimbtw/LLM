"""Which sub-block is actually doing the work.

The loss curve says a model learned something; it does not say *where* the
learning went. This replaces one sub-block at a time with a no-op and measures
how much the output degrades, which is the cheapest way to see that (say) an
under-deep model has dumped everything into its first FFN and is carrying three
more layers that contribute almost nothing.

Ablating means contributing zero to the residual stream, not deleting the
module: every block here is `x = x + sublayer(norm(x))`, so a sublayer that
returns zeros leaves the stream untouched and the rest of the network intact.
That is the closest thing to "what would this model be without this piece"
that does not require retraining.

One reading rule, and it is load-bearing. Ablation deltas are only interpretable
relative to how much the model knows: a baseline cross-entropy near ln(vocab) is
a model still at chance, and every sub-block will look equally unimportant
because none of them are doing anything yet. `learned_nats` is reported next to
the deltas so that case is visible rather than inferred.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from llm.model.transformer import Transformer


@dataclass
class AblationReport:
    model_name: str
    layers: int
    window_tokens: int
    intact_ce: float
    # ln(vocab) — the cross-entropy of a uniform guess, and the ceiling this
    # model is measured down from.
    chance_ce: float
    # How many nats below chance the intact model sits. Near zero means the
    # ablation table below is noise.
    learned_nats: float
    units: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(vars(self))


class _Silent(torch.nn.Module):
    """Stands in for a sublayer and contributes nothing to the residual."""

    def __init__(self, attention: bool):
        super().__init__()
        self.attention = attention

    def forward(self, x, *args, **kwargs):
        zeros = torch.zeros_like(x)
        # Attention returns (output, cache); the feed-forward returns a bare
        # tensor. The stand-in has to match or the block's unpacking fails.
        return (zeros, kwargs.get("cache", args[2] if len(args) > 2 else None)) \
            if self.attention else zeros


@contextmanager
def _silenced(block, attribute: str):
    original = getattr(block, attribute)
    setattr(block, attribute, _Silent(attribute == "attn"))
    try:
        yield
    finally:
        setattr(block, attribute, original)


def _cross_entropy(logits: torch.Tensor, ids: torch.Tensor) -> float:
    """Next-token cross-entropy over the window, in nats."""
    return float(
        F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
        )
    )


def ablate(
    model: Transformer,
    window: torch.Tensor,
    model_name: str = "",
) -> AblationReport:
    """Silence each sublayer in turn and measure the damage.

    `window` is [1, seq] token ids — one real context, not a batch: the question
    is which unit matters for this model, and averaging over contexts costs a
    forward pass each to sharpen a number that is already unambiguous.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            intact_logits, _ = model(window)
            intact = _cross_entropy(intact_logits, window)
            reference = intact_logits.argmax(-1)

            units = []
            for index, block in enumerate(model.blocks):
                for label, attribute in (("attention", "attn"), ("ffn", "ff")):
                    with _silenced(block, attribute), torch.no_grad():
                        logits, _ = model(window)

                    # KL(ablated || intact), per query position: how much
                    # probability mass the ablation moved, not just whether it
                    # moved the argmax.
                    kl = float(
                        F.kl_div(
                            F.log_softmax(intact_logits, dim=-1),
                            F.log_softmax(logits, dim=-1),
                            log_target=True,
                            reduction="sum",
                        )
                        / (intact_logits.shape[0] * intact_logits.shape[1])
                    )
                    ce = _cross_entropy(logits, window)
                    units.append({
                        "unit": f"layer{index}_{label}",
                        "layer": index,
                        "kind": label,
                        "kl_per_query": round(kl, 4),
                        "cross_entropy": round(ce, 4),
                        "ce_delta": round(ce - intact, 4),
                        "top1_agreement": round(
                            float((logits.argmax(-1) == reference).float().mean()), 4
                        ),
                    })
    finally:
        model.train(was_training)

    chance = math.log(model.vocab)
    units.sort(key=lambda u: -u["ce_delta"])
    return AblationReport(
        model_name=model_name,
        layers=len(model.blocks),
        window_tokens=int(window.shape[1]),
        intact_ce=round(intact, 4),
        chance_ce=round(chance, 4),
        learned_nats=round(chance - intact, 4),
        units=units,
    )
