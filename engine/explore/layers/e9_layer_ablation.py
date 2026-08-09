"""E9 — Layer / residual ablation: how much does each block *block* contribute?

The E2 report said heads stay diffuse and the residual stream keeps growing. The
natural question is: does the trained model actually *use* both layers, and how
much of its behaviour sits in the attention sub-block vs the FFN?

We surgically remove one sub-block at a time — replace that block's attention
(or its FFN) with an identity at *inference* — and record:
  * the mean logit shift (KL divergence) versus the intact model on a real
    window, one value per source (query) position;
  * how the downstream loss (next-token cross-entropy vs. the reference tokens)
    moves when each sub-block is removed.

This is a runtime monkey-patch inside the probe only; engine source is untouched.

Run (from repo root):
    engine/.venv/bin/python engine/explore/layers/e9_layer_ablation.py \\
        > engine/explore/layers/e9_layer_ablation.json
"""

import json
import os
import sys
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ENGINE_ROOT))
os.environ.setdefault("DATA_DIR", str(ENGINE_ROOT / "data"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm import storage
from llm.corpus import bpe, pack
from llm.model.checkpoint import load as ckpt_load


def target_model() -> str:
    """The checkpoint to probe: `argv[1]`, else the model the report was written
    against. Passing a name is how this probe gets re-run after a real training
    run — the numbers in report.html came off a 45-step model and describe that
    model, not the architecture."""
    return sys.argv[1] if len(sys.argv) > 1 else "diagfix"


def emit(name, data):
    print(json.dumps({"experiment": name, **data},
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def nll(logits, target):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1])[:-1],
                           target.reshape(-1)[1:], reduction="mean")


class ZeroAttention(nn.Module):
    """Replaces an Attention module; returns a 0 output of the same shape."""

    def __init__(self, ref):
        super().__init__()
        self.dim = ref.wo.in_features

    def forward(self, x, cos, sin, cache=None):
        out = torch.zeros_like(x)
        return out, None


class ZeroFF(nn.Module):
    """Replaces a FeedForward; returns a 0 output of the same shape."""

    def forward(self, x):
        return torch.zeros_like(x)


def main():
    model_name = target_model()
    model, _ = ckpt_load(storage.weights_path(model_name), "cpu")
    model.eval()
    tok = bpe.load()
    corpus = pack.load_corpus()
    _, docs, _ = pack.load_segments()

    d = docs[0]
    window = torch.from_numpy(corpus[d.start : d.start + 64].astype("int64"))
    xs = window.unsqueeze(0)

    with torch.no_grad():
        ref_logits, _ = model(xs)
        ref_nll = float(nll(ref_logits, xs).item())

    results = {}
    for layer in range(len(model.blocks)):
        block = model.blocks[layer]
        cases = {"attention": ("attn", ZeroAttention(block.attn)),
                 "ffn": ("ff", ZeroFF())}
        for name, (attr, repl) in cases.items():
            original = getattr(block, attr)
            with torch.no_grad():
                setattr(block, attr, repl)
                try:
                    logits, _ = model(xs)
                except Exception:
                    logits = None
                setattr(block, attr, original)
            if logits is None:
                continue
            kl = float(
                F.kl_div(
                    F.log_softmax(ref_logits, dim=-1),
                    F.log_softmax(logits, dim=-1),
                    log_target=True,
                    reduction="sum",
                )
                / (ref_logits.shape[0] * ref_logits.shape[1])
            )
            n = float(nll(logits, xs))
            # Top-1 agreement: same argmax at each query position.
            agree = float((logits.argmax(-1) == ref_logits.argmax(-1)).float().mean())
            results[f"layer{layer}_{name}"] = {
                "kl_per_query": round(kl, 4),
                "cross_entropy": round(n, 4),
                "ce_delta": round(n - ref_nll, 4),
                "top1_agreement": round(agree, 4),
            }

    emit("e9_layer_ablation", {
        # Ablation deltas only mean anything relative to how much the model had
        # learned: a baseline near ln(vocab) is a model still at chance, and
        # "which sub-block matters" is not yet a question it can answer.
        "model": model_name,
        "layers": len(model.blocks),
        "window": {"doc": 0, "seq": 63, "ref_ce": round(ref_nll, 4)},
        "ablation": results,
        "baseline": {"intact_ce": round(ref_nll, 4)},
    })


if __name__ == "__main__":
    main()