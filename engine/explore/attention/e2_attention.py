"""E2 — Attention anatomy: uniforms at init, specializers after training.

On real text from the packed corpus, capture the explicit attention weights of:
  * a randomly-initialised model of the same architecture as `diagfix`
  * the trained `diagfix` model
and derive for every (layer, head):
  * entropy (normalised to [0,1], same unit the engine's diagnostics report)
  * sink mass — probability on position 0 (a "read global state" head)
  * recency mass — probability on the last 2 visible tokens
  * mean attention distance in tokens (weighted by key position)
  * peak mass — mass on the single most-attended key

The init-vs-trained contrast is the whole point: does the cold-start graph
already contain the specialisations training will amplify?
"""

import json
import os
import sys
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ENGINE_ROOT))
os.environ.setdefault("DATA_DIR", str(ENGINE_ROOT / "data"))

import torch

from llm.corpus import bpe, pack
from llm.model.transformer import Transformer
from llm.train.diagnostics import _entropy


def emit(name, data):
    print(json.dumps({"experiment": name, **data},
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def tokenizer():
    return bpe.load()


def _attention(model, ids):
    """Forward one window with the recording attention path on."""
    for block in model.blocks:
        block.attn.record_attention = True
    x = torch.tensor([ids])
    with torch.no_grad():
        model(x)
    out = []
    for block in model.blocks:
        out.append(block.attn.last_attention.clone())
        block.attn.last_attention = None
    return out  # per layer: [1, heads, seq, seq]


def _describe(weights):
    """Per-head metrics over one layer."""
    w = weights[0].float()  # [heads, seq, seq]
    heads, seq, _ = w.shape
    visible = torch.arange(1, seq + 1).float()
    entropy = _entropy(weights)  # engine's own unit
    sink = w[:, :, 0].mean(dim=1)
    recent = w[:, :, -2:].sum(dim=-1).mean(dim=1)
    peak = w.max(dim=-1).values.mean(dim=1)
    # Mean attended distance: row i has keys [0..i], distance = i - j.
    j = torch.arange(seq).float()
    # (pos - key), causal parts >= 0
    pos_minus_key = j[None, :, None] - j[None, None, :]
    dist = (w * pos_minus_key.clamp(min=0)).sum(-1)
    mean_dist = dist.mean(dim=1)
    return {
        "entropy": entropy,
        "sink_mass": sink,
        "recent_mass": recent,
        "peak_mass": peak,
        "mean_distance": mean_dist,
    }


def window_ids():
    tok = tokenizer()
    corpus = pack.load_corpus()
    ids = corpus[2_800:2_800 + 256].tolist()
    # Clip at a document boundary roughly: fine for a probe.
    return ids


if __name__ == "__main__":
    from llm import storage
    from llm.model.checkpoint import load as ckpt_load

    ids = window_ids()
    init = Transformer(8192, 128, 4, 2, 4, 256)
    trained, _ = ckpt_load(storage.weights_path("diagfix"), "cpu")
    trained.eval()

    layers = {}
    for tag, model in (("init", init), ("trained", trained)):
        atts = _attention(model, ids)
        layers[tag] = []
        for li, attn in enumerate(atts):
            d = _describe(attn)
            layers[tag].append(d)

    out = {
        "window_tokens": len(ids),
        "layer_count": len(atts),
        "heads": 4,
        "per_layer": layers,
    }
    emit("e2_attention", out)