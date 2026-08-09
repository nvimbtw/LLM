"""E8 — Per-head attention temperature: what would a learnable temperature do?

The engine's attention uses a fixed 1/sqrt(head_dim) scale — one softmax
"temperature" shared by every head. The E2 report found training widens layer-1
heads (entropy ~0.94–0.96, peak mass ~0.05) and that no sharp copy head ever
forms. A common idea (see the report) is a *learnable per-head softmax
temperature* so a head can choose sharp or diffuse behaviour on its own.

This probe answers the "would that idea work" question on the frozen `diagfix`
model using only recorded attention weights, with the identity

    softmax(tau * scores) == softmax(tau * log(w)),   w = recorded attention weights

so no engine change or re-training is needed: for each (layer, head) we sweep a
temperature tau over the real window and report entropy + peak-mass sensitivity
to tau, the tau at which a diffuse head becomes "sharp" (peak >= 0.30), and the
converse (diffuse at tau = 0.25).

Read-only: the weights come from the engine's own attention-recording path.
"""

import json
import os
import sys
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ENGINE_ROOT))
os.environ.setdefault("DATA_DIR", str(ENGINE_ROOT / "data"))

import torch

from llm import storage
from llm.corpus import bpe, pack
from llm.model.checkpoint import load as ckpt_load
from llm.train.diagnostics import _entropy

TEMPS = [0.25, 0.5, 1.0, 2.0, 4.0]
SHARP_PEAK = 0.30


def emit(name, data):
    print(json.dumps({"experiment": name, **data},
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def set_recording(model, on):
    for block in model.blocks:
        block.record_activations = on
        block.attn.record_attention = on


def main():
    model, _ = ckpt_load(storage.weights_path("diagfix"), "cpu")
    model.eval()
    tok = bpe.load()
    corpus = pack.load_corpus()
    _, docs, _ = pack.load_segments()

    d = docs[0]
    arr = corpus[d.start : d.start + 40].astype("int64")
    window = torch.from_numpy(arr.copy()).long().unsqueeze(0)
    with torch.no_grad():
        set_recording(model, True)
        model(window)
        weights = [b.attn.last_attention.detach() for b in model.blocks]
        set_recording(model, False)

    sharpest_head = None
    sharpest_at = 0.0
    per_layer = []
    for layer, w in enumerate(weights):
        w = w[0]  # [heads, seq, seq], causal already zeroed
        heads = []
        for h in range(w.shape[0]):
            wh = w[h].unsqueeze(0).unsqueeze(0)  # -> [1,1,seq,keys] for _entropy
            logw = torch.log(wh.clamp_min(1e-9))
            ent_tau, peak_tau = {}, {}
            sharp_at = None
            for tau in TEMPS:
                wt = torch.softmax(tau * logw, dim=-1)
                ent_tau[str(tau)] = round(float(_entropy(wt)[0]), 4)
                p = float(wt.max(dim=-1).values.mean())
                peak_tau[str(tau)] = round(p, 4)
                if tau == 4.0 and p > sharpest_at:
                    sharpest_at = p
                    sharpest_head = (layer, h)
                if p >= SHARP_PEAK and sharp_at is None:
                    sharp_at = tau
            heads.append({
                "head": h,
                "entropy_by_tau": ent_tau,
                "peak_by_tau": peak_tau,
                "first_sharp_tau": sharp_at,
            })
        per_layer.append({"layer": layer, "heads": heads})

    flat = [h for L in per_layer for h in L["heads"]]
    sharp1 = [h for h in flat if h["peak_by_tau"]["1.0"] >= SHARP_PEAK]
    sharp4 = [h for h in flat if h["peak_by_tau"]["4.0"] >= SHARP_PEAK]
    summary = {
        "heads": len(flat),
        "sharp_at_tau1": len(sharp1),
        "sharp_by_tau4": len(sharp4),
        "sharp_at_tau4_to_0_5": sum(1 for h in flat if h["peak_by_tau"]["4.0"] >= 0.5),
        "entropy_lift_0_25": round(float(
            torch.tensor([h["entropy_by_tau"]["0.25"] - h["entropy_by_tau"]["1.0"] for h in flat]).float().mean()), 4),
        "mean_peak_gain_tau1_to_tau4": round(float(
            torch.tensor([h["peak_by_tau"]["4.0"] - h["peak_by_tau"]["1.0"] for h in flat]).float().mean()), 4),
    }

    emit("e8_attention_temperature", {
        "window": {"doc": 0, "seq": int(window.shape[1])},
        "temps": TEMPS,
        "sharp_peak_threshold": SHARP_PEAK,
        "per_layer": per_layer,
        "summary": {
            **summary,
            "sharpest_head_at_tau4": sharpest_head,
            "sharpest_peak_at_tau4": round(sharpest_at, 4),
        },
    })


if __name__ == "__main__":
    main()