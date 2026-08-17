"""E3 — Embedding geometry of the trained model.

Uses the engine's own diagnostics (frequent_tokens, _embedding_metrics) on the
trained `diagfix` model:
  * pairwise cosine among the most frequent tokens — centred vs uncentred, to
    expose how much of the number is a shared offset
  * norm of the mean row relative to per-row spread
  * the PCA spectrum of the embedding table — how many dimensions it really uses
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
from llm.corpus import pack
from llm.model.checkpoint import load as ckpt_load
from llm.train.diagnostics import _embedding_metrics, frequent_tokens


def emit(name, data):
    print(json.dumps({"experiment": name, **data},
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def trained_model(name):
    return ckpt_load(storage.weights_path(name), "cpu")


def _cos_stats(rows: torch.Tensor) -> dict:
    """Mean / p5 / p95 pairwise cosine over the rows, excluding self-diag."""
    unit = torch.nn.functional.normalize(rows.float(), dim=1)
    cos = unit @ unit.T
    mask = ~torch.eye(len(rows), dtype=torch.bool)
    on = cos[mask]
    p = torch.quantile(on, torch.tensor([0.05, 0.95])).tolist()
    return {"mean": float(on.mean()), "p5": p[0], "p95": p[1]}


def main():
    # The checkpoint to probe: argv[1], else the model the report describes.
    model_name = sys.argv[1] if len(sys.argv) > 1 else "diagfix"
    model, arch = trained_model(model_name)
    tokens = frequent_tokens(pack.load_corpus(pack.latest_key()), 32)
    metrics = _embedding_metrics(model, tokens)

    with torch.no_grad():
        weight = model.embedding.weight.float()          # [vocab, dim]
        mu = weight.mean(dim=0, keepdim=True)            # shared offset
        centred = weight - mu

        uncentred = _cos_stats(weight[:1000])
        centred_stats = _cos_stats(centred[:1000])

        # PCA spectrum over the first 2000 rows.
        rows = centred[:2000] - centred[:2000].mean(dim=0, keepdim=True)
        S = torch.linalg.svdvals(rows)
        var = (S ** 2).tolist()
        total = sum(var)
        cum = []
        acc = 0.0
        for v in var:
            acc += v
            cum.append(acc / total)

        def dims_for(fraction):
            for i, v in enumerate(cum):
                if v >= fraction:
                    return i + 1
            return len(cum)

        offset_mag = float(mu.norm())
        spread = float(centred.norm(dim=1).mean())

    emit("e3_geometry", {
        "model": model_name,
        "arch": arch,
        "frequent_token_metrics": {
            "emb_cos_mean": metrics["emb_cos_mean"],
            "emb_cos_p5": metrics["emb_cos_p5"],
            "emb_cos_p95": metrics["emb_cos_p95"],
        },
        "cosine_uncentred": uncentred,
        "cosine_centred": centred_stats,
        "shared_offset": {
            "mean_row_norm": round(offset_mag, 3),
            "mean_centred_row_norm": round(spread, 3),
            "offset_share_of_total_norm": round(offset_mag / (offset_mag + spread), 3),
        },
        "pca_spectrum": {
            "dims": len(var),
            "top_12_variance_share": [round(v, 4) for v in cum[:12]],
            "dims_for_50pct": dims_for(0.5),
            "dims_for_90pct": dims_for(0.9),
        },
    })


if __name__ == "__main__":
    main()