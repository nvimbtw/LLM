"""E6 — A tiny real training run through engine/llm/train/loop, on CPU.

Not a benchmark of speed (that was settled on the GPU box) but a *behavioral*
harness: it runs the actual training loop on the actual corpus and records
  * the loss curve and lr schedule
  * a couple of full diagnostics snapshots (grad ratios, update ratios,
    attention entropy, embedding anisotropy) via the engine's own Collector
so that ideas in the report can be checked against numbers the engine itself
produces — e.g. whether attention entropy or update:weight ratio is close to a
collapse value that a "residual" or "sink-token" layer idea should fix.

Read-only w.r.t. engine code. Checkpoints are written to a SCRATCH dir.
"""

import json
import os
import sys
import time
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ENGINE_ROOT))

import numpy as np

import bench  # the engine's own harness

STEPS = 40


def emit(name, data):
    print(json.dumps({"experiment": name, **data}, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def main():
    device = bench.loop.pick_device()
    model, corpus, plan, cfg = bench.build(
        "shakedown",
        STEPS,
        model_name="__explore__",
        epochs=1,
        warmup_steps=2,
        shuffle=False,
        dimensions=64,
        heads=4,
        layers=2,
        ff_multiplier=4,
        ctx_window=64,
        batch_size=16,
    )

    losses = []
    diags = []

    def on_progress(progress):
        if not progress.epoch_done and progress.loss is not None:
            losses.append({"step": progress.step, "loss": progress.loss, "lr": progress.lr})
        if progress.diag:
            diags.append({"step": progress.step, "diag": dict(progress.diag)})

    started = time.perf_counter()
    with bench.scratch_storage():
        result = bench.loop.run(model, corpus, plan, cfg, device, on_progress=on_progress)
    elapsed = time.perf_counter() - started
    n = result["steps"]

    ms_per_step = 1000.0 * elapsed / max(1, n)
    tokens_per_s = n * cfg.batch_size * cfg.ctx_window / max(1e-6, elapsed)

    # Summarise the first and last diagnostics to something tabular.
    def tidy(diag):
        return {
            "step": diag["step"],
            "grad_global": diag["diag"].get("grad_global"),
            "update_ratio_layers": diag["diag"].get("update_ratio_layers"),
            "attn_entropy": diag["diag"].get("attn_entropy"),
            "emb_cos_mean": diag["diag"].get("emb_cos_mean"),
            "act_rms_layers": diag["diag"].get("act_rms_layers"),
        }

    tidy_diags = [tidy(d) for d in diags]
    first_diag = tidy_diags[0] if tidy_diags else None
    last_diag = tidy_diags[-1] if tidy_diags else None

    emit("e6_tiny_run", {
        "device": device,
        "arch": {"d": 64, "h": 4, "L": 2, "ctx": 64, "bs": 16, "vocab": 8192},
        "summary": result,
        "step_stats": {
            "steps": n,
            "ms_per_step": round(ms_per_step, 2),
            "tokens_per_sec": round(tokens_per_s, 1),
        },
        "curve": losses[:5] + losses[-5:],
        "diag_first": first_diag,
        "diag_last": last_diag,
    })


if __name__ == "__main__":
    main()