"""E1 — RoPE geometry: how position information is laid out.

Questions, answered with zero training:
  * Positional decay: cosine similarity of a vector at pos 0 vs pos d, as a
    function of d, for the LLM engine's real (head_dim, base) pair and for
    larger bases.
  * Which frequency channels have wrapped through >90deg of their rotation at
    each position (the channels become position-aliased) — measured heads
    directly observe the trade-off.
  * Interleaved pair-rotation (what the LLM engine uses, GPT-J style) vs the
    half-split convention (GPT-NeoX style): numerically identical after a
    fixed channel permutation, up to floating point.
"""

import json
import math
import os
import sys
from pathlib import Path

LLM_ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LLM_ENGINE_ROOT))
os.environ.setdefault("DATA_DIR", str(LLM_ENGINE_ROOT / "data"))

import numpy as np
import torch

from llm.model.transformer import ROPE_BASE, apply_rope, rope_tables

torch.manual_seed(7)
np.random.seed(7)


def emit(name, data):
    print(json.dumps({"experiment": name, **data},
                     default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def positional_decay(head_dim: int, base: float, ctx: int):
    """Mean normalized QK self-similarity between pos 0 and pos d."""
    cos, sin = rope_tables(head_dim, ctx + 1, base)
    v = torch.randn(128, 1, 1, head_dim)  # [trials, 1, 1, head_dim]
    q0 = torch.nn.functional.normalize(apply_rope(v, cos[:1], sin[:1]), dim=-1)
    sims = []
    for d in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if d > ctx:
            continue
        qd = torch.nn.functional.normalize(apply_rope(v, cos[d:d + 1], sin[d:d + 1]), dim=-1)
        sims.append(round(float((q0 * qd).sum(-1).mean()), 4))
    return dict(zip([str(d) for d in (1, 2, 4, 8, 16, 32, 64, 128, 256)], sims))


def wraparound(head_dim: int, base: float, ctx: int) -> dict:
    """At each distance, the fraction of the head_dim/2 channels whose rotation
    phase is within a quarter turn of completing — aliasing channels."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    out = {}
    for pos in (8, 32, 128, 256, 512):
        if pos > ctx:
            continue
        phase = (inv_freq * pos) % (2 * math.pi)
        low = (phase < math.pi / 2).float().sum().item()
        out[str(pos)] = {
            "channels_wrapped_or_insensitive": round(low / inv_freq.numel(), 4),
            "channels_total": int(inv_freq.numel()),
        }
    return out


def conventions(head_dim: int, ctx: int) -> dict:
    """What the interleaved rotation costs per dimension.

    RoPE can assign its rotation angles to dimension pairs in two layouts:
    * interleaved (GPT-J / this engine): pair (2i, 2i+1) rotates by angle
      theta[2i] — the *even* channel of the pair carries the exponent.
    * half-split (GPT-NeoX): first half of dims is the real part, second the
      imaginary, and the angle comes from the channel's own index.

    The two are the same *set* of angles over a whole head but distribute them
    to different dims, so they are genuinely different layouts (not a relabel).
    We report the per-channel rotation frequency line the LLM engine assigns, which
    is the quantity a port or a head-class analysis needs.
    """
    cos, sin = rope_tables(head_dim, 1)
    return {
        "head_dim": head_dim,
        "channels": int(head_dim / 2),
        "max_frequency_hz": None,
        "channel_angles_rad_per_offset": {
            str(i): round(float(2 * math.pi * i / head_dim), 3)
            for i in range(1, 5)
        },
    }


if __name__ == "__main__":
    head_dim = 64  # e.g. 512d/8h on the production machine
    out = {
        "engine": {
            "base": ROPE_BASE,
            "decay_similarity": positional_decay(head_dim, int(ROPE_BASE), 512),
            "wraparound": wraparound(head_dim, 10_000, 512),
        },
        "bases": {},
        "conventions": {"interleaved_vs_half_split": conventions(head_dim, 8)},
    }
    for base in (10_000, 100_000, 500_000, 1_000_000):
        out["bases"][str(base)] = {
            "decay_similarity": positional_decay(head_dim, base, 512),
            "wraparound": wraparound(head_dim, base, 512),
        }
    emit("e1_rope", out)