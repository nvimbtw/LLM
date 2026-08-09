"""Saving and loading weights.

The file carries its own architecture. The Rust side learned this the hard way:
its `MODEL_VERSION` exists because pre-v5 weights trained with no final
LayerNorm and pre-v6 weights with an untied output projection, and a LayerNorm
at gamma=1/beta=0 is not the identity — so a loader that assumes the current
architecture silently changes what an old model outputs. Storing the shape
alongside the weights means loading never needs out-of-band knowledge.
"""

from pathlib import Path

import torch

from llm.model.transformer import Transformer

# Bump when a change makes previously-saved weights mean something different.
CHECKPOINT_VERSION = 1

# `rope_base` is last and the Transformer defaults it: a checkpoint saved before
# it existed simply omits the key and loads at the original 10000.0, which is
# what it was trained with. That is why this needs no version bump.
_ARCH_FIELDS = (
    "vocab", "dimensions", "heads", "layers", "ff_multiplier", "ctx_window", "rope_base",
    "learned_temperature",
)


def save(model: Transformer, path: Path, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "arch": {field: getattr(model, field) for field in _ARCH_FIELDS},
            "state_dict": model.state_dict(),
            **extra,
        },
        path,
    )
    return path


def load(path: Path, device: str = "cpu") -> tuple[Transformer, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"No weights at {path}")

    # weights_only=True: a checkpoint is data, and unpickling arbitrary objects
    # from one is remote code execution if the file ever comes from elsewhere.
    blob = torch.load(path, map_location=device, weights_only=True)

    version = blob.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} is checkpoint v{version}, this build reads v{CHECKPOINT_VERSION}"
        )

    model = Transformer(**blob["arch"])
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob["arch"]
