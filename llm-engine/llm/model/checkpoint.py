"""Saving and loading weights, and the training state beside them.

The file carries its own architecture. The Rust side learned this the hard way:
its `MODEL_VERSION` exists because pre-v5 weights trained with no final
LayerNorm and pre-v6 weights with an untied output projection, and a LayerNorm
at gamma=1/beta=0 is not the identity — so a loader that assumes the current
architecture silently changes what an old model outputs. Storing the shape
alongside the weights means loading never needs out-of-band knowledge.

`save`/`load` are what inference reads and their meaning does not move.
`save_state`/`load_state` are the training pair: the same `version`, `arch` and
training state needed to continue. Recovery files reference sibling weights
without duplicating them; old self-contained files remain readable. Both
loaders support these forms, while plain weights carry no optimizer history.
"""

import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
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
    _publish(
        {
            "version": CHECKPOINT_VERSION,
            "arch": {name: getattr(model, name) for name in _ARCH_FIELDS},
            "state_dict": model.state_dict(),
            **extra,
        },
        path,
    )
    return path


def _publish(payload: dict, path: Path) -> None:
    """Replace a checkpoint only after its new bytes have reached disk.

    Staging beside the destination keeps rename on the same filesystem. Readers
    opening the destination see either complete generation, never a partial zip.
    A hard process kill can leave a hidden staging file; it is never a recovery
    candidate and must be reconciled by recovery-storage cleanup.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".pending", delete=False
        ) as output:
            staging = Path(output.name)
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(staging, path)
        sync_directory(path.parent)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def sync_directory(directory: Path) -> None:
    """Persist rename/link directory entries after their file bytes are synced."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load(path: Path, device: str = "cpu") -> tuple[Transformer, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"No weights at {path}")

    # weights_only=True: a checkpoint is data, and unpickling arbitrary objects
    # from one is remote code execution if the file ever comes from elsewhere.
    blob = torch.load(path, map_location=device, weights_only=True)

    if "state_dict" not in blob:
        weights = torch.load(path.with_name("model.pt"), map_location=device, weights_only=True)
        if weights.get("arch") != blob.get("arch") or weights.get("version") != blob.get("version"):
            raise ValueError("Recovery weights do not match optimizer state")
        blob["state_dict"] = weights["state_dict"]

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


@dataclass
class TrainState:
    """Everything a run needs to be continued, as opposed to merely loaded.

    `optimizer` is absent for a state read out of a plain weights file — a
    finished model has no moments to hand over, and resuming from one starts
    AdamW cold. That is a real difference in what the continuation *is*, so it is
    left visible here rather than papered over with an empty dict.

    `extra` is the orchestrator's: what corpus and vocabulary this state belongs
    to, the config it was produced under, and the loss at the moment it was
    taken. Nothing in `llm/train/` reads it — it is what tells a *later* run
    whether continuing here is the same run or a different one.
    """

    arch: dict
    state_dict: dict
    step: int = 0
    epoch: int = 0
    stage: int = 0
    total_steps: int = 0
    optimizer: dict | None = None
    rng: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def rng_state() -> dict:
    """Every generator a step's arithmetic can depend on.

    Numpy's is stored as a tensor rather than as the tuple `get_state` hands
    back: that tuple holds an `ndarray`, and `weights_only=True` refuses to
    construct one. Widening the unpickler to admit it would mean every
    checkpoint file this build reads becomes able to construct arbitrary numpy
    objects, which is a large price for 624 integers. Converting costs nothing.
    """
    keys, position, has_gauss, cached_gaussian = np.random.get_state()[1:]
    state = {
        "torch": torch.get_rng_state(),
        "numpy_keys": torch.from_numpy(keys.astype(np.int64)),
        "numpy_position": int(position),
        "numpy_has_gauss": int(has_gauss),
        "numpy_cached_gaussian": float(cached_gaussian),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    """Put every generator back where `rng_state` found it.

    A missing entry is skipped rather than defaulted: a state saved on a CUDA box
    and resumed on a CPU one is a legitimate continuation, and so is the reverse.
    The device count is checked for the same reason — restoring four streams onto
    one card is not a resume, it is a crash thousands of steps from anything a
    person could read.
    """
    if not state:
        return
    torch.set_rng_state(state["torch"].to(torch.uint8))
    np.random.set_state((
        "MT19937",
        state["numpy_keys"].numpy().astype(np.uint32),
        state["numpy_position"],
        state["numpy_has_gauss"],
        state["numpy_cached_gaussian"],
    ))
    random.setstate(_as_python_rng(state["python"]))
    cuda = state.get("cuda")
    if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([one.to(torch.uint8) for one in cuda])


def _as_python_rng(state):
    """`random.setstate` insists on tuples; the unpickler hands lists back."""
    version, keys, gauss = state
    return (version, tuple(keys), gauss)


def save_state(
    path: Path,
    *,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    stage: int,
    total_steps: int,
    rng: dict,
    extra: dict,
    include_weights: bool = True,
) -> Path:
    """The full training state, beside the weights rather than instead of them.

    It costs about three times a `model.pt`: the weights again, plus AdamW's two
    moments per parameter. That is the number `progress.TARGET_CHECKPOINTS`
    trades against.
    """
    _publish(
        {
            "version": CHECKPOINT_VERSION,
            "arch": {name: getattr(model, name) for name in _ARCH_FIELDS},
            **({"state_dict": model.state_dict()} if include_weights else {}),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "stage": stage,
            "total_steps": total_steps,
            "rng": rng,
            "extra": extra,
        },
        path,
    )
    return path


def load_state(path: Path, device: str = "cpu") -> TrainState:
    """A training state, or a plain weights file read as the start of one.

    The version does not move for the added keys, because they change nothing
    about what a v1 file means — so this reads a `model.pt` written years of
    commits ago and answers with a state whose optimizer is None, which is a
    continuation from cold moments and says so.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No training state at {path}")

    blob = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" not in blob:
        weights = torch.load(path.with_name("model.pt"), map_location=device, weights_only=True)
        if weights.get("version") != blob.get("version") or weights.get("arch") != blob.get("arch"):
            raise ValueError("Recovery weights and optimizer architecture differ")
        blob["state_dict"] = weights["state_dict"]
    version = blob.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} is checkpoint v{version}, this build reads v{CHECKPOINT_VERSION}"
        )

    return TrainState(
        arch=blob["arch"],
        state_dict=blob["state_dict"],
        step=blob.get("step", 0),
        epoch=blob.get("epoch", 0),
        stage=blob.get("stage", 0),
        total_steps=blob.get("total_steps", 0),
        optimizer=blob.get("optimizer"),
        rng=blob.get("rng") or {},
        extra=blob.get("extra") or {},
    )
