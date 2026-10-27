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
from typing import BinaryIO

import numpy as np
import torch

from llm.model.transformer import Transformer

# Bump when a change makes previously-saved weights mean something different.
CHECKPOINT_VERSION = 1
MODULAR_CHECKPOINT_VERSION = 2
MODULAR_FORMAT = "slate.modular-weights"

# `rope_base` is last and the Transformer defaults it: a checkpoint saved before
# it existed simply omits the key and loads at the original 10000.0, which is
# what it was trained with. That is why this needs no version bump.
_ARCH_FIELDS = (
    "vocab", "dimensions", "heads", "layers", "ff_multiplier", "ctx_window", "rope_base",
    "learned_temperature",
)


def save(model: Transformer, path: Path, **extra) -> Path:
    """Atomically publish a legacy version-one Transformer without changing its file meaning."""
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


class _BoundedTensorOutput:
    """Enforce the admitted serialized file ceiling before each PyTorch archive write."""

    def __init__(self, stream: BinaryIO, max_bytes: int):
        """Retain one caller-owned output and its positive exact write ceiling."""
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("checkpoint byte budget must be positive")
        self.stream, self.max_bytes = stream, max_bytes
        self.exceeded = False

    def write(self, data) -> int:
        """Refuse an over-budget archive block before it can grow the partial file."""
        if self.exceeded or self.stream.tell() + len(data) > self.max_bytes:
            self.exceeded = True
            raise ValueError("checkpoint exceeds serialized file byte budget")
        return self.stream.write(data)

    def flush(self) -> None:
        """Flush the owning stream when PyTorch finishes its archive."""
        self.stream.flush()


def _publish(payload: dict, path: Path, *, max_bytes: int | None = None) -> None:
    """Replace a checkpoint only after its new bytes have reached disk.

    Staging beside the destination keeps rename on the same filesystem. Readers
    opening the destination see either complete generation, never a partial zip.
    A hard process kill can leave a hidden staging file; it is never a recovery
    candidate and must be reconciled by recovery-storage cleanup.
    max_bytes, when supplied, limits serialized blocks before writes; refusal
    preserves the previous file and removes partial output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".pending", delete=False
        ) as output:
            staging = Path(output.name)
            bounded = _BoundedTensorOutput(output, max_bytes) if max_bytes is not None else None
            try:
                torch.save(payload, bounded or output)
            except Exception as error:
                if bounded is not None and bounded.exceeded:
                    raise ValueError("checkpoint exceeds serialized file byte budget") from error
                raise
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


def _is_modular_checkpoint(path: Path) -> bool:
    """Recognize modular metadata without unpickling tensors, including orphan standalone v2 files."""
    import pickletools
    import zipfile

    if path.with_name("package.json").exists():
        return True
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 16384:
            raise ValueError("Checkpoint archive metadata exceeds entry budget")
        metadata = [entry for entry in entries if entry.filename.endswith("/data.pkl")]
        if len(metadata) != 1 or metadata[0].file_size > 4 << 20:
            raise ValueError("Checkpoint metadata is missing or exceeds byte budget")
        data = archive.read(metadata[0])
    return any(argument == MODULAR_FORMAT for _, argument, _ in pickletools.genops(data))


def load(path: Path, device: str = "cpu") -> tuple[Transformer, dict]:
    """Load legacy weights safely; modular construction requires the explicit worker-only package API."""
    if not path.is_file():
        raise FileNotFoundError(f"No weights at {path}")

    if _is_modular_checkpoint(path):
        raise ValueError("Modular checkpoints require the admitted worker package loader")
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
    definition_id: str | None = None
    optimizer_parameter_groups: tuple[tuple[str, ...], ...] = ()


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

    CPU Torch, NumPy and Python generators are required for a nonempty state.
    RNG byte/key tensors return to CPU even when checkpoint map_location moved
    them to CUDA with the weights. Optional CUDA streams restore only when the
    current device count matches; cross-device continuation remains supported.
    """
    if not state:
        return
    torch.set_rng_state(state["torch"].to(device="cpu", dtype=torch.uint8))
    np.random.set_state((
        "MT19937",
        state["numpy_keys"].cpu().numpy().astype(np.uint32),
        state["numpy_position"],
        state["numpy_has_gauss"],
        state["numpy_cached_gaussian"],
    ))
    random.setstate(_as_python_rng(state["python"]))
    cuda = state.get("cuda")
    if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([one.to(device="cpu", dtype=torch.uint8) for one in cuda])


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


def optimizer_layout(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[tuple[str, ...], ...]:
    """Record stable group/parameter ownership, refusing duplicate or foreign optimizer Parameters."""
    return group_layout(model, optimizer.param_groups)


def group_layout(model: torch.nn.Module, groups: list[dict]) -> tuple[tuple[str, ...], ...]:
    """Validate stable optimizer ownership/order from prepared groups before optimizer construction or state loading."""
    names = {id(parameter): name for name, parameter in reversed(tuple(model.named_parameters()))}
    seen = set()
    layout = []
    for group in groups:
        ordered = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in names or identity in seen:
                raise ValueError("Optimizer must own each model Parameter at most once")
            seen.add(identity)
            ordered.append(names[identity])
        layout.append(tuple(ordered))
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if trainable - seen:
        raise ValueError("Optimizer omits intended trainable Parameters")
    return tuple(layout)


def save_modular_weights(model: torch.nn.Module, path: Path, definition_id: str, *, max_bytes: int = 16 << 30) -> Path:
    """Write discriminated tensor-only weights inside an unpublished self-contained package."""
    from llm import storage

    storage.definition_dir(definition_id)
    _publish({"format": MODULAR_FORMAT, "version": MODULAR_CHECKPOINT_VERSION,
              "definition_id": definition_id, "state_dict": model.state_dict()}, path, max_bytes=max_bytes)
    return path


def save_modular_state(
    path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    definition_id: str, step: int, epoch: int, stage: int, total_steps: int,
    rng: dict, extra: dict, max_bytes: int = 16 << 30,
) -> Path:
    """Write data-only optimizer/RNG/cursor state pinned to exact sibling weights and definition."""
    from llm.architecture.package_integrity import file_identity

    weights = file_identity(path.with_name("model.pt"), 16 << 30)
    _publish({"format": MODULAR_FORMAT, "version": MODULAR_CHECKPOINT_VERSION,
              "definition_id": definition_id, "weights_sha256": weights.sha256,
              "optimizer": optimizer.state_dict(), "optimizer_parameter_groups": optimizer_layout(model, optimizer),
              "step": step, "epoch": epoch, "stage": stage, "total_steps": total_steps,
              "rng": rng, "extra": extra}, path, max_bytes=max_bytes)
    return path


def _load_modular_state(path: Path, blob: dict, device: str) -> TrainState:
    """Verify package identity before recovering data-only state and exact matching sibling tensors."""
    from llm.architecture import package

    checked = package.verify(path.parent)
    if (blob.get("format") != MODULAR_FORMAT or blob.get("version") != MODULAR_CHECKPOINT_VERSION
            or blob.get("definition_id") != checked.manifest.definition_id):
        raise ValueError("Modular checkpoint definition/version mismatch")
    weights = blob if path.name == "model.pt" else torch.load(path.with_name("model.pt"), map_location=device, weights_only=True)
    if (weights.get("format") != MODULAR_FORMAT or weights.get("version") != MODULAR_CHECKPOINT_VERSION
            or weights.get("definition_id") != checked.manifest.definition_id):
        raise ValueError("Modular recovery weights definition mismatch")
    if "optimizer" in blob and blob.get("weights_sha256") != checked.integrity.files["model.pt"].sha256:
        raise ValueError("Modular recovery state references different weights")
    groups = tuple(tuple(group) for group in blob.get("optimizer_parameter_groups", ()))
    if "optimizer" in blob and tuple(name for group in groups for name in group) != checked.manifest.optimizer_parameter_order:
        raise ValueError("Modular optimizer order differs from package manifest")
    return TrainState(arch={}, state_dict=weights["state_dict"], step=blob.get("step", 0),
                      epoch=blob.get("epoch", 0), stage=blob.get("stage", 0), total_steps=blob.get("total_steps", 0),
                      optimizer=blob.get("optimizer"), rng=blob.get("rng") or {}, extra=blob.get("extra") or {},
                      definition_id=checked.manifest.definition_id, optimizer_parameter_groups=groups)


def load_state(path: Path, device: str = "cpu") -> TrainState:
    """A training state, or a plain weights file read as the start of one.

    The version does not move for the added keys, because they change nothing
    about what a v1 file means — so this reads a `model.pt` written years of
    commits ago and answers with a state whose optimizer is None, which is a
    continuation from cold moments and says so.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No training state at {path}")

    if _is_modular_checkpoint(path):
        if os.environ.get("LLM_RUNTIME_CHILD") != "1":
            raise ValueError("Modular recovery loading requires an admitted worker")
        from llm.architecture import package

        manifest = package.verify(path.parent).manifest
        if path.name not in {manifest.weights_path, manifest.training_state_path}:
            raise ValueError("Unexpected modular checkpoint artifact path")
    blob = torch.load(path, map_location=device, weights_only=True)
    if blob.get("format") == MODULAR_FORMAT or blob.get("version") == MODULAR_CHECKPOINT_VERSION:
        return _load_modular_state(path, blob, device)
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
