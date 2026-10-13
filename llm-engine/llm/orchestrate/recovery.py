"""Publish complete recovery artifacts before announcing them to the API."""

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from llm import storage
from llm.model.checkpoint import sync_directory
from llm.model import checkpoint


def resume(config, kind: str):
    if not config.resume_from:
        return None
    saved = checkpoint.load_state(storage.train_state_path(config.resume_from))
    if saved.optimizer is None or saved.extra.get("training_kind") != kind:
        raise ValueError(f"Recovery point does not contain resumable {kind} state")
    original = type(config)(**saved.extra["config"])
    mutable = {"model_name", "resume_from", "checkpoint_target", "rebuild_dataset"}
    if original.model_dump(exclude=mutable) != config.model_dump(exclude=mutable):
        raise ValueError("Recovery requires the original training configuration")
    return saved


@contextmanager
def stage(destination: Path):
    """Publish a new, immutable model directory with a single rename.

    Staging lives outside the model listing, on the same filesystem. A failed
    save never exposes a partial model or changes an earlier recovery point.
    The caller must supply a fresh destination, not an existing model ID.
    """
    staging_root = storage.recovery_staging_dir()
    staging_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=staging_root))
    try:
        yield temporary
        for artifact in temporary.iterdir():
            with artifact.open("rb") as source:
                os.fsync(source.fileno())
        sync_directory(temporary)
        if destination.exists():
            raise FileExistsError(f"Recovery destination already exists: {destination}")
        os.rename(temporary, destination)
        sync_directory(destination.parent)
        sync_directory(staging_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def fingerprint(*arrays, plan=None) -> str:
    """Hash packed bytes once; equal-length rebuilt data is still different data."""
    import hashlib
    import json
    digest = hashlib.sha256()
    if plan is not None:
        digest.update(json.dumps([plan.eot_id,plan.shuffle,plan.seed,plan.documents.names],sort_keys=True).encode())
        arrays = (*arrays,plan.documents.rows)
        for stage in plan.stages:
            digest.update(json.dumps([stage.name,stage.epochs,stage.documents.names]).encode())
            arrays = (*arrays,stage.documents.rows)
    for array in arrays:
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()
