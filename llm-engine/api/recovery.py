"""Reconcile disk publication with model registration before the runner starts."""

import json
import shutil
import uuid

from sqlalchemy import select

import store
from api import handlers, rl_recovery, sft_authority, training_publications, validation_recovery
from llm import storage
from llm.architecture import package
from llm.train.progress import Checkpoint
from store import queries as q


def reconcile() -> None:
    """Reconcile complete parent-owned model/validation publications after exclusive worker cleanup.

    Modular packages use their verified job/generation metadata, without entering
    legacy tensor handling. Also finish physical retirement after a crash between
    registration and deletion, retaining published/kept/referenced generations.
    """
    validation_recovery.reconcile()
    package.remove_staging()
    staging = storage.recovery_staging_dir()
    if staging.exists():
        shutil.rmtree(staging)
    for pending in storage.models_dir().glob("*/.*.pending"):
        pending.unlink(missing_ok=True)
    for directory in storage.models_dir().glob("*"):
        metadata = directory / "meta.json"
        if not metadata.is_file():
            continue
        try:
            if (directory / "package.json").is_file():
                training_publications.reconcile(directory)
                continue
            result = json.loads(metadata.read_text()).get("result", {})
            if result.get("recovery"):
                _adopt(directory, result)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"recovery: cannot reconcile {directory.name}: {error}")

    with store.session() as db:
        parents = set(db.scalars(select(store.Model.parent_id).where(store.Model.parent_id.is_not(None))))
        retired = []
        for parent in parents:
            siblings = q.checkpoints_of(db, parent)
            if siblings:
                retired.extend(q.retire_recovery(db, parent, siblings[-1].id))
    for model_id in retired:
        shutil.rmtree(storage.model_dir(str(model_id)), ignore_errors=True)
    sft_authority.reconcile()
    rl_recovery.reconcile()


def _adopt(directory, result: dict) -> None:
    for path in (storage.weights_path(directory.name), storage.train_state_path(directory.name), storage.tokenizer_file(directory)):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"incomplete recovery artifact: {path.name}")
    model_id = uuid.UUID(directory.name)
    parent_id = uuid.UUID(result["output_model_id"])
    saved = result["checkpoint"]
    with store.session() as db:
        existing = q.get_model(db, model_id)
        siblings = q.checkpoints_of(db, parent_id)
        newer = any(row.checkpoint_index > saved["index"] for row in siblings)
        run_id = next((run.id for run in db.scalars(select(store.Run))
                       if (run.config or {}).get("model_name") == str(parent_id)), None)
    if existing is not None:
        return
    if newer:
        shutil.rmtree(directory)
    elif run_id is not None:
        handlers.record_checkpoint(run_id, Checkpoint(
            model_id=str(model_id), index=saved["index"], step=saved["step"],
            total_steps=saved["total_steps"], epoch=saved["epoch"], stage=saved["stage"],
            stage_name=None, path=str(storage.train_state_path(str(model_id))),
            loss=saved.get("loss"), architecture=result.get("architecture", {}),
        ))
