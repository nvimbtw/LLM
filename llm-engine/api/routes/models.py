"""Trained models and what they scored.

Disk is the authority on what can be loaded and the row is the authority on
what a person calls it, so the listing is the join of the two: a directory with
no row is a model trained outside the queue, and a row with no directory is one
whose weights were removed.
"""

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError

import store
from api import auth
from api.model_metadata import read_model_meta, resolve_model_metadata
from api.visibility import public_model
from llm import storage
from store import queries as q
from store.schema import Model

router = APIRouter(tags=["models"])


class ModelEdit(BaseModel):
    display_name: str | None = Field(default=None, max_length=120)
    description: str | None = None
    family: str | None = Field(default=None, max_length=120)
    published: bool | None = None
    recovery_keep: bool | None = None
    size: Literal["fast", "balanced", "flagship"] | None = None

    @field_validator("display_name", "family")
    @classmethod
    def nonblank_label(cls, value: str | None) -> str | None:
        """Trim supplied identity labels and reject whitespace-only names or families."""
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("name and family cannot be empty")
        return value


@router.get("/models")
def list_models(include_private: bool = True, checkpoints: bool = False, administration: bool = Depends(auth.administration_visible)):
    """Join indexed models and weight directories, applying requested visibility and checkpoint filters."""
    with store.session() as db:
        rows = {
            m.id: m
            for m in q.list_models(db, include_private=include_private if administration else False, checkpoints=checkpoints)
        }
        directories = _joinable_dirs(db, checkpoints)
        listed = [
            _model(db, _row_for(rows, name), d) for name, d in directories.items()
            if administration or _row_for(rows, name) is not None
        ]
        listed += [_model(db, m, None) for m in rows.values() if str(m.id) not in directories]
    return {"models": listed if administration else [public_model(model) for model in listed]}


@router.get("/models/{model_id}")
def get_model(model_id: uuid.UUID, administration: bool = Depends(auth.administration_visible)):
    """Return one indexed model with resolved metadata, or 404 for an unknown identity."""
    with store.session() as db:
        model = q.get_model(db, model_id)
        if model is None or (not administration and not model.published):
            raise HTTPException(status_code=404, detail=f"no model {model_id}")
        value = _model(db, model, storage.model_dir(str(model_id)))
        return value if administration else public_model(value)


@router.patch("/models/{model_id}")
def edit_model(model_id: uuid.UUID, body: ModelEdit):
    """Apply authorized metadata edits; a supplied null size clears the admin's category."""
    fields = body.model_dump(exclude_none=True)
    if "display_name" in fields:
        fields["name"] = fields.pop("display_name")
    with store.session() as db:
        model = q.get_model(db, model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f"no model {model_id}")
        if "recovery_keep" in fields:
            model.parameters = {**(model.parameters or {}), "recovery_keep": fields.pop("recovery_keep")}
        if "size" in body.model_fields_set:
            fields.pop("size", None)
            model.parameters = {**(model.parameters or {}), "display_size": body.size}
        for key, value in fields.items():
            setattr(model, key, value)
        try:
            db.flush()
        except IntegrityError as error:
            raise HTTPException(status_code=409, detail="A model with that name already exists.") from error
        return _model(db, model, storage.model_dir(str(model_id)))


@router.delete("/models/{model_id}", status_code=200)
def delete_model(model_id: uuid.UUID, weights: bool = True, _=Depends(auth.require_service)):
    """Drop the row and its checkpoints', and by default the weights with them —
    a model kept only on disk is one nothing can find again."""
    with store.session() as db:
        intermediates = [c.id for c in q.checkpoints_of(db, model_id)]
        name = q.delete_model(db, model_id)
    if name is None:
        raise HTTPException(status_code=404, detail=f"no model {model_id}")
    removed = _remove_weights(model_id) if weights else False
    if weights:
        for checkpoint_id in intermediates:
            _remove_weights(checkpoint_id)
    return {
        "model_id": str(model_id),
        "name": name,
        "weights_removed": removed,
        "checkpoints_removed": len(intermediates),
    }


@router.get("/models/{model_id}/checkpoints")
def list_checkpoints(model_id: uuid.UUID, administration: bool = Depends(auth.administration_visible)):
    """One model's intermediates, in the order they were taken.

    Its own route rather than a flag on the detail: a model detail is loaded on
    every chat, and a run's ten extra rows are wanted on exactly one page.
    """
    with store.session() as db:
        parent = q.get_model(db, model_id)
        if parent is None or (not administration and not parent.published):
            raise HTTPException(status_code=404, detail=f"no model {model_id}")
        found = [c for c in q.checkpoints_of(db, model_id) if administration or c.published]
        return {
            "checkpoints": [
                _model(db, c, storage.model_dir(str(c.id))) if administration else public_model(
                    _model(db, c, storage.model_dir(str(c.id)))
                ) for c in found
            ]
        }


def _weight_dirs() -> list:
    """Return model artifact directories in stable name order without loading weights."""
    return sorted((p for p in storage.models_dir().glob("*") if p.is_dir()), key=lambda p: p.name)


def _joinable_dirs(db, checkpoints: bool) -> dict:
    """The weights directories this listing is the join against.

    A checkpoint's directory is laid out exactly like a finished model's, so
    filtering only the *rows* would let every one of them straight back in
    through the other half of the join — reported as a directory with no row,
    which is how a model trained outside the queue is offered.
    """
    directories = {d.name: d for d in _weight_dirs()}
    if checkpoints:
        return directories
    hidden = {str(i) for i in q.checkpoint_ids(db)}
    return {name: d for name, d in directories.items() if name not in hidden}


def _row_for(rows: dict[uuid.UUID, Model], directory_name: str) -> Model | None:
    """Match an artifact directory to its indexed row, tolerating legacy non-UUID names."""
    try:
        return rows.get(uuid.UUID(directory_name))
    except ValueError:
        return None


def _model(db, model: Model | None, directory) -> dict:
    """A model as the API reports it — whichever half of it exists."""
    meta = read_model_meta(directory)
    facts = resolve_model_metadata(db, model, meta, directory)
    entry = {
        "model_id": str(model.id) if model else None,
        "name": model.name if model else (directory.name if directory else None),
        "description": model.description if model else None,
        "family": model.family if model else None,
        "context": facts.context,
        "architecture": facts.architecture,
        "vocab_size": facts.vocabulary_size,
        "parameter_count": facts.parameter_count,
        "definition_id": facts.definition_id,
        "context_extension_supported": facts.definition_id is None and facts.architecture is not None,
        "size": (model.parameters or {}).get("display_size") if model else None,
        "published": model.published if model else False,
        "run_id": str(model.run_id) if model and model.run_id else None,
        "parent_id": str(model.parent_id) if model and model.parent_id else None,
        "checkpoint_step": model.checkpoint_step if model else None,
        "checkpoint_index": model.checkpoint_index if model else None,
        "recovery_kept": bool(model and ((model.parameters or {}).get("recovery_keep") or (model.parameters or {}).get("recovery_referenced") or model.published)),
        "parameters": model.parameters if model else None,
        "created_at": model.created_at.isoformat() if model and model.created_at else None,
        "has_weights": bool(directory and (directory / "model.pt").is_file()),
        "config": meta.get("config", {}),
    }
    if directory is not None and model is None:
        entry["model_id"] = directory.name
    return entry


def _remove_weights(model_id: uuid.UUID) -> bool:
    """Remove this model's artifact directory if present, returning whether anything was removed."""
    import shutil

    directory = storage.model_dir(str(model_id))
    if not directory.is_dir():
        return False
    shutil.rmtree(directory)
    return True
