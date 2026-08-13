"""Trained models and what they scored.

Disk is the authority on what can be loaded and the row is the authority on
what a person calls it, so the listing is the join of the two: a directory with
no row is a model trained outside the queue, and a row with no directory is one
whose weights were removed.
"""

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import store
from api import auth
from llm import storage
from store import queries as q
from store.schema import Benchmark, Model, Suite

router = APIRouter(tags=["models"])


class ModelEdit(BaseModel):
    display_name: str | None = Field(default=None, max_length=120)
    description: str | None = None
    published: bool | None = None


@router.get("/models")
def list_models(include_private: bool = True):
    directories = {d.name: d for d in _weight_dirs()}
    with store.session() as db:
        rows = {m.id: m for m in q.list_models(db, include_private=include_private)}
        listed = [_model(_row_for(rows, name), d) for name, d in directories.items()]
        listed += [_model(m, None) for m in rows.values() if str(m.id) not in directories]
    return {"models": listed}


@router.get("/models/{model_id}")
def get_model(model_id: uuid.UUID):
    with store.session() as db:
        model = q.get_model(db, model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f"no model {model_id}")
        return {
            **_model(model, storage.model_dir(str(model_id))),
            "benchmarks": [_benchmark(b) for b in q.benchmarks_of(db, model_id)],
        }


@router.patch("/models/{model_id}")
def edit_model(model_id: uuid.UUID, body: ModelEdit):
    fields = body.model_dump(exclude_none=True)
    if "display_name" in fields:
        fields["name"] = fields.pop("display_name")
    with store.session() as db:
        model = q.get_model(db, model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f"no model {model_id}")
        for key, value in fields.items():
            setattr(model, key, value)
        return _model(model, storage.model_dir(str(model_id)))


@router.delete("/models/{model_id}", status_code=200)
def delete_model(model_id: uuid.UUID, weights: bool = True, _=Depends(auth.require_service)):
    """Drop the row, and by default the weights with it — a model kept only on
    disk is one nothing can find again."""
    with store.session() as db:
        name = q.delete_model(db, model_id)
    if name is None:
        raise HTTPException(status_code=404, detail=f"no model {model_id}")
    removed = _remove_weights(model_id) if weights else False
    return {"model_id": str(model_id), "name": name, "weights_removed": removed}


@router.get("/models/{model_id}/benchmarks")
def list_benchmarks(model_id: uuid.UUID, limit: int = 50):
    with store.session() as db:
        return {"benchmarks": [_benchmark(b) for b in q.benchmarks_of(db, model_id, limit)]}


@router.get("/models/{model_id}/benchmarks/{suite}")
def latest_benchmark(model_id: uuid.UUID, suite: Suite, version: int = 1):
    """The newest comparable score. Suite and version both matter: a scoring
    change moves the number without the model having changed."""
    with store.session() as db:
        found = q.latest_benchmark(db, model_id, suite, version)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no {suite.value} score for {model_id}")
        return _benchmark(found)


def _weight_dirs() -> list:
    return sorted((p for p in storage.models_dir().glob("*") if p.is_dir()), key=lambda p: p.name)


def _row_for(rows: dict[uuid.UUID, Model], directory_name: str) -> Model | None:
    try:
        return rows.get(uuid.UUID(directory_name))
    except ValueError:
        return None


def _model(model: Model | None, directory) -> dict:
    """A model as the API reports it — whichever half of it exists."""
    entry = {
        "model_id": str(model.id) if model else None,
        "name": model.name if model else (directory.name if directory else None),
        "description": model.description if model else None,
        "family": model.family if model else None,
        "context": model.context if model else None,
        "published": model.published if model else False,
        "created_at": model.created_at.isoformat() if model and model.created_at else None,
        "has_weights": bool(directory and (directory / "model.pt").is_file()),
        "config": _meta(directory),
    }
    if directory is not None and model is None:
        entry["model_id"] = directory.name
    return entry


def _meta(directory) -> dict:
    if directory is None:
        return {}
    meta = directory / "meta.json"
    if not meta.is_file():
        return {}
    try:
        return json.loads(meta.read_text()).get("config", {})
    except ValueError:
        return {}


def _remove_weights(model_id: uuid.UUID) -> bool:
    import shutil

    directory = storage.model_dir(str(model_id))
    if not directory.is_dir():
        return False
    shutil.rmtree(directory)
    return True


def _benchmark(run: Benchmark) -> dict:
    return {
        "benchmark_id": str(run.id),
        "model_id": str(run.model_id),
        "model_name": run.model_name,
        "suite": run.suite.value,
        "suite_version": run.suite_version,
        "status": run.status.value,
        "score": run.score,
        "eval_tokens": run.eval_tokens,
        "eval_conversations": run.eval_conversations,
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }
