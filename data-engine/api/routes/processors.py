"""The processor catalogue, and the preview that makes it usable.

The preview is the point of this module. Choosing a cleaning step by reading its
description is guesswork; choosing it by watching what it does to twenty real
documents is not.
"""

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import processors
import store
from api import auth
from processors import source
from processors.loader import MAX_SOURCE_BYTES
from store import queries as q
from store.schema import DatasetKind

router = APIRouter(tags=["processors"])

#: How many documents a preview may look at. Small on purpose: this runs on every
#: keystroke in a parameter field, and the answer stops being more convincing
#: somewhere well below a hundred.
MAX_SAMPLE = 50


class PreviewRequest(BaseModel):
    dataset_id: uuid.UUID
    steps: list[dict[str, Any]] = Field(max_length=32)
    limit: int = Field(default=10, ge=1, le=MAX_SAMPLE)


def _param(param: processors.Param) -> dict:
    """Render one declared parameter for the browser's generated form."""
    return {
        "name": param.name,
        "label": param.label,
        "type": param.type,
        "default": param.default,
        "description": param.description,
        "minimum": param.minimum,
        "maximum": param.maximum,
    }


def _view(processor: processors.Processor) -> dict:
    """Render an active processor with its loaded source identity."""
    return {
        "name": processor.name,
        "title": processor.title,
        "description": processor.description,
        "applies_to": [k.value for k in processor.applies_to],
        "params": [_param(p) for p in processor.params],
        "file": f"processors/{'custom/' if processors.REGISTRY.editable(processor) else ''}{Path(processor.file).name}",
        "version_hash": processor.version_hash,
    }


@router.get("/processors")
def index(kind: DatasetKind | None = None, _: str = Depends(auth.require_service)):
    """List the active catalogue, optionally filtered by dataset kind."""
    listed = (
        processors.for_kind(kind)
        if kind
        else list(processors.REGISTRY.snapshot().values())
    )
    return {"processors": [_view(p) for p in listed]}


class SourceRequest(BaseModel):
    source: str = Field(max_length=MAX_SOURCE_BYTES)


@router.post("/processors", status_code=201)
def create_source(body: SourceRequest, _: str = Depends(auth.require_service)):
    """Create a valid first custom version; Rust gates this operation with AdminUser."""
    try:
        return asdict(source.create(processors.REGISTRY, body.source))
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/processors/{name}/source")
def get_source(name: str, _: str = Depends(auth.require_service)):
    """Return saved source and active metadata; unknown processors/files return 404."""
    try:
        return asdict(source.read(processors.REGISTRY, name))
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(
            status_code=404, detail="no such processor source"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.put("/processors/{name}/source")
def put_source(name: str, body: SourceRequest, _: str = Depends(auth.require_service)):
    """Save a known custom file as supplied; Rust owns the AdminUser authorization gate."""
    try:
        saved = source.save(processors.REGISTRY, name, body.source)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="no such processor") from error
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "name": name,
        "saved": True,
        "version_hash": saved.version_hash,
        "active_version_hash": saved.active_version_hash,
        "reload_error": saved.reload_error,
    }


@router.post("/preview")
def preview(body: PreviewRequest, _: str = Depends(auth.require_service)):
    """Run a pipeline over the first `limit` documents and report both sides.

    Always the *first* documents by ordinal, never a random sample: the user is
    comparing successive runs against each other as they tune a parameter, and a
    sample that moved between them compares nothing.
    One shared sample state keeps cross-document filters consistent with a run.
    """
    try:
        steps = processors.resolve(
            [{**step, "version_hash": None} for step in body.steps],
            processors.REGISTRY.snapshot(),
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    with store.session() as db:
        dataset = q.dataset(db, body.dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="no such dataset")
        rows = q.documents(db, body.dataset_id, offset=0, limit=body.limit)
        sample = [
            {
                "ordinal": r.ordinal,
                "content": r.content,
                "messages": r.messages,
                "meta": r.meta,
            }
            for r in rows
        ]

    state: processors.State = {}
    results, dropped = [], 0
    for document in sample:
        outcome = processors.run(
            steps, {k: v for k, v in document.items() if k != "ordinal"}, state
        )
        dropped += 1 if outcome.dropped else 0
        results.append(
            {
                "ordinal": document["ordinal"],
                "before": outcome.before,
                "after": outcome.after,
                "dropped": outcome.dropped,
                "dropped_by": outcome.dropped_by,
                "changed": outcome.changed,
            }
        )

    return {
        "kind": dataset.kind.value,
        "sampled": len(results),
        "dropped": dropped,
        "results": results,
        "steps": [step.provenance() for step in steps],
    }
