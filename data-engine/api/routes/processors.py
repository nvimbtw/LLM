"""The processor catalogue, and the preview that makes it usable.

The preview is the point of this module. Choosing a cleaning step by reading its
description is guesswork; choosing it by watching what it does to twenty real
documents is not.
"""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import processors
import store
from api import auth
from store import queries as q
from store.schema import DatasetKind

router = APIRouter(tags=["processors"])

#: How many documents a preview may look at. Small on purpose: this runs on every
#: keystroke in a parameter field, and the answer stops being more convincing
#: somewhere well below a hundred.
MAX_SAMPLE = 50


class PreviewRequest(BaseModel):
    dataset_id: uuid.UUID
    steps: list[dict[str, Any]]
    limit: int = Field(default=10, ge=1, le=MAX_SAMPLE)


def _param(param: processors.Param) -> dict:
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
    return {
        "name": processor.name,
        "title": processor.title,
        "description": processor.description,
        "applies_to": [k.value for k in processor.applies_to],
        "params": [_param(p) for p in processor.params],
    }


@router.get("/processors")
def index(kind: DatasetKind | None = None, _: str = Depends(auth.require_service)):
    listed = processors.for_kind(kind) if kind else list(processors.REGISTRY.values())
    return {"processors": [_view(p) for p in listed]}


@router.post("/preview")
def preview(body: PreviewRequest, _: str = Depends(auth.require_service)):
    """Run a pipeline over the first `limit` documents and report both sides.

    Always the *first* documents by ordinal, never a random sample: the user is
    comparing successive runs against each other as they tune a parameter, and a
    sample that moved between them compares nothing.
    """
    try:
        steps = processors.resolve(list(body.steps), processors.REGISTRY)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    with store.session() as db:
        dataset = q.dataset(db, body.dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="no such dataset")
        rows = q.documents(db, body.dataset_id, offset=0, limit=body.limit)
        sample = [
            {"ordinal": r.ordinal, "content": r.content, "messages": r.messages, "meta": r.meta}
            for r in rows
        ]

    # One state across the sample, so a stateful step like dedup behaves here the
    # way it will over the whole dataset.
    state: processors.State = {}
    results, dropped = [], 0
    for document in sample:
        outcome = processors.run(steps, {k: v for k, v in document.items() if k != "ordinal"}, state)
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
    }
