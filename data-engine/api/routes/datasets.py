"""Datasets: what has been pulled, what it contains, and what to make of it."""

import json
import re
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import processors
import store
from api import auth
from api.runner import RUNNER
from store import queries as q
from store.schema import Dataset, DatasetKind, DatasetOrigin, DatasetStatus, RunKind

router = APIRouter(tags=["datasets"])

#: A dataset name is used in URLs and, one day, as a directory — so it is kept to
#: what is safe in both rather than to what Postgres would accept.
NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 _-]{0,118}[a-zA-Z0-9]$")

#: The most documents one page may ask for. A preview needs ten; a browser that
#: asks for a million is a mistake, not a request.
MAX_PAGE = 200


class PullRequest(BaseModel):
    repo: str
    name: str
    kind: DatasetKind = DatasetKind.pretrain
    description: str | None = None
    prefix: str | None = None
    column: str | None = None
    config: str | None = None
    split: str | None = None
    #: Absent means the handler's default. There is no "everything".
    limit_rows: int | None = Field(default=None, ge=1)


class ProcessRequest(BaseModel):
    name: str
    description: str | None = None
    steps: list[dict[str, Any]] = Field(max_length=32)


def _view(row: Dataset) -> dict:
    return {
        "id": str(row.id),
        "name": row.name,
        "description": row.description,
        "kind": row.kind.value,
        "origin": row.origin.value,
        "status": row.status.value,
        "source_repo": row.source_repo,
        "source_config": row.source_config,
        "source_split": row.source_split,
        "parent_id": str(row.parent_id) if row.parent_id else None,
        "steps": row.steps,
        "document_count": row.document_count,
        "byte_size": row.byte_size,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def _document_view(row: store.Document) -> dict:
    return {
        "id": str(row.id),
        "ordinal": row.ordinal,
        "content": row.content,
        "messages": row.messages,
        "meta": row.meta,
        "byte_size": row.byte_size,
    }


def _validate_name(name: str) -> str:
    cleaned = name.strip()
    if not NAME.match(cleaned):
        raise HTTPException(status_code=422, detail=f"unusable dataset name: {name!r}")
    return cleaned


def _claim_name(db, name: str) -> str:
    if q.dataset_by_name(db, name) is not None:
        raise HTTPException(status_code=409, detail=f"a dataset called {name!r} already exists")
    return name


def _require(db, dataset_id: uuid.UUID) -> Dataset:
    row = q.dataset(db, dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such dataset")
    return row


@router.get("/datasets")
def index(_: str = Depends(auth.require_service)):
    with store.session() as db:
        return {"datasets": [_view(row) for row in q.list_datasets(db)]}


@router.get("/datasets/{dataset_id}")
def show(dataset_id: uuid.UUID, _: str = Depends(auth.require_service)):
    with store.session() as db:
        return _view(_require(db, dataset_id))


@router.get("/datasets/{dataset_id}/documents")
def documents(
    dataset_id: uuid.UUID,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=MAX_PAGE),
    _: str = Depends(auth.require_service),
):
    with store.session() as db:
        dataset = _require(db, dataset_id)
        rows = q.documents(db, dataset_id, offset=offset, limit=limit)
        return {
            "documents": [_document_view(row) for row in rows],
            "offset": offset,
            "total": dataset.document_count,
        }


@router.post("/datasets/pull", status_code=202)
def pull(body: PullRequest, owner: uuid.UUID = Depends(auth.require_owner)):
    """Create the dataset row, then queue the run that fills it.

    The row first so the page has something to show as `pulling` — a queued run
    with nothing attached to it is invisible until it starts.
    """
    name = _validate_name(body.name)
    with store.session() as db:
        _claim_name(db, name)
        row = q.create_dataset(
            db,
            name=name,
            kind=body.kind,
            origin=DatasetOrigin.huggingface,
            owner_id=owner,
            description=body.description,
            source_repo=body.repo,
            source_config=body.config,
            source_split=body.split,
        )
        dataset_id = row.id

    run_id = RUNNER.enqueue(
        RunKind.pull,
        label=f"pull {body.repo} → {name}",
        owner_id=owner,
        ref_id=dataset_id,
        config={
            "dataset_id": str(dataset_id),
            "repo": body.repo,
            "kind": body.kind.value,
            "prefix": body.prefix,
            "column": body.column,
            "limit_rows": body.limit_rows,
        },
    )
    return {"dataset_id": str(dataset_id), "run_id": str(run_id)}


@router.post("/datasets/{dataset_id}/process", status_code=202)
def process(
    dataset_id: uuid.UUID, body: ProcessRequest, owner: uuid.UUID = Depends(auth.require_owner)
):
    """Queue a pipeline over a dataset, into a new one. The source is untouched."""
    name = _validate_name(body.name)
    with store.session() as db:
        source = _require(db, dataset_id)
        if source.status != DatasetStatus.ready:
            raise HTTPException(status_code=409, detail=f"{source.name} is {source.status.value}")
        _claim_name(db, name)
        steps = _resolved_steps(body.steps, source.kind)
        target = q.create_dataset(
            db,
            name=name,
            kind=source.kind,
            origin=DatasetOrigin.derived,
            owner_id=owner,
            description=body.description,
            source_repo=source.source_repo,
            source_config=source.source_config,
            source_split=source.source_split,
            parent_id=source.id,
            steps=steps,
        )
        target_id, label = target.id, f"process {source.name} → {name}"

    run_id = RUNNER.enqueue(
        RunKind.process,
        label=label,
        owner_id=owner,
        ref_id=target_id,
        config={"source_id": str(dataset_id), "target_id": str(target_id), "steps": steps},
    )
    return {"dataset_id": str(target_id), "run_id": str(run_id)}


def _resolved_steps(steps: list[dict], kind: DatasetKind) -> list[dict]:
    """The pipeline as it will actually run, with defaults filled in.

    Resolved here rather than in the worker so a bad step is a 422 the user sees
    while the form is still open, and so the provenance recorded on the dataset
    is the parameters used and not the parameters typed.
    """
    if not steps:
        raise HTTPException(
            status_code=422, detail="a pipeline needs at least one step"
        )

    resolved = []
    registry = processors.REGISTRY.snapshot()
    for step in steps:
        processor = registry.get(str(step.get("processor")))
        if processor is None:
            raise HTTPException(
                status_code=422, detail=f"unknown processor: {step.get('processor')!r}"
            )
        if kind not in processor.applies_to:
            raise HTTPException(
                status_code=422,
                detail=f"{processor.name} does not apply to a {kind.value} dataset",
            )
        resolved.append(
            {
                "processor": processor.name,
                "params": processor.resolve(step.get("params")),
                "version_hash": processor.version_hash,
            }
        )
    return resolved


@router.delete("/datasets/{dataset_id}", status_code=204)
def destroy(dataset_id: uuid.UUID, _: uuid.UUID = Depends(auth.require_owner)):
    with store.session() as db:
        if not q.delete_dataset(db, dataset_id):
            raise HTTPException(status_code=404, detail="no such dataset")


@router.get("/datasets/{dataset_id}/export")
def export(dataset_id: uuid.UUID, _: str = Depends(auth.require_service)):
    """The whole dataset as JSON lines.

    Streamed, not assembled: this is the one route whose response is the size of
    the corpus. It is also the seam the LLM engine will one day read instead of
    scanning a directory — built and exercised now, wired to nothing yet.
    """
    with store.session() as db:
        dataset = _require(db, dataset_id)
        name, kind = dataset.name, dataset.kind.value

    async def lines() -> AsyncIterator[str]:
        with store.session() as db:
            for row in q.stream_documents(db, dataset_id):
                payload = (
                    {"text": row.content}
                    if row.content is not None
                    else {"messages": row.messages}
                )
                yield json.dumps(payload, ensure_ascii=False) + "\n"

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{name}.jsonl"',
            "X-Dataset-Kind": kind,
        },
    )
