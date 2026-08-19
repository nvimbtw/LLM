"""The queue: what is running, what it is doing, and how to stop it."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

import store
from api import auth, sse
from api.runner import RUNNER
from api.runs import REGISTRY
from store import queries as q
from store.schema import Run

router = APIRouter(tags=["runs"])

DEFAULT_LIMIT = 50


def _view(row: Run) -> dict:
    return {
        "id": str(row.id),
        "kind": row.kind.value,
        "status": row.status.value,
        "label": row.label,
        "progress": row.progress,
        "stage": row.stage,
        "ref_id": str(row.ref_id) if row.ref_id else None,
        "result": row.result,
        "error": row.error,
        "created_at": row.created_at.isoformat(),
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }


@router.get("/runs")
def index(limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=200), _: str = Depends(auth.require_service)):
    with store.session() as db:
        return {"runs": [_view(row) for row in q.list_runs(db, limit)]}


@router.get("/runs/{run_id}")
def show(run_id: uuid.UUID, _: str = Depends(auth.require_service)):
    with store.session() as db:
        row = q.run(db, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="no such run")
        return _view(row)


@router.post("/runs/{run_id}/cancel")
def cancel(run_id: uuid.UUID, _: str = Depends(auth.require_service)):
    if not RUNNER.cancel(run_id):
        raise HTTPException(status_code=409, detail="this run is already over")
    return {"cancelling": str(run_id)}


@router.get("/runs/{run_id}/events")
def events(run_id: uuid.UUID, _: str = Depends(auth.require_service)):
    """Every frame of a run, replayed from the start.

    A run with no live state still answers, with one frame saying where it got
    to: queued and not yet started, or finished in a previous process. A caller
    that attached late must not be left holding an open stream that says nothing.
    """
    with store.session() as db:
        row = q.run(db, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="no such run")
        settled = _view(row)

    live = REGISTRY.get(str(run_id))
    if live is None:
        kind = "done" if settled["status"] == "done" else "error" if settled["status"] == "failed" else "progress"
        stream = sse.once(kind, settled)
    else:
        stream = sse.run_events(REGISTRY, live)

    return StreamingResponse(stream, media_type=sse.MEDIA_TYPE, headers=sse.HEADERS)
