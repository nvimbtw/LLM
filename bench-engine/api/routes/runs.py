import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import store
from api import auth, sse
from api.runner import Runner
from api.runs import REGISTRY
from store import queries as q
from store.schema import RunKind
from pydantic import BaseModel
router = APIRouter(tags=["runs"])
class EvalRequest(BaseModel):
    model_name: str
    suite: str = "language-v3"
    max_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    seed: int = 1234
@router.post("/eval", status_code=202)
async def evaluate(cfg: EvalRequest, owner: str = Depends(auth.require_service)):
    if cfg.suite not in ("language-v3", "language"):
        raise HTTPException(status_code=400, detail="only language-v3 supported")
    from api.handlers import RUNNER
    config = cfg.model_dump()
    run_id = RUNNER.enqueue(RunKind.benchmark, label=cfg.model_name, config=config, owner_id=_owner_id(owner))
    return snapshot(run_id)
@router.get("/runs")
def list_runs(limit: int = 50, offset: int = 0):
    with store.session() as db:
        return {"active": [row(r) for r in q.active_runs(db)], "history": [row(r) for r in q.run_history(db, limit, offset)], "history_offset": offset, "history_limit": limit, "history_total": q.count_history(db), "paused": q.is_paused(db)}
@router.get("/runs/{run_id}")
def get_run(run_id: uuid.UUID):
    return snapshot(run_id)
@router.get("/runs/{run_id}/events")
async def run_events(run_id: uuid.UUID):
    live = REGISTRY.get(str(run_id))
    if live is None:
        require_row(run_id)
        return StreamingResponse(sse.once("state", snapshot(run_id)), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS)
    return StreamingResponse(sse.run_events(REGISTRY, live), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS)
@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: uuid.UUID):
    require_row(run_id)
    from api.handlers import RUNNER
    if not RUNNER.cancel(run_id):
        raise HTTPException(status_code=409, detail="run is already finished")
    return {"run_id": str(run_id), "cancelling": True}
@router.delete("/runs/{run_id}")
def delete_run(run_id: uuid.UUID):
    require_row(run_id)
    with store.session() as db:
        if not q.delete_run(db, run_id):
            raise HTTPException(status_code=409, detail="a live run is cancelled, not deleted")
    return {"run_id": str(run_id), "deleted": True}
@router.patch("/runs/{run_id}")
def reorder_run(run_id: uuid.UUID, after: uuid.UUID | None = None):
    with store.session() as db:
        if not q.reorder(db, run_id, after):
            raise HTTPException(status_code=409, detail="only a queued run can be reordered")
    return snapshot(run_id)
@router.post("/runs/pause")
def set_pause(paused: bool = True):
    with store.session() as db:
        q.set_paused(db, paused)
    if not paused:
        from api.handlers import RUNNER
        RUNNER.wake()
    return {"paused": paused}
def _owner_id(caller: str):
    try:
        return uuid.UUID(caller)
    except ValueError:
        return None
def row(run):
    return {"run_id": str(run.id), "kind": run.kind.value, "status": run.status.value, "label": run.label, "rank": run.rank, "progress": run.progress, "stage": run.stage, "ref_id": str(run.ref_id) if run.ref_id else None, "depends_on": str(run.depends_on) if run.depends_on else None, "result": run.result, "error": run.error, "config": run.config, "created_at": run.created_at.isoformat() if run.created_at else None, "started_at": run.started_at.isoformat() if run.started_at else None, "finished_at": run.finished_at.isoformat() if run.finished_at else None}
def require_row(run_id: uuid.UUID):
    with store.session() as db:
        run = db.get(store.Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id}")
        return row(run)
def snapshot(run_id: uuid.UUID):
    seen = require_row(run_id)
    live = REGISTRY.get(str(run_id))
    seen["frames"] = len(live.history) if live else 0
    return seen
