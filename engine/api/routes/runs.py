"""The queue: what is waiting, what is running, and what it did.

Job routes return a run id rather than a result. There is one device and one
line, and a caller holding an id can follow the run whether it starts now or in
an hour — so a busy engine is not an error.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import store
from api import auth, sse
from api.runner import RUNNER
from api.runs import REGISTRY
from llm.config import EvalConfig, SftConfig, TrainConfig
from store import queries as q
from store.schema import Run, RunKind

router = APIRouter(tags=["runs"])


@router.post("/train", status_code=202)
async def train(
    cfg: TrainConfig,
    display_name: str | None = None,
    wait: bool = False,
    owner: str = Depends(auth.require_service),
):
    """Queue a pretraining run."""
    config = {**cfg.model_dump(), "display_name": display_name or cfg.model_name}
    return await _queue(RunKind.train, config["display_name"], config, wait, owner)


@router.post("/sft", status_code=202)
async def sft(
    cfg: SftConfig,
    display_name: str | None = None,
    wait: bool = False,
    owner: str = Depends(auth.require_service),
):
    """Queue a fine-tune. The same kind of run as /train — what makes it a
    fine-tune is naming a base model to start from."""
    config = {**cfg.model_dump(), "display_name": display_name or cfg.model_name}
    return await _queue(RunKind.train, config["display_name"], config, wait, owner)


@router.post("/eval", status_code=202)
async def evaluate(cfg: EvalConfig, wait: bool = False, owner: str = Depends(auth.require_service)):
    """Queue a benchmark."""
    return await _queue(RunKind.benchmark, _label(cfg.model_name), cfg.model_dump(), wait, owner)


class Step(BaseModel):
    kind: RunKind
    config: dict


@router.post("/runs/chain", status_code=202)
def queue_chain(steps: list[Step], owner: str = Depends(auth.require_service)):
    """Queue a sequence, each step running against the model the one before it
    produced.

    A flat list rather than a tree: "then, then, then" is what a person means,
    and a step naming its own model would be a second answer to the question the
    chain exists to leave open — so the first config names a model and the rest
    have theirs filled in when their predecessor lands.
    """
    if not steps:
        raise HTTPException(status_code=400, detail="a chain needs at least one step")

    queued, after = [], None
    for step in steps:
        run_id = RUNNER.enqueue(
            step.kind,
            label=_step_label(step),
            config=step.config,
            owner_id=_owner_id(owner),
            depends_on=after,
        )
        queued.append(snapshot(run_id))
        after = run_id
    return queued


@router.get("/runs")
def list_runs(limit: int = 50, offset: int = 0):
    """The live band and the history behind it.

    Two lists rather than one: the band is short and complete, the history is
    long and paged, and a client renders them differently.
    """
    with store.session() as db:
        return {
            "active": [row(r) for r in q.active_runs(db)],
            "history": [row(r) for r in q.run_history(db, limit, offset)],
            "history_offset": offset,
            "history_limit": limit,
            "history_total": q.count_history(db),
            "paused": q.is_paused(db),
        }


@router.get("/runs/{run_id}")
def get_run(run_id: uuid.UUID):
    return snapshot(run_id)


@router.get("/runs/{run_id}/events")
async def run_events(run_id: uuid.UUID):
    """Stream a run's progress as SSE, replaying what it has already emitted.

    A queued run has no frames yet and no live state; it gets the durable row
    and nothing else until the runner reaches it.
    """
    live = REGISTRY.get(str(run_id))
    if live is None:
        require_row(run_id)
        return StreamingResponse(
            sse.once("state", snapshot(run_id)), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS
        )
    return StreamingResponse(
        sse.run_events(REGISTRY, live), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS
    )


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: uuid.UUID):
    """Ask a run to stop.

    A queued run stops at once. A running one stops at its next step boundary,
    where the weights are consistent — so this returns while it is still
    running, and the caller learns it stopped from the event stream.
    """
    require_row(run_id)
    if not RUNNER.cancel(run_id):
        raise HTTPException(status_code=409, detail="run is already finished")
    return {"run_id": str(run_id), "cancelling": True}


@router.delete("/runs/{run_id}")
def delete_run(run_id: uuid.UUID):
    """Forget a finished run. A live one is cancelled first."""
    require_row(run_id)
    with store.session() as db:
        if not q.delete_run(db, run_id):
            raise HTTPException(status_code=409, detail="a live run is cancelled, not deleted")
    return {"run_id": str(run_id), "deleted": True}


@router.patch("/runs/{run_id}")
def reorder_run(run_id: uuid.UUID, after: uuid.UUID | None = None):
    """Move a queued run within the line, to sit after another or at the front."""
    with store.session() as db:
        if not q.reorder(db, run_id, after):
            raise HTTPException(status_code=409, detail="only a queued run can be reordered")
    return snapshot(run_id)


@router.post("/runs/pause")
def set_pause(paused: bool = True):
    """Stop the runner picking up new work. Survives a restart, because the
    point of pausing is to leave the machine alone."""
    with store.session() as db:
        q.set_paused(db, paused)
    if not paused:
        RUNNER.wake()
    return {"paused": paused}


def _step_label(step: "Step") -> str:
    if step.kind is RunKind.benchmark:
        return _label(str(step.config.get("model_name") or ""))
    return str(step.config.get("display_name") or step.config.get("model_name") or "train")


def _label(model_name: str) -> str:
    """What to call a run in a list. A model's id is its weights directory, so
    the label has to be looked up or the board reads as a column of uuids."""
    with store.session() as db:
        try:
            model = q.get_model(db, uuid.UUID(model_name))
        except ValueError:
            model = q.find_model_by_name(db, model_name)
        return model.name if model else model_name


async def _queue(kind: RunKind, label: str, config: dict, wait: bool, owner: str) -> dict:
    run_id = RUNNER.enqueue(kind, label=label, config=config, owner_id=_owner_id(owner))
    if wait:
        await RUNNER.wait(run_id)
    return snapshot(run_id)


def _owner_id(caller: str) -> uuid.UUID | None:
    """Who asked for this run, when anyone in particular did. Training is often
    the backend acting as itself, which owns nothing."""
    try:
        return uuid.UUID(caller)
    except ValueError:
        return None


def row(run: Run) -> dict:
    """One run as the API reports it. The durable row is the whole answer for
    everything except a run that is live right now."""
    return {
        "run_id": str(run.id),
        "kind": run.kind.value,
        "status": run.status.value,
        "label": run.label,
        "rank": run.rank,
        "progress": run.progress,
        "stage": run.stage,
        "ref_id": str(run.ref_id) if run.ref_id else None,
        "depends_on": str(run.depends_on) if run.depends_on else None,
        "result": run.result,
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def require_row(run_id: uuid.UUID) -> dict:
    with store.session() as db:
        run = db.get(store.Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id}")
        return row(run)


def snapshot(run_id: uuid.UUID) -> dict:
    """The row, plus how many frames the live run has emitted if it is live."""
    seen = require_row(run_id)
    live = REGISTRY.get(str(run_id))
    seen["frames"] = len(live.history) if live else 0
    return seen
