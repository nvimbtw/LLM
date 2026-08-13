"""The API layer. Parses a request, calls an orchestrator, returns its result.

Nothing else belongs here — no path building, no pipeline steps, no torch
import. If a route grows past a few lines, the logic belongs in an orchestrator
or in `api/`.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.responses import StreamingResponse

import store
from api import auth, sse, ws
from api.handlers import HANDLERS
from api.runner import Runner
from api.runs import REGISTRY
from store import queries as q
from store.schema import RunKind
from llm.config import (
    ChatConfig,
    CountRequest,
    EvalConfig,
    SftConfig,
    TrainConfig,
)
from llm.orchestrate import chat as chat_pipeline
from llm.orchestrate import evaluate as eval_pipeline
from llm.orchestrate import sft as sft_pipeline
from llm.orchestrate import train as train_pipeline


RUNNER = Runner(REGISTRY, HANDLERS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Worker threads publish progress onto this loop; see Registry.bind_loop.
    REGISTRY.bind_loop(asyncio.get_running_loop())
    store.configure()

    # Runs still marked live belong to a process that is gone. Queued rows are
    # deliberately untouched — draining them is the point of a durable queue.
    with store.session() as db:
        stranded = q.fail_orphaned(db)
    if stranded:
        print(f"engine: failed {stranded} run(s) stranded by a restart")

    if not auth.enforced():
        print("engine: ENGINE_SERVICE_SECRET unset — accepting unauthenticated callers")

    drain = asyncio.create_task(RUNNER.drain_forever())
    try:
        yield
    finally:
        drain.cancel()


# Applied to the whole app rather than per route: a new endpoint is
# authenticated because it exists, not because someone remembered to decorate
# it. A WebSocket upgrade is an HTTP request, so the socket routes are covered
# by the same check — an unsigned handshake is refused before the socket exists.
app = FastAPI(
    title="LLM engine",
    lifespan=lifespan,
    dependencies=[Depends(auth.require_service)],
    # FastAPI serves /docs, /redoc and /openapi.json itself, and its own routes
    # bypass the dependency above — so the whole API surface was enumerable by
    # anything that could reach the port. The engine has one client and it calls
    # typed methods; the schema is still reachable in-process via app.openapi()
    # for anything that needs to generate against it.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/")
def health():
    active = REGISTRY.active()
    return {"status": "ok", "active_run": active.id if active else None}


@app.post("/train", status_code=202)
async def train(cfg: TrainConfig, wait: bool = False):
    """Queue a pretraining run.

    Queued rather than started: there is one device and one line, and a caller
    that gets a run id back can follow it whether it starts now or in an hour.
    """
    return await _queue(RunKind.train, cfg.model_name, cfg.model_dump(), wait)


@app.post("/sft", status_code=202)
async def sft(cfg: SftConfig, wait: bool = False):
    """Queue a fine-tune. The same kind of run as /train — what makes it a
    fine-tune is naming a base model to start from."""
    return await _queue(RunKind.train, cfg.model_name, cfg.model_dump(), wait)


@app.post("/eval", status_code=202)
async def evaluate(cfg: EvalConfig, wait: bool = False):
    """Queue a benchmark."""
    return await _queue(RunKind.benchmark, cfg.model_name, cfg.model_dump(), wait)


async def _queue(kind: RunKind, label: str, config: dict, wait: bool) -> dict:
    run_id = RUNNER.enqueue(kind, label=label, config=config)
    if wait:
        await RUNNER.wait(run_id)
    return _run_snapshot(run_id)


@app.get("/runs")
def list_runs(limit: int = 50, offset: int = 0):
    """The live band and the history behind it.

    Two lists rather than one: the band is short and complete, the history is
    long and paged, and a client renders them differently.
    """
    with store.session() as db:
        return {
            "active": [_row(r) for r in q.active_runs(db)],
            "history": [_row(r) for r in q.run_history(db, limit, offset)],
        }


@app.get("/runs/{run_id}")
def get_run(run_id: uuid.UUID):
    return _run_snapshot(run_id)


@app.get("/runs/{run_id}/events")
async def run_events(run_id: uuid.UUID):
    """Stream a run's progress as SSE, replaying what it has already emitted.

    A queued run has no frames yet and no live state; it gets the durable row
    and nothing else until the runner reaches it.
    """
    live = REGISTRY.get(str(run_id))
    if live is None:
        _require_row(run_id)
        return StreamingResponse(
            sse.once("state", _run_snapshot(run_id)),
            media_type=sse.MEDIA_TYPE,
            headers=sse.HEADERS,
        )
    return StreamingResponse(
        sse.run_events(REGISTRY, live), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS
    )


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: uuid.UUID):
    """Ask a run to stop.

    A queued run stops at once. A running one stops at its next step boundary,
    where the weights are consistent — so this returns while it is still
    running, and the caller learns it stopped from the event stream.
    """
    _require_row(run_id)
    if not RUNNER.cancel(run_id):
        raise HTTPException(status_code=409, detail="run is already finished")
    return {"run_id": str(run_id), "cancelling": True}


@app.patch("/runs/{run_id}")
def reorder_run(run_id: uuid.UUID, after: uuid.UUID | None = None):
    """Move a queued run within the line, to sit after another or at the front."""
    with store.session() as db:
        if not q.reorder(db, run_id, after):
            raise HTTPException(status_code=409, detail="only a queued run can be reordered")
    return _run_snapshot(run_id)


@app.post("/runs/pause")
def set_pause(paused: bool = True):
    """Stop the runner picking up new work. Survives a restart, because the
    point of pausing is to leave the machine alone."""
    with store.session() as db:
        q.set_paused(db, paused)
    if not paused:
        RUNNER.wake()
    return {"paused": paused}


@app.post("/chat")
def chat(cfg: ChatConfig):
    """Generate the next assistant turn for a conversation."""
    try:
        return chat_pipeline.run(cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(cfg: ChatConfig):
    """The same turn, streamed token by token as SSE."""

    def advance(generator):
        try:
            return next(generator), None
        except StopIteration as done:
            return None, done.value or {}

    async def frames():
        generator = chat_pipeline.stream(cfg)
        while True:
            try:
                piece, summary = await asyncio.to_thread(advance, generator)
            except (FileNotFoundError, ValueError) as e:
                yield sse.token_frame("error", {"error": str(e)})
                return
            if summary is not None:
                yield sse.token_frame("done", summary)
                return
            yield sse.token_frame("token", piece)

    return StreamingResponse(frames(), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS)


@app.get("/datasets")
def datasets():
    """What corpora are on disk, for a caller composing a curriculum.

    The engine owns the data outright — these folders are filled by the fetch
    scripts, not uploaded through the API — so this is a read-only view.
    """
    from llm.corpus import sources

    return {"datasets": sources.catalogue()}


@app.get("/sft/datasets")
def sft_datasets():
    """What SFT sets are on disk, for a caller choosing which to fine-tune on.

    Read-only for the same reason `/datasets` is — the fetch scripts fill these
    folders, nothing is uploaded through the API. File count and bytes rather
    than a conversation count: the latter costs a full pass over hundreds of
    megabytes of JSONL per request, and tells sets apart no better.
    """
    from llm.corpus import sft as sft_corpus

    return {"datasets": sft_corpus.catalogue()}


@app.get("/models")
def models():
    """Trained models on disk, with the architecture each was built with."""
    import json as _json

    from llm import storage

    out = []
    for d in sorted(p for p in storage.models_dir().glob("*") if p.is_dir()):
        meta = d / "meta.json"
        entry = {"name": d.name, "has_weights": (d / "model.pt").is_file()}
        if meta.is_file():
            try:
                entry["config"] = _json.loads(meta.read_text()).get("config", {})
            except ValueError:
                pass
        out.append(entry)
    return {"models": out}


@app.post("/tokenize/count")
def count_tokens(req: CountRequest):
    """Token length of `text` under a model's tokenizer. Loads no weights."""
    try:
        return chat_pipeline.count_tokens(req.model_name, req.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _row(run: store.Run) -> dict:
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
        "result": run.result,
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def _require_row(run_id: uuid.UUID) -> dict:
    with store.session() as db:
        run = db.get(store.Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id}")
        return _row(run)


def _run_snapshot(run_id: uuid.UUID) -> dict:
    """The row, plus how many frames the live run has emitted if it is live."""
    snapshot = _require_row(run_id)
    live = REGISTRY.get(str(run_id))
    snapshot["frames"] = len(live.history) if live else 0
    return snapshot
