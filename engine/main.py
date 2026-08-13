"""The API layer. Parses a request, calls an orchestrator, returns its result.

Nothing else belongs here — no path building, no pipeline steps, no torch
import. If a route grows past a few lines, the logic belongs in an orchestrator
or in `llm.server`.
"""

import asyncio

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import StreamingResponse

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
from llm.server import sse, ws
from llm.server.runs import REGISTRY, EngineBusy

app = FastAPI(title="LLM engine")


@app.on_event("startup")
async def _bind_loop():
    # Worker threads publish progress onto this loop; see Registry.bind_loop.
    REGISTRY.bind_loop(asyncio.get_running_loop())


@app.get("/")
def health():
    active = REGISTRY.active()
    return {"status": "ok", "active_run": active.id if active else None}


@app.post("/train", status_code=202)
async def train(cfg: TrainConfig, wait: bool = False):
    """Start a training run. Returns immediately with a run id.

    `wait=true` blocks until the run finishes and returns its result instead —
    for curl and tests, not for the job queue, which wants the event stream.
    """
    try:
        run = REGISTRY.start(
            "train",
            cfg.model_dump(),
            lambda on_progress, cancel: train_pipeline.run(cfg, on_progress, cancel),
        )
    except EngineBusy as e:
        raise HTTPException(status_code=409, detail=str(e))

    if wait:
        await REGISTRY.wait(run)
    return run.snapshot()


@app.websocket("/train")
async def train_socket(websocket: WebSocket):
    """The same run as `POST /train`, start to finish over one socket.

    Send `{"config": {...}}`; receive a `started` frame carrying the run id,
    then every progress frame as it happens, then one of done/cancelled/error.
    Send `{"type": "cancel"}` at any point to stop it.

    It is the same registry either way, so a run started here is listed at
    /runs and can still be followed or cancelled over HTTP.
    """
    await ws.serve_run(
        websocket,
        REGISTRY,
        "train",
        TrainConfig,
        lambda cfg: lambda on_progress, cancel: train_pipeline.run(cfg, on_progress, cancel),
    )


@app.post("/sft", status_code=202)
async def sft(cfg: SftConfig, wait: bool = False):
    """Fine-tune a pretrained model on chat conversations, with loss on the
    assistant replies only. Same run semantics as /train."""
    try:
        run = REGISTRY.start(
            "sft",
            cfg.model_dump(),
            lambda on_progress, cancel: sft_pipeline.run(cfg, on_progress, cancel),
        )
    except EngineBusy as e:
        raise HTTPException(status_code=409, detail=str(e))

    if wait:
        await REGISTRY.wait(run)
    return run.snapshot()


@app.post("/eval", status_code=202)
async def evaluate(cfg: EvalConfig, wait: bool = False):
    """Benchmark a model. Same run semantics as /train — it holds the device."""
    try:
        run = REGISTRY.start(
            "eval",
            cfg.model_dump(),
            lambda on_progress, cancel: eval_pipeline.run(cfg, on_progress, cancel),
        )
    except EngineBusy as e:
        raise HTTPException(status_code=409, detail=str(e))

    if wait:
        await REGISTRY.wait(run)
    return run.snapshot()


@app.get("/runs")
def list_runs():
    return REGISTRY.list()


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    return _require(run_id).snapshot()


@app.get("/runs/{run_id}/events")
async def run_events(run_id: str):
    """Stream a run's progress as SSE, replaying what it has already emitted."""
    run = _require(run_id)
    return StreamingResponse(
        sse.run_events(REGISTRY, run), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS
    )


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    """Ask a run to stop at its next step boundary, where weights are consistent.

    Returns immediately — the run is still 'running' until it reaches that
    boundary, and the caller learns it stopped from the event stream.
    """
    run = _require(run_id)
    if run.is_terminal:
        raise HTTPException(status_code=409, detail=f"run is already {run.status}")
    run.cancel.set()
    return {"run_id": run.id, "cancelling": True}


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


def _require(run_id: str):
    run = REGISTRY.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    return run
