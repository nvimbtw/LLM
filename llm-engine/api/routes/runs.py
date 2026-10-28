"""The queue: what is waiting, what is running, and what it did.

Job routes return a run id rather than a result. There is one device and one
line, and a caller holding an id can follow the run whether it starts now or in
an hour — so a busy LLM engine is not an error.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

import store
from api import auth, sse
from api import training_jobs, validation_jobs
from api.runner import RUNNER
from api.runs import REGISTRY
from api.visibility import public_run, visible_events
from llm.config import LEGACY_STRUCTURE_FIELDS, EvalConfig, RlConfig, SftConfig, TokenizerConfig, TrainConfig
from llm.architecture.validation_records import ValidationSnapshot
from store import architecture_queries as catalog, queries as q
from store.schema import Run, RunKind

router = APIRouter(tags=["runs"])

MAX_CHAIN_STEPS = 100


@router.post("/train", status_code=202)
async def train(
    cfg: TrainConfig,
    display_name: str | None = None,
    wait: bool = False,
    request_id: uuid.UUID | None = None,
    owner: str = Depends(auth.require_service),
    administration: bool = Depends(auth.administration_visible),
):
    """Queue a pretraining run.

    Stored as it was *stated*, not as it validates. A run is queued now and
    started later, so this dict is the whole of what the pipeline is handed —
    and `resumed_architecture` decides what a resume inherits by asking pydantic
    which fields the request set. Dumping the defaults in alongside made every
    field "set" by the time `handlers.train` rebuilt the config, so a resume
    from any checkpoint that was not the composer's default shape was refused
    for restating an architecture nobody had typed.
    """
    payload = {"config": cfg.model_dump(mode="json", exclude_unset=True), "display_name": display_name}
    with store.session() as db:
        definition_id = training_jobs.saved_definition(db, cfg.resume_from)
    if cfg.architecture is not None or definition_id is not None:
        _require_recipe_request(request_id, owner, administration)
    if definition_id is not None and cfg.model_fields_set & LEGACY_STRUCTURE_FIELDS:
        raise HTTPException(422, "saved recipe training inherits its structural definition")
    config = _identified(payload["config"], display_name)
    return await _queue(RunKind.train, _train_label(config), config, wait, owner,
                        request_id=request_id, payload=payload, definition_id=definition_id, administration=administration)


@router.post("/sft", status_code=202)
async def sft(
    cfg: SftConfig,
    display_name: str | None = None,
    wait: bool = False,
    request_id: uuid.UUID | None = None,
    owner: str = Depends(auth.require_service),
    administration: bool = Depends(auth.administration_visible),
):
    """Queue SFT with immutable saved-definition authority and actor/request checks for modular source execution."""
    payload = {"config": cfg.model_dump(mode="json", exclude_unset=True), "display_name": display_name}
    with store.session() as db:
        definition_id = training_jobs.saved_definition(db, cfg.resume_from or cfg.base_model)
    if definition_id is not None:
        _require_recipe_request(request_id, owner, administration)
    config = _identified(payload["config"], display_name)
    return await _queue(RunKind.train, _train_label(config), config, wait, owner, request_id=request_id,
        payload=payload, definition_id=definition_id, administration=administration)


@router.post("/rl", status_code=202)
async def rl(
    cfg: RlConfig,
    display_name: str | None = None,
    wait: bool = False,
    request_id: uuid.UUID | None = None,
    owner: str = Depends(auth.require_service),
    administration: bool = Depends(auth.administration_visible),
):
    """Queue RL with sparse original configuration, inherited saved-definition authority and modular actor/request authorization."""
    payload = {"config": cfg.model_dump(mode="json", exclude_unset=True), "display_name": display_name}
    with store.session() as db:
        definition_id = training_jobs.saved_definition(db, cfg.resume_from or cfg.base_model)
    if definition_id is not None:
        _require_recipe_request(request_id, owner, administration)
    config = _identified(payload["config"], display_name)
    return await _queue(RunKind.rl, _train_label(config), config, wait, owner, request_id=request_id,
        payload=payload, definition_id=definition_id, administration=administration)


@router.post("/tokenize", status_code=202)
async def tokenize(
    cfg: TokenizerConfig,
    display_name: str | None = None,
    wait: bool = False,
    owner: str = Depends(auth.require_service),
):
    """Queue a vocabulary build: one reusable tokenizer, no packed corpus.

    The corpus is packed by the run that trains on it, keyed by this vocabulary
    and that run's own selection — so one build serves every run that names it,
    whatever each of them trains on.
    """
    config = _identified(cfg.model_dump(), display_name, field="tokenizer_id")
    return await _queue(RunKind.tokenize, _train_label(config, "tokenizer_id"), config, wait, owner)


@router.post("/eval", status_code=202)
async def evaluate(cfg: EvalConfig, wait: bool = False, owner: str = Depends(auth.require_service)):
    """Queue a benchmark."""
    return await _queue(RunKind.benchmark, _label(cfg.model_name), cfg.model_dump(), wait, owner)


class Step(BaseModel):
    kind: RunKind
    config: dict


@router.post("/runs/chain", status_code=202)
def queue_chain(steps: Annotated[list[Step], Body(max_length=MAX_CHAIN_STEPS)],
                request_id: uuid.UUID | None = None,
                owner: str = Depends(auth.require_service),
                administration: bool = Depends(auth.administration_visible)) -> list[dict]:
    """Validate every step, then commit the complete bounded chain before waking admission.

    Later training steps fine-tune their predecessor's eventual model. Benchmark
    dispatch cannot precede another step because its output is a report. Invalid
    configuration or a late storage failure leaves no jobs from this submission.
    """
    if not steps:
        raise HTTPException(status_code=400, detail="a chain needs at least one step")
    if any(step.kind is RunKind.model_validation for step in steps):
        raise HTTPException(
            status_code=400,
            detail="model validation does not produce a model and cannot be chained",
        )
    if any(step.kind is RunKind.tokenize for step in steps):
        raise HTTPException(
            status_code=400,
            detail="a tokenize run cannot be chained — build the vocabulary first, then "
            "name it on the runs that use it",
        )

    if any(step.kind is RunKind.benchmark for step in steps[:-1]):
        raise HTTPException(400, "a benchmark produces a report and must be the final chain step")
    with store.session() as db:
        definition_id = (training_jobs.saved_definition(db, steps[0].config.get("resume_from") or steps[0].config.get("base_model"))
                         if steps[0].kind in (RunKind.train, RunKind.rl) else None)
    if any(step.config.get("architecture") is not None for step in steps) or definition_id is not None:
        _require_recipe_request(request_id, owner, administration)
    actor = _owner_id(owner)
    if request_id is not None and actor is None:
        raise HTTPException(403, "idempotent training submission requires an actor UUID")
    with store.session() as db:
        def enqueue() -> catalog.MutationOutcome:
            """Prepare and insert the full chain while this request owns its one queue transaction."""
            return _enqueue_chain(db, steps, actor, definition_id=definition_id)
        outcome = (catalog.idempotent_mutation(db, actor_id=actor, request_id=request_id,
            operation="training_chain", payload={"steps": [step.model_dump(mode="json") for step in steps]},
            execute=enqueue) if request_id is not None else enqueue())
        ids = outcome.job_ids
        if any(training_jobs.inherited_pin(db, run_id) is not None for run_id in ids) and not administration:
            raise HTTPException(403, "recipe training chain requires administrator access")
        if any(db.get(Run, run_id) is None for run_id in ids):
            raise catalog.CatalogError("operation_pruned", "original chain history was pruned; use a fresh request UUID")
    RUNNER.wake()
    return [snapshot(run_id) for run_id in ids]


def _enqueue_chain(db: Session, steps: list[Step], actor: uuid.UUID | None, *, definition_id: str | None = None) -> catalog.MutationOutcome:
    """Validate and freeze every waiting-step selection before creating any linked queue/domain rows."""
    configs = []
    for index, step in enumerate(steps):
        predecessor = configs[-1]["model_name"] if configs else None
        configs.append(_chain_config(step, index, predecessor))
    labels = [_step_label(step.kind, config) for step, config in zip(steps, configs)]
    if definition_id is not None and LEGACY_STRUCTURE_FIELDS.intersection(configs[0]):
        raise HTTPException(422, "saved recipe training inherits its structural definition")
    ids = tuple(uuid.uuid4() for _ in steps)
    pins = []
    for index, (step, config, job_id) in enumerate(zip(steps, configs, ids)):
        pin = training_jobs.prepare_selection(db, config, job_id) if config.get("architecture") is not None else None
        if index == 0 and pin is None and definition_id is not None:
            pin = catalog.prepare_saved_training_pin(db, definition_id)
        if index and step.kind in (RunKind.train, RunKind.rl):
            pin = pins[-1]
        pins.append(pin)
    for index, (step, config, label, job_id, pin) in enumerate(zip(steps, configs, labels, ids, pins)):
        run = q.enqueue(db, kind=step.kind, label=label, config=config, id=job_id,
                        owner_id=actor, depends_on=ids[index - 1] if index else None)
        if pin is not None:
            catalog.attach_training_pin(db, run, pin)
    return catalog.MutationOutcome("training", ids[0], ids)


def _chain_config(step: Step, index: int, predecessor: str | None) -> dict:
    """Validate one effective step without persisting defaults or a future model input.

    A predecessor's reserved output UUID satisfies validation for waiting
    consumers; promotion supplies its actual produced model. Errors retain the
    failing step and config field in the standard HTTP validation response.
    """
    config = _identified(step.config) if step.kind in (RunKind.train, RunKind.rl) else dict(step.config)
    effective = dict(config)
    if predecessor is not None:
        effective["model_name" if step.kind is RunKind.benchmark else "base_model"] = predecessor
    if step.kind is RunKind.train:
        model = SftConfig if effective.get("base_model") else TrainConfig
    else:
        model = RlConfig if step.kind is RunKind.rl else EvalConfig
    try:
        parsed = model.model_validate(effective)
    except ValidationError as error:
        errors = [{**item, "loc": ("body", index, "config", *item["loc"])} for item in error.errors()]
        raise RequestValidationError(errors) from error
    if isinstance(parsed, TrainConfig):
        _require_vocabulary(parsed.tokenizer_id)
    return config


@router.get("/runs")
def list_runs(
    limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0),
    administration: bool = Depends(auth.administration_visible),
):
    """The live band and the history behind it.

    Two lists rather than one: the band is short and complete, the history is
    long and paged, and a client renders them differently.
    """
    with store.session() as db:
        return {
            "active": [row(r, administration=administration) for r in q.active_runs(db, include_validation=administration)],
            "history": [row(r, administration=administration) for r in q.run_history(db, limit, offset, include_validation=administration)],
            "history_offset": offset,
            "history_limit": limit,
            "history_total": q.count_history(db, include_validation=administration),
            "paused": RUNNER.paused,
        }


@router.get("/runs/{run_id}")
def get_run(run_id: uuid.UUID, administration: bool = Depends(auth.administration_visible)):
    """Read an authorized job snapshot; validation identities are private before serialization."""
    return snapshot(run_id, administration=administration)


@router.get("/runs/{run_id}/events")
async def run_events(run_id: uuid.UUID, administration: bool = Depends(auth.administration_visible)):
    """Stream a run's progress as SSE, replaying what it has already emitted.

    A queued run has no frames yet and no live state; it gets the durable row
    and nothing else until the runner reaches it.
    """
    seen = snapshot(run_id, administration=administration)
    live = REGISTRY.get(str(run_id))
    if live is None:
        return StreamingResponse(
            sse.once("state", seen), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS
        )
    return StreamingResponse(
        _visible_frames(live, administration), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS
    )


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: uuid.UUID, administration: bool = Depends(auth.administration_visible)):
    """Ask a run to stop.

    A queued run stops at once. A running one stops at its next step boundary,
    where the weights are consistent — so this returns while it is still
    running, and the caller learns it stopped from the event stream.
    """
    require_row(run_id,administration=administration)
    if not RUNNER.cancel(run_id):
        raise HTTPException(status_code=409, detail="run is already finished")
    return {"run_id": str(run_id), "cancelling": True}


@router.post("/runs/{run_id}/retry", status_code=202)
async def retry_run(run_id: uuid.UUID, owner: str = Depends(auth.require_service),
                    administration: bool = Depends(auth.administration_visible),
                    request_id: uuid.UUID | None = None):
    """Retry a finished job, preserving validation pins while minting fresh request/job identities."""
    with store.session() as db:
        replay = _replayed_training(db, request_id, owner, administration, {"retry_of": str(run_id)})
        if replay is not None:
            return replay
        run = db.get(Run, run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        if run.kind == RunKind.model_validation and not administration:
            raise HTTPException(403,"validation requires administrator access")
        if not run.status.is_terminal:
            raise HTTPException(409, "only finished runs can retry")
        config, kind, label = dict(run.config), run.kind, run.label
        pin_from = run_id if training_jobs.inherited_pin(db, run_id) is not None else None
        if pin_from is not None and not administration:
            raise HTTPException(403, "recipe retry requires administrator access")
        if kind == RunKind.model_validation:
            actor = _owner_id(owner)
            if actor is None:
                raise HTTPException(403,"validation retry requires an actor UUID")
            retried = validation_jobs.retry_validation(db,run,actor,request_id)
            response = row(retried)
    if kind == RunKind.model_validation:
        RUNNER.wake()
        return response
    config.pop("execution_id", None)
    if kind == RunKind.benchmark:
        if config.get("suite", "language") != "language":
            raise HTTPException(409, "this historical suite is read-only")
    else:
        field = "tokenizer_id" if kind == RunKind.tokenize else "model_name"
        config[field] = str(uuid.uuid4())
    return await _queue(kind, label, config, False, owner, request_id=request_id,
                        payload={"retry_of": str(run_id)}, pin_from=pin_from, administration=administration)


@router.post("/runs/{run_id}/recover", status_code=202)
async def recover_run(run_id: uuid.UUID, owner: str = Depends(auth.require_service),
                      administration: bool = Depends(auth.administration_visible),
                      request_id: uuid.UUID | None = None):
    """Queue exact recovery from the last registered generation, retaining any original immutable recipe pin."""
    with store.session() as db:
        replay = _replayed_training(db, request_id, owner, administration, {"recover_of": str(run_id)})
        if replay is not None:
            return replay
        run = db.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        if run.kind not in (RunKind.train, RunKind.rl) or not run.status.is_terminal:
            raise HTTPException(status_code=409, detail="only finished training runs can recover")
        config = dict(run.config or {})
        pin_from = run_id if training_jobs.inherited_pin(db, run_id) is not None else None
        if pin_from is not None and not administration:
            raise HTTPException(403, "recipe recovery requires administrator access")
        try:
            output_id = uuid.UUID(config.get("model_name", ""))
        except ValueError:
            raise HTTPException(status_code=409, detail="legacy run has no recovery output ID")
        checkpoints = q.checkpoints_of(db, output_id)
        if not checkpoints:
            raise HTTPException(status_code=409, detail="no completed recovery point")
        latest = checkpoints[-1]
        config["resume_from"] = str(latest.id)
        if pin_from is not None:
            config.pop("architecture", None)
        kind = run.kind
        if kind == RunKind.train and not config.get("base_model"):
            config["recover_exact"] = True
    config["model_name"] = str(uuid.uuid4())
    config = _identified(config, config.get("display_name"))
    return await _queue(kind, _train_label(config), config, False, owner, request_id=request_id,
                        payload={"recover_of": str(run_id)}, pin_from=pin_from, administration=administration)


@router.delete("/runs/{run_id}")
def delete_run(run_id: uuid.UUID, administration: bool = Depends(auth.administration_visible)):
    """Forget a finished run. A live one is cancelled first."""
    require_row(run_id,administration=administration)
    with store.session() as db:
        if not q.delete_run(db, run_id):
            raise HTTPException(status_code=409, detail="a live run is cancelled, not deleted")
    return {"run_id": str(run_id), "deleted": True}


@router.patch("/runs/{run_id}")
def reorder_run(run_id: uuid.UUID, after: uuid.UUID | None = None,
                administration: bool = Depends(auth.administration_visible)):
    """Move a queued run within the line, to sit after another or at the front."""
    require_row(run_id,administration=administration)
    with store.session() as db:
        if not q.reorder(db, run_id, after):
            raise HTTPException(status_code=409, detail="only a queued run can be reordered")
    return snapshot(run_id,administration=administration)


@router.post("/runs/pause")
def set_pause(paused: bool = True):
    """Pause admission in this process; every engine restart starts running."""
    RUNNER.set_paused(paused)
    return {"paused": paused}


def _identified(config: dict, display_name: str | None = None, *, field: str = "model_name") -> dict:
    """Give a run the id its artifact will live under, and keep the name a person
    typed.

    `model_name` *is* the model id — it names the directory under
    `artifacts/models/` and the row keyed by it — while a composer sends what the
    model should be *called*. So an id is minted here unless the caller already
    named one, and the typed name travels as `display_name`. `tokenizer_id` is
    the same arrangement one directory over.

    Taking the typed name for an id is what failed at the very end of a run: the
    weights were saved, registering them raised `badly formed hexadecimal UUID
    string`, and a run that had finished (or been cancelled) was recorded as
    failed with no usable model to show for it.
    """
    named = str(config.get(field) or "").strip()
    try:
        artifact_id = str(uuid.UUID(named))
    except ValueError:
        artifact_id = str(uuid.uuid4())

    identified = {**config, field: artifact_id}
    name = display_name or named
    # Absent rather than empty: a step that was never named falls back to the id,
    # and an empty string would take that fallback away from everyone downstream.
    if name and name != artifact_id:
        identified["display_name"] = name
    return identified


def _require_vocabulary(tokenizer_id: str | None) -> None:
    """Refuse a run naming a vocabulary that does not exist.

    Here rather than in the handler: a typo would otherwise sit in the queue and
    fail hours later, at the front of the line, having waited for the device it
    never needed.
    """
    if not tokenizer_id:
        return
    from llm.corpus import vocab

    if vocab.info(tokenizer_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"no tokenizer {tokenizer_id} — build one, or leave it unset to learn "
            f"one from this run's own corpora",
        )


def _train_label(config: dict, field: str = "model_name") -> str:
    return str(config.get("display_name") or config.get(field) or "train")


def _step_label(kind: RunKind, config: dict) -> str:
    if kind is RunKind.benchmark:
        return _label(str(config.get("model_name") or ""))
    return _train_label(config)


def _label(model_name: str) -> str:
    """What to call a run in a list. A model's id is its weights directory, so
    the label has to be looked up or the board reads as a column of uuids."""
    with store.session() as db:
        try:
            model = q.get_model(db, uuid.UUID(model_name))
        except ValueError:
            model = q.find_model_by_name(db, model_name)
        return model.name if model else model_name


async def _queue(kind: RunKind, label: str, config: dict, wait: bool, owner: str, *,
                 request_id: uuid.UUID | None = None, payload: dict | None = None,
                 pin_from: uuid.UUID | None = None, definition_id: str | None = None,
                 administration: bool = False) -> dict:
    """Commit a reserved output and optional immutable recipe pin before waking or awaiting execution."""
    if request_id is None and pin_from is None and definition_id is None:
        if kind is RunKind.train and not config.get("base_model"):
            _require_vocabulary(config.get("tokenizer_id"))
        run_id = RUNNER.enqueue(kind, label=label, config=config, owner_id=_owner_id(owner))
    else:
        actor = _owner_id(owner)
        if actor is None:
            raise HTTPException(403, "idempotent training submission requires an actor UUID")
        with store.session() as db:
            def enqueue() -> catalog.MutationOutcome:
                """Reserve one output and pin the accepted source/evidence under the request's transaction."""
                job_id = uuid.uuid4()
                pin = training_jobs.inherited_pin(db, pin_from) if pin_from is not None else None
                if pin_from is not None and pin is None:
                    raise catalog.CatalogError("operation_pruned", "original immutable training pin was pruned")
                if pin is None and config.get("architecture") is not None:
                    pin = training_jobs.prepare_selection(db, config, job_id)
                if pin is None and definition_id is not None:
                    pin = catalog.prepare_saved_training_pin(db, definition_id)
                if kind is RunKind.train and not config.get("base_model") and pin is None:
                    _require_vocabulary(config.get("tokenizer_id"))
                run = q.enqueue(db, kind=kind, label=label, config=config, owner_id=actor, id=job_id)
                if pin is not None:
                    catalog.attach_training_pin(db, run, pin)
                return catalog.MutationOutcome("training", run.id)
            outcome = (catalog.idempotent_mutation(db, actor_id=actor, request_id=request_id,
                operation="training", payload=payload or {"kind": kind.value, "config": config}, execute=enqueue)
                if request_id is not None else enqueue())
            run_id = outcome.id
            if training_jobs.inherited_pin(db, run_id) is not None and not administration:
                raise HTTPException(403, "recipe training requires administrator access")
            if db.get(Run, run_id) is None:
                raise catalog.CatalogError("operation_pruned", "original training history was pruned; use a fresh request UUID")
        RUNNER.wake()
    if wait:
        await RUNNER.wait(run_id)
    return snapshot(run_id)


def _require_recipe_request(request_id: uuid.UUID | None, owner: str, administration: bool) -> None:
    """Require verified administrative actor attribution and retry identity for new recipe selections."""
    if not administration or _owner_id(owner) is None:
        raise HTTPException(403, "recipe selection requires an administrative actor UUID")
    if request_id is None:
        raise HTTPException(422, "recipe submission requires a request_id UUID")


def _replayed_training(db: Session, request_id: uuid.UUID | None, owner: str,
                       administration: bool, payload: dict) -> dict | None:
    """Return a prior retry/recovery output before looking up its potentially pruned input job."""
    if request_id is None:
        return None
    actor = _owner_id(owner)
    if actor is None:
        raise HTTPException(403, "idempotent training submission requires an actor UUID")
    outcome = catalog.completed_mutation(db, actor_id=actor, request_id=request_id, operation="training", payload=payload)
    if outcome is None:
        return None
    if outcome.kind == "validation" and not administration:
        raise HTTPException(403, "validation retry requires administrator access")
    if db.get(Run, outcome.id) is None:
        raise catalog.CatalogError("operation_pruned", "original training history was pruned; use a fresh request UUID")
    if training_jobs.inherited_pin(db, outcome.id) is not None and not administration:
        raise HTTPException(403, "recipe retry/recovery requires administrator access")
    return snapshot(outcome.id)


def _owner_id(caller: str) -> uuid.UUID | None:
    """Who asked for this run, when anyone in particular did. Training is often
    the backend acting as itself, which owns nothing."""
    try:
        return uuid.UUID(caller)
    except ValueError:
        return None


def row(run: Run, *, administration: bool = True) -> dict:
    """One run as the API reports it. The durable row is the whole answer for
    everything except a run that is live right now."""
    config = run.config
    if run.kind == RunKind.model_validation and isinstance(config,dict) and "request" in config and "run_id" in config:
        config = ValidationSnapshot.model_validate(config).request.model_dump(mode="json")
    value = {
        "run_id": str(run.id),
        "kind": run.kind.value,
        "config": config,
        "status": run.status.value,
        "label": run.label,
        "rank": run.rank,
        "progress": run.progress,
        "stage": run.stage,
        "ref_id": str(run.ref_id) if run.ref_id else None,
        "output": run.output,
        "depends_on": str(run.depends_on) if run.depends_on else None,
        "result": run.result,
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }
    return value if administration else public_run(value)


def require_row(run_id: uuid.UUID, *, administration: bool = True) -> dict:
    """Look up a durable row and reject private validation before projecting any metadata."""
    with store.session() as db:
        run = db.get(store.Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id}")
        if run.kind == RunKind.model_validation and not administration:
            raise HTTPException(status_code=403, detail="validation requires administrator access")
        return row(run, administration=administration)


def snapshot(run_id: uuid.UUID, *, administration: bool = True) -> dict:
    """The row, plus how many frames the live run has emitted if it is live."""
    seen = require_row(run_id, administration=administration)
    live = REGISTRY.get(str(run_id))
    seen["frames"] = len(live.history) if live else 0
    return seen


async def _visible_frames(live, administration: bool):
    """Frame visibility-filtered replay and live events for an already authorized durable job."""
    async for event in visible_events(REGISTRY, live, administration):
        yield sse.frame(event)
