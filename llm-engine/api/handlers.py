"""What each kind of run does, and what it writes when it lands.

The pipelines return a summary and leave weights on disk; these turn that into
the row the rest of the system reads. A handler is the only place that knows
both — `llm/` has never heard of the database, and `store/` has never heard of
a pipeline.
"""

import uuid
import shutil

from llm import storage

import store
from llm.config import EvalConfig, RlConfig, SftConfig, TokenizerConfig, TrainConfig
from llm.orchestrate import rl as rl_pipeline
from llm.orchestrate import sft as sft_pipeline
from llm.orchestrate import tokenize as tokenize_pipeline
from llm.orchestrate import train as train_pipeline
from llm.train.progress import Checkpoint
from store import queries as q
from store.schema import RunKind

#: A model's weights live at `artifacts/models/<id>/`, so the pipeline's
#: `model_name` is the id and `display_name` is what a person calls it.
DISPLAY_NAME = "display_name"

#: Room reserved for the ` #n` a checkpoint's name ends with, so composing one
#: cannot push the row past `Model.name`'s width.
SUFFIX_ROOM = 8


def train(config: dict, progress, cancel, owner: str | None = None) -> dict:
    """Pretrain, or fine-tune when the config names a base to start from.

    `owner` is who the run was queued for. It travels this far because the text
    is fetched from the data engine while the corpus is packed, and the backend
    scopes that read to the person — a run reads exactly the datasets whoever
    queued it could have read.
    """
    fine_tune = bool(config.get("base_model"))
    if fine_tune:
        cfg = SftConfig(**config)
        result = sft_pipeline.run(cfg, progress, cancel, owner)
    else:
        cfg = TrainConfig(**config)
        result = train_pipeline.run(cfg, progress, cancel, owner)

    # The id was minted when the run was queued; a row from before that carries
    # the name someone typed, and `uuid.UUID` on it raises "badly formed
    # hexadecimal UUID string" here — after the weights are already on disk.
    # Saying so is all this can do: the directory is named for the id, so there
    # is no id this row's weights could be registered under.
    try:
        model_id = uuid.UUID(cfg.model_name)
    except ValueError:
        raise ValueError(
            f"model_name {cfg.model_name!r} is not a model id. This run was queued "
            f"before ids were minted at enqueue — queue it again to register a model."
        ) from None

    with store.session() as db:
        q.upsert_model(
            db,
            model_id=model_id,
            name=config.get(DISPLAY_NAME) or cfg.model_name,
            parameters={"config": config, "result": _summarize(result)},
            # From what the run *trained*, not from what it asked for: a
            # resumed run states no architecture (it inherits the
            # checkpoint's) and a fine-tune has no `ctx_window` field at all,
            # so both would land here with the column empty or, worse, holding
            # a default nobody chose.
            context=_context(result, config),
            family="sft" if fine_tune else "pretrain",
        )
        if not fine_tune and not config.get("tokenizer_id"):
            _record_derived_vocabulary(db, result)
    return {**result, "model_id": str(model_id)}


def _context(result: dict, config: dict) -> int | None:
    """How long a window this model was trained to read.

    `result.architecture` is what the run actually built, recorded by both
    pipelines; the config is the fallback for a row written before that field
    existed.
    """
    architecture = result.get("architecture") or {}
    return architecture.get("ctx_window") or result.get("ctx_window") or config.get("ctx_window")


def _record_derived_vocabulary(db, result: dict) -> None:
    """Index the vocabulary a pretrain learned for itself.

    It is a real build in a real directory that other runs will reuse, so it
    belongs in the list rather than being an unnamed directory nothing accounts
    for. Named after its id because that is the only thing about it that is
    unique — two runs over the same corpora *are* this one build.
    """
    try:
        tokenizer_id = uuid.UUID(result["tokenizer_id"])
    except (KeyError, TypeError, ValueError):
        return
    q.upsert_tokenizer(
        db,
        tokenizer_id=tokenizer_id,
        name=f"auto {str(tokenizer_id)[:8]}",
        vocab_size=result.get("vocab"),
        fingerprint=result.get("tokenizer_fingerprint"),
        implicit=True,
    )


def record_checkpoint(run_id: uuid.UUID, checkpoint: Checkpoint) -> uuid.UUID | None:
    """Register one intermediate save as a model of its own.

    A checkpoint is a model — its own id, its own directory, its own row — which
    is what lets it be chatted with, benchmarked and continued from without any
    of those growing a second code path. What it is not is a row anybody should
    meet by accident: `published=False` keeps it out of chat, and
    `checkpoint_index` is what every listing filters on.

    `final` frames announce the run's own finished model, whose row `train`
    writes when the pipeline returns; there is nothing to do with one here.

    Registration failures return None after logging. The API runner then stops
    training; the complete artifact remains available for startup adoption and
    further unregistered generations cannot accumulate.
    """
    if checkpoint.final:
        return None
    try:
        model_id = uuid.UUID(checkpoint.model_id)
    except ValueError:
        return None

    retired = []
    try:
        with store.session() as db:
            run = db.get(store.Run, run_id)
            if run is None:
                return None
            config = run.config or {}
            q.upsert_model(
                db,
                model_id=model_id,
                name=_checkpoint_name(config, run.label, checkpoint.index),
                published=False,
                run_id=run_id,
                parent_id=_model_id(config.get("model_name")),
                checkpoint_step=checkpoint.step,
                checkpoint_index=checkpoint.index,
                # The shape comes off the frame, which read it off the weights
                # it had just saved. The run's config cannot answer for a
                # resumed run — it states no architecture, because inheriting
                # the checkpoint's is what saying nothing means — so a row
                # sourced from it advertised the request's defaults as this
                # model's shape, and served a `context` nobody had chosen.
                context=checkpoint.architecture.get("ctx_window"),
                # The row carries where in the run this was taken and what the
                # loss was there, because nothing else can answer it. The same
                # numbers are in the checkpoint's own `meta.json`, but the
                # listing serves only that file's `config` half — and sourcing
                # them from the run instead would lose them the moment the run
                # row is pruned, which is the very reason `run_id` is not a
                # foreign key.
                parameters={
                    "config": config,
                    "result": {
                        "recovery": True,
                        # Absent from a producer that does not measure it, the
                        # way every unmeasured figure is absent here.
                        **(
                            {"architecture": checkpoint.architecture}
                            if checkpoint.architecture
                            else {}
                        ),
                        "checkpoint": {
                            "index": checkpoint.index,
                            "step": checkpoint.step,
                            "total_steps": checkpoint.total_steps,
                            "epoch": checkpoint.epoch,
                            "stage": checkpoint.stage,
                            "stage_name": checkpoint.stage_name,
                            "loss": checkpoint.loss,
                        }
                    },
                },
            )
            parent_id = _model_id(config.get("model_name"))
            if parent_id is not None:
                retired = q.retire_recovery(db, parent_id, model_id)
        for old_id in retired:
            directory = storage.model_dir(str(old_id))
            if directory.exists():
                shutil.rmtree(directory)
    except Exception as error:  # noqa: BLE001 — see the docstring
        # First line only: a driver error carries the statement, its parameters
        # and a documentation link, and this is going into a running run's log
        # between two loss reports.
        reason = str(error).splitlines()[0] if str(error) else type(error).__name__
        print(
            f"  checkpoint #{checkpoint.index} at step {checkpoint.step} was written to "
            f"{checkpoint.path} but could not be registered: {reason}",
            flush=True,
        )
        return None
    return model_id


def _checkpoint_name(config: dict, label: str, index: int) -> str:
    """`<the run's name> #n`.

    A hash and not the ` (n)` `_free_model_name` spells a collision with: the
    two mean opposite things — one is the third checkpoint of a run, the other
    is the third model somebody gave the same name — and a reader who cannot
    tell them apart is worse off than one who has never seen either.
    """
    display = config.get(DISPLAY_NAME) or config.get("model_name") or label
    return f"{str(display)[: q.NAME_LIMIT - SUFFIX_ROOM].rstrip()} #{index}"


def attach_run(run_id: uuid.UUID, model_id: uuid.UUID) -> None:
    """Credit a finished model to the run that produced it.

    Here rather than in `train`, which is handed a config and never a run id. It
    is the same value the run's checkpoints carry, so a run's whole output — the
    intermediates and the model they led to — groups by one column.
    """
    with store.session() as db:
        model = q.get_model(db, model_id)
        if model is not None:
            model.run_id = run_id


def _model_id(value) -> uuid.UUID | None:
    """A run's `model_name` as an id, or nothing for a run queued before ids
    were minted at enqueue — those name a directory nothing can parse."""
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


def tokenize(config: dict, progress, cancel, owner: str | None = None) -> dict:
    """Learn one reusable vocabulary and record it."""
    cfg = TokenizerConfig(**config)
    result = tokenize_pipeline.run(cfg, progress, cancel, owner)

    try:
        tokenizer_id = uuid.UUID(cfg.tokenizer_id)
    except ValueError:
        raise ValueError(
            f"tokenizer_id {cfg.tokenizer_id!r} is not an id. This run was queued before "
            f"ids were minted at enqueue — queue it again to register a vocabulary."
        ) from None

    with store.session() as db:
        q.upsert_tokenizer(
            db,
            tokenizer_id=tokenizer_id,
            name=config.get(DISPLAY_NAME) or cfg.tokenizer_id,
            vocab_size=result["vocab_size"],
            merges=result["merges"],
            max_vocab=result["max_vocab"],
            fingerprint=result["fingerprint"],
            datasets=result["datasets"],
            implicit=False,
        )
    return result


def benchmark(config: dict, progress, cancel, owner: str | None = None) -> dict:
    """Dispatch through Rust; this queue remains the execution-state authority."""
    from llm.backend import Hop
    cfg = EvalConfig(**config)
    benchmark_id = config["execution_id"]
    with Hop(owner) as backend:
        backend.request("POST", f"/internal/bench/execute/{benchmark_id}", json=cfg.model_dump())
        while True:
            row, _ = backend.request("GET", f"/internal/bench/benchmarks/{benchmark_id}")
            if row["status"] in ("done", "cancelled"):
                return {**(row.get("report") or {}), "benchmark_id": benchmark_id}
            if row["status"] == "failed":
                raise RuntimeError(row.get("error") or "benchmark failed")
            if cancel.is_set():
                backend.request("POST", f"/internal/bench/execute/{benchmark_id}/cancel")
            progress({"stage": "benchmark", "detail": "measuring English language"})
            # An already-set cancellation event must not turn this into a busy loop.
            import time
            time.sleep(0.5)


def _summarize(result: dict) -> dict:
    """The run summary without its bulky series — the row is an index, and the
    full report is on disk beside the weights."""
    return {k: v for k, v in result.items() if not isinstance(v, list)}


def reinforce(config: dict, progress, cancel, owner: str | None = None) -> dict:
    """Reinforcement learning against a gym, and the model it leaves behind.

    The same shape as `train`: a pipeline runs, weights land under the id, and
    this writes the row. `family` is `rl` so a listing can tell a model that was
    *practised* from one that was fine-tuned — they are otherwise the same
    artifact, which is the point.
    """
    cfg = RlConfig(**config)
    result = rl_pipeline.run(cfg, progress, cancel, owner)

    try:
        model_id = uuid.UUID(cfg.model_name)
    except ValueError:
        raise ValueError(
            f"model_name {cfg.model_name!r} is not a model id — queue the run again"
        ) from None

    with store.session() as db:
        q.upsert_model(
            db,
            model_id=model_id,
            name=config.get(DISPLAY_NAME) or cfg.model_name,
            parameters={"config": config, "result": _summarize(result)},
            context=_context(result, config),
            family="rl",
        )
    return {**result, "model_id": str(model_id)}


HANDLERS = {
    RunKind.train: train,
    RunKind.benchmark: benchmark,
    RunKind.tokenize: tokenize,
    RunKind.rl: reinforce,
}
