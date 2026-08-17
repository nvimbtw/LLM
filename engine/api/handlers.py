"""What each kind of run does, and what it writes when it lands.

The pipelines return a summary and leave weights on disk; these turn that into
the row the rest of the system reads. A handler is the only place that knows
both — `llm/` has never heard of the database, and `store/` has never heard of
a pipeline.
"""

import uuid

import store
from llm.config import EvalConfig, SftConfig, TokenizerConfig, TrainConfig
from llm.orchestrate import evaluate as eval_pipeline
from llm.orchestrate import sft as sft_pipeline
from llm.orchestrate import tokenize as tokenize_pipeline
from llm.orchestrate import train as train_pipeline
from store import queries as q
from store.schema import BenchmarkStatus, RunKind, Suite

#: A model's weights live at `artifacts/models/<id>/`, so the pipeline's
#: `model_name` is the id and `display_name` is what a person calls it.
DISPLAY_NAME = "display_name"


def train(config: dict, progress, cancel) -> dict:
    """Pretrain, or fine-tune when the config names a base to start from."""
    fine_tune = bool(config.get("base_model"))
    if fine_tune:
        cfg = SftConfig(**config)
        result = sft_pipeline.run(cfg, progress, cancel)
    else:
        cfg = TrainConfig(**config)
        result = train_pipeline.run(cfg, progress, cancel)

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
            context=result.get("ctx_window") or config.get("ctx_window"),
            family="sft" if fine_tune else "pretrain",
        )
        if not fine_tune and not config.get("tokenizer_id"):
            _record_derived_vocabulary(db, result)
    return {**result, "model_id": str(model_id)}


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


def tokenize(config: dict, progress, cancel) -> dict:
    """Learn one reusable vocabulary and record it."""
    cfg = TokenizerConfig(**config)
    result = tokenize_pipeline.run(cfg, progress, cancel)

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


def benchmark(config: dict, progress, cancel) -> dict:
    """Score one model on one suite.

    The row opens here rather than at enqueue: a benchmark row means a
    measurement was attempted, so a run cancelled in the queue leaves none.
    """
    cfg = EvalConfig(**config)
    model_id = uuid.UUID(cfg.model_name)

    with store.session() as db:
        model = q.get_model(db, model_id)
        if model is None:
            raise FileNotFoundError(f"no model {cfg.model_name}")
        opened = q.open_benchmark(db, model=model, suite=Suite(cfg.suite))
        benchmark_id, display_name = opened.id, model.name

    try:
        report = eval_pipeline.run(cfg, progress, cancel)
    except Exception as e:
        with store.session() as db:
            q.close_benchmark(
                db, benchmark_id, status=BenchmarkStatus.failed, error=f"{type(e).__name__}: {e}"
            )
        raise

    # The pipeline knows the model by its directory; the name is a row it has
    # never heard of, so it is filled in here.
    report["model"]["name"] = display_name

    score = report.get("score") or {}
    with store.session() as db:
        q.close_benchmark(
            db,
            benchmark_id,
            status=BenchmarkStatus.done,
            score=score.get("score"),
            suite_version=score.get("version", 1),
            report=report,
            eval_tokens=report.get("tokens_used"),
            eval_conversations=report.get("conversations"),
        )
    return {**report, "benchmark_id": str(benchmark_id)}


def _summarize(result: dict) -> dict:
    """The run summary without its bulky series — the row is an index, and the
    full report is on disk beside the weights."""
    return {k: v for k, v in result.items() if not isinstance(v, list)}


HANDLERS = {
    RunKind.train: train,
    RunKind.benchmark: benchmark,
    RunKind.tokenize: tokenize,
}
