"""What each kind of run does, and what it writes when it lands.

The pipelines return a summary and leave weights on disk; these turn that into
the row the rest of the system reads. A handler is the only place that knows
both — `llm/` has never heard of the database, and `store/` has never heard of
a pipeline.
"""

import uuid

import store
from llm.config import EvalConfig, SftConfig, TrainConfig
from llm.orchestrate import evaluate as eval_pipeline
from llm.orchestrate import sft as sft_pipeline
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

    model_id = uuid.UUID(cfg.model_name)
    with store.session() as db:
        q.upsert_model(
            db,
            model_id=model_id,
            name=config.get(DISPLAY_NAME) or cfg.model_name,
            parameters={"config": config, "result": _summarize(result)},
            context=result.get("ctx_window") or config.get("ctx_window"),
            family="sft" if fine_tune else "pretrain",
        )
    return {**result, "model_id": str(model_id)}


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
        benchmark_id = opened.id

    try:
        report = eval_pipeline.run(cfg, progress, cancel)
    except Exception as e:
        with store.session() as db:
            q.close_benchmark(
                db, benchmark_id, status=BenchmarkStatus.failed, error=f"{type(e).__name__}: {e}"
            )
        raise

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
}
