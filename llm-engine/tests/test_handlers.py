"""What a finished run writes.

The pipelines are exercised in `test_pipelines.py`; what is checked here is the
seam — a summary and some weights on disk becoming the rows the rest of the
system reads.
"""

import threading
import uuid

import pytest

import store
from api import handlers
from store import queries as q
from store.schema import BenchmarkStatus, RunKind, Suite


@pytest.fixture
def progress():
    return lambda payload: None


@pytest.fixture
def cancel():
    return threading.Event()


@pytest.fixture
def pipelines(monkeypatch):
    """Stand in for the pipelines, recording the config each was called with."""
    called = {}

    def train(cfg, on_progress, stop, owner=None):
        called["train"] = cfg
        return {"ctx_window": 128, "vocab_size": 320, "loss_series": [1.0, 0.5]}

    def sft(cfg, on_progress, stop, owner=None):
        called["sft"] = cfg
        return {
            "ctx_window": 64,
            "base_model": cfg.base_model,
            "architecture": {
                "dimensions": 256,
                "heads": 8,
                "layers": 4,
                "ff_multiplier": 4,
                "ctx_window": 64,
            },
        }

    def tokenize(cfg, on_progress, stop, owner=None):
        called["tokenize"] = cfg
        return {
            "tokenizer_id": cfg.tokenizer_id,
            "vocab_size": 512,
            "merges": 250,
            "max_vocab": 512,
            "files": 2,
            "datasets": [["Tiny", None]],
            "fingerprint": "abcdef0123456789",
        }

    monkeypatch.setattr(handlers.train_pipeline, "run", train)
    monkeypatch.setattr(handlers.sft_pipeline, "run", sft)
    monkeypatch.setattr(handlers.tokenize_pipeline, "run", tokenize)
    return called


def test_a_pretraining_run_writes_a_model_row(store_session, pipelines, progress, cancel):
    model_id = uuid.uuid4()
    config = {"model_name": str(model_id), "display_name": "First", "datasets": []}

    result = handlers.train(config, progress, cancel)

    assert result["model_id"] == str(model_id)
    with store.session() as db:
        model = q.get_model(db, model_id)
        assert model.name == "First"
        assert model.family == "pretrain"
        assert model.context == 128


def test_a_fine_tune_is_told_apart_by_naming_a_base(store_session, pipelines, progress, cancel):
    model_id = uuid.uuid4()
    config = {"model_name": str(model_id), "base_model": str(uuid.uuid4())}

    handlers.train(config, progress, cancel)

    assert "sft" in pipelines and "train" not in pipelines
    with store.session() as db:
        model = q.get_model(db, model_id)
        assert model.family == "sft"
        # An SFT run's own config carries no architecture — it comes from the
        # base checkpoint — so the API must be able to serve one back from
        # `result` instead, the same place it now finds `base_model`.
        stored = model.parameters["result"]
        assert stored["base_model"] == config["base_model"]
        assert stored["architecture"] == {
            "dimensions": 256,
            "heads": 8,
            "layers": 4,
            "ff_multiplier": 4,
            "ctx_window": 64,
        }


def test_the_row_keeps_the_summary_without_its_series(
    store_session, pipelines, progress, cancel
):
    """The row is an index; the full report is on disk beside the weights."""
    model_id = uuid.uuid4()
    handlers.train({"model_name": str(model_id)}, progress, cancel)

    with store.session() as db:
        stored = q.get_model(db, model_id).parameters["result"]
        assert "loss_series" not in stored
        assert stored["vocab_size"] == 320


def test_retraining_under_the_same_id_updates_its_row(
    store_session, pipelines, progress, cancel
):
    """The id is the weights directory, so a second run writes the same one."""
    model_id = uuid.uuid4()
    handlers.train({"model_name": str(model_id), "display_name": "One"}, progress, cancel)
    handlers.train({"model_name": str(model_id), "display_name": "Two"}, progress, cancel)

    with store.session() as db:
        assert q.get_model(db, model_id).name == "Two"
        assert len(q.list_models(db, include_private=True)) == 1


def test_a_benchmark_opens_a_row_when_it_reaches_the_device(
    store_session, monkeypatch, progress, cancel
):
    model_id = uuid.uuid4()
    with store.session() as db:
        q.upsert_model(db, model_id=model_id, name="scored")

    monkeypatch.setattr(
        handlers.eval_pipeline,
        "run",
        lambda cfg, on_progress, stop, owner=None: {
            "model": {"id": cfg.model_name, "name": None, "layers": 2},
            "score": {"score": 62.5, "version": 3},
            "tokens_used": 900,
            "conversations": 12,
        },
    )

    result = handlers.benchmark(
        {"model_name": str(model_id), "suite": "markup"}, progress, cancel
    )

    with store.session() as db:
        [run] = q.benchmarks_of(db, model_id)
        assert str(run.id) == result["benchmark_id"]
        assert (run.score, run.suite_version) == (62.5, 3)
        assert run.suite is Suite.markup
        assert run.status is BenchmarkStatus.done
        assert run.eval_tokens == 900


def test_a_report_learns_the_name_the_pipeline_could_not_know(
    store_session, monkeypatch, progress, cancel
):
    """The pipeline knows a model by its weights directory. What a person calls
    it is a row, so the handler is where the two meet."""
    model_id = uuid.uuid4()
    with store.session() as db:
        q.upsert_model(db, model_id=model_id, name="Smoke 2")

    monkeypatch.setattr(
        handlers.eval_pipeline,
        "run",
        lambda cfg, on_progress, stop, owner=None: {
            "model": {"id": cfg.model_name, "name": None},
            "score": {"score": 1.0},
        },
    )

    report = handlers.benchmark({"model_name": str(model_id)}, progress, cancel)

    assert report["model"]["id"] == str(model_id)
    assert report["model"]["name"] == "Smoke 2"
    with store.session() as db:
        assert q.benchmarks_of(db, model_id)[0].report["model"]["name"] == "Smoke 2"


def test_a_failed_benchmark_closes_its_row_with_the_reason(
    store_session, monkeypatch, progress, cancel
):
    model_id = uuid.uuid4()
    with store.session() as db:
        q.upsert_model(db, model_id=model_id, name="doomed")

    def explode(cfg, on_progress, stop):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(handlers.eval_pipeline, "run", explode)

    with pytest.raises(RuntimeError):
        handlers.benchmark({"model_name": str(model_id)}, progress, cancel)

    with store.session() as db:
        [run] = q.benchmarks_of(db, model_id)
        assert run.status is BenchmarkStatus.failed
        assert "kaboom" in run.error


def test_benchmarking_an_unknown_model_opens_no_row(store_session, progress, cancel):
    """A benchmark row means a measurement was attempted."""
    model_id = uuid.uuid4()

    with pytest.raises(FileNotFoundError):
        handlers.benchmark({"model_name": str(model_id)}, progress, cancel)

    with store.session() as db:
        assert q.benchmarks_of(db, model_id) == []


def test_a_vocabulary_build_writes_a_tokenizer_row(store_session, pipelines, progress, cancel):
    tokenizer_id = uuid.uuid4()
    config = {"tokenizer_id": str(tokenizer_id), "display_name": "Shared 8k"}

    result = handlers.tokenize(config, progress, cancel)

    assert result["vocab_size"] == 512
    with store.session() as db:
        row = q.get_tokenizer(db, tokenizer_id)
        assert row.name == "Shared 8k"
        assert row.fingerprint == "abcdef0123456789"
        assert row.implicit is False


def test_rebuilding_a_vocabulary_updates_its_row(store_session, pipelines, progress, cancel):
    tokenizer_id = uuid.uuid4()
    handlers.tokenize({"tokenizer_id": str(tokenizer_id), "display_name": "One"}, progress, cancel)
    handlers.tokenize({"tokenizer_id": str(tokenizer_id), "display_name": "Two"}, progress, cancel)

    with store.session() as db:
        assert q.get_tokenizer(db, tokenizer_id).name == "Two"
        assert len(q.list_tokenizers(db)) == 1


def test_a_pretrain_indexes_the_vocabulary_it_learned_for_itself(
    store_session, pipelines, progress, cancel, monkeypatch
):
    """It is a real build other runs will reuse, so it belongs in the list —
    marked apart from the ones built on purpose."""
    derived = uuid.uuid4()

    def train(cfg, on_progress, stop, owner=None):
        return {"ctx_window": 128, "tokenizer_id": str(derived), "vocab": 512}

    monkeypatch.setattr(handlers.train_pipeline, "run", train)
    handlers.train({"model_name": str(uuid.uuid4())}, progress, cancel)

    with store.session() as db:
        assert q.get_tokenizer(db, derived).implicit is True
        assert q.list_tokenizers(db, include_implicit=False) == []


def test_a_run_that_named_its_vocabulary_does_not_reindex_it(
    store_session, pipelines, progress, cancel, monkeypatch
):
    """Naming a build consumes it. Only the vocabulary a run derived for itself
    is this run's to record."""
    named = uuid.uuid4()

    def train(cfg, on_progress, stop, owner=None):
        return {"ctx_window": 128, "tokenizer_id": str(named), "vocab": 512}

    monkeypatch.setattr(handlers.train_pipeline, "run", train)
    handlers.train(
        {"model_name": str(uuid.uuid4()), "tokenizer_id": str(named)}, progress, cancel
    )

    with store.session() as db:
        assert q.get_tokenizer(db, named) is None


def test_a_training_run_is_indexed_by_what_it_produced(
    store_session, pipelines, progress, cancel, monkeypatch
):
    """Its result names the vocabulary it *consumed* as well as the model it
    made. `ref_id` is the model — reading the tokenizer first would point every
    training row at a vocabulary."""
    from api.runner import _produced

    consumed = uuid.uuid4()
    monkeypatch.setattr(
        handlers.train_pipeline,
        "run",
        lambda cfg, on_progress, stop, owner=None: {"ctx_window": 128, "tokenizer_id": str(consumed)},
    )
    result = handlers.train(
        {"model_name": str(uuid.uuid4()), "tokenizer_id": str(consumed)}, progress, cancel
    )

    assert result["tokenizer_id"] == str(consumed)
    assert str(_produced(result)) == result["model_id"]


def test_every_kind_the_queue_accepts_has_a_handler():
    """A run with no handler fails, so the two lists have to agree.

    Stated as a subtraction rather than a roster: adding a kind that needs a
    handler and forgetting to write one should fail here, which a hand-kept list
    only does if somebody also remembers to extend the list.
    """
    assert set(handlers.HANDLERS) == set(RunKind) - {RunKind.chat}


def test_a_chat_is_the_one_kind_with_no_handler():
    """It is a row so a conversation shows up on the board beside the runs, and
    nothing about it is queued — the reason the subtraction above has exactly one
    term, and the thing that would otherwise make it look like an oversight."""
    assert RunKind.chat not in handlers.HANDLERS
