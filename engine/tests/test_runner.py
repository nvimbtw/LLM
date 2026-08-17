"""The durable queue: what survives, what does not, and what stops a run."""

import asyncio
import threading
import uuid

import pytest

import store
from api.runner import Runner
from api.runs import Registry
from llm.train.progress import Progress, Tokenize
from store import queries as q
from store.schema import RunKind, RunStatus


@pytest.fixture
async def registry():
    """Async so it can see the loop the tests actually run on — worker threads
    publish onto it, and binding the wrong one shows up as a hang."""
    reg = Registry()
    reg.bind_loop(asyncio.get_running_loop())
    return reg


@pytest.fixture
def ran():
    """Every config a handler was called with, in order."""
    return []


def handler_recording(ran, result=None):
    def handler(config, progress, cancel):
        ran.append(config)
        progress({"fraction": 0.5, "stage": "working"})
        return result or {"ok": True}

    return handler


async def drain(runner: Runner, ticks: int = 1):
    """Run the loop's body directly — the real loop never returns."""
    for _ in range(ticks):
        await runner._drain_once()


async def test_a_queued_run_executes_and_lands_done(store_session, registry, ran):
    runner = Runner(registry, {RunKind.train: handler_recording(ran)})
    run_id = runner.enqueue(RunKind.train, label="t", config={"model_name": "x"})

    await drain(runner)

    assert ran == [{"model_name": "x"}]
    with store.session() as db:
        run = db.get(store.Run, run_id)
        assert run.status is RunStatus.done
        assert run.result == {"ok": True}
        assert run.finished_at is not None


async def test_runs_execute_in_queue_order(store_session, registry, ran):
    runner = Runner(registry, {RunKind.train: handler_recording(ran)})
    runner.enqueue(RunKind.train, label="first", config={"n": 1})
    runner.enqueue(RunKind.train, label="second", config={"n": 2})

    await drain(runner)

    assert [c["n"] for c in ran] == [1, 2]


async def test_a_failing_handler_fails_only_its_own_run(store_session, registry, ran):
    def explode(config, progress, cancel):
        raise RuntimeError("kaboom")

    runner = Runner(registry, {RunKind.train: explode, RunKind.benchmark: handler_recording(ran)})
    bad = runner.enqueue(RunKind.train, label="bad", config={})
    good = runner.enqueue(RunKind.benchmark, label="good", config={"n": 1})

    await drain(runner)

    with store.session() as db:
        assert db.get(store.Run, bad).status is RunStatus.failed
        assert "kaboom" in db.get(store.Run, bad).error
        assert db.get(store.Run, good).status is RunStatus.done


async def test_a_kind_with_no_handler_fails_rather_than_hanging(store_session, registry):
    runner = Runner(registry, {})
    run_id = runner.enqueue(RunKind.chat, label="orphan", config={})

    await drain(runner)

    with store.session() as db:
        run = db.get(store.Run, run_id)
        assert run.status is RunStatus.failed
        assert "no handler" in run.error


async def test_cancelling_a_queued_run_stops_it_before_it_starts(store_session, registry, ran):
    runner = Runner(registry, {RunKind.train: handler_recording(ran)})
    run_id = runner.enqueue(RunKind.train, label="t", config={})

    assert runner.cancel(run_id) is True
    await drain(runner)

    assert ran == []
    with store.session() as db:
        assert db.get(store.Run, run_id).status is RunStatus.cancelled


async def test_cancelling_a_running_run_is_cooperative(store_session, registry):
    """It stops at a step boundary, so `cancelling` is a real state rather than
    a client-side fiction."""
    entered = threading.Event()
    saw_cancel = threading.Event()

    def slow(config, progress, cancel):
        entered.set()
        if cancel.wait(timeout=5):
            saw_cancel.set()
        return {"stopped": True}

    runner = Runner(registry, {RunKind.train: slow})
    run_id = runner.enqueue(RunKind.train, label="slow", config={})

    task = asyncio.create_task(drain(runner))
    await asyncio.to_thread(entered.wait, 5)
    assert runner.cancel(run_id) is True
    await task

    assert saw_cancel.is_set()
    with store.session() as db:
        assert db.get(store.Run, run_id).status is RunStatus.cancelled


async def test_cancelling_a_finished_run_is_refused(store_session, registry, ran):
    runner = Runner(registry, {RunKind.train: handler_recording(ran)})
    run_id = runner.enqueue(RunKind.train, label="t", config={})
    await drain(runner)

    assert runner.cancel(run_id) is False


async def test_a_paused_queue_holds_the_line(store_session, registry, ran):
    runner = Runner(registry, {RunKind.train: handler_recording(ran)})
    runner.enqueue(RunKind.train, label="t", config={"n": 1})

    with store.session() as db:
        q.set_paused(db, True)
    await drain(runner)
    assert ran == []

    with store.session() as db:
        q.set_paused(db, False)
    await drain(runner)
    assert len(ran) == 1


async def test_what_a_run_produced_is_recorded_on_its_row(store_session, registry):
    produced = uuid.uuid4()

    def trains(config, progress, cancel):
        return {"model_id": str(produced)}

    runner = Runner(registry, {RunKind.train: trains})
    run_id = runner.enqueue(RunKind.train, label="t", config={})

    await drain(runner)

    with store.session() as db:
        assert db.get(store.Run, run_id).ref_id == produced


async def test_coarse_progress_reaches_the_row(store_session, registry, ran):
    """A page loading before its socket attaches still gets a sane bar."""
    runner = Runner(registry, {RunKind.train: handler_recording(ran)})
    run_id = runner.enqueue(RunKind.train, label="t", config={})

    await drain(runner)

    with store.session() as db:
        run = db.get(store.Run, run_id)
        assert run.stage == "working"
        assert run.progress == 1.0


async def test_a_training_report_travels_as_itself(store_session, registry):
    """The loop reports a `Progress`, not a dict, and every field of it has to
    reach the chart — the frame name is `train` and the payload is the whole
    dataclass. Publishing it as a generic `progress` frame is what once left a
    running job with no statistics and no diagnostics at all."""

    def trains(config, progress, cancel):
        progress(
            Progress(
                step=3,
                total_steps=10,
                epoch=1,
                total_epochs=2,
                chunk=3,
                total_chunks=5,
                loss=1.5,
                lr=1e-3,
                elapsed_secs=2.0,
                stage_name="warmup",
                diag={"step": 3, "grad_global": 0.1},
            )
        )
        return {}

    runner = Runner(registry, {RunKind.train: trains})
    run_id = runner.enqueue(RunKind.train, label="t", config={})

    await drain(runner)

    live = registry.get(str(run_id))
    frame = next(e for e in live.history if e.kind == "train")
    assert frame.data["loss"] == 1.5
    assert frame.data["diag"] == {"step": 3, "grad_global": 0.1}
    assert frame.data["fraction"] == 0.3

    # The row's stage column is a label, so a curriculum run writes its name
    # there rather than the index the charts group by.
    with store.session() as db:
        assert db.get(store.Run, run_id).stage == "warmup"


async def test_a_phase_one_report_says_only_what_it_measured(store_session, registry):
    """A pretrain counts files and merges, a fine-tune counts conversations. What
    a phase did not measure is absent from the frame rather than zero, so the
    panel shows "—" instead of a count of nothing."""

    def builds(config, progress, cancel):
        progress(Tokenize(files=3, vocab=8192, merges=7900, progress=0.6))
        return {}

    runner = Runner(registry, {RunKind.train: builds})
    run_id = runner.enqueue(RunKind.train, label="t", config={})

    await drain(runner)

    live = registry.get(str(run_id))
    frame = next(e for e in live.history if e.kind == "tokenize")
    assert frame.data == {
        "files": 3,
        "vocab": 8192,
        "merges": 7900,
        "progress": 0.6,
        "done": False,
    }


async def test_a_vocabulary_build_takes_its_turn_like_anything_else(store_session, registry, ran):
    """It never touches the GPU, but it is minutes of CPU — one line means it
    cannot start while the device is training."""
    runner = Runner(registry, {RunKind.tokenize: handler_recording(ran)})
    run_id = runner.enqueue(RunKind.tokenize, label="v", config={"tokenizer_id": "x"})

    await drain(runner)

    with store.session() as db:
        assert db.get(store.Run, run_id).status is RunStatus.done


async def test_the_queue_survives_a_restart(store_session, registry, ran):
    """Rows outlive the process: an in-flight run is failed, the line is not."""
    runner = Runner(registry, {RunKind.train: handler_recording(ran)})
    runner.enqueue(RunKind.train, label="was-running", config={"n": 1})
    with store.session() as db:
        q.claim_next(db)
    still_queued = runner.enqueue(RunKind.train, label="waiting-its-turn", config={"n": 2})

    with store.session() as db:
        assert q.fail_orphaned(db) == 1

    await drain(runner)

    assert [c["n"] for c in ran] == [2]
    with store.session() as db:
        assert db.get(store.Run, still_queued).status is RunStatus.done
