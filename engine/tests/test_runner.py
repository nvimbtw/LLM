"""The durable queue: what survives, what does not, and what stops a run."""

import asyncio
import threading
import uuid

import pytest

import store
from api.runner import Runner
from api.runs import Registry
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
