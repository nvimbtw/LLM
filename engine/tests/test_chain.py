"""A run that follows another, against the model that one produces."""

import asyncio
import uuid

import pytest

import store
from api import chain
from api.runner import Runner
from api.runs import Registry
from store import queries as q
from store.schema import RunKind, RunStatus


@pytest.fixture
async def registry():
    reg = Registry()
    reg.bind_loop(asyncio.get_running_loop())
    return reg


def produces(model_id):
    def handler(config, progress, cancel):
        return {"model_id": str(model_id)}

    return handler


async def test_a_follow_up_waits_until_its_model_exists(store_session, registry):
    produced = uuid.uuid4()
    runner = Runner(registry, {RunKind.train: produces(produced)})

    first = runner.enqueue(RunKind.train, label="base", config={"model_name": "base"})
    second = runner.enqueue(
        RunKind.train, label="sft", config={"model_name": "sft"}, depends_on=first
    )

    with store.session() as db:
        assert db.get(store.Run, second).status is RunStatus.waiting

    await runner._drain_once()

    with store.session() as db:
        follow_up = db.get(store.Run, second)
        assert follow_up.status is RunStatus.done
        assert follow_up.config["base_model"] == str(produced)


async def test_a_benchmark_follow_up_scores_what_the_run_before_it_made(
    store_session, registry
):
    produced = uuid.uuid4()
    scored = []

    def benchmark(config, progress, cancel):
        scored.append(config["model_name"])
        return {}

    runner = Runner(
        registry, {RunKind.train: produces(produced), RunKind.benchmark: benchmark}
    )
    first = runner.enqueue(RunKind.train, label="base", config={})
    runner.enqueue(RunKind.benchmark, label="score", config={}, depends_on=first)

    await runner._drain_once()

    assert scored == [str(produced)]


async def test_a_follow_up_of_a_run_that_made_nothing_fails(store_session, registry):
    """It was written against a model that is never going to exist."""

    def explode(config, progress, cancel):
        raise RuntimeError("kaboom")

    runner = Runner(registry, {RunKind.train: explode})
    first = runner.enqueue(RunKind.train, label="base", config={})
    second = runner.enqueue(RunKind.train, label="sft", config={}, depends_on=first)

    await runner._drain_once()

    with store.session() as db:
        follow_up = db.get(store.Run, second)
        assert follow_up.status is RunStatus.failed
        assert str(first) in follow_up.error


async def test_promoting_nothing_is_not_an_error(store_session):
    assert chain.promote(uuid.uuid4(), uuid.uuid4()) == 0


async def test_a_waiting_run_can_be_cancelled_before_its_turn(store_session, registry):
    runner = Runner(registry, {RunKind.train: produces(uuid.uuid4())})
    first = runner.enqueue(RunKind.train, label="base", config={})
    second = runner.enqueue(RunKind.train, label="sft", config={}, depends_on=first)

    assert runner.cancel(second) is True
    with store.session() as db:
        assert db.get(store.Run, second).status is RunStatus.cancelled
        # And it stays cancelled: promoting must not resurrect it.
        assert q.waiting_on(db, first) == []

    await runner._drain_once()
    with store.session() as db:
        assert db.get(store.Run, second).status is RunStatus.cancelled
