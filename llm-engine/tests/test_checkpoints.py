"""A checkpoint is a model, and this is what keeps it one.

Everything a finished model gets — a row, an id, a directory — an intermediate
gets too, so nothing downstream needs a second code path. What it gets
*differently* is all in this file: it is written the moment its frame lands
rather than on the progress timer, it is named apart from a collision suffix,
it is kept out of the listings, and it dies with its parent.
"""

import asyncio
import shutil
import uuid

import pytest

import store
from api import auth, handlers
from api.runner import Runner, _frame
from api.runs import Registry
from llm import storage
from llm.train.progress import Checkpoint, Progress
from store import queries as q
from store.schema import RunKind

# The drift check's scratch Postgres, reused rather than restated — the round
# trip below is about the same migration it guards.
from tests.test_migrations import scratch_database  # noqa: F401


#: What the frame says the weights were saved at — deliberately nothing like a
#: `TrainConfig`'s defaults, since telling the two apart is the point.
ARCHITECTURE = {
    "dimensions": 512,
    "heads": 8,
    "layers": 16,
    "ff_multiplier": 4,
    "ctx_window": 1024,
}


def checkpoint(index: int, *, model_id: str, step: int, final: bool = False) -> Checkpoint:
    return Checkpoint(
        model_id=model_id,
        index=index,
        step=step,
        total_steps=100,
        epoch=1,
        stage=0,
        stage_name=None,
        path=f"{model_id}/train_state.pt",
        loss=1.25,
        architecture=dict(ARCHITECTURE),
        final=final,
    )


def test_a_checkpoint_row_carries_the_shape_the_frame_measured(store_session):
    """The run's config cannot answer this. A resumed run states no
    architecture — inheriting the checkpoint's is what saying nothing means — so
    a row sourced from the config advertised pydantic's defaults as this model's
    shape, and served a `context` nobody had chosen. The frame reads it off the
    weights it just saved instead."""
    parent_id = uuid.uuid4()
    with store.session() as db:
        run_id = q.enqueue(
            db,
            kind=RunKind.train,
            label="t",
            config={"model_name": str(parent_id), "resume_from": "somewhere"},
        ).id

    made = handlers.record_checkpoint(run_id, checkpoint(1, model_id=str(uuid.uuid4()), step=10))

    with store.session() as db:
        model = q.get_model(db, made)
        assert model.parameters["result"]["architecture"] == ARCHITECTURE
        assert model.context == ARCHITECTURE["ctx_window"]


# ── the migration ───────────────────────────────────────────────────────────


@pytest.mark.postgres
def test_the_checkpoint_columns_survive_forward_ownership_migrations(scratch_database):  # noqa: F811
    """Upgrade to the ownership schema, then write a real row through it. `compare_metadata` in
    `test_migrations.py` proves the columns are *declared*; only writing one
    proves the migration built something a checkpoint fits in."""
    from alembic import command
    from alembic.config import Config

    config = Config("alembic.ini")
    command.upgrade(config, "head")

    store.configure(scratch_database)
    run_id, parent_id = uuid.uuid4(), uuid.uuid4()
    with store.session() as db:
        q.upsert_model(db, model_id=parent_id, name="parent", run_id=run_id)
        q.upsert_model(
            db,
            model_id=uuid.uuid4(),
            name="parent #1",
            run_id=run_id,
            parent_id=parent_id,
            checkpoint_step=40,
            checkpoint_index=1,
        )

    with store.session() as db:
        [only] = q.checkpoints_of(db, parent_id)
        assert (only.checkpoint_step, only.checkpoint_index) == (40, 1)
        assert only.run_id == run_id
        assert q.list_models(db, include_private=True) == [
            q.get_model(db, parent_id)
        ]


# ── the frame ───────────────────────────────────────────────────────────────


@pytest.fixture
async def registry():
    reg = Registry()
    reg.bind_loop(asyncio.get_running_loop())
    return reg


def test_a_checkpoint_travels_under_its_own_kind():
    kind, frame = _frame(checkpoint(2, model_id=str(uuid.uuid4()), step=40))
    assert kind == "checkpoint"
    assert (frame["index"], frame["step"], frame["loss"]) == (2, 40, 1.25)
    assert frame["final"] is False


async def test_a_checkpoint_row_does_not_wait_on_the_progress_throttle(
    store_session, registry
):
    """`PROGRESS_WRITE_SECS` exists so a progress bar does not cost a row a step.
    A checkpoint frame it swallowed would be weights nothing can name, so both of
    these land inside one throttle window that a `Progress` has just closed."""
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    parent_id = uuid.uuid4()

    def trains(config, progress, cancel, owner=None):
        progress(
            Progress(
                step=1,
                total_steps=100,
                epoch=1,
                total_epochs=1,
                chunk=1,
                total_chunks=100,
                loss=2.0,
                lr=1e-3,
                elapsed_secs=0.1,
            )
        )
        progress(checkpoint(1, model_id=first, step=10))
        progress(checkpoint(2, model_id=second, step=20))
        with store.session() as db:
            q.upsert_model(db, model_id=parent_id, name="Finished")
        return {"model_id": str(parent_id)}

    runner = Runner(registry, {RunKind.train: trains})
    run_id = runner.enqueue(
        RunKind.train,
        label="t",
        config={"model_name": str(parent_id), "display_name": "Sonnet"},
    )
    await runner._drain_once()

    with store.session() as db:
        rows = q.checkpoints_of(db, parent_id)
        assert [r.name for r in rows] == ["Sonnet #2"]
        assert [r.checkpoint_step for r in rows] == [20]
        assert [str(r.id) for r in rows] == [second]
        assert q.get_model(db, uuid.UUID(first)) is None
        assert all(r.run_id == run_id and r.published is False for r in rows)


async def test_a_runs_whole_output_groups_by_one_value(store_session, registry):
    """The finished model carries the same `run_id` its checkpoints do, which is
    the only thing that makes "what did this run produce" one query."""
    parent_id = uuid.uuid4()

    def trains(config, progress, cancel, owner=None):
        progress(checkpoint(1, model_id=str(uuid.uuid4()), step=10))
        with store.session() as db:
            q.upsert_model(db, model_id=parent_id, name="Finished")
        return {"model_id": str(parent_id)}

    runner = Runner(registry, {RunKind.train: trains})
    run_id = runner.enqueue(
        RunKind.train, label="t", config={"model_name": str(parent_id)}
    )
    await runner._drain_once()

    with store.session() as db:
        parent = q.get_model(db, parent_id)
        assert parent.run_id == run_id
        assert parent.checkpoint_index is None
        assert q.checkpoints_of(db, parent_id)[0].run_id == run_id


def test_the_runs_own_model_is_not_registered_twice(store_session):
    """A `final` frame announces the model `train` is about to write a row for.
    Recording it here as well would leave the run's output as its own child."""
    parent_id = uuid.uuid4()
    with store.session() as db:
        run = q.enqueue(
            db, kind=RunKind.train, label="t", config={"model_name": str(parent_id)}
        )
        run_id = run.id

    landed = checkpoint(0, model_id=str(parent_id), step=100, final=True)
    assert handlers.record_checkpoint(run_id, landed) is None

    with store.session() as db:
        assert q.get_model(db, parent_id) is None


def test_a_checkpoint_of_a_run_that_is_gone_is_dropped_not_raised(store_session):
    """This is called from inside a training run's progress callback. Losing the
    run over a row it cannot compose would cost the weights still being made."""
    unnameable = checkpoint(1, model_id="not-a-uuid", step=1)
    assert handlers.record_checkpoint(uuid.uuid4(), unnameable) is None


# ── naming ──────────────────────────────────────────────────────────────────


def test_a_checkpoint_is_numbered_with_a_hash_not_a_collision_suffix(store_session):
    """` (2)` already means "the second model somebody gave this name". A reader
    who cannot tell that from "the second checkpoint of this run" is worse off
    than one who has seen neither spelling."""
    parent_id = uuid.uuid4()
    with store.session() as db:
        # The name a person typed, already taken twice over by other runs.
        q.upsert_model(db, model_id=uuid.uuid4(), name="Draft")
        q.upsert_model(db, model_id=uuid.uuid4(), name="Draft (2)")
        run_id = q.enqueue(
            db,
            kind=RunKind.train,
            label="t",
            config={"model_name": str(parent_id), "display_name": "Draft"},
        ).id

    made = handlers.record_checkpoint(run_id, checkpoint(2, model_id=str(uuid.uuid4()), step=20))

    with store.session() as db:
        assert q.get_model(db, made).name == "Draft #2"


def test_two_runs_of_one_name_still_get_distinguishable_checkpoints(store_session):
    """`models.name` is unique, so the second run's `Draft #1` has to become
    something. It becomes what every other collision in this table becomes."""
    names = []
    for _ in range(2):
        parent_id = uuid.uuid4()
        with store.session() as db:
            run_id = q.enqueue(
                db,
                kind=RunKind.train,
                label="t",
                config={"model_name": str(parent_id), "display_name": "Draft"},
            ).id
        first = checkpoint(1, model_id=str(uuid.uuid4()), step=10)
        made = handlers.record_checkpoint(run_id, first)
        with store.session() as db:
            names.append(q.get_model(db, made).name)

    assert names == ["Draft #1", "Draft #1 (2)"]


def test_a_composed_name_stays_inside_the_column(store_session):
    parent_id = uuid.uuid4()
    with store.session() as db:
        run_id = q.enqueue(
            db,
            kind=RunKind.train,
            label="t",
            config={"model_name": str(parent_id), "display_name": "n" * q.NAME_LIMIT},
        ).id

    made = handlers.record_checkpoint(run_id, checkpoint(7, model_id=str(uuid.uuid4()), step=70))

    with store.session() as db:
        name = q.get_model(db, made).name
    assert len(name) <= q.NAME_LIMIT
    assert name.endswith(" #7")


# ── the listings ────────────────────────────────────────────────────────────


@pytest.fixture
def client(engine_client):
    """Read all recovery artifacts as an administrator already authorized by Rust."""
    import main

    main.app.dependency_overrides[auth.require_service] = lambda: auth.SYSTEM
    main.app.dependency_overrides[auth.administration_visible] = lambda: True
    yield engine_client
    main.app.dependency_overrides.clear()


@pytest.fixture
def run_output(data_root):
    """A finished model and two checkpoints of it, rows and directories both.

    The directories are real because the listing is a join against them, and
    the half of that join this feature can get wrong is the disk half.
    """
    parent_id, run_id = uuid.uuid4(), uuid.uuid4()
    made = []
    with store.session() as db:
        q.upsert_model(db, model_id=parent_id, name="Whole", run_id=run_id, published=True)
        for index, step in ((1, 10), (2, 20)):
            row = q.upsert_model(
                db,
                model_id=uuid.uuid4(),
                name=f"Whole #{index}",
                run_id=run_id,
                parent_id=parent_id,
                checkpoint_step=step,
                checkpoint_index=index,
            )
            made.append(row.id)
    for model_id in (parent_id, *made):
        storage.model_dir(str(model_id)).mkdir(parents=True, exist_ok=True)
    try:
        yield parent_id, made
    finally:
        for model_id in (parent_id, *made):
            shutil.rmtree(storage.model_dir(str(model_id)), ignore_errors=True)


async def test_checkpoints_are_kept_out_of_the_listing(client, run_output):
    parent_id, made = run_output

    listed = (await client.get("/models")).json()["models"]
    ids = {m["model_id"] for m in listed}

    assert str(parent_id) in ids
    # Neither by its row nor, since the directory is real, by the other half of
    # the join — which reports a directory with no row as a model.
    assert ids.isdisjoint({str(m) for m in made})


async def test_a_caller_that_asks_for_checkpoints_gets_them(client, run_output):
    _, made = run_output
    listed = (await client.get("/models?checkpoints=true")).json()["models"]
    assert {str(m) for m in made} <= {m["model_id"] for m in listed}


async def test_a_models_checkpoints_come_back_in_the_order_they_were_taken(
    client, run_output
):
    parent_id, made = run_output
    body = (await client.get(f"/models/{parent_id}/checkpoints")).json()

    assert [c["model_id"] for c in body["checkpoints"]] == [str(m) for m in made]
    assert [c["checkpoint_step"] for c in body["checkpoints"]] == [10, 20]
    assert [c["parent_id"] for c in body["checkpoints"]] == [str(parent_id)] * 2


async def test_a_model_with_no_checkpoints_answers_with_none(client, run_output):
    """Existing empty parents return an empty list, while missing parents are not found."""
    model_id = uuid.uuid4()
    with store.session() as db:
        q.upsert_model(db, model_id=model_id, name="No checkpoints")
    body = (await client.get(f"/models/{model_id}/checkpoints")).json()
    assert body == {"checkpoints": []}
    assert (await client.get(f"/models/{uuid.uuid4()}/checkpoints")).status_code == 404


# ── deletion ────────────────────────────────────────────────────────────────


async def test_deleting_a_model_takes_its_checkpoints_with_it(client, run_output):
    """Rows and directories both: nothing can reach a checkpoint whose parent is
    gone, so one left on disk is storage nothing accounts for."""
    parent_id, made = run_output

    body = (await client.delete(f"/models/{parent_id}")).json()

    assert body["checkpoints_removed"] == 2
    with store.session() as db:
        assert q.get_model(db, parent_id) is None
        assert [q.get_model(db, m) for m in made] == [None, None]
    assert not any(storage.model_dir(str(m)).exists() for m in made)


async def test_sparing_the_weights_spares_the_checkpoints_too(client, run_output):
    parent_id, made = run_output

    await client.delete(f"/models/{parent_id}?weights=false")

    with store.session() as db:
        assert q.checkpoints_of(db, parent_id) == []
    assert all(storage.model_dir(str(m)).exists() for m in made)


def test_deleting_a_checkpoint_leaves_its_siblings(store_session):
    """It is a model, so it deletes like one — `parent_id` points up, and
    nothing points at a checkpoint."""
    parent_id = uuid.uuid4()
    with store.session() as db:
        q.upsert_model(db, model_id=parent_id, name="Whole")
        ids = [
            q.upsert_model(
                db,
                model_id=uuid.uuid4(),
                name=f"Whole #{i}",
                parent_id=parent_id,
                checkpoint_step=i * 10,
                checkpoint_index=i,
            ).id
            for i in (1, 2)
        ]

        assert q.delete_model(db, ids[0]) == "Whole #1"
        assert [c.id for c in q.checkpoints_of(db, parent_id)] == [ids[1]]
        assert q.get_model(db, parent_id) is not None
