"""The queue over HTTP: authoring, watching, reordering and stopping."""

import uuid

import pytest

import store
from api import auth
from store import queries as q
from store.schema import RunStatus


@pytest.fixture
async def client(engine_client):
    """Paused throughout: these are assertions about the line, not about what
    training does, and letting a run execute would leave a worker behind."""
    import main

    main.app.dependency_overrides[auth.require_service] = lambda: auth.SYSTEM
    await engine_client.post("/runs/pause?paused=true")
    yield engine_client
    main.app.dependency_overrides.clear()


async def queue_train(client, name=None) -> str:
    body = {"model_name": name or str(uuid.uuid4())}
    response = await client.post("/train", json=body)
    assert response.status_code == 202
    return response.json()["run_id"]


async def test_pausing_survives_the_answer(client):
    assert (await client.get("/runs")).json()["paused"] is True
    assert (await client.post("/runs/pause?paused=false")).json()["paused"] is False
    assert (await client.get("/runs")).json()["paused"] is False


async def test_a_sft_run_is_a_train_run_that_names_a_base(client):
    base = str(uuid.uuid4())
    response = await client.post(
        "/sft", json={"model_name": str(uuid.uuid4()), "base_model": base}
    )
    assert response.status_code == 202
    assert response.json()["kind"] == "train"

    with store.session() as db:
        run = db.get(store.Run, uuid.UUID(response.json()["run_id"]))
        assert run.config["base_model"] == base


async def test_a_benchmark_is_its_own_kind(client):
    response = await client.post("/eval", json={"model_name": str(uuid.uuid4())})
    assert response.json()["kind"] == "benchmark"


async def test_the_history_is_paged_and_counted(client):
    with store.session() as db:
        for n in range(3):
            q.enqueue(
                db,
                kind=store.RunKind.train,
                label=f"old-{n}",
                config={},
                status=RunStatus.done,
            )

    page = (await client.get("/runs?limit=2&offset=0")).json()
    assert len(page["history"]) == 2
    assert page["history_total"] == 3
    assert page["history_limit"] == 2

    assert len((await client.get("/runs?limit=2&offset=2")).json()["history"]) == 1


async def test_a_run_can_be_moved_to_the_front(client):
    """No `after` is the front of the line, not a missing argument."""
    first = await queue_train(client)
    second = await queue_train(client)

    assert (await client.patch(f"/runs/{second}")).status_code == 200

    band = (await client.get("/runs")).json()["active"]
    assert [r["run_id"] for r in band] == [second, first]


async def test_a_run_can_be_moved_behind_another(client):
    first = await queue_train(client)
    second = await queue_train(client)
    third = await queue_train(client)

    assert (await client.patch(f"/runs/{third}?after={first}")).status_code == 200

    band = (await client.get("/runs")).json()["active"]
    assert [r["run_id"] for r in band] == [first, third, second]


async def test_moving_a_run_behind_one_that_is_not_queued_is_refused(client):
    run_id = await queue_train(client)
    assert (await client.patch(f"/runs/{run_id}?after={uuid.uuid4()}")).status_code == 409


async def test_only_a_queued_run_can_be_reordered(client):
    run_id = await queue_train(client)
    await client.post(f"/runs/{run_id}/cancel")

    assert (await client.patch(f"/runs/{run_id}")).status_code == 409


async def test_a_finished_run_can_be_forgotten(client):
    run_id = await queue_train(client)
    await client.post(f"/runs/{run_id}/cancel")

    assert (await client.delete(f"/runs/{run_id}")).json()["deleted"] is True
    assert (await client.get(f"/runs/{run_id}")).status_code == 404


async def test_a_live_run_is_cancelled_rather_than_deleted(client):
    run_id = await queue_train(client)
    assert (await client.delete(f"/runs/{run_id}")).status_code == 409


async def test_cancelling_twice_is_refused(client):
    run_id = await queue_train(client)
    assert (await client.post(f"/runs/{run_id}/cancel")).status_code == 200
    assert (await client.post(f"/runs/{run_id}/cancel")).status_code == 409


async def test_a_chain_queues_a_sequence_that_waits(client):
    steps = [
        {"kind": "train", "config": {"model_name": str(uuid.uuid4())}},
        {"kind": "benchmark", "config": {"model_name": str(uuid.uuid4())}},
    ]
    queued = (await client.post("/runs/chain", json=steps)).json()

    assert [r["status"] for r in queued] == ["queued", "waiting"]
    assert queued[1]["depends_on"] == queued[0]["run_id"]


async def test_an_empty_chain_is_refused(client):
    assert (await client.post("/runs/chain", json=[])).status_code == 400


async def test_a_queued_run_streams_its_row_and_ends(client):
    """It has no live state yet, so there is nothing to follow — the row is the
    whole story and the stream says so once."""
    run_id = await queue_train(client)

    async with client.stream("GET", f"/runs/{run_id}/events") as response:
        payload = "".join([chunk async for chunk in response.aiter_text()])

    assert "event: state" in payload
    assert run_id in payload


async def test_the_event_stream_of_an_absent_run_404s(client):
    assert (await client.get(f"/runs/{uuid.uuid4()}/events")).status_code == 404


async def test_a_run_records_who_asked_for_it(engine_client):
    import main

    owner = uuid.uuid4()
    main.app.dependency_overrides[auth.require_service] = lambda: str(owner)
    try:
        await engine_client.post("/runs/pause?paused=true")
        response = await engine_client.post("/train", json={"model_name": str(uuid.uuid4())})
        with store.session() as db:
            run = db.get(store.Run, uuid.UUID(response.json()["run_id"]))
            assert run.owner_id == owner
    finally:
        main.app.dependency_overrides.clear()
