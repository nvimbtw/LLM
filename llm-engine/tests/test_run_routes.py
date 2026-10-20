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


async def test_a_typed_name_becomes_a_model_id_and_a_display_name(client):
    """`model_name` is the directory the weights land in and the id the model row
    is keyed by, but a composer sends what to call the model. Queueing the typed
    name as if it were an id is what failed at the end of a run — after training,
    on registration — and marked a finished run failed."""
    response = await client.post("/train", json={"model_name": "my-model"})
    assert response.status_code == 202

    with store.session() as db:
        config = db.get(store.Run, uuid.UUID(response.json()["run_id"])).config

    uuid.UUID(config["model_name"])  # raises if it is not an id
    assert config["display_name"] == "my-model"
    assert response.json()["label"] == "my-model"


async def test_a_chained_train_step_is_given_an_id_too(client):
    steps = [{"kind": "train", "config": {"model_name": "chained"}}]
    queued = (await client.post("/runs/chain", json=steps)).json()

    with store.session() as db:
        config = db.get(store.Run, uuid.UUID(queued[0]["run_id"])).config

    uuid.UUID(config["model_name"])
    assert config["display_name"] == "chained"


async def test_an_id_a_caller_named_is_kept(client):
    """Re-running against an existing model has to write that model's directory,
    so an id that was asked for is never replaced with a fresh one."""
    model_id = str(uuid.uuid4())
    response = await client.post("/train", json={"model_name": model_id})

    with store.session() as db:
        config = db.get(store.Run, uuid.UUID(response.json()["run_id"])).config

    assert config["model_name"] == model_id


async def test_a_vocabulary_build_is_its_own_kind(client):
    response = await client.post("/tokenize", json={"tokenizer_id": "my-vocab", "max_vocab": 512})

    assert response.status_code == 202
    assert response.json()["kind"] == "tokenize"
    assert response.json()["label"] == "my-vocab"

    with store.session() as db:
        config = db.get(store.Run, uuid.UUID(response.json()["run_id"])).config
    uuid.UUID(config["tokenizer_id"])  # raises if the typed name was taken for an id
    assert config["display_name"] == "my-vocab"


async def test_a_run_naming_a_vocabulary_that_does_not_exist_is_refused(client):
    """At enqueue, not at claim time: a typo would otherwise wait its turn in the
    queue and fail hours later, having held up the device it never needed."""
    response = await client.post(
        "/train", json={"model_name": "x", "tokenizer_id": str(uuid.uuid4())}
    )

    assert response.status_code == 400
    assert "build one" in response.json()["detail"]


async def test_a_tokenize_step_cannot_be_chained(client):
    """Promotion hands a successor a *model*. A chained build would leave the run
    after it pretraining against a vocabulary nobody chose."""
    steps = [
        {"kind": "tokenize", "config": {"tokenizer_id": "v"}},
        {"kind": "train", "config": {"model_name": str(uuid.uuid4())}},
    ]

    assert (await client.post("/runs/chain", json=steps)).status_code == 400


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


async def test_a_queued_config_keeps_what_the_request_stated(client):
    """The stored config is the request, not the request plus every default.

    A run is queued now and started later, so this dict is the whole of what the
    pipeline is handed — and `resumed_architecture` decides what a resume
    inherits by asking pydantic which fields were set. Dumping the defaults in
    alongside made every field "set" by the time the config was rebuilt, and a
    resume from a checkpoint of any other shape was then refused for restating
    an architecture nobody had typed.
    """
    resume = str(uuid.uuid4())
    response = await client.post(
        "/train", json={"model_name": str(uuid.uuid4()), "resume_from": resume, "epochs": 3}
    )
    assert response.status_code == 202

    with store.session() as db:
        config = db.get(store.Run, uuid.UUID(response.json()["run_id"])).config

    assert config["resume_from"] == resume
    assert config["epochs"] == 3
    for field in ("dimensions", "heads", "layers", "ff_multiplier", "ctx_window"):
        assert field not in config, f"{field} was filled in behind the request"


async def test_validation_cannot_be_a_model_producing_chain_step(client):
    """Recognizing the new kind cannot partially enqueue a chain that consumes no model."""
    steps = [
        {"kind": "train", "config": {"model_name": "must-not-be-queued"}},
        {"kind": "model_validation", "config": {}},
    ]
    response = await client.post("/runs/chain", json=steps)
    assert response.status_code == 400
    with store.session() as db:
        assert q.active_runs(db) == []
