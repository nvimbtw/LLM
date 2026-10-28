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
    main.app.dependency_overrides[auth.administration_visible] = lambda: True
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


@pytest.mark.parametrize("final", [
    {"kind": "train", "config": {"lr": "invalid"}},
    {"kind": "rl", "config": {"task_types": []}},
    {"kind": "benchmark", "config": {"suite": "unsupported"}},
])
async def test_every_chain_config_is_validated_before_any_job_is_inserted(client, monkeypatch, final):
    """A malformed later consumer returns its field location without inserting even the first step."""
    inserted = []
    enqueue = q.enqueue

    def record_enqueue(*args, **kwargs):
        """Record actual queue writes while preserving the production insert behavior."""
        inserted.append(kwargs["kind"])
        return enqueue(*args, **kwargs)

    monkeypatch.setattr(q, "enqueue", record_enqueue)
    response = await client.post("/runs/chain", json=[{"kind": "train", "config": {}}, final])
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][:3] == ["body", 1, "config"]
    assert inserted == []
    with store.session() as db:
        assert q.active_runs(db) == []


async def test_a_chain_rejects_a_missing_first_step_vocabulary_before_storage(client):
    """Chain pretraining applies the same owned-tokenizer check as direct submission."""
    steps = [
        {"kind": "train", "config": {"tokenizer_id": str(uuid.uuid4())}},
        {"kind": "benchmark", "config": {}},
    ]
    response = await client.post("/runs/chain", json=steps)
    assert response.status_code == 400
    with store.session() as db:
        assert q.active_runs(db) == []


async def test_a_late_chain_insert_failure_rolls_back_the_first_job(client, monkeypatch):
    """An error after a real first insert rolls back the transaction and never wakes admission."""
    from api.runner import RUNNER

    enqueue = q.enqueue
    inserts, wakes = [], []

    def fail_second_insert(*args, **kwargs):
        """Insert one real job, then simulate a later storage failure in the same transaction."""
        if inserts:
            raise RuntimeError("late chain insertion failed")
        result = enqueue(*args, **kwargs)
        inserts.append(result.id)
        return result

    monkeypatch.setattr(q, "enqueue", fail_second_insert)
    monkeypatch.setattr(RUNNER, "wake", lambda: wakes.append(True))
    steps = [{"kind": "train", "config": {}}, {"kind": "benchmark", "config": {}}]
    with pytest.raises(RuntimeError, match="late chain insertion failed"):
        await client.post("/runs/chain", json=steps)
    assert len(inserts) == 1
    assert wakes == []
    with store.session() as db:
        assert q.active_runs(db) == []
        assert db.get(store.Run, inserts[0]) is None


async def test_chain_admission_wakes_once_after_all_rows_commit(client, monkeypatch):
    """A wake reader sees the complete linked chain from a separate committed session."""
    from api.runner import RUNNER

    observed = []

    def observe_commit():
        """Read durable jobs exactly when submission asks the runner to admit work."""
        with store.session() as db:
            observed.append([(run.id, run.depends_on, run.status) for run in q.active_runs(db)])

    monkeypatch.setattr(RUNNER, "wake", observe_commit)
    steps = [
        {"kind": "train", "config": {"model_name": "base", "epochs": 3}},
        {"kind": "train", "config": {"model_name": "tuned", "lr": 0.00001}},
        {"kind": "benchmark", "config": {}},
    ]
    response = await client.post("/runs/chain", json=steps)
    assert response.status_code == 202
    rows = response.json()
    ids = [uuid.UUID(item["run_id"]) for item in rows]
    assert observed == [[
        (ids[0], None, RunStatus.queued),
        (ids[1], ids[0], RunStatus.waiting),
        (ids[2], ids[1], RunStatus.waiting),
    ]]
    assert rows[0]["config"]["epochs"] == 3
    assert "ctx_window" not in rows[0]["config"]
    assert "base_model" not in rows[1]["config"]
    assert "model_name" not in rows[2]["config"]


async def test_a_benchmark_cannot_supply_a_model_to_another_chain_step(client):
    """Reject a report-producing predecessor before any rows are stored."""
    steps = [
        {"kind": "benchmark", "config": {"model_name": str(uuid.uuid4())}},
        {"kind": "train", "config": {}},
    ]
    response = await client.post("/runs/chain", json=steps)
    assert response.status_code == 400
    with store.session() as db:
        assert q.active_runs(db) == []


async def test_chain_admission_is_bounded(client):
    """Refuse an oversized operation at the request boundary without partial queue rows."""
    from api.routes.runs import MAX_CHAIN_STEPS

    response = await client.post("/runs/chain", json=[{"kind": "train", "config": {}}] * (MAX_CHAIN_STEPS + 1))
    assert response.status_code == 422
    with store.session() as db:
        assert q.active_runs(db) == []
