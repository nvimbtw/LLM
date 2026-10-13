"""Models and scores over HTTP: the join of the row and the weights on disk."""

import uuid

import pytest

import store
from api import auth
from store import queries as q
from tests.fixtures.tiny_env import TINY_MODEL


@pytest.fixture
def client(engine_client):
    """Signed in as the backend acting for itself — models are shared, and
    editing one is admin work the backend has already authorized."""
    import main

    main.app.dependency_overrides[auth.require_service] = lambda: auth.SYSTEM
    yield engine_client
    main.app.dependency_overrides.clear()


@pytest.fixture
def recorded():
    """A model row whose id is a directory that does not exist."""
    with store.session() as db:
        model = q.upsert_model(
            db,
            model_id=uuid.uuid4(),
            name="rowed",
            family="pretrain",
            context=256,
            parameters={"config": {"dimensions": 32, "layers": 2}},
        )
        return model.id


async def test_a_directory_with_no_row_is_still_listed(client):
    """Disk is the authority on what can be loaded, so a model trained outside
    the queue is offered rather than hidden."""
    listed = {m["name"]: m for m in (await client.get("/models")).json()["models"]}
    assert listed[TINY_MODEL]["has_weights"] is True
    assert listed[TINY_MODEL]["model_id"] == TINY_MODEL


async def test_a_row_with_no_directory_is_listed_without_weights(client, recorded):
    listed = {m["name"]: m for m in (await client.get("/models")).json()["models"]}
    assert listed["rowed"]["has_weights"] is False
    assert listed["rowed"]["model_id"] == str(recorded)


async def test_only_published_models_when_the_caller_asks_for_those(client, recorded):
    body = (await client.get("/models?include_private=false")).json()
    assert "rowed" not in [m["name"] for m in body["models"]]

    await client.patch(f"/models/{recorded}", json={"published": True})
    body = (await client.get("/models?include_private=false")).json()
    assert "rowed" in [m["name"] for m in body["models"]]




async def test_an_absent_model_404s(client):
    assert (await client.get(f"/models/{uuid.uuid4()}")).status_code == 404
    assert (await client.patch(f"/models/{uuid.uuid4()}", json={})).status_code == 404
    assert (await client.delete(f"/models/{uuid.uuid4()}")).status_code == 404


async def test_renaming_a_model_leaves_the_directory_alone(client, recorded):
    """The id is the directory; the name is what a person calls it."""
    body = (await client.patch(f"/models/{recorded}", json={"display_name": "Renamed"})).json()
    assert body["name"] == "Renamed"
    assert body["model_id"] == str(recorded)


async def test_deleting_a_model_takes_its_weights(client, data_root):
    from llm import storage

    model_id = uuid.uuid4()
    directory = storage.model_dir(str(model_id))
    directory.mkdir(parents=True)
    (directory / "model.pt").write_bytes(b"weights")
    with store.session() as db:
        q.upsert_model(db, model_id=model_id, name="doomed")

    body = (await client.delete(f"/models/{model_id}")).json()

    assert body["weights_removed"] is True
    assert not directory.exists()


async def test_deleting_a_model_can_spare_its_weights(client, data_root):
    from llm import storage

    model_id = uuid.uuid4()
    directory = storage.model_dir(str(model_id))
    directory.mkdir(parents=True)
    with store.session() as db:
        q.upsert_model(db, model_id=model_id, name="kept")

    body = (await client.delete(f"/models/{model_id}?weights=false")).json()

    assert body["weights_removed"] is False
    assert directory.exists()




async def test_a_fine_tune_reports_the_window_of_the_base_it_started_from(client, recorded):
    """A fine-tune does not choose a context window — it inherits the one its
    base's positions already encode, and nothing writes that onto its own row.
    Without this the chat UI has no window to measure a conversation against
    and simply hides the meter when you pick such a model."""
    with store.session() as db:
        tuned = q.upsert_model(
            db,
            model_id=uuid.uuid4(),
            name="tuned",
            family="sft",
            parameters={"config": {"base_model": str(recorded)}},
        ).id

    listed = {m["name"]: m for m in (await client.get("/models")).json()["models"]}
    assert listed["tuned"]["context"] == listed["rowed"]["context"] == 256


async def test_a_fine_tune_of_itself_does_not_resolve_forever(client):
    """A config that names its own model as the base is nonsense, but it is
    reachable nonsense — the chain has to terminate rather than recurse."""
    model_id = uuid.uuid4()
    with store.session() as db:
        q.upsert_model(
            db,
            model_id=model_id,
            name="ouroboros",
            family="sft",
            parameters={"config": {"base_model": str(model_id)}},
        )

    listed = {m["name"]: m for m in (await client.get("/models")).json()["models"]}
    assert listed["ouroboros"]["context"] is None
