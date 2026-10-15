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


@pytest.mark.parametrize("family,ancestor_field", [("sft", "base_model"), ("rl", "base_model"), ("pretrain", "resume_from")])
async def test_legacy_continuations_resolve_shape_vocabulary_and_measured_parameters(client, family, ancestor_field):
    """List and detail inherit missing facts, preserving each continuation's own training provenance."""
    architecture = {"dimensions": 512, "heads": 8, "layers": 16, "ff_multiplier": 4, "ctx_window": 1024}
    with store.session() as db:
        base = q.upsert_model(db, model_id=uuid.uuid4(), name="base", parameters={
            "config": {**architecture, "max_vocab": 1024},
            "result": {"vocab": 8192, "parameters": 54600704},
        })
        child = q.upsert_model(db, model_id=uuid.uuid4(), name="child", family=family, parameters={
            "config": {ancestor_field: str(base.id), "epochs": 2},
            "result": {"dataset": {"conversations": 20}},
        }).id
    listed = next(row for row in (await client.get("/models")).json()["models"] if row["model_id"] == str(child))
    detail = (await client.get(f"/models/{child}")).json()
    assert listed == detail
    assert detail["architecture"] == architecture
    assert detail["context"] == 1024
    assert detail["vocab_size"] == 8192
    assert detail["parameter_count"] == 54600704
    assert detail["parameters"]["result"]["dataset"] == {"conversations": 20}
    assert "dimensions" not in detail["parameters"]["config"]


async def test_results_override_stale_requested_shape_and_do_not_require_a_live_base(client):
    """Saved results remain authoritative after an ancestor is deleted or a request used defaults."""
    architecture = {"dimensions": 32, "heads": 2, "layers": 2, "ff_multiplier": 4, "ctx_window": 256}
    with store.session() as db:
        model_id = q.upsert_model(db, model_id=uuid.uuid4(), name="resolved", parameters={
            "config": {**architecture, "dimensions": 512, "base_model": str(uuid.uuid4())},
            "result": {"architecture": architecture, "parameters": 50000, "dataset": {"vocab_size": 400}},
        }).id
    body = (await client.get(f"/models/{model_id}")).json()
    assert body["architecture"] == architecture
    assert body["parameter_count"] == 50000
    assert body["vocab_size"] == 400


async def test_legacy_disk_metadata_supplies_resolved_facts(client, data_root):
    """A row without provenance can recover its own facts from meta.json without loading weights."""
    import json
    from llm import storage

    model_id = uuid.uuid4()
    directory = storage.model_dir(str(model_id))
    directory.mkdir(parents=True)
    architecture = {"dimensions": 32, "heads": 2, "layers": 2, "ff_multiplier": 4, "ctx_window": 256}
    (directory / "meta.json").write_text(json.dumps({"config": architecture, "dataset": {"vocab_size": 400, "parameters": 50000}}))
    with store.session() as db:
        q.upsert_model(db, model_id=model_id, name="disk-metadata")
    body = (await client.get(f"/models/{model_id}")).json()
    assert body["architecture"] == architecture
    assert body["vocab_size"] == 400
    assert body["parameter_count"] == 50000


async def test_ancestry_cycles_and_invalid_ids_preserve_unavailable_facts(client):
    """Corrupt ancestry terminates without fabricating shape or count from unrelated defaults."""
    first, second = uuid.uuid4(), uuid.uuid4()
    with store.session() as db:
        q.upsert_model(db, model_id=first, name="cycle-a", parameters={"config": {"base_model": str(second)}})
        q.upsert_model(db, model_id=second, name="cycle-b", parameters={"config": {"resume_from": str(first)}})
        invalid = q.upsert_model(db, model_id=uuid.uuid4(), name="invalid-base", parameters={"config": {"base_model": "not-a-uuid", "max_vocab": 8192}}).id
    for model_id in (first, invalid):
        body = (await client.get(f"/models/{model_id}")).json()
        assert body["architecture"] is body["vocab_size"] is body["parameter_count"] is None


async def test_admin_size_category_is_explicit_persisted_and_clearable(client, recorded):
    """Display size is independent of parameter count and does not overwrite training or recovery metadata."""
    response = await client.patch(f"/models/{recorded}", json={"size": "flagship", "recovery_keep": True})
    assert response.status_code == 200
    body = (await client.get(f"/models/{recorded}")).json()
    assert body["size"] == "flagship"
    assert body["recovery_kept"] is True
    assert body["parameters"]["config"] == {"dimensions": 32, "layers": 2}
    assert (await client.patch(f"/models/{recorded}", json={"description": "Edited"})).json()["size"] == "flagship"
    assert (await client.patch(f"/models/{recorded}", json={"size": "automatic"})).status_code == 422
    assert (await client.patch(f"/models/{recorded}", json={"size": None})).json()["size"] is None
    assert (await client.get(f"/models/{recorded}")).json()["parameters"]["recovery_keep"] is True


async def test_invalid_and_conflicting_display_edits_preserve_model_metadata(client, recorded):
    """Invalid fields and name collisions fail without partially applying a size or description edit."""
    with store.session() as db:
        q.upsert_model(db, model_id=uuid.uuid4(), name="taken")
    assert (await client.patch(f"/models/{recorded}", json={"display_name": "  "})).status_code == 422
    assert (await client.patch(f"/models/{recorded}", json={"family": "  "})).status_code == 422
    assert (await client.patch(f"/models/{recorded}", json={"display_name": "taken", "size": "flagship"})).status_code == 409
    body = (await client.get(f"/models/{recorded}")).json()
    assert body["name"] == "rowed"
    assert body["size"] is None
