"""Vocabulary metadata edits preserve artifact identity and recorded provenance."""

import uuid

import pytest

import store
from api import auth
from store import queries as q


@pytest.fixture
def application(tmp_path, monkeypatch):
    """Mount the real router and service guard against a disposable store and artifact tree."""
    from fastapi import Depends, FastAPI
    from api.routes.tokenizers import router

    previous = store._sessions
    factory = store.configure(f"sqlite+pysqlite:///{tmp_path / 'store.sqlite3'}", create=True)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "artifacts"))
    app = FastAPI(dependencies=[Depends(auth.require_service)])
    app.include_router(router)
    app.dependency_overrides[auth.require_service] = lambda: auth.SYSTEM
    try:
        yield app
    finally:
        factory.kw["bind"].dispose()
        store._sessions = previous


@pytest.fixture
async def client(application):
    """Exercise vocabulary HTTP contracts without starting model execution or a queue runner."""
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://engine") as caller:
        yield caller


@pytest.fixture
def vocabulary(application):
    """Index a vocabulary whose artifacts are unavailable, retaining its measured provenance."""
    identity = uuid.uuid4()
    with store.session() as db:
        q.upsert_tokenizer(db, tokenizer_id=identity, name="Original", description="Before",
                           vocab_size=512, merges=192, fingerprint="f" * 64, implicit=True)
    return identity


async def test_metadata_persists_without_changing_identity_or_model_references(client, vocabulary):
    """Renaming survives a separate read and does not rewrite model configuration or fingerprint."""
    with store.session() as db:
        base = q.upsert_model(db, model_id=uuid.uuid4(), name="Base",
                              parameters={"result": {"tokenizer_id": str(vocabulary)}})
        child = q.upsert_model(db, model_id=uuid.uuid4(), name="Child",
                               parameters={"config": {"base_model": str(base.id)}})
    response = await client.patch(f"/tokenizers/{vocabulary}",
                                  json={"display_name": "  Renamed  ", "description": "After"})
    assert response.status_code == 200
    saved = (await client.get(f"/tokenizers/{vocabulary}")).json()
    assert saved["name"] == "Renamed"
    assert saved["description"] == "After"
    assert saved["tokenizer_id"] == str(vocabulary)
    assert saved["fingerprint"] == "f" * 64
    assert saved["implicit"] is True
    assert {entry["model_id"] for entry in saved["models"]} == {str(base.id), str(child.id)}
    with store.session() as db:
        assert q.get_model(db, base.id).parameters["result"]["tokenizer_id"] == str(vocabulary)


@pytest.mark.parametrize("name", ["", "  ", "x" * 121])
async def test_invalid_names_do_not_partially_change_description(client, vocabulary, name):
    """Boundary validation rejects blank and oversized labels before any field is saved."""
    response = await client.patch(f"/tokenizers/{vocabulary}",
                                  json={"display_name": name, "description": "Lost"})
    assert response.status_code == 422
    assert (await client.get(f"/tokenizers/{vocabulary}")).json()["description"] == "Before"


async def test_duplicate_name_rolls_back_all_edits(client, vocabulary):
    """A uniqueness conflict returns 409 and preserves the prior name and description."""
    with store.session() as db:
        q.upsert_tokenizer(db, tokenizer_id=uuid.uuid4(), name="Taken")
    response = await client.patch(f"/tokenizers/{vocabulary}",
                                  json={"display_name": "Taken", "description": "Lost"})
    assert response.status_code == 409
    saved = (await client.get(f"/tokenizers/{vocabulary}")).json()
    assert (saved["name"], saved["description"]) == ("Original", "Before")


async def test_unknown_identity_and_non_system_actor_are_refused(client, application):
    """Unknown resources return 404; an ordinary acting user cannot edit shared metadata."""
    path = f"/tokenizers/{uuid.uuid4()}"
    assert (await client.patch(path, json={"display_name": "New"})).status_code == 404
    application.dependency_overrides[auth.require_service] = lambda: str(uuid.uuid4())
    assert (await client.patch(path, json={"display_name": "New"})).status_code == 403


async def test_disk_only_build_gains_metadata_without_writing_artifacts(client, monkeypatch):
    """An unregistered vocabulary can be renamed while its existing fingerprint stays fixed."""
    from llm.corpus import vocab

    identity = uuid.uuid4()
    built = {"tokenizer_id": str(identity), "fingerprint": "a" * 64,
             "vocab_size": 512, "merges": 192, "max_vocab": 512, "datasets": []}
    monkeypatch.setattr(vocab, "info", lambda value: built if value == str(identity) else None)
    response = await client.patch(f"/tokenizers/{identity}", json={"display_name": "Indexed"})
    assert response.status_code == 200
    assert response.json()["fingerprint"] == built["fingerprint"]
    with store.session() as db:
        assert q.get_tokenizer(db, identity).name == "Indexed"
