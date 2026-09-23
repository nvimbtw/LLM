"""Tokens counted from the turns themselves, per day and per model."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

import store
from api import auth
from store import queries as q
from store.schema import Role

OWNER = uuid.uuid4()
STRANGER = uuid.uuid4()


@pytest.fixture
def client(engine_client):
    import main

    main.app.dependency_overrides[auth.require_service] = lambda: str(OWNER)
    yield engine_client
    main.app.dependency_overrides.clear()


def turn(db, chat_id, role, tokens, *, days_ago=0, model_id=None):
    message = q.add_message(
        db, chat_id=chat_id, role=role, content="x", tokens=tokens, model_id=model_id
    )
    message.created_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return message


async def test_a_question_is_asked_and_a_reply_is_generated(client):
    with store.session() as db:
        chat = q.create_chat(db, OWNER, "c")
        turn(db, chat.id, Role.user, 3)
        turn(db, chat.id, Role.assistant, 17)

    [day] = (await client.get("/usage")).json()["series"]
    assert (day["asked"], day["generated"], day["replies"]) == (3, 17, 1)


async def test_another_owners_turns_are_not_counted(client):
    with store.session() as db:
        theirs = q.create_chat(db, STRANGER, "not yours")
        turn(db, theirs.id, Role.assistant, 99)

    assert (await client.get("/usage")).json()["series"] == []


async def test_the_window_is_what_the_caller_asks_for(client):
    with store.session() as db:
        chat = q.create_chat(db, OWNER, "c")
        turn(db, chat.id, Role.assistant, 5, days_ago=1)
        turn(db, chat.id, Role.assistant, 5, days_ago=20)

    assert len((await client.get("/usage?days=7")).json()["series"]) == 1
    assert len((await client.get("/usage?days=30")).json()["series"]) == 2


async def test_replies_are_grouped_by_the_model_that_wrote_them(client):
    with store.session() as db:
        model = q.upsert_model(db, model_id=uuid.uuid4(), name="tiny")
        chat = q.create_chat(db, OWNER, "c")
        turn(db, chat.id, Role.assistant, 10, model_id=model.id)
        turn(db, chat.id, Role.assistant, 4, model_id=model.id)
        turn(db, chat.id, Role.assistant, 1)

    by_model = {m["name"]: m for m in (await client.get("/usage")).json()["by_model"]}

    assert by_model["tiny"]["tokens"] == 14
    assert by_model["tiny"]["replies"] == 2
    # A reply whose model is gone still owns the tokens it generated.
    assert by_model[None]["tokens"] == 1


async def test_the_backend_acting_as_itself_has_no_usage(engine_client):
    import main

    main.app.dependency_overrides[auth.require_service] = lambda: auth.SYSTEM
    try:
        assert (await engine_client.get("/usage")).status_code == 403
    finally:
        main.app.dependency_overrides.clear()


async def test_a_window_outside_the_allowed_range_is_refused(client):
    assert (await client.get("/usage?days=0")).status_code == 422
    assert (await client.get("/usage?days=400")).status_code == 422
