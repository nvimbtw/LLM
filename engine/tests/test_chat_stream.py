"""The streamed turn: what reaches the caller, and what is stored when."""

import uuid

import pytest

import store
from api import auth
from store import queries as q
from tests.fixtures import trace_spec
from tests.fixtures.tiny_env import TINY_MODEL

OWNER = uuid.uuid4()


@pytest.fixture
def client(engine_client):
    import main

    main.app.dependency_overrides[auth.require_service] = lambda: str(OWNER)
    yield engine_client
    main.app.dependency_overrides.clear()


@pytest.fixture
def chat_id():
    with store.session() as db:
        return q.create_chat(db, OWNER, "streamed").id


def ask(model: str = TINY_MODEL) -> dict:
    return {
        "model_name": model,
        "text": trace_spec.QUESTION,
        "system": trace_spec.SYSTEM,
        "max_tokens": trace_spec.MAX_TOKENS,
        "seed": trace_spec.SEED,
    }


async def stream(client, chat_id, body) -> list[dict]:
    from tests.test_routes import _sse

    async with client.stream(
        "POST", f"/chats/{chat_id}/messages/stream", json=body
    ) as response:
        assert response.status_code == 200
        payload = "".join([chunk async for chunk in response.aiter_text()])
    return _sse(payload)


async def test_the_frames_concatenate_to_the_stored_reply(client, chat_id):
    events = await stream(client, chat_id, ask())

    tokens = "".join(e["data"]["token"] for e in events if e["event"] == "token")
    [done] = [e for e in events if e["event"] == "done"]

    assert done["data"]["content"] == tokens
    assert done["data"]["role"] == "assistant"

    with store.session() as db:
        stored = q.messages_of(db, chat_id)
        assert [m.role.value for m in stored] == ["user", "assistant"]
        assert stored[1].content == tokens


async def test_the_reply_is_written_once_the_last_token_lands(client, chat_id):
    """Not per token: a dropped connection loses the reply rather than storing
    half of one."""
    before = await stream(client, chat_id, ask())
    assert [e["event"] for e in before][-1] == "done"

    with store.session() as db:
        assert len(q.messages_of(db, chat_id)) == 2


async def test_an_unknown_model_is_refused_before_the_question_is_stored(client, chat_id):
    response = await client.post(
        f"/chats/{chat_id}/messages/stream", json=ask("no-such-model")
    )

    assert response.status_code == 404
    with store.session() as db:
        assert q.messages_of(db, chat_id) == []


async def test_a_stranger_cannot_stream_into_someone_elses_chat(client):
    with store.session() as db:
        theirs = q.create_chat(db, uuid.uuid4(), "not yours").id

    assert (await client.post(f"/chats/{theirs}/messages/stream", json=ask())).status_code == 404


async def test_the_usage_of_a_turn_is_reported_with_it(client, chat_id):
    events = await stream(client, chat_id, ask())
    [done] = [e for e in events if e["event"] == "done"]

    usage = done["data"]["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["generated_tokens"] == done["data"]["tokens"]


async def test_the_question_is_counted_in_the_models_own_tokenizer(
    client, chat_id, recorded_trace
):
    """Counted once at write time, so a chat's cost sums without re-tokenizing."""
    await client.post(f"/chats/{chat_id}/messages", json=ask())

    with store.session() as db:
        asked = q.messages_of(db, chat_id)[0]
        assert asked.tokens == len(recorded_trace["question_ids"])
