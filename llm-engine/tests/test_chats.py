"""Conversations: who can see them, what a turn writes, and what erases them."""

import uuid

import pytest

import store
from api import auth
from store import queries as q
from store.schema import Role

OWNER = uuid.uuid4()
STRANGER = uuid.uuid4()


@pytest.fixture
def acting(engine_client):
    """A client that speaks for `OWNER`, and can be told to speak for someone else.

    The LLM engine reads the actor from the service token, so a test overrides the
    dependency rather than minting one — the token itself is test_auth's subject.
    """
    import main

    def act_as(caller):
        main.app.dependency_overrides[auth.require_service] = lambda: str(caller)

    act_as(OWNER)
    engine_client.act_as = act_as
    yield engine_client
    main.app.dependency_overrides.clear()


async def new_chat(client, title="Untitled") -> str:
    response = await client.post("/chats", json={"title": title})
    assert response.status_code == 201
    return response.json()["chat_id"]


async def test_a_chat_belongs_to_whoever_the_token_acts_for(acting):
    chat_id = await new_chat(acting, "first")

    listed = (await acting.get("/chats")).json()["chats"]
    assert [c["chat_id"] for c in listed] == [chat_id]

    with store.session() as db:
        assert q.owned_chat(db, uuid.UUID(chat_id), OWNER) is not None
        assert q.owned_chat(db, uuid.UUID(chat_id), STRANGER) is None


async def test_another_owners_chat_is_absent_rather_than_forbidden(acting):
    """404, not 403 — telling a stranger the id exists is the leak."""
    with store.session() as db:
        theirs = q.create_chat(db, STRANGER, "not yours").id

    assert (await acting.get(f"/chats/{theirs}")).status_code == 404
    assert (await acting.delete(f"/chats/{theirs}")).status_code == 404


async def test_the_backend_acting_as_itself_owns_no_chats(acting):
    acting.act_as(auth.SYSTEM)
    assert (await acting.get("/chats")).status_code == 403


async def test_a_turn_stores_the_question_and_the_reply(acting, recorded_trace):
    from tests.fixtures.tiny_env import TINY_MODEL

    chat_id = await new_chat(acting)
    response = await acting.post(
        f"/chats/{chat_id}/messages",
        json={"model_name": TINY_MODEL, "text": "the cat sat on the", "max_tokens": 8},
    )
    assert response.status_code == 201
    assert response.json()["role"] == "assistant"

    stored = (await acting.get(f"/chats/{chat_id}")).json()["messages"]
    assert [m["role"] for m in stored] == ["user", "assistant"]
    assert stored[1]["content"] == response.json()["content"]


async def test_context_floor_reports_nothing_to_prune_for_a_short_chat(acting):
    from tests.fixtures.tiny_env import TINY_MODEL

    chat_id = await new_chat(acting)
    with store.session() as db:
        first = q.add_message(
            db, chat_id=uuid.UUID(chat_id), role=Role.user, content="hi", tokens=3
        )
        q.add_message(
            db, chat_id=uuid.UUID(chat_id), role=Role.assistant, content="hello", tokens=3
        )

    response = await acting.get(
        f"/chats/{chat_id}/context-floor", params={"model_name": TINY_MODEL}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pruned_turns"] == 0
    assert body["pruned_tokens"] == 0
    assert body["floor_message_id"] == str(first.id)


async def test_context_floor_prunes_the_oldest_turns_past_ctx_window(acting):
    """`tiny`'s ctx_window is 128, so a chat many times that size has real
    history to cut — and the sum of what's reported pruned has to match
    exactly what actually got left out, not just be "some positive number"."""
    from tests.fixtures.tiny_env import TINY_MODEL

    filler = "the quick brown fox jumps over the lazy dog and then does it again " * 3
    chat_id = await new_chat(acting)
    with store.session() as db:
        for i in range(12):
            q.add_message(
                db,
                chat_id=uuid.UUID(chat_id),
                role=Role.user if i % 2 == 0 else Role.assistant,
                content=filler,
                tokens=50,
            )

    # The authoritative order — same query `context-floor` and every prompt
    # builds `history` from — rather than trusting these inserts landed in the
    # order this loop made them: same-transaction timestamps can tie.
    messages = (await acting.get(f"/chats/{chat_id}")).json()["messages"]

    response = await acting.get(
        f"/chats/{chat_id}/context-floor", params={"model_name": TINY_MODEL}
    )
    assert response.status_code == 200
    body = response.json()

    assert 0 < body["pruned_turns"] < len(messages)
    kept_from = body["pruned_turns"]
    assert body["floor_message_id"] == messages[kept_from]["message_id"]
    assert body["pruned_tokens"] == 50 * body["pruned_turns"]


async def test_history_floor_excludes_pruned_turns_from_the_next_prompt(acting):
    """The floor `/context-floor` reports, resent as `history_floor`, actually
    shrinks what the next turn is built from — not just what a report claims.

    Both requests carry the same raised `context_multiplier`, so a floored
    request has just as much *room* to reach back for the pruned turns as the
    unfloored one does; the only thing that can be stopping it is the floor
    itself excluding them before `build_prompt` ever runs."""
    from tests.fixtures.tiny_env import TINY_MODEL

    filler = "the quick brown fox jumps over the lazy dog and then does it again "

    async def seeded_chat() -> str:
        chat_id = await new_chat(acting)
        with store.session() as db:
            for i in range(12):
                q.add_message(
                    db,
                    chat_id=uuid.UUID(chat_id),
                    role=Role.user if i % 2 == 0 else Role.assistant,
                    content=filler,
                    tokens=15,
                )
        return chat_id

    full_chat = await seeded_chat()
    full = await acting.post(
        f"/chats/{full_chat}/messages",
        json={
            "model_name": TINY_MODEL,
            "text": "one more thing",
            "max_tokens": 1,
            "context_multiplier": 4,
        },
    )
    assert full.status_code == 201

    floored_chat = await seeded_chat()
    floor = (
        await acting.get(
            f"/chats/{floored_chat}/context-floor", params={"model_name": TINY_MODEL}
        )
    ).json()
    assert floor["pruned_turns"] > 0, "nothing pruned — the two requests can't differ"

    floored = await acting.post(
        f"/chats/{floored_chat}/messages",
        json={
            "model_name": TINY_MODEL,
            "text": "one more thing",
            "max_tokens": 1,
            "context_multiplier": 4,
            "history_floor": floor["floor_message_id"],
        },
    )
    assert floored.status_code == 201
    assert floored.json()["usage"]["prompt_tokens"] < full.json()["usage"]["prompt_tokens"]


async def test_an_unrecognised_history_floor_falls_back_to_the_full_history(acting):
    """A stale floor — another chat's message id, or one from a chat since
    cleared — is a client's memory being wrong, not this turn's problem."""
    from tests.fixtures.tiny_env import TINY_MODEL

    chat_id = await new_chat(acting)
    response = await acting.post(
        f"/chats/{chat_id}/messages",
        json={
            "model_name": TINY_MODEL,
            "text": "hello",
            "max_tokens": 1,
            "history_floor": str(uuid.uuid4()),
        },
    )
    assert response.status_code == 201


async def test_a_tool_result_is_a_turn_of_its_own(acting):
    """The second pass of a tool round: the backend hands back what the tool
    said, and the conversation grows a `tool` message the next prompt renders."""
    from tests.fixtures.tiny_env import TINY_MODEL

    chat_id = await new_chat(acting)
    await acting.post(
        f"/chats/{chat_id}/messages",
        json={"model_name": TINY_MODEL, "text": "where did the cat sit?", "max_tokens": 4},
    )

    answered = await acting.post(
        f"/chats/{chat_id}/messages",
        json={
            "model_name": TINY_MODEL,
            "max_tokens": 4,
            "tool_result": {
                "tool": "web-search",
                "argument": "where the cat sat",
                "body": "<|par|>on the mat<|/par|>",
            },
        },
    )
    assert answered.status_code == 201

    stored = (await acting.get(f"/chats/{chat_id}")).json()["messages"]
    assert [m["role"] for m in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[2]["content"] == (
        '<|tool-call|>{"tool": "web-search", "argument": "where the cat sat"}<|/tool-call|>'
        "<|par|>on the mat<|/par|>"
    )


async def test_an_interjection_rides_alongside_its_tool_result(acting):
    """A message that arrived mid-turn is stored right after the tool result
    it followed, and both are in the turns the model is prompted with."""
    from tests.fixtures.tiny_env import TINY_MODEL

    chat_id = await new_chat(acting)
    await acting.post(
        f"/chats/{chat_id}/messages",
        json={"model_name": TINY_MODEL, "text": "where did the cat sit?", "max_tokens": 4},
    )

    answered = await acting.post(
        f"/chats/{chat_id}/messages",
        json={
            "model_name": TINY_MODEL,
            "max_tokens": 4,
            "tool_result": {
                "tool": "web-search",
                "argument": "where the cat sat",
                "body": "<|par|>on the mat<|/par|>",
            },
            "interject": "actually, tell me about the dog instead",
        },
    )
    assert answered.status_code == 201

    stored = (await acting.get(f"/chats/{chat_id}")).json()["messages"]
    assert [m["role"] for m in stored] == ["user", "assistant", "tool", "user", "assistant"]
    assert stored[3]["content"] == "actually, tell me about the dog instead"


async def test_a_continuation_with_only_a_tool_result_is_unchanged(acting):
    """No `interject` on the request means exactly one new message, same as
    before this field existed."""
    from tests.fixtures.tiny_env import TINY_MODEL

    chat_id = await new_chat(acting)
    await acting.post(
        f"/chats/{chat_id}/messages",
        json={"model_name": TINY_MODEL, "text": "where did the cat sit?", "max_tokens": 4},
    )

    answered = await acting.post(
        f"/chats/{chat_id}/messages",
        json={
            "model_name": TINY_MODEL,
            "max_tokens": 4,
            "tool_result": {
                "tool": "web-search",
                "argument": "where the cat sat",
                "body": "<|par|>on the mat<|/par|>",
            },
        },
    )
    assert answered.status_code == 201

    stored = (await acting.get(f"/chats/{chat_id}")).json()["messages"]
    assert [m["role"] for m in stored] == ["user", "assistant", "tool", "assistant"]


async def test_an_interjection_without_a_tool_result_is_refused(acting):
    from tests.fixtures.tiny_env import TINY_MODEL

    chat_id = await new_chat(acting)
    response = await acting.post(
        f"/chats/{chat_id}/messages",
        json={"model_name": TINY_MODEL, "text": "hello", "interject": "wait"},
    )
    assert response.status_code == 400
    assert "interject" in response.json()["detail"]


async def test_a_turn_is_either_asked_or_answered_but_not_both(acting):
    from tests.fixtures.tiny_env import TINY_MODEL

    chat_id = await new_chat(acting)
    result = {"tool": "web-search", "argument": "x", "body": "<|par|>y<|/par|>"}

    for body in (
        {"model_name": TINY_MODEL},
        {"model_name": TINY_MODEL, "text": "hello", "tool_result": result},
    ):
        response = await acting.post(f"/chats/{chat_id}/messages", json=body)
        assert response.status_code == 400
        assert "exactly one" in response.json()["detail"]


async def test_an_unknown_tool_is_refused_rather_than_stored(acting):
    from tests.fixtures.tiny_env import TINY_MODEL

    chat_id = await new_chat(acting)
    response = await acting.post(
        f"/chats/{chat_id}/messages",
        json={
            "model_name": TINY_MODEL,
            "tool_result": {"tool": "telepathy", "argument": "x", "body": "y"},
        },
    )
    assert response.status_code == 400
    assert (await acting.get(f"/chats/{chat_id}")).json()["messages"] == []


async def test_a_chats_token_count_is_the_sum_of_its_turns(acting):
    chat_id = await new_chat(acting)
    with store.session() as db:
        q.add_message(db, role=Role.user, chat_id=uuid.UUID(chat_id), content="a", tokens=3)
        q.add_message(db, role=Role.user, chat_id=uuid.UUID(chat_id), content="b", tokens=4)

    listed = (await acting.get("/chats")).json()["chats"]
    assert listed[0]["tokens"] == 7


async def test_renaming_and_deleting_a_chat(acting):
    chat_id = await new_chat(acting, "before")

    renamed = await acting.patch(f"/chats/{chat_id}", json={"title": "after"})
    assert renamed.json()["title"] == "after"

    assert (await acting.delete(f"/chats/{chat_id}")).status_code == 204
    assert (await acting.get(f"/chats/{chat_id}")).status_code == 404


async def test_feedback_is_scoped_through_the_chat(acting):
    """One owner cannot rate another's reply, so this goes through the join
    rather than straight at the message id."""
    with store.session() as db:
        theirs = q.create_chat(db, STRANGER, "not yours")
        reply = q.add_message(db, role=Role.user, chat_id=theirs.id, content="hi", model_id=None)
        message_id = reply.id

    rated = await acting.put(f"/messages/{message_id}/feedback", json={"feedback": "up"})
    assert rated.status_code == 404


async def test_erasing_an_owner_is_the_backends_call_alone(acting):
    with store.session() as db:
        q.create_chat(db, OWNER, "one")
        q.create_chat(db, OWNER, "two")

    assert (await acting.delete(f"/owners/{OWNER}")).status_code == 403

    acting.act_as(auth.SYSTEM)
    erased = await acting.delete(f"/owners/{OWNER}")
    assert erased.status_code == 200
    assert erased.json()["chats_deleted"] == 2


async def test_erasing_an_owner_twice_is_not_an_error(acting):
    """A half-failed account deletion is fixed by retrying it."""
    with store.session() as db:
        q.create_chat(db, OWNER, "one")

    acting.act_as(auth.SYSTEM)
    assert (await acting.delete(f"/owners/{OWNER}")).json()["chats_deleted"] == 1
    assert (await acting.delete(f"/owners/{OWNER}")).json()["chats_deleted"] == 0
