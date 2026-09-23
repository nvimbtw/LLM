"""Uploads: who can reach one, what binds it to a turn, and what nothing does
with it.

The last is the point of the feature and the easiest thing to regress: an
attachment is stored and served back, and no part of a prompt is built from it.
"""

import uuid

import pytest

import store
from api import auth
from api.routes import attachments
from store import queries as q
from store.schema import Role

OWNER = uuid.uuid4()
STRANGER = uuid.uuid4()


@pytest.fixture
def acting(engine_client):
    import main

    def act_as(caller):
        main.app.dependency_overrides[auth.require_service] = lambda: str(caller)

    act_as(OWNER)
    engine_client.act_as = act_as
    yield engine_client
    main.app.dependency_overrides.clear()


async def upload(client, name="note.txt", body=b"hello", media_type="text/plain") -> dict:
    response = await client.post(
        "/attachments", files={"file": (name, body, media_type)}
    )
    assert response.status_code == 201, response.text
    return response.json()


async def test_an_upload_comes_back_byte_for_byte(acting):
    body = b"\x89PNG\r\n\x1a\n not really a png"
    stored = await upload(acting, "shot.png", body, "image/png")

    assert stored["filename"] == "shot.png"
    assert stored["size"] == len(body)

    fetched = await acting.get(f"/attachments/{stored['attachment_id']}")
    assert fetched.status_code == 200
    assert fetched.content == body
    assert fetched.headers["content-type"].startswith("image/png")


async def test_another_owners_upload_is_absent_rather_than_forbidden(acting):
    theirs = await upload(acting)
    acting.act_as(STRANGER)

    assert (await acting.get(f"/attachments/{theirs['attachment_id']}")).status_code == 404
    assert (await acting.delete(f"/attachments/{theirs['attachment_id']}")).status_code == 404


async def test_an_oversized_upload_is_refused(acting, monkeypatch):
    monkeypatch.setattr(attachments, "MAX_BYTES", 8)
    response = await acting.post(
        "/attachments", files={"file": ("big.bin", b"x" * 9, "application/octet-stream")}
    )

    assert response.status_code == 413


async def test_an_empty_file_is_not_an_attachment(acting):
    response = await acting.post("/attachments", files={"file": ("empty.txt", b"", "text/plain")})

    assert response.status_code == 400


async def test_a_filename_is_a_label_and_never_a_path(acting):
    stored = await upload(acting, "../../etc/passwd")

    assert stored["filename"] == "passwd"


async def test_a_turn_binds_its_attachments_and_the_history_reports_them(
    acting, recorded_trace
):
    from tests.fixtures.tiny_env import TINY_MODEL

    staged = await upload(acting, "diagram.png", b"bytes", "image/png")
    chat_id = (await acting.post("/chats", json={"title": "with a file"})).json()["chat_id"]

    reply = await acting.post(
        f"/chats/{chat_id}/messages",
        json={
            "model_name": TINY_MODEL,
            "text": "what is this",
            "attachment_ids": [staged["attachment_id"]],
            "max_tokens": 4,
        },
    )
    assert reply.status_code == 201

    messages = (await acting.get(f"/chats/{chat_id}")).json()["messages"]
    asked = next(m for m in messages if m["role"] == "user")
    assert [a["attachment_id"] for a in asked["attachments"]] == [staged["attachment_id"]]
    # The reply carries none: a file is attached to a question, not to an answer.
    assert all(m["attachments"] == [] for m in messages if m["role"] == "assistant")


async def test_the_question_the_model_is_given_says_nothing_about_the_file(
    acting, recorded_trace
):
    """The whole bargain: stored, shown back, never prompted with."""
    from tests.fixtures.tiny_env import TINY_MODEL

    staged = await upload(acting, "secret-report.pdf", b"%PDF-1.7", "application/pdf")
    chat_id = (await acting.post("/chats", json={"title": "quiet"})).json()["chat_id"]

    await acting.post(
        f"/chats/{chat_id}/messages",
        json={
            "model_name": TINY_MODEL,
            "text": "summarise it",
            "attachment_ids": [staged["attachment_id"]],
            "max_tokens": 4,
        },
    )

    with store.session() as db:
        asked = next(
            m for m in q.messages_of(db, uuid.UUID(chat_id)) if m.role is Role.user
        )
        assert asked.content == "summarise it"
        assert "secret-report" not in asked.content


async def test_an_attachment_cannot_be_sent_twice(acting, recorded_trace):
    from tests.fixtures.tiny_env import TINY_MODEL

    staged = await upload(acting)
    chat_id = (await acting.post("/chats", json={"title": "once"})).json()["chat_id"]
    body = {
        "model_name": TINY_MODEL,
        "text": "again",
        "attachment_ids": [staged["attachment_id"]],
        "max_tokens": 4,
    }

    assert (await acting.post(f"/chats/{chat_id}/messages", json=body)).status_code == 201
    assert (await acting.post(f"/chats/{chat_id}/messages", json=body)).status_code == 400


async def test_another_owners_attachment_cannot_be_bound(acting, recorded_trace):
    from tests.fixtures.tiny_env import TINY_MODEL

    with store.session() as db:
        theirs = q.create_attachment(
            db, owner_id=STRANGER, filename="theirs.txt", media_type="text/plain", content=b"x"
        ).id

    chat_id = (await acting.post("/chats", json={"title": "borrowed"})).json()["chat_id"]
    response = await acting.post(
        f"/chats/{chat_id}/messages",
        json={
            "model_name": TINY_MODEL,
            "text": "whose is this",
            "attachment_ids": [str(theirs)],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 400


async def test_a_staged_upload_can_be_discarded_and_a_sent_one_cannot(
    acting, recorded_trace
):
    from tests.fixtures.tiny_env import TINY_MODEL

    staged = await upload(acting)
    assert (await acting.delete(f"/attachments/{staged['attachment_id']}")).status_code == 204

    sent = await upload(acting)
    chat_id = (await acting.post("/chats", json={"title": "kept"})).json()["chat_id"]
    await acting.post(
        f"/chats/{chat_id}/messages",
        json={
            "model_name": TINY_MODEL,
            "text": "keep it",
            "attachment_ids": [sent["attachment_id"]],
            "max_tokens": 4,
        },
    )

    assert (await acting.delete(f"/attachments/{sent['attachment_id']}")).status_code == 404


async def test_deleting_a_chat_takes_its_attachments_with_it(acting, recorded_trace):
    from tests.fixtures.tiny_env import TINY_MODEL

    sent = await upload(acting)
    chat_id = (await acting.post("/chats", json={"title": "doomed"})).json()["chat_id"]
    await acting.post(
        f"/chats/{chat_id}/messages",
        json={
            "model_name": TINY_MODEL,
            "text": "bye",
            "attachment_ids": [sent["attachment_id"]],
            "max_tokens": 4,
        },
    )

    assert (await acting.delete(f"/chats/{chat_id}")).status_code == 204
    with store.session() as db:
        assert q.owned_attachment(db, uuid.UUID(sent["attachment_id"]), OWNER) is None


async def test_a_tool_result_carries_no_attachments(acting):
    from tests.fixtures.tiny_env import TINY_MODEL

    staged = await upload(acting)
    chat_id = (await acting.post("/chats", json={"title": "tools"})).json()["chat_id"]

    response = await acting.post(
        f"/chats/{chat_id}/messages",
        json={
            "model_name": TINY_MODEL,
            "tool_result": {"tool": "web-search", "argument": "x", "body": "y"},
            "attachment_ids": [staged["attachment_id"]],
        },
    )

    assert response.status_code == 400
