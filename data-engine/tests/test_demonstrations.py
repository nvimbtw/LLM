"""Gym demonstrations, arriving as a dataset like any other.

The reason they come here at all rather than into a corpus path in the
LLM engine: it owns no corpora, so a demonstration set that lived beside one
would be the LLM engine owning a corpus again. Landing them here means the SFT build
needs no new reader — it selects, caps and keys these exactly as it does
something pulled from the Hub, and they can be mixed into one fine-tune with a
chat corpus, which matters because three task types on their own train a model
that only answers gym questions.
"""

import uuid

import store
from store import queries as q
from store.schema import DatasetKind, DatasetOrigin, DatasetStatus

BODY = "<|codeblock|>notes.txt:3:ANSWER_2ee433<|/codeblock|>"


def conversation(answer: str = "notes.txt") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You have a shell."},
            {"role": "user", "content": "Which file contains ANSWER_2ee433?"},
            # The assistant names the call it makes, the tool the one it
            # answers. Both halves are what the backend now writes, and the
            # first is what puts the command in the region SFT trains on.
            {
                "role": "assistant",
                "content": "<|par|>Searching.<|/par|>",
                "tool": "bash",
                "argument": "grep -rn ANSWER_2ee433 .",
            },
            {
                "role": "tool",
                "content": BODY,
                "tool": "bash",
                "argument": "grep -rn ANSWER_2ee433 .",
            },
            {"role": "assistant", "content": answer},
        ],
        "meta": {"task_type": "bash_grep", "seed": 4},
    }


def payload(**over) -> dict:
    return {
        "name": "bash-gym-demonstrations",
        "description": "Expert trajectories for the bash gym.",
        "steps": [{"gym": "bash", "task_types": ["bash_grep"], "max_output_chars": 4000}],
        "conversations": [conversation()],
        **over,
    }


async def test_a_batch_creates_the_set_and_lands_in_it(client, store_session):
    response = await client.post("/datasets/demonstrations", json=payload())

    assert response.status_code == 201
    body = response.json()
    assert body["written"] == 1
    assert body["document_count"] == 1

    with store.session() as db:
        dataset = q.dataset_by_name(db, "bash-gym-demonstrations")
        assert dataset.kind is DatasetKind.sft
        assert dataset.origin is DatasetOrigin.gym
        # Ready on arrival: every row is complete when it is written, so there
        # is no interval where calling this half-written would be true.
        assert dataset.status is DatasetStatus.ready
        # A demonstration is a fact about a task, not about who asked for it.
        assert dataset.owner_id is None


async def test_a_second_batch_appends_rather_than_starting_over(client, store_session):
    await client.post("/datasets/demonstrations", json=payload())
    second = await client.post(
        "/datasets/demonstrations",
        json=payload(conversations=[conversation("main.py"), conversation("utils.py")]),
    )

    assert second.json()["document_count"] == 3
    with store.session() as db:
        dataset = q.dataset_by_name(db, "bash-gym-demonstrations")
        rows = q.documents(db, dataset.id, offset=0, limit=10)
        # Ordinals continue rather than restarting, which is what makes a cap of
        # "the first N documents" mean anything across two writes.
        assert [row.ordinal for row in rows] == [0, 1, 2]


async def test_the_counters_the_corpus_key_reads_are_kept_current(client, store_session):
    """`Selected.identity` keys on `document_count` and `updated_at`. A set that
    grew without moving either would let a run reuse a corpus built from fewer
    demonstrations and report success."""
    await client.post("/datasets/demonstrations", json=payload())
    with store.session() as db:
        before = q.dataset_by_name(db, "bash-gym-demonstrations")
        first_count, first_bytes = before.document_count, before.byte_size

    await client.post("/datasets/demonstrations", json=payload())
    with store.session() as db:
        after = q.dataset_by_name(db, "bash-gym-demonstrations")

    assert after.document_count > first_count
    assert after.byte_size > first_bytes


async def test_a_tool_turn_keeps_the_call_it_answers(client, store_session):
    """Stored unrendered, deliberately. The body was rendered by the backend's
    Rust and the `<|tool-call|>` header is added by the LLM engine at build time, so
    this service never spells a layout it cannot verify."""
    await client.post("/datasets/demonstrations", json=payload())

    with store.session() as db:
        dataset = q.dataset_by_name(db, "bash-gym-demonstrations")
        messages = q.documents(db, dataset.id, offset=0, limit=1)[0].messages

    tool = next(m for m in messages if m["role"] == "tool")
    assert tool["tool"] == "bash"
    assert tool["argument"] == "grep -rn ANSWER_2ee433 ."
    assert tool["content"] == BODY
    assert "<|tool-call|>" not in tool["content"]


async def test_the_provenance_says_what_produced_the_set(client, store_session):
    """Including the output budget. A body is clamped when it is written, so a
    set rendered at one budget is not interchangeable with one rendered at
    another — and nothing in the LLM engine's corpus key can see that."""
    await client.post("/datasets/demonstrations", json=payload())

    with store.session() as db:
        steps = q.dataset_by_name(db, "bash-gym-demonstrations").steps

    assert steps[0]["gym"] == "bash"
    assert steps[0]["max_output_chars"] == 4000


async def test_a_conversation_that_does_not_end_on_an_answer_is_refused(client):
    """The LLM engine would drop it at build time — a conversation trailing off on a
    tool result ends on tokens nothing is trained on. Refusing here makes it a
    422 naming the row instead of a lower conversation count noticed weeks
    later."""
    ending_on_the_shell = conversation()
    ending_on_the_shell["messages"] = ending_on_the_shell["messages"][:-1]

    response = await client.post(
        "/datasets/demonstrations", json=payload(conversations=[ending_on_the_shell])
    )

    assert response.status_code == 422
    assert "conversation 0" in response.json()["detail"]


async def test_half_a_tool_call_is_refused(client):
    """The pairing `ChatTurn` enforces in the LLM engine. Half of it says nothing —
    and would silently store as an already-rendered turn, which it is not."""
    half = conversation()
    half["messages"][3] = {"role": "tool", "content": BODY, "tool": "bash"}

    response = await client.post("/datasets/demonstrations", json=payload(conversations=[half]))

    assert response.status_code == 422


async def test_an_assistant_turn_may_name_the_call_it_makes(client, store_session):
    """This used to be refused, and refusing it was the bug.

    SFT trains assistant turns and masks tool turns, so a demonstration that
    could only hang a command on the tool turn was a demonstration whose command
    carried no gradient — it taught a model to narrate a search and then end its
    turn without calling anything. Both turns naming the same call is what a
    live chat already looks like.
    """
    response = await client.post(
        "/datasets/demonstrations", json=payload(conversations=[conversation()])
    )

    assert response.status_code == 201
    with store.session() as db:
        dataset_id = uuid.UUID(response.json()["dataset_id"])
        stored = q.documents(db, dataset_id, offset=0, limit=1)[0].messages
    calling = [m for m in stored if m["role"] == "assistant" and m.get("tool")]
    assert len(calling) == 1
    assert calling[0]["argument"] == "grep -rn ANSWER_2ee433 ."


async def test_a_user_turn_may_not_name_a_call(client):
    """The rule that survives: only a turn that makes or answers a call names
    one. A user turn does neither, and a call on one would render a marker into
    a region the model is asked to condition on."""
    named_user = conversation()
    named_user["messages"][1] = {
        "role": "user", "content": "hi", "tool": "bash", "argument": "ls",
    }

    response = await client.post(
        "/datasets/demonstrations", json=payload(conversations=[named_user])
    )

    assert response.status_code == 422


async def test_nothing_is_written_when_one_conversation_in_the_batch_is_bad(client, store_session):
    """Validation runs over the whole batch before the session is opened. A
    partial write would leave a set whose count is right and whose contents are
    a prefix, which is indistinguishable from a smaller run."""
    broken = conversation()
    broken["messages"][-1] = {"role": "user", "content": "?"}

    response = await client.post(
        "/datasets/demonstrations", json=payload(conversations=[conversation(), broken])
    )

    assert response.status_code == 422
    with store.session() as db:
        assert q.dataset_by_name(db, "bash-gym-demonstrations") is None
