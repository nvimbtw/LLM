"""Where a gym's expert trajectories land, so they can be trained on.

A demonstration is a conversation: the task as a user turn, then the model's
commands and what the shell said back, then the answer alone as the last turn.
That is an `sft` dataset like any other here, which is the whole point of
routing it through this service rather than inventing a corpus path in the LLM
engine — the engine owns no corpora, and a demonstration set that lived beside
one would be exactly that. Selected, capped, keyed, previewed, processed and
deleted like anything pulled from the Hub.

**Nothing is rendered here.** Turns arrive carrying a call as `tool` and
`argument`: the `assistant` turns name the call they make, the `tool` turns the
one they answer along with a body the backend already rendered with the same
Rust function that serves a live round. The engine spells the `<|tool-call|>`
block at build time, and where it puts it depends on which of those two a turn
is. This service adds no third spelling of a layout it cannot verify — see
`contracts/shell-observation.json` for what it costs when one does.

The writer is the backend, never the gyms service: the two Python services never
speak, and the backend is the one process that already talks to both.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import store
from api import auth
from store import queries as q

router = APIRouter(tags=["demonstrations"])

#: One request's worth. A demonstration batch is generated, not collected, so a
#: caller wanting more can simply ask again into the same name — which appends.
#: The ceiling bounds validation and the dataset row lock while still allowing
#: a multi-thousand expert batch to land in one request.
MAX_CONVERSATIONS = 5_000

#: Matches `datasets.name`'s column width.
MAX_NAME_CHARS = 120

ROLES = ("system", "user", "assistant", "tool")


class Message(BaseModel):
    """One turn. `tool`/`argument` are given together or not at all, the same
    pairing `ChatTurn` enforces in the engine and for the same reason: they are
    what tells the build to render this body into the block that answers a call,
    and half of that pair says nothing."""

    role: str
    content: str
    tool: str | None = None
    argument: str | None = None


class Conversation(BaseModel):
    messages: list[Message] = Field(min_length=2)
    meta: dict | None = None


class NewDemonstrations(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_NAME_CHARS)
    description: str = ""
    #: Provenance, stored on the dataset the way a derived one records its
    #: pipeline: which gym, which task types, which seeds, and the output budget
    #: the observations were rendered at. That last one matters and is easy to
    #: forget — a body is clamped when it is written, so a set rendered at one
    #: budget is not interchangeable with one rendered at another, and nothing
    #: in the engine's corpus key can see the difference.
    steps: list = Field(default_factory=list)
    conversations: list[Conversation] = Field(min_length=1, max_length=MAX_CONVERSATIONS)


def _validate(conversation: Conversation, index: int) -> list[dict]:
    """One conversation as it will be stored, or a 422 naming which one failed.

    Refused up front rather than skipped at build time. A malformed row in a
    corpus pulled from the Hub is somebody else's mistake and costs that row;
    one here is our own, arriving from a service we wrote, and silently storing
    it would mean discovering it as a lower conversation count in a fine-tune
    weeks later.
    """
    messages = []
    for message in conversation.messages:
        where = f"conversation {index}"
        if message.role not in ROLES:
            raise HTTPException(422, f"{where}: unusable role {message.role!r}")
        if (message.tool is None) != (message.argument is None):
            raise HTTPException(422, f"{where}: a tool and its argument go together")
        # An assistant turn names the call it *makes*; a tool turn names the one
        # it *answers*. Both are legitimate and they render into different
        # positions — the engine puts the assistant's marker after its text,
        # inside the trained region, and the tool's at the head of the masked
        # result. A user or system turn names neither.
        if message.tool is not None and message.role not in ("tool", "assistant"):
            raise HTTPException(422, f"{where}: only a tool or assistant turn names a call")
        stored = {"role": message.role, "content": message.content}
        if message.tool is not None:
            stored["tool"] = message.tool
            stored["argument"] = message.argument
        messages.append(stored)

    # The engine drops a conversation that does not end on an assistant turn:
    # one ending anywhere else trails off on tokens nothing is trained on, so it
    # costs context and teaches no ending. A demonstration that ends on the
    # shell is a bug in the generator, not data.
    if messages[-1]["role"] != "assistant":
        raise HTTPException(422, f"conversation {index}: ends on {messages[-1]['role']}, not an answer")
    return messages


@router.post("/datasets/demonstrations", status_code=201)
def demonstrations(body: NewDemonstrations, _: str = Depends(auth.require_service)):
    """Append a batch of demonstrations to a named set, creating it on first use.

    A 201 either way. The set is not a resource a caller creates and then fills
    — it is a name that accumulates, so "did this exist already" is not a
    distinction the caller has to make or can act on.
    """
    rows = [
        {"messages": _validate(conversation, index), "meta": conversation.meta}
        for index, conversation in enumerate(body.conversations)
    ]

    with store.session() as db:
        dataset = q.gym_dataset(db, body.name, body.description, body.steps)
        written = q.add_conversations(db, dataset.id, rows)
        return {
            "dataset_id": str(dataset.id),
            "name": dataset.name,
            "written": written,
            "document_count": dataset.document_count,
        }
