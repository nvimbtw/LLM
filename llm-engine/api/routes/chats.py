"""Conversations, scoped to whoever the caller is acting for.

The LLM engine owns these rows because it is what writes them: a reply and the
token count it cost are produced here, and shipping them back to another
database to be stored would make the write a round trip that can half-fail.

`owner_id` comes from the service token, never from the request body — the
backend is trusted to say who it acts for, and nothing else gets a say.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import store
from api import auth, sse
from api.routes.attachments import MAX_PER_MESSAGE, describe
from llm.chat import tools
from llm.config import ChatConfig, ChatTurn
from llm.orchestrate import chat as chat_pipeline
from store import queries as q
from store.schema import Chat, Feedback, Message, Role

router = APIRouter(tags=["chats"])


class NewChat(BaseModel):
    title: str = Field(default="New chat", max_length=200)


class Rename(BaseModel):
    title: str = Field(max_length=200)


class Rating(BaseModel):
    feedback: Feedback | None = None


class ToolResult(BaseModel):
    """What a tool the last reply called came back with.

    `body` is already in the marker vocabulary — the caller that ran the tool
    knows the shape of its answer, and this side only decides where the block
    sits in the conversation.
    """

    tool: str
    argument: str
    body: str


class Ask(BaseModel):
    """A turn to answer. The history comes from the stored messages, so a client
    that lost its copy of the conversation still gets a coherent reply.

    Exactly one of `text` and `tool_result` says what the new turn is: a person
    asking, or a tool answering a call the last reply made. Both is a caller
    that has lost track of which; neither is nothing to answer.

    `interject` rides alongside `tool_result` only: a person's message that
    arrived while the round was generating, spliced in by the backend right
    after the tool result and before the model is asked to continue — see
    `docs/backend/README.md`'s "Tool calls" section for where that buffering
    happens. It has no `attachment_ids` of its own; an interjection is text
    steering mid-turn, not a new question with its own uploads, and keeping
    that off the wire keeps this side's contract to one attachment path
    instead of two.
    """

    model_name: str
    text: str | None = None
    tool_result: ToolResult | None = None
    interject: str | None = None
    #: Files already uploaded to `/attachments`, bound to the message this
    #: request stores. They are not rendered, not counted and not prompted with
    #: — the conversation carries them, the model never sees them. See
    #: `../decisions/attachments-are-stored-not-read.md`.
    attachment_ids: list[uuid.UUID] = Field(default_factory=list, max_length=MAX_PER_MESSAGE)
    system: str | None = None
    #: "low" | "med" | "high". Absent is not a fourth value — it renders no
    #: marker, which is the prompt shape the model was trained under.
    budget: str | None = None
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    seed: int | None = None
    context_multiplier: float = 1.0
    #: A message id from a `/chats/{id}/context-floor` response, resent by the
    #: client on every later turn. History strictly before it is excluded from
    #: what the model is prompted with — see `context_floor` above for why this
    #: lives on the request rather than being written down anywhere.
    history_floor: uuid.UUID | None = None


@router.get("/chats")
def list_chats(owner: uuid.UUID = Depends(auth.require_owner)):
    with store.session() as db:
        return {
            "chats": [_chat(c, tokens) for c, tokens in q.list_chats(db, owner)],
        }


@router.post("/chats", status_code=201)
def create_chat(body: NewChat, owner: uuid.UUID = Depends(auth.require_owner)):
    with store.session() as db:
        return _chat(q.create_chat(db, owner, body.title), 0)


@router.get("/chats/{chat_id}")
def get_chat(chat_id: uuid.UUID, owner: uuid.UUID = Depends(auth.require_owner)):
    with store.session() as db:
        chat = _owned(db, chat_id, owner)
        messages = q.messages_of(db, chat_id)
        return {
            **_chat(chat, sum(m.tokens for m in messages)),
            "messages": [_message(m) for m in messages],
        }


@router.get("/chats/{chat_id}/context-floor")
def context_floor(
    chat_id: uuid.UUID, model_name: str, owner: uuid.UUID = Depends(auth.require_owner)
):
    """Where `/prune` (the frontend command) would cut this chat's history if
    `model_name` were fitting its own `ctx_window` rather than whatever
    `context_multiplier` extends it to right now.

    Read-only, on both sides of this call: nothing is written here, and the
    frontend is what remembers the answer and resends it as `history_floor` on
    every later turn (`Ask.history_floor` below) — see
    `docs/frontend/README.md`. If that ever needs to survive a lost
    localStorage or a second device, this is the point to reconsider: persist
    the floor against the chat instead of trusting the client to keep it.
    """
    with store.session() as db:
        _owned(db, chat_id, owner)
        messages = q.messages_of(db, chat_id)

    try:
        turns = [ChatTurn(role=m.role.value, text=m.content) for m in messages]
        start = chat_pipeline.history_floor(model_name, turns)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    pruned = messages[:start]
    return {
        "floor_message_id": str(messages[start].id) if start < len(messages) else None,
        "pruned_turns": len(pruned),
        "pruned_tokens": sum(m.tokens for m in pruned),
    }


@router.patch("/chats/{chat_id}")
def rename_chat(
    chat_id: uuid.UUID, body: Rename, owner: uuid.UUID = Depends(auth.require_owner)
):
    with store.session() as db:
        chat = q.rename_chat(db, chat_id, owner, body.title)
        if chat is None:
            raise HTTPException(status_code=404, detail=f"no chat {chat_id}")
        return _chat(chat, 0)


@router.delete("/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: uuid.UUID, owner: uuid.UUID = Depends(auth.require_owner)):
    with store.session() as db:
        if not q.delete_chat(db, chat_id, owner):
            raise HTTPException(status_code=404, detail=f"no chat {chat_id}")


@router.post("/chats/{chat_id}/messages", status_code=201)
def ask(chat_id: uuid.UUID, body: Ask, owner: uuid.UUID = Depends(auth.require_owner)):
    """Store the question, answer it, store the answer."""
    cfg, model_id = _prepare(chat_id, owner, body)
    try:
        reply = chat_pipeline.run(cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with store.session() as db:
        stored = q.add_message(
            db,
            chat_id=chat_id,
            role=Role.assistant,
            content=reply["text"],
            tokens=reply["generated_tokens"],
            model_id=model_id,
        )
        return {**_message(stored), "usage": _usage(reply)}


@router.post("/chats/{chat_id}/messages/stream")
async def ask_streaming(
    chat_id: uuid.UUID, body: Ask, owner: uuid.UUID = Depends(auth.require_owner)
):
    """The same turn, streamed. The row is written when the last token lands, so
    a dropped connection loses the reply rather than storing half of one."""
    import asyncio

    cfg, model_id = _prepare(chat_id, owner, body)

    def advance(generator):
        try:
            return next(generator), None
        except StopIteration as done:
            return None, done.value or {}

    async def frames():
        generator = chat_pipeline.stream(cfg)
        while True:
            try:
                piece, reply = await asyncio.to_thread(advance, generator)
            except (FileNotFoundError, ValueError) as e:
                yield sse.token_frame("error", {"error": str(e)})
                return
            if reply is not None:
                with store.session() as db:
                    stored = q.add_message(
                        db,
                        chat_id=chat_id,
                        role=Role.assistant,
                        content=reply["text"],
                        tokens=reply["generated_tokens"],
                        model_id=model_id,
                    )
                    yield sse.token_frame("done", {**_message(stored), "usage": _usage(reply)})
                return
            yield sse.token_frame(*piece)

    return StreamingResponse(frames(), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS)


@router.put("/messages/{message_id}/feedback")
def rate(
    message_id: uuid.UUID, body: Rating, owner: uuid.UUID = Depends(auth.require_owner)
):
    with store.session() as db:
        if not q.set_feedback(db, message_id, owner, body.feedback):
            raise HTTPException(status_code=404, detail=f"no message {message_id}")
    return {"message_id": str(message_id), "feedback": body.feedback}


@router.delete("/owners/{owner_id}", status_code=200)
def forget_owner(owner_id: uuid.UUID, caller: str = Depends(auth.require_service)):
    """Erase everything one person has here.

    The backend calls this when it deletes the account, and only ever as itself:
    an account deletion is not something a user token should be able to aim at
    someone else's id.
    """
    if caller != auth.SYSTEM:
        raise HTTPException(status_code=403, detail="only the backend may erase an owner")
    with store.session() as db:
        return {"owner_id": str(owner_id), "chats_deleted": q.delete_chats_of(db, owner_id)}


def _new_turns(body: Ask) -> list[tuple[Role, str]]:
    """The turn(s) this request adds: a person asking, or a tool answering —
    optionally with a second, `user`-role turn right after it when the
    request carries an `interject`.

    The "exactly one" rule lives here rather than in a validator so it sits
    beside the branch that depends on it. Order matters for the second case:
    the tool result must precede the interjection in the list this returns,
    because that is the order both are stored in and the order the model
    reads them in.
    """
    if (body.text is None) == (body.tool_result is None):
        raise HTTPException(
            status_code=400, detail="exactly one of text and tool_result is required"
        )
    if body.interject is not None and body.tool_result is None:
        raise HTTPException(
            status_code=400, detail="interject is only valid alongside a tool_result"
        )
    if body.text is not None:
        return [(Role.user, body.text)]

    # A tool answering a call is not something a person attached a file to. The
    # backend's `carry_settings` already drops the field on a follow-up round;
    # this is what makes that a contract rather than a coincidence.
    if body.attachment_ids:
        raise HTTPException(
            status_code=400, detail="a tool result carries no attachments"
        )

    result = body.tool_result
    if tools.by_name(result.tool) is None:
        raise HTTPException(status_code=400, detail=f"unknown tool {result.tool!r}")
    turns = [(Role.tool, tools.render_result(result.tool, result.argument, result.body))]
    if body.interject:
        turns.append((Role.user, body.interject))
    return turns


def _floored(messages: list[Message], floor: uuid.UUID | None) -> list[Message]:
    """`messages`, or the suffix from `floor` onward when the client sent one.

    A floor that doesn't match any stored message — the chat since deleted
    down to before it, or a stale id from another chat entirely — falls back
    to the full history rather than erroring: a client's memory of a prune is
    a convenience, never something the turn it is answering should fail over.
    """
    if floor is None:
        return messages
    for i, message in enumerate(messages):
        if message.id == floor:
            return messages[i:]
    return messages


def _prepare(
    chat_id: uuid.UUID, owner: uuid.UUID, body: Ask
) -> tuple[ChatConfig, uuid.UUID | None]:
    """Store the new turn(s) and build the config that answers them.

    Counting first also proves the model is loadable, so an unknown one is a
    404 rather than a chat left holding a question nothing answered. There is
    almost always exactly one new turn; a continuation carrying an `interject`
    is the one case with two, and both are stored and prompted with, in order.
    """
    new_turns = _new_turns(body)

    try:
        token_counts = [
            chat_pipeline.count_tokens(body.model_name, text)["tokens"]
            for _, text in new_turns
        ]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    with store.session() as db:
        _owned(db, chat_id, owner)
        model_id = _credit(db, body.model_name)
        history = [
            ChatTurn(role=m.role.value, text=m.content)
            for m in _floored(q.messages_of(db, chat_id), body.history_floor)
        ]
        stored = [
            q.add_message(db, chat_id=chat_id, role=role, content=content, tokens=tokens)
            for (role, content), tokens in zip(new_turns, token_counts)
        ]
        # Only a plain `text` turn — the first and only entry on that path —
        # can carry attachments; a tool result and any interjection beside it
        # never do (checked in `_new_turns`).
        if body.text is not None:
            _bind(db, stored[0], owner, body.attachment_ids)

    return (
        ChatConfig(
            model_name=body.model_name,
            turns=[
                *history,
                *(ChatTurn(role=role.value, text=content) for role, content in new_turns),
            ],
            system=body.system,
            budget=body.budget,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            seed=body.seed,
            context_multiplier=body.context_multiplier,
        ),
        model_id,
    )


def _bind(db, message: Message, owner: uuid.UUID, attachment_ids: list[uuid.UUID]) -> None:
    """Tie staged uploads to the turn being stored.

    A 400 rather than a silent skip: an id that is not this owner's, or that was
    already sent with another turn, means the client and this side disagree
    about what the person just sent.
    """
    if q.bind_attachments(
        db, message_id=message.id, owner_id=owner, attachment_ids=attachment_ids
    ) is None:
        raise HTTPException(
            status_code=400, detail="an attachment is unknown or already sent"
        )


def _credit(db, model_name: str) -> uuid.UUID | None:
    """Which row the reply is credited to.

    A model's weights directory is its id, but one trained before it had a row —
    or dropped in by hand — still answers; it just credits nobody.
    """
    try:
        model = q.get_model(db, uuid.UUID(model_name))
    except ValueError:
        model = q.find_model_by_name(db, model_name)
    return model.id if model else None


def _owned(db, chat_id: uuid.UUID, owner: uuid.UUID) -> Chat:
    chat = q.owned_chat(db, chat_id, owner)
    if chat is None:
        raise HTTPException(status_code=404, detail=f"no chat {chat_id}")
    return chat


def _chat(chat: Chat, tokens: int) -> dict:
    return {
        "chat_id": str(chat.id),
        "title": chat.title,
        "tokens": tokens,
        "created_at": chat.created_at.isoformat() if chat.created_at else None,
    }


def _message(message: Message) -> dict:
    return {
        "message_id": str(message.id),
        "chat_id": str(message.chat_id),
        "role": message.role.value,
        "content": message.content,
        "tokens": message.tokens,
        "model_id": str(message.model_id) if message.model_id else None,
        "model_name": message.model.name if message.model else None,
        "feedback": message.feedback.value if message.feedback else None,
        "attachments": [describe(a) for a in message.attachments],
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def _usage(reply: dict) -> dict:
    return {
        "prompt_tokens": reply["prompt_tokens"],
        "generated_tokens": reply["generated_tokens"],
        # Repeated on the terminal frame so a caller that came in over a
        # non-streaming request, or reattached late, still learns what the reply
        # asked for. The streaming caller has already seen each as it closed.
        "tool_calls": reply["tool_calls"],
        "stopped": reply["stopped"],
        "elapsed_secs": reply["elapsed_secs"],
    }
