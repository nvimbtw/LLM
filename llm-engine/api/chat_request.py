"""Stateless validation and rendering of incoming conversation turns."""
import uuid
from fastapi import HTTPException
from pydantic import BaseModel, Field
from llm.chat import tools

MAX_PER_MESSAGE = 5

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


def _new_turns(body: Ask) -> list[tuple[str, str]]:
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
        return [("user", body.text)]

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
    turns = [("tool", tools.render_result(result.tool, result.argument, result.body))]
    if body.interject:
        turns.append(("user", body.interject))
    return turns


