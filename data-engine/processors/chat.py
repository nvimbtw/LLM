"""Processors for conversations.

An SFT row is only useful if its turns alternate sensibly and end somewhere the
model can learn from, so most of these are about shape rather than wording.
"""

from processors.spec import Document, Param, Processor, State
from store.schema import DatasetKind

SFT = (DatasetKind.sft,)

#: What the roles in this project are called. Anything a source calls its turns
#: has to land on one of these or the conversation cannot be templated.
_CANONICAL = {
    "user": "user",
    "human": "user",
    "prompter": "user",
    "question": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "answer": "assistant",
    "system": "system",
    "tool": "tool",
}


def _turns(doc: Document) -> list[dict]:
    return list(doc.get("messages") or [])


def _normalise_roles(doc: Document, params: dict, state: State) -> Document | None:
    """Map a source's role names onto this project's.

    A turn whose role is unrecognised drops the whole conversation: guessing
    would put words in the wrong mouth, and one bad exchange is cheaper to lose
    than a corpus that teaches the model to answer itself.
    """
    messages = []
    for turn in _turns(doc):
        role = _CANONICAL.get(str(turn.get("role", "")).strip().lower())
        if role is None:
            return None
        messages.append({**turn, "role": role})
    return {**doc, "messages": messages}


def _drop_empty_messages(doc: Document, params: dict, state: State) -> Document | None:
    messages = [t for t in _turns(doc) if str(t.get("content") or "").strip()]
    return {**doc, "messages": messages} if messages else None


def _min_turns(doc: Document, params: dict, state: State) -> Document | None:
    return doc if len(_turns(doc)) >= params["turns"] else None


def _require_roles(doc: Document, params: dict, state: State) -> Document | None:
    """Keep only conversations that actually contain an exchange.

    A file of prompts with no replies, or replies with no prompts, imports
    cleanly and teaches nothing — this is the check that catches a mis-mapped
    column before it becomes a training run.
    """
    roles = {t.get("role") for t in _turns(doc)}
    if params["needs_user"] and "user" not in roles:
        return None
    if params["needs_assistant"] and "assistant" not in roles:
        return None
    return doc


PROCESSORS = (
    Processor(
        name="normalise-roles",
        title="Normalise roles",
        description="Map human/gpt/prompter and friends onto user and assistant.",
        applies_to=SFT,
        apply=_normalise_roles,
    ),
    Processor(
        name="drop-empty-messages",
        title="Drop empty turns",
        description="Remove blank turns, and the conversation if nothing is left.",
        applies_to=SFT,
        apply=_drop_empty_messages,
    ),
    Processor(
        name="min-turns",
        title="Require a minimum length",
        description="Drop conversations shorter than a number of turns.",
        applies_to=SFT,
        apply=_min_turns,
        params=(
            Param(
                name="turns",
                label="Minimum turns",
                type="int",
                default=2,
                minimum=1,
                maximum=100,
            ),
        ),
    ),
    Processor(
        name="require-roles",
        title="Require an exchange",
        description="Drop conversations missing a question or an answer.",
        applies_to=SFT,
        apply=_require_roles,
        params=(
            Param(name="needs_user", label="Must have a user turn", type="bool", default=True),
            Param(
                name="needs_assistant",
                label="Must have an assistant turn",
                type="bool",
                default=True,
            ),
        ),
    ),
)
