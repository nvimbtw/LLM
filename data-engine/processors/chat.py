"""Processors for conversations.

An SFT row is only useful if its turns alternate sensibly and end somewhere the
model can learn from, so most of these are about shape rather than wording. The
exception is `convert-chat`, which is the wording step for a corpus that is
conversation and nothing else — the one every plain chat source needs and none
of the per-source converters is right for.
"""

from processors.markup import blocks, neutralise, reply_markup, split_thinking
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


#: A tool call in either spelling — the source's own tag and this project's
#: marker — so the filter reads the same before and after a converter has run.
_TOOL_MARKS = ("<tool_call>", "<|tool-call|>", "<tool_response>", "<|tool-result|>")


def _drop_tool_calls(doc: Document, params: dict, state: State) -> Document | None:
    """Drop a conversation in which anybody calls a tool.

    A `tool` turn counts as well as the markers: a result whose call has already
    been converted away is still a conversation about using tools.

    Dropped whole rather than trimmed, because a reply with its call cut out is
    a model acting on a result it was never shown — a worse lesson than the row
    is worth. This exists for a first SFT that teaches the chat template and
    nothing else, where the point is a corpus with no tool use in it at all.
    """
    for turn in _turns(doc):
        if turn.get("role") == "tool":
            return None
        content = str(turn.get("content") or "")
        if any(mark in content for mark in _TOOL_MARKS):
            return None
    return doc


def _convert_chat(doc: Document, params: dict, state: State) -> Document | None:
    """A plain chat corpus as training text.

    User and system turns keep their wording and are only made safe — a marker
    the source wrote is broken so it cannot end a block early. Assistant turns
    become markup, because the reply is the only side the model is trained to
    produce: `<think>` bodies to `<|thinking|>`, paragraphs to `<|par|>`, fenced
    code to `<|codeblock|>`.

    No tool call is recognised and none is invented. `convert-smoltalk` is what
    turns a source's own function calls into calls this project can make, and
    `drop-tool-calls` is what removes them — a `tool` turn reaching here means
    the corpus is not the plain chat this step is for, so the row is dropped
    rather than passed through unconverted.

    Roles must already be canonical: `normalise-roles` owns that mapping, and
    doing it twice in two places is how the two spellings drift apart.
    """
    # Anything but the one opt-out keeps the reasoning, matching how `_coerce`
    # treats a value it cannot use: the reading that loses no text wins.
    strip = params["reasoning"] == "strip"
    messages: list[dict] = []
    for turn in _turns(doc):
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(content, str) or role not in ("system", "user", "assistant"):
            return None

        if role == "assistant":
            reasoning, body = split_thinking(content)
            rendered = blocks(body) if strip else reply_markup(reasoning, body)
        else:
            rendered = neutralise(content).strip()
        if rendered:
            messages.append({"role": role, "content": rendered})

    # A question left hanging at the end teaches nothing and costs window.
    while messages and messages[-1]["role"] == "user":
        messages.pop()
    return {**doc, "messages": messages} if messages else None


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
        name="drop-tool-calls",
        title="Drop tool use",
        description="Drop conversations that call a tool, in either spelling.",
        applies_to=SFT,
        apply=_drop_tool_calls,
    ),
    Processor(
        name="convert-chat",
        title="Convert a plain chat corpus",
        description=(
            "Assistant replies become <|par|>/<|codeblock|> markup and <think> becomes "
            "<|thinking|>. Run normalise-roles first."
        ),
        applies_to=SFT,
        apply=_convert_chat,
        params=(
            Param(
                name="reasoning",
                label="Reasoning blocks",
                type="string",
                default="convert",
                description=(
                    "convert: <think> becomes <|thinking|>, which the LLM engine's "
                    "strip_thinking can still drop per run. strip: discard it here, for "
                    "one dataset in a blend that should carry no reasoning at all."
                ),
            ),
        ),
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
