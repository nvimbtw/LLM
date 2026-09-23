"""The tools a model may call, and how a call and its answer are spelled.

One table. The grammar reads it to decide whether a call names something real,
the generation loop reads it to report a call it just saw, and the SFT renderer
reads it to lay a result out — so adding a tool is one row here and **no
vocabulary change at all**.

A call is a leaf block inside an assistant turn, and every tool is called
through the same marker pair::

    <|tool-call|>{"tool": "web-search", "argument": "tallest mountain in europe"}<|/tool-call|>
    <|tool-call|>{"tool": "bash", "argument": "grep -rn cat . | head -3"}<|/tool-call|>

Its answer comes back as a block of its own, between that assistant turn and
the next one, in the position the system block occupies at the front of a
conversation::

    <|tool-result|><|tool-call|>{"tool": "web-search", "argument": "tallest mountain in europe"}<|/tool-call|>
                   <|list|><|item|>Mount Elbrus — 5,642 m<|/item|><|/list|><|/tool-result|>

Repeating the call inside the result is what tells the model *which* of the
calls it made is being answered, without spending a marker on a correlation id
it cannot count.

## One marker for every tool

This replaces a marker pair per tool (`<|web-search|>query<|/web-search|>`).
The trade is deliberate and it is not free — see
`docs/decisions/generic-tool-calls.md`. What it buys is that the tool table is
no longer part of the vocabulary: a model trained today can be served a tool
that did not exist when its tokenizer was built, because the only thing new
about that tool is a name inside a JSON object. What it costs is that the
grammar can no longer refuse a call to a tool this build does not have — the
name is ordinary text now — and that the model has to keep a small JSON object
balanced. `parse` is deliberately forgiving about the second.

`{"tool", "argument"}` is exactly the shape of the `tool_call` frame the
generation loop emits and the backend consumes, so nothing between the marker
and the tool has to map one onto the other.
"""

import json
from dataclasses import dataclass

from llm.corpus import special

TOOL_CALL_OPEN = "<|tool-call|>"
TOOL_CALL_CLOSE = "<|/tool-call|>"

TOOL_RESULT_OPEN = "<|tool-result|>"
TOOL_RESULT_CLOSE = "<|/tool-result|>"

#: Keys a call's argument is accepted under. `argument` is what this project
#: writes and what SFT teaches; the rest are what a small model drifts into
#: saying when the tool it is calling is named `web-search` or `bash`. Reading
#: them costs nothing and saves the whole round a refusal would have wasted.
_ARGUMENT_KEYS = ("argument", "query", "command", "input", "text")

#: Keys the tool's own name arrives under, and therefore the ones that are never
#: its argument.
_NAME_KEYS = ("tool", "name")


@dataclass(frozen=True)
class Tool:
    """A callable tool. `name` is the wire name *and* the name inside the call,
    so the two cannot drift — the LLM engine never maps between them."""

    name: str


TOOLS: list[Tool] = [Tool("web-search"), Tool("bash")]


def by_name(name: str) -> Tool | None:
    return next((tool for tool in TOOLS if tool.name == name), None)


def call(name: str, argument: str) -> str:
    """One call, rendered the way SFT teaches it and serving reads it back.

    `json.dumps` rather than an f-string: an argument is a search phrase or a
    shell command, and both routinely contain the quotes and backslashes that
    would otherwise close the object early.
    """
    payload = json.dumps({"tool": name, "argument": argument.strip()}, ensure_ascii=False)
    return f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}"


def parse(body: str) -> tuple[str, str] | None:
    """The `(tool, argument)` a call's body names, or None if it names none.

    Forgiving in three ways, each of them a thing a model at this size actually
    does rather than a hypothetical:

    - the argument may be under any of `_ARGUMENT_KEYS`,
    - it may be wrapped in an `args`/`arguments` object, which is how every
      function-calling corpus on the Hub spells it,
    - a non-string argument (a number, a list) is rendered back to text rather
      than refused.

    What it will not do is guess at a tool name. A body that parses to no name,
    or to a name this build has never heard of, is not a call — the generation
    loop drops it, and the reply that contains it still stands.
    """
    try:
        payload = json.loads(body.strip())
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None

    name = next((payload[k] for k in _NAME_KEYS if isinstance(payload.get(k), str)), None)
    if not isinstance(name, str) or by_name(name.strip()) is None:
        return None

    source = payload
    for wrapper in ("args", "arguments", "parameters"):
        nested = payload.get(wrapper)
        if isinstance(nested, dict):
            source = nested
            break
        if isinstance(nested, str):
            return name.strip(), nested.strip()

    # Whatever is left once the name is accounted for. A call that named its
    # tool and nothing else has no argument — `{"tool": "web-search"}` must not
    # come back as a search for the word "web-search".
    fields = {k: v for k, v in source.items() if source is not payload or k not in _NAME_KEYS}

    for key in _ARGUMENT_KEYS:
        if key in fields:
            return name.strip(), _as_text(fields[key])

    # One field left is unambiguous whatever it is called — `{"pattern": "cat"}`
    # for a tool that takes one argument can only mean it.
    if len(fields) == 1:
        return name.strip(), _as_text(next(iter(fields.values())))
    return None


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def render_result(name: str, argument: str, body: str) -> str:
    """The *inside* of the `<|tool-result|>` block answering one call.

    The delimiters are the template's to add, the same way it owns `<|user|>`
    around a question — this is the content, and what makes it legible is that
    it opens by repeating the call it answers.

    The single renderer for both the live path and the SFT data. Two copies that
    disagreed would train the model on a layout it is never served.
    """
    return f"{call(name, argument)}{body}"


_MISSING = [
    marker
    for marker in (TOOL_CALL_OPEN, TOOL_CALL_CLOSE, TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE)
    if marker not in special.SPECIAL_TOKENS
]
assert not _MISSING, f"tool markers with no reserved slot: {_MISSING}"
