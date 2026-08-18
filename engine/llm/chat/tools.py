"""The tools a model may call, and how a call and its answer are spelled.

One table. The grammar reads it to decide which markers may be emitted, the
generation loop reads it to name a call it just saw, and the SFT renderer reads
it to lay a result out — so adding a tool is two reserved slots in
`corpus.special` and one row here, and nothing else in the engine changes.

A call is a leaf block inside an assistant turn::

    <|web-search|>tallest mountain in europe<|/web-search|>

Its answer comes back as a block of its own, between that assistant turn and
the next one, in the position the system block occupies at the front of a
conversation::

    <|tool-result|><|web-search|>tallest mountain in europe<|/web-search|>
                   <|list|><|item|>Mount Elbrus — 5,642 m<|/item|><|/list|><|/tool-result|>

Repeating the call inside the result is what tells the model *which* of the
calls it made is being answered, without spending a marker on a correlation id
it cannot count.
"""

from dataclasses import dataclass

from llm.corpus import special

TOOL_RESULT_OPEN = "<|tool-result|>"
TOOL_RESULT_CLOSE = "<|/tool-result|>"


@dataclass(frozen=True)
class Tool:
    """A callable tool. `name` is the wire name *and* the marker stem, so the
    two cannot drift — the engine never maps between them."""

    name: str

    @property
    def open(self) -> str:
        return f"<|{self.name}|>"

    @property
    def close(self) -> str:
        return f"<|/{self.name}|>"

    def call(self, argument: str) -> str:
        return f"{self.open}{argument.strip()}{self.close}"


TOOLS: list[Tool] = [Tool("web-search")]


def by_name(name: str) -> Tool | None:
    return next((tool for tool in TOOLS if tool.name == name), None)


def render_result(tool: Tool, argument: str, body: str) -> str:
    """The *inside* of the `<|tool-result|>` block answering one call.

    The delimiters are the template's to add, the same way it owns `<|user|>`
    around a question — this is the content, and what makes it legible is that
    it opens by repeating the call it answers.

    The single renderer for both the live path and the SFT data. Two copies that
    disagreed would train the model on a layout it is never served.
    """
    return f"{tool.call(argument)}{body}"


_MISSING = [
    marker
    for tool in TOOLS
    for marker in (tool.open, tool.close)
    if marker not in special.SPECIAL_TOKENS
]
assert not _MISSING, f"tools with no reserved slot: {_MISSING}"
