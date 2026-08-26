"""Source prose as the block markup the engine reads.

`markdown.py` is about the dialect a *source* wrote in; this is about the one a
reply is generated in. The engine draws the same line, and nothing here may
import across it: the two services have separate virtualenvs, so the marker
layout is written out on both sides and `engine/llm/corpus/special.py` stays the
source of truth for it.

Shared by every converter that turns a chat corpus into training text. A
paragraph, a fenced code block and a `<think>` body mean the same thing whatever
repository they came out of, and a second copy of this would be a second set of
rules about what a marker in somebody else's text is allowed to do.
"""

import re

THINKING_OPEN, THINKING_CLOSE = "<|thinking|>", "<|/thinking|>"
PAR_OPEN, PAR_CLOSE = "<|par|>", "<|/par|>"
CODEBLOCK_OPEN, CODEBLOCK_CLOSE = "<|codeblock|>", "<|/codeblock|>"

#: Fenced code, one capture group so `re.split` lays the bodies at the odd
#: indices. Split out before paragraphs are: a blank line inside a fence is part
#: of the code, not a paragraph break.
_FENCE = re.compile(r"```[^\n]*\n(.*?)(?:^[ \t]*```[ \t]*$|\Z)", re.DOTALL | re.MULTILINE)

_BLANK_LINE = re.compile(r"\n[ \t]*\n")

#: Any literal marker in the source text, broken the same way (and with the same
#: zero-width space) as `scripts/corpus_shards.py::neutralize` — chat corpora
#: quote chat templates and tokenizer configs, and a `<|endoftext|>` inside a
#: quoted YAML file would otherwise tokenize as a real document boundary.
#: Underscores are matched where that pattern omits them, because
#: `<|reserved_7|>` is a real name for a real slot in `corpus.special`.
_MARKER = re.compile(r"<\|(/?[a-zA-Z0-9:_-]+)\|>")

#: Written as an escape rather than the character itself: it is invisible, and a
#: literal one here is a byte nobody reviewing this file could see.
_ZERO_WIDTH = "\u200b"

#: C0 controls other than tab, newline and carriage return. These arrive *made*
#: rather than copied: a source that holds a tool result as JSON inside a string
#: turns an escaped NUL — six harmless characters in the corpus — into a real one
#: the moment `json.loads` reads the response. Postgres `jsonb` cannot store a
#: NUL at all and rejects the whole insert, which takes the batch and the run
#: with it; the rest are bytes no tokenizer has a use for.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

#: The source's own reasoning tag — `<think>`, not `<|thinking|>`. Closed at the
#: end of the turn if the source never closed it, so an unterminated block
#: becomes reasoning rather than leaking its tag into the reply as prose.
_SOURCE_THINK = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.DOTALL)

#: A `<think>` or `</think>` the pass above could not pair off — a turn that
#: opens the block twice leaves the inner opener inside the captured body.
#: Dropped, so no turn carries a reasoning tag through as text.
_STRAY_THINK = re.compile(r"</?think>")


def strip_control(text: str) -> str:
    """The control bytes out, and nothing else touched.

    Separate from `neutralise` because a shell command is not prose: it must
    lose the bytes Postgres cannot store, but a marker inside it is already
    JSON-encoded into a call's payload and cannot end a block early.
    """
    return _CONTROL.sub("", text)


def neutralise(text: str) -> str:
    """Source text made safe to embed: markers broken, control bytes dropped."""
    return _MARKER.sub(f"<|\\1|{_ZERO_WIDTH}>", strip_control(text))


def blocks(text: str) -> str:
    """Prose as markup: paragraphs `<|par|>`, fenced code `<|codeblock|>`.

    Whitespace inside a paragraph is collapsed and preserved inside a code
    block, which is the whole reason the two are told apart here. Inline
    markdown is left alone — that is `markdown-emphasis`'s job, run after this
    one, and it knows to skip what is inside a `<|codeblock|>`.
    """
    out: list[str] = []
    for index, chunk in enumerate(_FENCE.split(text)):
        if index % 2:
            code = neutralise(chunk).strip("\n")
            if code.strip():
                out.append(f"{CODEBLOCK_OPEN}{code}{CODEBLOCK_CLOSE}")
            continue
        for paragraph in _BLANK_LINE.split(chunk):
            body = " ".join(neutralise(paragraph).split())
            if body:
                out.append(f"{PAR_OPEN}{body}{PAR_CLOSE}")
    return "".join(out)


def split_thinking(content: str) -> tuple[str, str]:
    """One assistant turn as `(reasoning, reply)`, both still raw text.

    Every `<think>` body joined in the order it was written, and the turn with
    those bodies cut out. Joined rather than kept apart because a reply here
    opens with its thinking and never returns to it, which is what the
    template's `strip_thinking` and the grammar's budget markers are built
    around.
    """
    reasoning = "\n\n".join(m.group(1) for m in _SOURCE_THINK.finditer(content))
    body = _SOURCE_THINK.sub("", content)
    return _STRAY_THINK.sub("", reasoning), _STRAY_THINK.sub("", body)


def reply_markup(reasoning: str, body: str) -> str:
    """An assistant turn as `<|thinking|>`-then-prose, either half allowed empty."""
    thinking = blocks(reasoning)
    return (f"{THINKING_OPEN}{thinking}{THINKING_CLOSE}" if thinking else "") + blocks(body)
