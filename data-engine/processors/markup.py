"""Source prose as the block markup the LLM engine reads.

`markdown.py` is about the dialect a *source* wrote in; this is about the one a
reply is generated in. The LLM engine draws the same line, and nothing here may
import across it: the two services have separate virtualenvs, so the marker
layout is written out on both sides and `llm-engine/llm/corpus/special.py` stays the
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
TITLE_OPEN, TITLE_CLOSE = "<|title|>", "<|/title|>"
HEADING_OPEN, HEADING_CLOSE = "<|heading|>", "<|/heading|>"
SUBHEADING_OPEN, SUBHEADING_CLOSE = "<|subheading|>", "<|/subheading|>"
LIST_OPEN, LIST_CLOSE = "<|list|>", "<|/list|>"
ITEM_OPEN, ITEM_CLOSE = "<|item|>", "<|/item|>"
QUOTE_OPEN, QUOTE_CLOSE = "<|quote|>", "<|/quote|>"
DIVIDER = "<|divider|>"

#: Markdown's six heading levels onto the three the vocabulary has. Clamped
#: rather than dropped past the third: a `####` line is a heading whatever depth
#: its author gave it, and rendering it as prose is the bug this table fixes.
#: `llm-engine/llm/corpus/special.py` has the markers; the frontend's
#: `message-parser.ts` is what fixes them as h1/h2/h3.
_HEADINGS = {
    1: (TITLE_OPEN, TITLE_CLOSE),
    2: (HEADING_OPEN, HEADING_CLOSE),
    3: (SUBHEADING_OPEN, SUBHEADING_CLOSE),
}

#: Fenced code, one capture group so `re.split` lays the bodies at the odd
#: indices. Split out before paragraphs are: a blank line inside a fence is part
#: of the code, not a paragraph break.
_FENCE = re.compile(r"```[^\n]*\n(.*?)(?:^[ \t]*```[ \t]*$|\Z)", re.DOTALL | re.MULTILINE)

#: An ATX heading — up to three spaces of indent, one to six hashes, a space,
#: the text. The optional run of trailing hashes is CommonMark's closing
#: sequence and is decoration, not part of the heading.
_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")

#: Three or more of the same -, * or _ alone on a line. CommonMark would read a
#: `---` directly under a line of text as a setext underline instead; replies
#: write a rule and put a blank line above it, and reading every one as a rule
#: costs a heading nobody writes that way.
_BREAK = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")

_QUOTE_LINE = re.compile(r"^ {0,3}> ?(.*)$")

#: A bullet or an ordered item, the marker captured apart from the text. The
#: marker itself is dropped: `<|list|>` already says the lines are a list, and a
#: rendered "1." inside an `<|item|>` is a number the reader sees twice. It is
#: still captured because a change of marker starts a *new* list. Indentation is
#: matched but not measured — see `_list` on why nesting flattens.
_ITEM_LINE = re.compile(r"^[ \t]*(?:(?P<bullet>[-*+])|\d{1,9}(?P<ordered>[.)]))[ \t]+(?P<text>.*)$")

_BLANK_LINE_ONLY = re.compile(r"^[ \t]*$")

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
    """Prose as markup: every block-level construct the vocabulary has a marker
    for, and `<|par|>` for the rest.

        # Title          ->  <|title|>Title<|/title|>
        ## Heading       ->  <|heading|>Heading<|/heading|>
        ### and deeper   ->  <|subheading|>...<|/subheading|>
        - a
        - b              ->  <|list|><|item|>a<|/item|><|item|>b<|/item|><|/list|>
        > quoted         ->  <|quote|>quoted<|/quote|>
        ---              ->  <|divider|>
        ```code```       ->  <|codeblock|>code<|/codeblock|>

    A source writes Markdown and the model must emit markers, so anything left
    unconverted is a `##` the model learns to type as prose — which is both
    wrong on screen and a waste of the reserved id sitting unused. The set of
    constructs handled here is therefore fixed by `corpus.special`, not by
    Markdown: what has no marker (tables, footnotes, images) stays prose.

    Whitespace inside a paragraph is collapsed and preserved inside a code
    block, which is the whole reason the two are told apart here. **Inline**
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
        out.extend(_prose(chunk))
    return "".join(out)


def _prose(text: str) -> list[str]:
    """One run of non-fenced text as a list of rendered blocks.

    A hand-written walk rather than a Markdown library: the output is not HTML
    and not a tree, it is this vocabulary and nothing else, and a dependency
    that renders everything would still have to be told what to do with the
    nine constructs that have markers and the many that do not.
    """
    lines = text.split("\n")
    out: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if _BLANK_LINE_ONLY.match(line):
            index += 1
        elif _BREAK.match(line):
            out.append(DIVIDER)
            index += 1
        elif (heading := _ATX.match(line)) is not None:
            opener, closer = _HEADINGS.get(len(heading.group(1)), _HEADINGS[3])
            if body := _inline(heading.group(2)):
                out.append(f"{opener}{body}{closer}")
            index += 1
        elif _QUOTE_LINE.match(line):
            index, block = _quote(lines, index)
            out.extend(block)
        elif _item(line) is not None:
            index, block = _list(lines, index)
            out.extend(block)
        else:
            index, block = _paragraph(lines, index)
            out.extend(block)

    return out


def _inline(text: str) -> str:
    """One block's text: markers broken, whitespace collapsed to single spaces.

    The line breaks a Markdown source wraps its prose at are not content, and a
    block marker already says where the block ends.
    """
    return " ".join(neutralise(text).split())


def _opens_block(line: str) -> bool:
    """Whether `line` begins a block of its own, and so ends the one before it.

    A blank line counts: it is what separates two paragraphs.
    """
    return bool(
        _BLANK_LINE_ONLY.match(line)
        or _BREAK.match(line)
        or _ATX.match(line)
        or _QUOTE_LINE.match(line)
        or _item(line) is not None
    )


def _item(line: str) -> tuple[str, str] | None:
    """A list item as `(marker, text)`, or None if `line` is not one.

    The marker is `-`, `*`, `+`, `.` or `)` — kept because Markdown reads a
    change of it as the start of a different list, which is the only thing
    telling a bulleted list from the ordered one written directly under it.

    `_BREAK` is tried first because `* * *` is a rule that a bullet pattern will
    happily read as an item whose text is `* *`.
    """
    if _BREAK.match(line):
        return None
    match = _ITEM_LINE.match(line)
    if match is None:
        return None
    return match.group("bullet") or match.group("ordered"), match.group("text")


def _paragraph(lines: list[str], start: int) -> tuple[int, list[str]]:
    index = start
    while index < len(lines) and not _opens_block(lines[index]):
        index += 1
    body = _inline(" ".join(lines[start:index]))
    return index, [f"{PAR_OPEN}{body}{PAR_CLOSE}"] if body else []


def _quote(lines: list[str], start: int) -> tuple[int, list[str]]:
    """Consecutive `>` lines as one `<|quote|>`.

    One block however many paragraphs the source quoted: the marker holds a run
    of text, and there is no `<|par|>` legal inside it that could hold two.
    """
    body: list[str] = []
    index = start
    while index < len(lines) and (match := _QUOTE_LINE.match(lines[index])):
        body.append(match.group(1))
        index += 1
    rendered = _inline(" ".join(body))
    return index, [f"{QUOTE_OPEN}{rendered}{QUOTE_CLOSE}"] if rendered else []


def _list(lines: list[str], start: int) -> tuple[int, list[str]]:
    """A run of items as one `<|list|>`, each item's text in an `<|item|>`.

    **Nesting flattens.** An indented sub-list becomes more items of the same
    list. `<|list|>` inside an `<|item|>` is legal — `chat.markup` allows it and
    bounds the depth at six — but no source this converts is reliable enough
    about indentation to tell a sub-list from a wrapped line, and a mis-read
    indent that opens a list changes what the model is taught about nesting.
    Flat is the reading that cannot be wrong about structure it invented.

    A blank line between two items of the same marker is a loose list, not the
    end of one; a blank line followed by anything else ends it. So does a change
    of marker — a numbered list under a bulleted one is a second list, and
    running them together would teach the model one list where the source drew
    two.
    """
    items: list[str] = []
    index = start
    marker: str | None = None

    while index < len(lines):
        found = _item(lines[index])
        if found is None or (marker is not None and found[0] != marker):
            if (
                found is None
                and _BLANK_LINE_ONLY.match(lines[index])
                and index + 1 < len(lines)
                and (following := _item(lines[index + 1])) is not None
                and following[0] == marker
            ):
                index += 1
                continue
            break

        marker, text = found
        body = [text]
        index += 1
        # A line under an item that starts nothing of its own is that item's
        # continuation, not the paragraph after the list.
        while index < len(lines) and not _opens_block(lines[index]):
            body.append(lines[index])
            index += 1
        if rendered := _inline(" ".join(body)):
            items.append(f"{ITEM_OPEN}{rendered}{ITEM_CLOSE}")

    if not items:
        return index, []
    return index, [LIST_OPEN + "".join(items) + LIST_CLOSE]


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
