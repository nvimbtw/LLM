"""Converting Markdown's inline formatting into the project's markup markers.

SmolTalk (like most instruct sources) answers with Markdown: `**bold**`,
`*italic*`, `_underscore italic_`, `` `code` ``. Our markers are
`<|bold|>…<|/bold|>`, `<|italic|>…<|/italic|>` and `<|code|>…<|/code|>` — the
model must *emit* those, and it can only learn to emit what it sees. Running
this processor in the same pipeline as `convert-smoltalk` is what turns the
source's styling into the vocabulary's.

**Inline only.** Headings, lists, quotes and rules are block structure and are
converted by `markup.blocks` while a turn is being rendered, before this runs —
by then a `##` is already inside a `<|par|>` and unwrapping one would mean
undoing that decision here. The split is what each module can see: `blocks` has
the line structure, this has the runs of text.

Conversion is deliberately conservative. It runs on **assistant turns only** —
responses are where the model learns to format; a user's `**bold**` is a
quoted word, not a style instruction, and rewriting it would change the
question. The protected-region split skips: fenced code blocks, inline code
spans (converted by the earlier pass, and protected from the later one),
already-rendered `<|...|>` markers (so a run after `convert-smoltalk`
keeps its `<|tool-call|>` and `<|par|>` blocks intact), and `tool-call` /
`tool-result` blocks whose contents are a JSON call or a result, not prose.

Emphasis must be flanked by non-space (CommonMark's rule), so `2 * 3` and
`base ** 2` survive; and underscore emphasis must sit between non-word
characters, so `snake_case` and URLs survive.
"""

import re

from processors.spec import Document, Processor, State
from store.schema import DatasetKind

SFT = (DatasetKind.sft,)

#: Regions whose contents must never be touched by the emphasis regexes:
#: fenced code, inline code, whole already-rendered leaf/tool blocks, and the
#: marker tokens themselves (their bodies stay open — a `<|par|>` may wrap text
#: that still wants converting). One capture group, so `re.split` lays the
#: protected regions at the odd indices.
_LEAF = r"<\|(?:tool-call|tool-result|code|codeblock)\|>.*?<\|/(?:tool-call|tool-result|code|codeblock)\|>"
_PROTECTED = re.compile(
    rf"({_LEAF}|```.*?```|`[^`\n]+`|<\|/?[a-zA-Z0-9-]+\|>)",
    re.DOTALL,
)

#: The same, minus the backtick span — this is what the code-span pass splits
#: on, and a span cannot be protected from the rule that converts it.
_PROTECTED_BLOCKS = re.compile(rf"({_LEAF}|```.*?```|<\|/?[a-zA-Z0-9-]+\|>)", re.DOTALL)

#: An inline code span. Single backticks only: the double-backtick form exists
#: to quote a backtick, and is rare enough in a reply that leaving it as text
#: beats a rule that swallows the pair. `<|code|>` is the marker for this —
#: `<|codeblock|>` is the fenced one, and `markup.blocks` owns that.
_CODE_SPAN = re.compile(r"`([^`\n]+)`")

CODE_OPEN, CODE_CLOSE = "<|code|>", "<|/code|>"

#: Emphasis forms, widest first so `***x***` is consumed before `**x**` and
#: `*x*` before either arc of a `**` pair can mis-fragment it. Each re-runs
#: until the text stops changing: a `**outer *inner* text**` needs the inner
#: pass to complete before the outer can close, and finishing the pass makes
#: that kind of nesting fall out for free.
#: A delimiter the source escaped is a literal character, not emphasis —
#: CommonMark's rule, and the one LaTeX depends on: `$a\_n$` is a subscript, and
#: reading its `\_` as an opener rewrites the formula. `(?<!\\)` on the opener is
#: what refuses that; the body then excludes an escaped delimiter so the *closer*
#: cannot be one either.
_EMPHASIS = [
    (re.compile(r"(?<!\\)\*\*\*(?=\S)((?:[^*\n\\]|\\.)+?)(?<=\S)\*\*\*"), "<|bold|><|italic|>", "<|/italic|><|/bold|>"),
    (re.compile(r"(?<![\w\\])___(?=\S)((?:[^_\n\\]|\\.)+?)(?<=\S)___(?!\w)"), "<|bold|><|italic|>", "<|/italic|><|/bold|>"),
    (re.compile(r"(?<!\\)\*\*(?=\S)((?:[^*\n\\]|\\.)+?)(?<=\S)\*\*"), "<|bold|>", "<|/bold|>"),
    (re.compile(r"(?<![\w\\])__(?=\S)((?:[^_\n\\]|\\.)+?)(?<=\S)__(?!\w)"), "<|bold|>", "<|/bold|>"),
    (re.compile(r"(?<![\w*\\])\*(?=\S)((?:[^*\n\\]|\\.)+?)(?<=\S)\*(?![\w*])"), "<|italic|>", "<|/italic|>"),
    (re.compile(r"(?<![\w\\])_(?=\S)((?:[^_\n\\]|\\.)+?)(?<=\S)_(?!\w)"), "<|italic|>", "<|/italic|>"),
]

#: Tolerable nesting depth of a replacement pass — two is plenty for real data,
#: the bound is only there so a pathological input cannot spin.
_MAX_PASSES = 4


def _emphasise(text: str) -> str:
    for _ in range(_MAX_PASSES):
        changed = False
        for pattern, opener, closer in _EMPHASIS:
            def _swap(match: re.Match, opener: str = opener, closer: str = closer) -> str:
                return opener + match.group(1) + closer

            replacement = pattern.sub(_swap, text)
            if replacement != text:
                changed = True
                text = replacement
        if not changed:
            break
    return text


def _code_spans(text: str) -> str:
    return _CODE_SPAN.sub(lambda m: f"{CODE_OPEN}{m.group(1)}{CODE_CLOSE}", text)


def _convert(content: str) -> str:
    """Code spans, then emphasis.

    Two passes because the first changes what the second must leave alone: once
    a span is `<|code|>…<|/code|>` it is a protected leaf, so `**` inside a code
    sample stays the characters the sample meant. Run the other way round, the
    emphasis pass would have converted them first.
    """
    spanned = "".join(
        part if index % 2 else _code_spans(part)
        for index, part in enumerate(_PROTECTED_BLOCKS.split(content))
    )
    return "".join(
        part if index % 2 else _emphasise(part)
        for index, part in enumerate(_PROTECTED.split(spanned))
    )


def _convert_markdown(doc: Document, params: dict, state: State) -> Document | None:
    messages = []
    for turn in doc.get("messages") or []:
        role = str(turn.get("role") or "").strip().lower()
        content = turn.get("content")
        if not isinstance(content, str):
            return None
        if role != "assistant":
            messages.append(turn)
            continue

        messages.append({**turn, "content": _convert(content)})

    return {**doc, "messages": messages}


PROCESSORS = (
    Processor(
        name="markdown-emphasis",
        title="Convert Markdown inline formatting",
        description=(
            "Turn **bold**, *italic* and `code` in assistant replies into <|bold|>, "
            "<|italic|> and <|code|> markers, leaving code blocks and existing markup "
            "alone. Block structure is converted earlier, by convert-chat."
        ),
        applies_to=SFT,
        apply=_convert_markdown,
    ),
)