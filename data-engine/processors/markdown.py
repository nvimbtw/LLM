"""Converting Markdown emphasis into the project's markup markers.

SmolTalk (like most instruct sources) answers with Markdown: `**bold**`,
`*italic*`, `_underscore italic_`. Our markers are `<|bold|>…<|/bold|>` and
`<|italic|>…<|/italic|>` — the model must *emit* those, and it can only learn
to emit what it sees. Running this processor in the same pipeline as
`convert-smoltalk` is what turns the source's styling into the vocabulary's.

Conversion is deliberately conservative. It runs on **assistant turns only** —
responses are where the model learns to format; a user's `**bold**` is a
quoted word, not a style instruction, and rewriting it would change the
question. The protected-region split skips: fenced code blocks, inline code
spans, already-rendered `<|...|>` markers (so a run after `convert-smoltalk`
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

#: Emphasis forms, widest first so `***x***` is consumed before `**x**` and
#: `*x*` before either arc of a `**` pair can mis-fragment it. Each re-runs
#: until the text stops changing: a `**outer *inner* text**` needs the inner
#: pass to complete before the outer can close, and finishing the pass makes
#: that kind of nesting fall out for free.
_EMPHASIS = [
    (re.compile(r"\*\*\*(?=\S)([^*\n]+?)(?<=\S)\*\*\*"), "<|bold|><|italic|>", "<|/italic|><|/bold|>"),
    (re.compile(r"(?<!\w)___(?=\S)([^_\n]+?)(?<=\S)___(?!\w)"), "<|bold|><|italic|>", "<|/italic|><|/bold|>"),
    (re.compile(r"\*\*(?=\S)([^*\n]+?)(?<=\S)\*\*"), "<|bold|>", "<|/bold|>"),
    (re.compile(r"(?<!\w)__(?=\S)([^_\n]+?)(?<=\S)__(?!\w)"), "<|bold|>", "<|/bold|>"),
    (re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])"), "<|italic|>", "<|/italic|>"),
    (re.compile(r"(?<!\w)_(?=\S)([^_\n]+?)(?<=\S)_(?!\w)"), "<|italic|>", "<|/italic|>"),
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

        parts = _PROTECTED.split(content)
        out: list[str] = []
        for index, part in enumerate(parts):
            out.append(part if index % 2 else _emphasise(part))
        messages.append({**turn, "content": "".join(out)})

    return {**doc, "messages": messages}


PROCESSORS = (
    Processor(
        name="markdown-emphasis",
        title="Convert Markdown emphasis",
        description=(
            "Turn **bold** and *italic* in assistant replies into <|bold|> and "
            "<|italic|> markers, leaving code and existing markup alone."
        ),
        applies_to=SFT,
        apply=_convert_markdown,
    ),
)