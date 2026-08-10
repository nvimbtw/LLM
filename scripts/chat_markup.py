"""Turning markdown chat data into this repo's marker vocabulary.

Shared by every SFT fetcher, because a second copy that drifted would produce
training data whose markup disagrees with what the frontend renders — and that
failure shows up as garbled assistant output long after the fetch that caused it.

## Markdown becomes markup

The source corpora are markdown; the models here speak the marker vocabulary in
`engine/llm/corpus/special.py`. Block level:

    blank-line separated  ->  <|par|>...<|/par|>
    ```lang ... ```       ->  <|codeblock|>...<|/codeblock|>
    -, *, +, "1." lines   ->  <|list|><|item|>...<|/item|>...<|/list|>
    # H / ## H            ->  <|heading|> / <|subheading|>
    > quoted              ->  <|quote|>...<|/quote|>
    --- / *** / ___       ->  <|divider|>

Inline, inside everything but a code block:

    `x`                   ->  <|code|>x<|/code|>
    **x**                 ->  <|bold|>x<|/bold|>
    *x*                   ->  <|italic|>x<|/italic|>
    [text](url)           ->  text <|link|>url<|/link|>

**Underscore emphasis (`_x_`, `__x__`) is deliberately not converted**: these
corpora run ~11% code and `sort_tuples` or `__init__` would be mangled into
italics. Bare URLs are left alone for the same reason — a URL inside a code span
is common and the inline pass would have to unpick which is which.

**Nested lists are flattened to one level.** The rendering has no nesting
convention below `list > item`, and a model at this size has no chance of
keeping two levels balanced anyway.

**Code spans are lifted out before the emphasis pass** and put back after, so
`**` and `*` inside code survive exactly as written.

## Fitting the context window

The whole rendered example — system, every turn, the scaffold — has to fit the
model's window. The engine refuses one that does not and counts it as skipped,
so nothing downstream breaks; cutting here as well keeps that count near zero,
because a conversation refused there is one that was imported, converted and
written for nothing.

Conversations are truncated to the longest whole-pair prefix that fits, which
keeps a 6-turn conversation as its first 2 or 4 turns rather than dropping it. A
pair is kept or dropped whole: cutting inside an assistant reply would teach the
model to stop mid-sentence. A conversation whose *first* pair alone overruns is
dropped entirely.

The budget is an *estimate*, because the real tokenizer is trained per-corpus and
is not available here. Two rates, because one alone is badly wrong:

    ASCII bytes      ~3.0 bytes per token   (tuned, see below)
    non-ASCII bytes  ~1.0 bytes per token   (no merge covers them; byte fallback)

Characters are the wrong unit and bytes alone are barely better. These corpora
contain Chinese, Russian and Korean conversations, and one measured 2,572
characters, 7,108 UTF-8 bytes and roughly 7,000 tokens under an 8192-vocabulary
that has never seen a Han character. A character budget admitted it; a flat byte
budget would have too.

That the two-rate estimate prices such conversations out is the right outcome
and not a workaround. A model with this vocabulary cannot represent them in less
than one token per byte, so they would spend a seventh of the window on one
sentence while teaching nothing an English model can use.
"""

import re

from corpus_shards import neutralize  # noqa: F401  (re-exported: one marker regex, not two)

# Bytes per token for ASCII text under the current 8192-merge vocabulary.
#
# Held-out prose measures 3.79 and these corpora average 3.71, but neither is the
# number to use: what overruns the window is the code-heavy tail, and code
# tokenizes far worse than the prose the merges were learned from. Tuned down
# against the exact check until the fraction the engine has to refuse settled:
# 3.5 left 1.18% overrunning, 3.0 leaves 0.29% while still keeping 93.7% of the
# source. Lower it further and the loss is in what gets dropped, not in what
# gets through.
ASCII_BYTES_PER_TOKEN = 3.0

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
QUOTE = re.compile(r"^\s{0,3}>\s?")
HR = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
BULLET = re.compile(r"^(\s*)(?:[-*+]|\d{1,3}[.)])\s+(.*)$")
FENCE = re.compile(r"^\s*```")

CODE_SPAN = re.compile(r"`([^`\n]+)`")
BOLD = re.compile(r"\*\*(\S(?:[^*]*\S)?)\*\*")
ITALIC = re.compile(r"(?<![*\w])\*(\S(?:[^*]*\S)?)\*(?![*\w])")
MDLINK = re.compile(r"\[([^\]\n]*)\]\(([^)\s]+)\)")

# Placeholder for a lifted code span. Control characters cannot appear in the
# source text, so this cannot collide with anything real.
HOLE = "\x00%d\x00"
HOLE_RE = re.compile(r"\x00(\d+)\x00")

# The canonical reasoning markers, from engine/llm/chat/template.py. Written out
# rather than imported because the engine is a separate package with its own
# virtualenv, and these scripts must run without it importable.
THINKING_OPEN = "<|thinking|>"
THINKING_CLOSE = "<|/thinking|>"

# How a source dataset spells a reasoning block before conversion.
SOURCE_THINK = re.compile(r"\A\s*<think>(.*?)</think>\s*", re.S | re.I)

# Structural markers the chat template supplies itself. Data containing them is
# rejected rather than escaped: it would be claiming to be scaffold.
STRUCTURAL = ("<|user|>", "<|assistant|>", "<|system|>",
              "<|/user|>", "<|/assistant|>", "<|/system|>")


def inline(text: str) -> str:
    """Emphasis, code spans and links, in the one order that composes.

    Code spans are lifted first so the emphasis passes cannot see inside them,
    and restored last so a `**` inside code survives verbatim.
    """
    spans: list[str] = []

    def lift(m: re.Match) -> str:
        spans.append(m.group(1))
        return HOLE % (len(spans) - 1)

    text = CODE_SPAN.sub(lift, text)
    text = MDLINK.sub(
        lambda m: (f"{m.group(1)} " if m.group(1).strip() else "")
        + f"<|link|>{m.group(2)}<|/link|>",
        text,
    )
    text = BOLD.sub(lambda m: f"<|bold|>{m.group(1)}<|/bold|>", text)
    text = ITALIC.sub(lambda m: f"<|italic|>{m.group(1)}<|/italic|>", text)
    return HOLE_RE.sub(lambda m: f"<|code|>{spans[int(m.group(1))]}<|/code|>", text)


def starts_block(line: str) -> bool:
    """Whether `line` begins a block, and so ends the one being accumulated."""
    return bool(
        not line.strip()
        or FENCE.match(line)
        or HEADING.match(line)
        or QUOTE.match(line)
        or HR.match(line)
        or BULLET.match(line)
    )


def split_blocks(text: str):
    """Markdown to a flat sequence of (kind, payload) blocks."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if FENCE.match(line):
            body = []
            i += 1
            while i < len(lines) and not FENCE.match(lines[i]):
                body.append(lines[i])
                i += 1
            i += 1  # the closing fence, or the end of the text
            yield "codeblock", "\n".join(body).strip("\n")
            continue

        if HR.match(line):
            yield "divider", ""
            i += 1
            continue

        m = HEADING.match(line)
        if m:
            kind = "heading" if len(m.group(1)) == 1 else "subheading"
            yield kind, m.group(2).strip()
            i += 1
            continue

        if QUOTE.match(line):
            body = []
            while i < len(lines) and QUOTE.match(lines[i]):
                body.append(QUOTE.sub("", lines[i]).strip())
                i += 1
            yield "quote", " ".join(x for x in body if x)
            continue

        if BULLET.match(line):
            items: list[str] = []
            while i < len(lines):
                m = BULLET.match(lines[i])
                if m:
                    # Nesting is flattened: an indented item joins the same list.
                    items.append(m.group(2).strip())
                    i += 1
                elif lines[i].strip() and not starts_block(lines[i]):
                    # A wrapped continuation line belongs to the item above it.
                    if items:
                        items[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            yield "list", [x for x in items if x]
            continue

        body = []
        while i < len(lines) and lines[i].strip() and not starts_block(lines[i]):
            body.append(lines[i].strip())
            i += 1
        yield "par", " ".join(body)


def to_markup(text: str) -> str:
    """An assistant reply, rendered in the marker vocabulary."""
    out = []
    for kind, payload in split_blocks(text):
        if kind == "codeblock":
            # Never passed through `inline`: its content is code, and a `*` in it
            # is multiplication rather than emphasis.
            if payload:
                out.append(f"<|codeblock|>{payload}<|/codeblock|>")
        elif kind == "divider":
            out.append("<|divider|>")
        elif kind == "list":
            if payload:
                items = "".join(f"<|item|>{inline(x)}<|/item|>" for x in payload)
                out.append(f"<|list|>{items}<|/list|>")
        elif payload:
            out.append(f"<|{kind}|>{inline(payload)}<|/{kind}|>")
    return "".join(out)


def split_reasoning(content: str) -> tuple[str | None, str]:
    """`(reasoning, reply)` from an assistant turn that may open with a
    `<think>` block, or `(None, content)` when it does not.

    Only a block at the very start is recognized, which is where every source
    that uses this convention puts it. A `<think>` that opens and never closes
    returns no reasoning and leaves the content alone — the caller drops such a
    turn rather than guessing where the author meant it to end, because a
    mismatched block survives conversion as literal text inside what is supposed
    to be reasoning-free output.
    """
    match = SOURCE_THINK.match(content)
    if not match:
        return None, content
    return match.group(1).strip(), content[match.end():]


def has_structural_markers(text: str) -> bool:
    """Whether text claims to be chat scaffold. The template supplies those."""
    return any(marker in text for marker in STRUCTURAL)


def est_tokens(text: str) -> int:
    """Roughly how many tokens `text` costs under the current vocabulary.

    Non-ASCII bytes are charged one token each because no merge covers them —
    the encoder falls back to raw bytes, so a Han character is three tokens and a
    paragraph of Chinese is its own byte length. Getting this wrong in the cheap
    direction (characters, or bytes at one flat rate) admits conversations that
    overrun the window by 6x.
    """
    raw = text.encode("utf-8")
    non_ascii = sum(1 for b in raw if b >= 0x80)
    return int((len(raw) - non_ascii) / ASCII_BYTES_PER_TOKEN + non_ascii) + 1


def fit_pairs(turns: list[dict], max_tokens: int, system: str | None = None) -> list[dict]:
    """The longest whole-pair prefix of `turns` that fits the window.

    Empty when even the opening exchange overruns — the caller drops the
    conversation, because truncating inside a reply teaches the model to stop
    mid-sentence.
    """
    budget = max_tokens - (est_tokens(system) if system else 0)
    kept: list[dict] = []
    used = 0
    for i in range(0, len(turns), 2):
        pair = turns[i : i + 2]
        cost = sum(est_tokens(t["text"]) for t in pair)
        if used + cost > budget:
            break
        kept.extend(pair)
        used += cost
    return kept
