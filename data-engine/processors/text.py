"""Text rewrites and document filters authored through the shared processor adapter."""

import html
import re
import unicodedata
from processors.helpers import Ctx, Param, processor

_TAG = re.compile("<[^>]+>")
_SCRIPT = re.compile("<(script|style)\\b[^>]*>.*?</\\1>", re.IGNORECASE | re.DOTALL)
_BLANK_RUN = re.compile("\\n{3,}")
_SPACE_RUN = re.compile("[ \\t]{2,}")
_TRAILING = re.compile("[ \\t]+$", re.MULTILINE)
_BOILERPLATE = re.compile(
    "^(share this|advertisement|read more|related articles?|cookie policy|all rights reserved|sign up|subscribe|click here)[.!]?$",
    re.IGNORECASE,
)


@processor(
    name="strip-html",
    title="Strip HTML",
    description="Remove tags, and the contents of script and style blocks.",
    params=(
        Param(
            name="unescape_entities",
            label="Unescape entities",
            type="bool",
            default=True,
            description="Turn &amp; and friends back into the characters they name.",
        ),
    ),
    kinds=["pretrain"],
)
def _strip_html(text: str, params: dict, ctx: Ctx) -> str:
    """Remove script/style bodies before tags, then optionally decode HTML entities."""
    text = _SCRIPT.sub(" ", text)
    text = _TAG.sub(" ", text)
    if params["unescape_entities"]:
        text = html.unescape(text)
    return text


@processor(
    name="normalise-unicode",
    title="Normalise Unicode",
    description="Put text into one canonical form and drop control characters.",
    params=(
        Param(
            name="form",
            label="Form",
            type="string",
            default="NFKC",
            description="NFC, NFD, NFKC or NFKD.",
        ),
        Param(
            name="strip_control",
            label="Strip control characters",
            type="bool",
            default=True,
            description="Newlines and tabs are kept.",
        ),
    ),
    kinds=["pretrain"],
)
def _normalise_unicode(text: str, params: dict, ctx: Ctx) -> str:
    """Put text into one canonical form and drop control characters. Return the transformed text."""
    text = unicodedata.normalize(params["form"], text)
    if params["strip_control"]:
        text = "".join(
            (ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
        )
    return text


@processor(
    name="collapse-whitespace",
    title="Collapse whitespace",
    description="Normalise line endings, trim trailing spaces, squeeze runs.",
    params=(
        Param(
            name="collapse_blank_lines",
            label="Collapse blank lines",
            type="bool",
            default=True,
            description="Reduce three or more newlines to a paragraph break.",
        ),
    ),
    kinds=["pretrain"],
)
def _collapse_whitespace(text: str, params: dict, ctx: Ctx) -> str:
    """Normalise line endings, trim trailing spaces, squeeze runs. Return the transformed text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING.sub("", text)
    text = _SPACE_RUN.sub(" ", text)
    if params["collapse_blank_lines"]:
        text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


@processor(
    name="strip-boilerplate",
    title="Strip boilerplate lines",
    description="Drop whole lines that are navigation furniture, not prose.",
    kinds=["pretrain"],
)
def _strip_boilerplate(text: str, params: dict, ctx: Ctx) -> str:
    """Drop whole lines that are navigation furniture, not prose. Return the transformed text."""
    kept = [line for line in text.split("\n") if not _BOILERPLATE.match(line.strip())]
    return "\n".join(kept)


@processor(
    name="join-broken-lines",
    title="Join broken lines",
    description="Drop every blank line and pull the surviving lines together. For plain-text dumps — Gutenberg's is one — that put a blank line after every line rather than only between paragraphs, so a document reads as one line per line instead of one sentence per blank-separated fragment.",
    kinds=["pretrain"],
)
def _join_broken_lines(text: str, params: dict, ctx: Ctx) -> str:
    """Normalize line endings and return nonblank lines joined by single newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


PROCESSORS = (
    _strip_html,
    _normalise_unicode,
    _collapse_whitespace,
    _strip_boilerplate,
    _join_broken_lines,
)
