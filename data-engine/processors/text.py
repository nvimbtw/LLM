"""Processors that rewrite prose in place.

None of these drop a document; they only change what it says. The ones that
decide whether a document is worth keeping live in `filter`.
"""

import html
import re
import unicodedata

from processors.spec import Document, Param, Processor, State
from store.schema import DatasetKind

PRETRAIN = (DatasetKind.pretrain,)

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_BLANK_RUN = re.compile(r"\n{3,}")
_SPACE_RUN = re.compile(r"[ \t]{2,}")
_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)

#: Lines that are navigation furniture rather than prose. Matched whole, after
#: stripping, so a sentence that merely contains one of these words survives.
_BOILERPLATE = re.compile(
    r"^(share this|advertisement|read more|related articles?|cookie policy|"
    r"all rights reserved|sign up|subscribe|click here)[.!]?$",
    re.IGNORECASE,
)


def _rewrite(doc: Document, text: str) -> Document:
    return {**doc, "content": text}


def _strip_html(doc: Document, params: dict, state: State) -> Document | None:
    text = doc.get("content") or ""
    # Script and style bodies go first: dropping only their tags would leave the
    # code itself sitting in the corpus as if it were prose.
    text = _SCRIPT.sub(" ", text)
    text = _TAG.sub(" ", text)
    if params["unescape_entities"]:
        text = html.unescape(text)
    return _rewrite(doc, text)


def _normalise_unicode(doc: Document, params: dict, state: State) -> Document | None:
    text = unicodedata.normalize(params["form"], doc.get("content") or "")
    if params["strip_control"]:
        text = "".join(
            ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
        )
    return _rewrite(doc, text)


def _collapse_whitespace(doc: Document, params: dict, state: State) -> Document | None:
    text = (doc.get("content") or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING.sub("", text)
    text = _SPACE_RUN.sub(" ", text)
    if params["collapse_blank_lines"]:
        text = _BLANK_RUN.sub("\n\n", text)
    return _rewrite(doc, text.strip())


def _strip_boilerplate(doc: Document, params: dict, state: State) -> Document | None:
    kept = [
        line
        for line in (doc.get("content") or "").split("\n")
        if not _BOILERPLATE.match(line.strip())
    ]
    return _rewrite(doc, "\n".join(kept))


PROCESSORS = (
    Processor(
        name="strip-html",
        title="Strip HTML",
        description="Remove tags, and the contents of script and style blocks.",
        applies_to=PRETRAIN,
        apply=_strip_html,
        params=(
            Param(
                name="unescape_entities",
                label="Unescape entities",
                type="bool",
                default=True,
                description="Turn &amp; and friends back into the characters they name.",
            ),
        ),
    ),
    Processor(
        name="normalise-unicode",
        title="Normalise Unicode",
        description="Put text into one canonical form and drop control characters.",
        applies_to=PRETRAIN,
        apply=_normalise_unicode,
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
    ),
    Processor(
        name="collapse-whitespace",
        title="Collapse whitespace",
        description="Normalise line endings, trim trailing spaces, squeeze runs.",
        applies_to=PRETRAIN,
        apply=_collapse_whitespace,
        params=(
            Param(
                name="collapse_blank_lines",
                label="Collapse blank lines",
                type="bool",
                default=True,
                description="Reduce three or more newlines to a paragraph break.",
            ),
        ),
    ),
    Processor(
        name="strip-boilerplate",
        title="Strip boilerplate lines",
        description="Drop whole lines that are navigation furniture, not prose.",
        applies_to=PRETRAIN,
        apply=_strip_boilerplate,
    ),
)
