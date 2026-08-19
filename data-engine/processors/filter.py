"""Processors that decide whether a document is worth keeping.

Each returns `None` to drop. They apply to both kinds, so length is measured
over whatever the document actually carries — prose for `pretrain`, the
concatenated turns for `sft`.
"""

import hashlib
import re

from processors.spec import Document, Param, Processor, State
from store.schema import DatasetKind

BOTH = (DatasetKind.pretrain, DatasetKind.sft)

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?)?\d{3}[ -]?\d{3,4}(?!\w)")
#: IPv4 only. A general IPv6 pattern matches far too much ordinary punctuation
#: to be worth the false positives on prose.
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def text_of(doc: Document) -> str:
    """What this document says, whichever shape it is in."""
    if doc.get("content") is not None:
        return doc["content"] or ""
    return "\n".join(str(m.get("content") or "") for m in (doc.get("messages") or []))


def _min_length(doc: Document, params: dict, state: State) -> Document | None:
    return doc if len(text_of(doc)) >= params["characters"] else None


def _max_length(doc: Document, params: dict, state: State) -> Document | None:
    text = text_of(doc)
    if len(text) <= params["characters"]:
        return doc
    if params["truncate"] and doc.get("content") is not None:
        return {**doc, "content": text[: params["characters"]]}
    return None


def _dedup_exact(doc: Document, params: dict, state: State) -> Document | None:
    """Keep the first copy of each document, drop every later one.

    Hashed rather than compared: holding every document seen so far in memory to
    compare against is the one thing a corpus-sized pass cannot afford.
    """
    seen: set[str] = state.setdefault("dedup-exact", set())
    text = text_of(doc)
    if params["ignore_whitespace"]:
        text = " ".join(text.split())
    if params["ignore_case"]:
        text = text.lower()

    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
    if digest in seen:
        return None
    seen.add(digest)
    return doc


def _redact(text: str, params: dict) -> str:
    if params["emails"]:
        text = _EMAIL.sub("[email]", text)
    if params["phone_numbers"]:
        text = _PHONE.sub("[phone]", text)
    if params["ip_addresses"]:
        text = _IP.sub("[ip]", text)
    return text


def _redact_pii(doc: Document, params: dict, state: State) -> Document | None:
    if doc.get("content") is not None:
        return {**doc, "content": _redact(doc["content"] or "", params)}
    messages = [
        {**m, "content": _redact(str(m.get("content") or ""), params)}
        for m in (doc.get("messages") or [])
    ]
    return {**doc, "messages": messages}


PROCESSORS = (
    Processor(
        name="min-length",
        title="Drop short documents",
        description="Remove anything below a character count.",
        applies_to=BOTH,
        apply=_min_length,
        params=(
            Param(
                name="characters",
                label="Minimum characters",
                type="int",
                default=200,
                minimum=0,
                maximum=100_000,
            ),
        ),
    ),
    Processor(
        name="max-length",
        title="Bound long documents",
        description="Drop or truncate anything above a character count.",
        applies_to=BOTH,
        apply=_max_length,
        params=(
            Param(
                name="characters",
                label="Maximum characters",
                type="int",
                default=50_000,
                minimum=1,
                maximum=1_000_000,
            ),
            Param(
                name="truncate",
                label="Truncate instead of dropping",
                type="bool",
                default=False,
                description="Conversations are always dropped — a half exchange is not data.",
            ),
        ),
    ),
    Processor(
        name="dedup-exact",
        title="Remove duplicates",
        description="Keep the first copy of each document and drop the rest.",
        applies_to=BOTH,
        apply=_dedup_exact,
        params=(
            Param(
                name="ignore_whitespace",
                label="Ignore whitespace",
                type="bool",
                default=True,
            ),
            Param(name="ignore_case", label="Ignore case", type="bool", default=False),
        ),
    ),
    Processor(
        name="redact-pii",
        title="Redact identifiers",
        description="Replace emails, phone numbers and IP addresses with placeholders.",
        applies_to=BOTH,
        apply=_redact_pii,
        params=(
            Param(name="emails", label="Emails", type="bool", default=True),
            Param(name="phone_numbers", label="Phone numbers", type="bool", default=False),
            Param(name="ip_addresses", label="IP addresses", type="bool", default=True),
        ),
    ),
)
