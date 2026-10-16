"""Text rewrites and document filters authored through the shared processor adapter."""

import hashlib
import re
from processors.helpers import Ctx, Param, drop, get_text, processor, set_text

_EMAIL = re.compile("\\b[\\w.+-]+@[\\w-]+\\.[\\w.-]+\\b")
_PHONE = re.compile(
    "(?<!\\w)(?:\\+\\d{1,3}[ -]?)?(?:\\(\\d{2,4}\\)[ -]?)?\\d{3}[ -]?\\d{3,4}(?!\\w)"
)
_IP = re.compile("\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b")


@processor(
    name="min-length",
    title="Drop short documents",
    description="Remove anything below a character count.",
    params=(
        Param(
            name="characters",
            label="Minimum characters",
            type="int",
            default=200,
            minimum=0,
            maximum=100000,
        ),
    ),
    kinds=["pretrain", "sft"],
    mode="document",
)
def _min_length(doc: dict, params: dict, ctx: Ctx) -> dict | None:
    """Remove anything below a character count. Return the document or a drop signal."""
    return doc if len(get_text(doc)) >= params["characters"] else drop()


@processor(
    name="max-length",
    title="Bound long documents",
    description="Drop or truncate anything above a character count.",
    params=(
        Param(
            name="characters",
            label="Maximum characters",
            type="int",
            default=50000,
            minimum=1,
            maximum=1000000,
        ),
        Param(
            name="truncate",
            label="Truncate instead of dropping",
            type="bool",
            default=False,
            description="Conversations are always dropped — a half exchange is not data.",
        ),
    ),
    kinds=["pretrain", "sft"],
    mode="document",
)
def _max_length(doc: dict, params: dict, ctx: Ctx) -> dict | None:
    """Drop or truncate anything above a character count. Return the document or a drop signal."""
    text = get_text(doc)
    if len(text) <= params["characters"]:
        return doc
    if params["truncate"] and doc.get("content") is not None:
        return set_text(doc, text[: params["characters"]])
    return drop()


@processor(
    name="dedup-exact",
    title="Remove duplicates",
    description="Keep the first copy of each document and drop the rest.",
    params=(
        Param(
            name="ignore_whitespace",
            label="Ignore whitespace",
            type="bool",
            default=True,
        ),
        Param(name="ignore_case", label="Ignore case", type="bool", default=False),
    ),
    kinds=["pretrain", "sft"],
    mode="document",
)
def _dedup_exact(doc: dict, params: dict, ctx: Ctx) -> dict | None:
    """Keep the first copy of each document, drop every later one.

    Hashed rather than compared: holding every document seen so far in memory to
    compare against is the one thing a corpus-sized pass cannot afford.
    """
    seen: set[str] = ctx.state.setdefault("dedup-exact", set())
    text = get_text(doc)
    if params["ignore_whitespace"]:
        text = " ".join(text.split())
    if params["ignore_case"]:
        text = text.lower()
    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
    if digest in seen:
        return drop()
    seen.add(digest)
    return doc


@processor(
    name="redact-pii",
    title="Redact identifiers",
    description="Replace emails, phone numbers and IP addresses with placeholders.",
    params=(
        Param(name="emails", label="Emails", type="bool", default=True),
        Param(name="phone_numbers", label="Phone numbers", type="bool", default=False),
        Param(name="ip_addresses", label="IP addresses", type="bool", default=True),
    ),
    kinds=["pretrain", "sft"],
)
def _redact_pii(text: str, params: dict, ctx: Ctx) -> str:
    """Replace emails, phone numbers and IP addresses with placeholders. Return the transformed text."""
    if params["emails"]:
        text = _EMAIL.sub("[email]", text)
    if params["phone_numbers"]:
        text = _PHONE.sub("[phone]", text)
    if params["ip_addresses"]:
        text = _IP.sub("[ip]", text)
    return text


PROCESSORS = (_min_length, _max_length, _dedup_exact, _redact_pii)
