"""The author-facing processor API and document shape adapters."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import signature
from typing import Any, Literal

from processors.spec import Apply, Document, Param, Processor, State
from store.schema import DatasetKind

__all__ = ["Ctx", "Param", "drop", "get_text", "map_messages", "processor", "set_text"]


@dataclass(frozen=True)
class Ctx:
    """Scratch state owned by one pipeline run, shared across its documents."""

    state: State


def get_text(doc: Document) -> str:
    """Return prose or newline-joined conversation text, preserving empty content."""
    if doc.get("content") is not None:
        return doc["content"] or ""
    return "\n".join(str(m.get("content") or "") for m in (doc.get("messages") or []))


def set_text(doc: Document, text: str) -> Document:
    """Return a prose document with new text and all other fields preserved."""
    return {**doc, "content": text}


def drop() -> None:
    """Return the pipeline's document-drop signal."""
    return None


def map_messages(doc: Document, fn: Callable[[str], object]) -> Document | None:
    """Rewrite every turn without mutating input; one dropped turn drops the exchange."""
    messages = []
    for message in doc.get("messages") or []:
        text = _text_result(fn(str(message.get("content") or "")))
        if text is None:
            return None
        messages.append({**message, "content": text})
    return {**doc, "messages": messages}


def _text_result(value: object) -> str | None:
    """Validate a text callback's output before adapting it to document storage."""
    if value is not None and not isinstance(value, str):
        raise TypeError("a text processor must return str or None")
    return value


def _rewrite(
    doc: Document, fn: Callable[[str], object], maps_turns: bool
) -> Document | None:
    """Adapt a text callback to the document's shape and validate its return value."""
    if maps_turns and doc.get("content") is None and "messages" in doc:
        return map_messages(doc, fn)
    text = _text_result(fn(doc.get("content") or ""))
    if text is None:
        return None
    return set_text(doc, text)


def _apply(fn: Callable[..., object], mode: str, maps_turns: bool) -> Apply:
    """Build the legacy Apply callable for a declared authoring mode."""

    def apply(doc: Document, params: dict[str, Any], state: State) -> Document | None:
        """Invoke author code with per-run context and preserve document metadata."""
        ctx = Ctx(state)
        if mode == "text":
            return _rewrite(doc, lambda text: fn(text, params, ctx), maps_turns)
        if mode == "document":
            result = fn(doc, params, ctx)
            if result is not None and not isinstance(result, dict):
                raise TypeError("a document processor must return dict or None")
            return result
        if doc.get("content") is not None:
            raise ValueError("turns mode requires a conversation")
        result = fn([dict(m) for m in doc.get("messages") or []], params, ctx)
        if result is None:
            return None
        if not isinstance(result, list) or any(not isinstance(m, dict) for m in result):
            raise TypeError("a turns processor must return a message list or None")
        return {**doc, "messages": result}

    return apply


def processor(
    *,
    name: str,
    title: str,
    kinds: Sequence[str | DatasetKind],
    params: Sequence[Param] = (),
    mode: Literal["text", "turns", "document"] = "text",
    description: str = "",
) -> Callable[[Callable[..., object]], Processor]:
    """Declare metadata and adapt run(text, params, ctx) to the existing pipeline.

    Text mode rewrites prose or each turn; turns mode receives a full message
    list. Document mode preserves whole-document filtering and legacy policy.
    Invalid modes/kinds fail at declaration; parameters retain Param semantics.
    """
    if mode not in ("text", "turns", "document"):
        raise ValueError(f"unknown processor mode: {mode}")
    applies_to = tuple(DatasetKind(kind) for kind in kinds)
    if not applies_to:
        raise ValueError("a processor needs at least one kind")

    def decorate(fn: Callable[..., object]) -> Processor:
        """Return a registry entry carrying the callback's declared metadata."""
        signature(fn).bind(None, {}, Ctx({}))
        return Processor(
            name=name,
            title=title,
            description=description,
            applies_to=applies_to,
            apply=_apply(fn, mode, DatasetKind.sft in applies_to),
            params=tuple(params),
            file=fn.__code__.co_filename,
        )

    return decorate
