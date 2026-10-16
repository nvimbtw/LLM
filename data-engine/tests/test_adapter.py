"""Author-facing adapters preserve document shape, metadata and run state."""

from processors.helpers import Ctx, get_text, processor
import pytest


def test_decorator_rejects_a_callback_with_the_wrong_call_contract():
    """A custom definition must accept the adapter's value, parameters and context."""
    with pytest.raises(TypeError):
        processor(name="bad", title="Bad", kinds=["pretrain"])(lambda text: text)


def test_text_rewrites_both_shapes_without_mutating_inputs():
    """A dual-kind callback maps each turn and retains turn/document metadata."""

    @processor(name="upper", title="Upper", kinds=["pretrain", "sft"])
    def upper(text: str, params: dict, ctx: Ctx) -> str:
        """Uppercase the supplied text."""
        return text.upper()

    prose = {"content": "hi", "meta": {"source": "test"}}
    chat = {"messages": [{"role": "user", "content": "hi", "extra": 1}], "meta": {}}
    assert upper.apply(prose, {}, {}) == {**prose, "content": "HI"}
    assert upper.apply(chat, {}, {})["messages"] == [
        {"role": "user", "content": "HI", "extra": 1}
    ]
    assert chat["messages"][0]["content"] == "hi"
    assert prose["content"] == "hi"


def test_one_dropped_turn_drops_the_conversation():
    """A failed text transform preserves exchange integrity by dropping its document."""

    @processor(name="nonempty", title="Nonempty", kinds=["sft"])
    def nonempty(text: str, params: dict, ctx: Ctx) -> str | None:
        """Keep nonempty text."""
        return text or None

    assert (
        nonempty.apply({"messages": [{"content": "hi"}, {"content": ""}]}, {}, {})
        is None
    )


def test_turns_mode_receives_and_returns_the_whole_list():
    """Conversation callbacks can reshape turns without changing the input list."""

    @processor(name="reverse", title="Reverse", kinds=["sft"], mode="turns")
    def reverse(turns: list[dict], params: dict, ctx: Ctx) -> list[dict]:
        """Reverse the supplied conversation."""
        return list(reversed(turns))

    doc = {"messages": [{"content": "one"}, {"content": "two"}], "meta": {"x": 1}}
    assert reverse.apply(doc, {}, {}) == {
        **doc,
        "messages": list(reversed(doc["messages"])),
    }


def test_document_mode_keeps_conversation_filters_and_per_run_state():
    """Whole-document dedup uses shared context without deduplicating individual turns."""

    @processor(name="dedup", title="Dedup", kinds=["pretrain", "sft"], mode="document")
    def dedup(doc: dict, params: dict, ctx: Ctx) -> dict | None:
        """Keep the first copy of the complete document text in this run."""
        seen = ctx.state.setdefault("seen", set())
        text = get_text(doc)
        if text in seen:
            return None
        seen.add(text)
        return doc

    state = {}
    doc = {"messages": [{"content": "same"}, {"content": "same"}]}
    assert dedup.apply(doc, {}, state) == doc
    assert dedup.apply(doc, {}, state) is None
    assert dedup.apply(doc, {}, {}) == doc
    assert get_text({"content": "", "messages": [{"content": "ignored"}]}) == ""
