"""The tool-call seam: the grammar, the relabelling, and what SFT is told.

The `tiny` fixture vocabulary predates the tool markers, which makes it exactly
the two cases worth testing — before relabelling it must still serve, and after
relabelling it must be able to call.
"""

import json

import pytest
import torch
from tokenizers import Tokenizer

from llm.chat import tools
from llm.chat.markup import OpenBlocks, Rules
from llm.chat.template import Reasoning, Role, Turn, training_segments
from llm.corpus import relabel, special
from llm.orchestrate import chat

WEB_SEARCH = tools.by_name("web-search")


@pytest.fixture
def stale(tokenizer_module) -> Tokenizer:
    """The fixture vocabulary as saved: tool slots still spelled as reserved."""
    return tokenizer_module


@pytest.fixture
def relabelled(tokenizer_module, tmp_path) -> Tokenizer:
    """The same vocabulary with its reserved slots adopted, as the migration
    script leaves them."""
    document = json.loads(tokenizer_module.to_str())
    relabel.apply_renames(document, relabel.slot_renames(document))
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return Tokenizer.from_file(str(path))


def _ids(rules: Rules, tokenizer: Tokenizer, *markers: str) -> list[int]:
    return [tokenizer.token_to_id(m) for m in markers]


def test_a_vocabulary_older_than_the_tool_still_builds_rules(stale):
    """A model trained before web search existed has to keep answering. It just
    cannot call anything."""
    rules = Rules.build(stale)
    assert rules.tool_open_for == {}
    assert rules.tool_close_for == {}


def test_relabelling_names_the_slots_without_moving_them(stale, relabelled):
    for marker in (tools.TOOL_RESULT_OPEN, WEB_SEARCH.open, WEB_SEARCH.close):
        slot = special.SPECIAL_TOKENS.index(marker)
        assert relabelled.token_to_id(marker) == slot
        assert stale.token_to_id(special.placeholder(slot)) == slot


def test_relabelling_is_idempotent(relabelled):
    assert relabel.slot_renames(json.loads(relabelled.to_str())) == []


def test_relabelling_refuses_a_vocabulary_it_did_not_build(relabelled):
    document = json.loads(relabelled.to_str())
    slot = special.SPECIAL_TOKENS.index(WEB_SEARCH.open)
    for entry in document["added_tokens"]:
        if entry["id"] == slot:
            entry["content"] = "<|something-else|>"
    with pytest.raises(ValueError, match="not a vocabulary built here"):
        relabel.slot_renames(document)


def test_an_unnamed_reserved_slot_is_still_masked(stale):
    """The placeholders have to stay forbidden. Naming a marker must not knock
    the slots around it out of the range the grammar polices."""
    rules = Rules.build(stale)
    for slot in range(len(special.SPECIAL_TOKENS), special.SPECIAL_BLOCK):
        token_id = stale.token_to_id(special.placeholder(slot))
        assert token_id in rules.forbidden
        assert not OpenBlocks(rules).allows(token_id)


def test_the_model_may_never_emit_a_tool_result(relabelled):
    rules = Rules.build(relabelled)
    for marker in (tools.TOOL_RESULT_OPEN, tools.TOOL_RESULT_CLOSE):
        assert relabelled.token_to_id(marker) in rules.forbidden


def test_a_tool_call_may_only_open_outside_every_block(relabelled):
    rules = Rules.build(relabelled)
    call, par = _ids(rules, relabelled, WEB_SEARCH.open, "<|par|>")

    blocks = OpenBlocks(rules)
    assert blocks.allows(call)

    blocks.observe(par)
    assert not blocks.allows(call)


def test_a_tool_call_admits_no_markup_inside_it(relabelled):
    """Its content is a query string. An opener in there would be searched for."""
    rules = Rules.build(relabelled)
    call, close, par, bold = _ids(
        rules, relabelled, WEB_SEARCH.open, WEB_SEARCH.close, "<|par|>", "<|bold|>"
    )

    blocks = OpenBlocks(rules)
    blocks.observe(call)
    assert not blocks.allows(par)
    assert not blocks.allows(bold)
    assert blocks.allows(close)


def test_a_turn_cannot_end_inside_a_tool_call(relabelled):
    rules = Rules.build(relabelled)
    blocks = OpenBlocks(rules)
    blocks.observe(relabelled.token_to_id(WEB_SEARCH.open))
    assert not blocks.allows(rules.end_of_assistant)
    assert blocks.closers() == [relabelled.token_to_id(WEB_SEARCH.close)]


def _drive(watch, ids: list[int]) -> list[dict]:
    """Feed `ids` through the watcher the way the generation loop does."""
    produced: list[int] = []
    seen = []
    for token_id in ids:
        produced.append(token_id)
        if (call := watch.observe(token_id, produced)) is not None:
            seen.append(call)
    return seen


def test_a_call_is_recognised_from_its_ids_as_it_closes(relabelled):
    rules = Rules.build(relabelled)
    watch = chat._ToolCalls(rules, relabelled)

    ids = [
        relabelled.token_to_id("<|par|>"),
        *relabelled.encode("the cat").ids,
        relabelled.token_to_id("<|/par|>"),
        relabelled.token_to_id(WEB_SEARCH.open),
        *relabelled.encode("the cat sat").ids,
        relabelled.token_to_id(WEB_SEARCH.close),
    ]

    assert _drive(watch, ids) == [{"tool": "web-search", "argument": "the cat sat"}]
    assert watch.made == [{"tool": "web-search", "argument": "the cat sat"}]


def test_a_turn_stops_being_allowed_to_call_once_it_has_called_enough(relabelled):
    rules = Rules.build(relabelled)
    watch = chat._ToolCalls(rules, relabelled)
    opener = relabelled.token_to_id(WEB_SEARCH.open)

    logits = torch.zeros(relabelled.get_vocab_size())
    watch.mask(logits)
    assert logits[opener] == 0.0

    watch.made = [{"tool": "web-search", "argument": "x"}] * chat.MAX_TOOL_CALLS_PER_TURN
    watch.mask(logits)
    assert logits[opener] == float("-inf")


def test_a_tool_result_is_never_a_training_target():
    """The model reads results and writes calls. Training the result would teach
    it to invent search output."""
    result = tools.render_result(WEB_SEARCH, "elbrus height", "<|par|>5,642 m<|/par|>")
    turns = [
        Turn(Role.USER, "how tall is elbrus?"),
        Turn(Role.ASSISTANT, WEB_SEARCH.call("elbrus height")),
        Turn(Role.TOOL, result),
        Turn(Role.ASSISTANT, "<|par|>5,642 m.<|/par|>"),
    ]
    segments = training_segments(None, turns)

    trained = [text for text, target in segments if target]
    assert WEB_SEARCH.call("elbrus height") in trained
    assert all(tools.TOOL_RESULT_OPEN not in text for text in trained)

    block = next(text for text, _ in segments if text.startswith(tools.TOOL_RESULT_OPEN))
    assert block == f"{tools.TOOL_RESULT_OPEN}{result}{tools.TOOL_RESULT_CLOSE}"
    assert dict(segments)[block] is False


def test_a_tool_result_survives_a_reasoning_strip():
    """STRIP drops thinking. It must not drop the call the next result answers,
    or the conversation stops making sense."""
    turns = [
        Turn(Role.USER, "how tall is elbrus?"),
        Turn(Role.ASSISTANT, "<|thinking|>look it up<|/thinking|>" + WEB_SEARCH.call("elbrus")),
        Turn(Role.TOOL, tools.render_result(WEB_SEARCH, "elbrus", "<|par|>5,642 m<|/par|>")),
        Turn(Role.ASSISTANT, "<|par|>5,642 m.<|/par|>"),
    ]
    rendered = "".join(text for text, _ in training_segments(None, turns, reasoning=Reasoning.STRIP))

    assert "<|thinking|>" not in rendered
    assert WEB_SEARCH.call("elbrus") in rendered
    assert tools.TOOL_RESULT_OPEN in rendered
