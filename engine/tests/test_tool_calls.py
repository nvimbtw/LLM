"""The tool-call seam: the grammar, the relabelling, and what SFT is told.

Every tool is called through one marker pair carrying a JSON object, so these
come in two halves: the grammar still guarantees the *shape* of a call (one leaf
block, outside everything, closed before the turn ends), and `tools.parse`
decides after the fact whether what is inside names anything real. A body that
does not is not a call — and must never be a crash.

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


def _ids(tokenizer: Tokenizer, *markers: str) -> list[int]:
    return [tokenizer.token_to_id(m) for m in markers]


def _body_of(call: str) -> str:
    """The JSON between the markers of a rendered call."""
    return call[len(tools.TOOL_CALL_OPEN) : -len(tools.TOOL_CALL_CLOSE)]


def _call_ids(tokenizer: Tokenizer, body: str) -> list[int]:
    """The ids the generation loop would produce for a call with this body."""
    return [
        tokenizer.token_to_id(tools.TOOL_CALL_OPEN),
        *tokenizer.encode(body).ids,
        tokenizer.token_to_id(tools.TOOL_CALL_CLOSE),
    ]


def _drive(watch, ids: list[int]) -> list[dict]:
    """Feed `ids` through the watcher the way the generation loop does."""
    produced: list[int] = []
    seen = []
    for token_id in ids:
        produced.append(token_id)
        if (call := watch.observe(token_id, produced)) is not None:
            seen.append(call)
    return seen


def test_a_vocabulary_older_than_tool_calls_still_builds_rules(stale):
    """A model trained before tool calls existed has to keep answering. It just
    cannot call anything."""
    rules = Rules.build(stale)
    assert rules.tool_call is None


def test_relabelling_names_the_slots_without_moving_them(stale, relabelled):
    for marker in (tools.TOOL_RESULT_OPEN, tools.TOOL_CALL_OPEN, tools.TOOL_CALL_CLOSE):
        slot = special.SPECIAL_TOKENS.index(marker)
        assert relabelled.token_to_id(marker) == slot
        assert stale.token_to_id(special.placeholder(slot)) == slot


def test_relabelling_is_idempotent(relabelled):
    assert relabel.slot_renames(json.loads(relabelled.to_str())) == []


def test_relabelling_refuses_a_vocabulary_it_did_not_build(relabelled):
    document = json.loads(relabelled.to_str())
    slot = special.SPECIAL_TOKENS.index(tools.TOOL_CALL_OPEN)
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


def test_the_retired_per_tool_markers_are_forbidden_not_merely_unused(relabelled):
    """`<|web-search|>` keeps its slot so that no id moves. Keeping a slot is
    not the same as leaving a marker callable: emitting one would be a call
    nothing in this build knows how to run."""
    rules = Rules.build(relabelled)
    for marker in ("<|web-search|>", "<|/web-search|>"):
        assert relabelled.token_to_id(marker) in rules.forbidden
        assert not OpenBlocks(rules).allows(relabelled.token_to_id(marker))


def test_a_tool_call_may_only_open_outside_every_block(relabelled):
    rules = Rules.build(relabelled)
    call, par = _ids(relabelled, tools.TOOL_CALL_OPEN, "<|par|>")

    blocks = OpenBlocks(rules)
    assert blocks.allows(call)

    blocks.observe(par)
    assert not blocks.allows(call)


def test_a_tool_call_admits_no_markup_inside_it(relabelled):
    """Its content is a JSON object. An opener in there would be sent to a tool."""
    rules = Rules.build(relabelled)
    call, close, par, bold = _ids(
        relabelled, tools.TOOL_CALL_OPEN, tools.TOOL_CALL_CLOSE, "<|par|>", "<|bold|>"
    )

    blocks = OpenBlocks(rules)
    blocks.observe(call)
    assert not blocks.allows(par)
    assert not blocks.allows(bold)
    assert blocks.allows(close)


def test_a_turn_cannot_end_inside_a_tool_call(relabelled):
    rules = Rules.build(relabelled)
    blocks = OpenBlocks(rules)
    blocks.observe(relabelled.token_to_id(tools.TOOL_CALL_OPEN))
    assert not blocks.allows(rules.end_of_assistant)
    assert blocks.closers() == [relabelled.token_to_id(tools.TOOL_CALL_CLOSE)]


def test_a_call_is_recognised_from_its_ids_as_it_closes(relabelled):
    rules = Rules.build(relabelled)
    watch = chat._ToolCalls(rules, relabelled)

    ids = [
        relabelled.token_to_id("<|par|>"),
        *relabelled.encode("the cat").ids,
        relabelled.token_to_id("<|/par|>"),
        *_call_ids(relabelled, _body_of(tools.call("web-search", "the cat sat"))),
    ]

    assert _drive(watch, ids) == [{"tool": "web-search", "argument": "the cat sat"}]
    assert watch.made == [{"tool": "web-search", "argument": "the cat sat"}]


def test_a_command_full_of_quotes_survives_the_round_trip(relabelled):
    """Why a call is JSON rather than a bare string: a shell command carries the
    quotes and backslashes that would otherwise close the object early."""
    rules = Rules.build(relabelled)
    watch = chat._ToolCalls(rules, relabelled)
    command = 'grep -rn "cat \\"the\\" sat" . | head -3 > hits.txt'

    ids = _call_ids(relabelled, _body_of(tools.call("bash", command)))

    assert _drive(watch, ids) == [{"tool": "bash", "argument": command}]


def test_a_body_that_is_not_a_call_costs_the_call_and_not_the_turn(relabelled):
    """Mangled JSON is what a model this size does. It has to read as a sentence
    nobody acted on, never as a crash — the price of one marker for every tool
    is that the grammar cannot catch this and something downstream must."""
    rules = Rules.build(relabelled)
    for body in ('{"tool": "bash", "argument"', "just some words", "[]", '{"argument": "ls"}'):
        watch = chat._ToolCalls(rules, relabelled)
        assert _drive(watch, _call_ids(relabelled, body)) == []
        assert watch.made == []


def test_a_tool_this_build_does_not_have_is_not_a_call(relabelled):
    rules = Rules.build(relabelled)
    watch = chat._ToolCalls(rules, relabelled)

    assert _drive(watch, _call_ids(relabelled, '{"tool": "rm_rf", "argument": "/"}')) == []


def test_the_argument_is_read_under_the_keys_a_model_actually_writes():
    """SFT teaches `argument`. The corpora these models were pretrained on are
    full of `query` and nested `arguments`, and refusing those spends a whole
    round teaching nothing."""
    assert tools.parse('{"tool": "web-search", "argument": "elbrus"}') == ("web-search", "elbrus")
    assert tools.parse('{"tool": "web-search", "query": "elbrus"}') == ("web-search", "elbrus")
    assert tools.parse('{"tool": "bash", "args": {"command": "ls -l"}}') == ("bash", "ls -l")
    assert tools.parse('{"name": "bash", "arguments": {"pattern": "cat"}}') == ("bash", "cat")
    # A name it cannot read is where the forgiveness stops.
    assert tools.parse('{"argument": "elbrus"}') is None
    assert tools.parse('{"tool": "web-search"}') is None


def test_a_turn_stops_being_allowed_to_call_once_it_has_called_enough(relabelled):
    rules = Rules.build(relabelled)
    watch = chat._ToolCalls(rules, relabelled)
    opener = relabelled.token_to_id(tools.TOOL_CALL_OPEN)

    logits = torch.zeros(relabelled.get_vocab_size())
    watch.mask(logits)
    assert logits[opener] == 0.0

    watch.made = [{"tool": "web-search", "argument": "x"}] * chat.MAX_TOOL_CALLS_PER_TURN
    watch.mask(logits)
    assert logits[opener] == float("-inf")


def test_a_tool_result_is_never_a_training_target():
    """The model reads results and writes calls. Training the result would teach
    it to invent search output."""
    call = tools.call("web-search", "elbrus height")
    result = tools.render_result("web-search", "elbrus height", "<|par|>5,642 m<|/par|>")
    turns = [
        Turn(Role.USER, "how tall is elbrus?"),
        Turn(Role.ASSISTANT, call),
        Turn(Role.TOOL, result),
        Turn(Role.ASSISTANT, "<|par|>5,642 m.<|/par|>"),
    ]
    segments = training_segments(None, turns)

    trained = [text for text, target in segments if target]
    assert call in trained
    assert all(tools.TOOL_RESULT_OPEN not in text for text in trained)

    block = next(text for text, _ in segments if text.startswith(tools.TOOL_RESULT_OPEN))
    assert block == f"{tools.TOOL_RESULT_OPEN}{result}{tools.TOOL_RESULT_CLOSE}"
    assert dict(segments)[block] is False


def test_a_tool_result_survives_a_reasoning_strip():
    """STRIP drops thinking. It must not drop the call the next result answers,
    or the conversation stops making sense."""
    call = tools.call("web-search", "elbrus")
    turns = [
        Turn(Role.USER, "how tall is elbrus?"),
        Turn(Role.ASSISTANT, "<|thinking|>look it up<|/thinking|>" + call),
        Turn(Role.TOOL, tools.render_result("web-search", "elbrus", "<|par|>5,642 m<|/par|>")),
        Turn(Role.ASSISTANT, "<|par|>5,642 m.<|/par|>"),
    ]
    rendered = "".join(text for text, _ in training_segments(None, turns, reasoning=Reasoning.STRIP))

    assert "<|thinking|>" not in rendered
    assert call in rendered
    assert tools.TOOL_RESULT_OPEN in rendered


def test_a_new_tool_costs_a_row_and_no_vocabulary_at_all(relabelled):
    """The whole point of one marker for every tool: a vocabulary frozen months
    ago can call something invented today."""
    rules = Rules.build(relabelled)
    watch = chat._ToolCalls(rules, relabelled)
    invented = tools.Tool("weather")
    tools.TOOLS.append(invented)
    try:
        body = '{"tool": "weather", "argument": "oslo"}'
        assert _drive(watch, _call_ids(relabelled, body)) == [
            {"tool": "weather", "argument": "oslo"}
        ]
    finally:
        tools.TOOLS.remove(invented)

    assert all("weather" not in marker for marker in special.SPECIAL_TOKENS)


def test_a_budget_of_zero_refuses_the_opener_before_the_first_token(relabelled):
    """The terminal round of a gym episode: masked rather than instructed,
    because a small model does not reliably follow "stop and answer" — and an
    episode that never answers scores zero for a reason that says nothing about
    whether it found the answer. See
    `docs/decisions/reinforcement-learning-in-the-engine.md`.
    """
    rules = Rules.build(relabelled)
    opener = rules.tool_call[0]

    spent = chat._ToolCalls(rules, relabelled, budget=0)
    logits = torch.zeros(relabelled.get_vocab_size())
    spent.mask(logits)
    assert logits[opener] == float("-inf")

    # The default budget leaves it reachable on a turn that has called nothing.
    fresh = chat._ToolCalls(rules, relabelled)
    logits = torch.zeros(relabelled.get_vocab_size())
    fresh.mask(logits)
    assert logits[opener] == 0.0
