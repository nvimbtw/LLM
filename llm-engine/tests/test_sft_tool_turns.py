"""A call in SFT data: which side renders it, and which turn it is stored on.

`chat.tools.render_result` calls itself "the single renderer for both the live
path and the SFT data", and this file is where that stops being a comment.

**Which turn a call is stored on decides whether tool use is trained at all.**
`training_segments` trains every assistant turn whole and masks every tool turn,
so a command stored only on the tool turn answering it carries no gradient. A
demonstration in that shape trains a model to narrate a search and then close
its turn without reaching for the shell — which is not a subtle degradation, it
is the opposite of the behaviour the gym exists to clone. So an assistant turn
names the call it *makes*, with the marker after its text where the model emits
it, and the tool turn names the call it *answers*, with the marker opening the
result. The same call, rendered into two positions, one trained and one not.

A `tool` turn reaches the corpus builder in one of two shapes and they must not
be confused:

- **Naming its call** (`tool` + `argument`) — the body alone, rendered here.
  This is the shape a gym demonstration lands in, because the observation body
  comes from `tools::observation` in Rust and the `<|tool-call|>` header should
  not be a second Rust spelling of `tools.call`.
- **Naming no call** — `text` is already the whole rendered block. This is how
  every converted corpus arrives, `hermes-agent-reasoning-traces-clean`
  included: all 138,110 of its tool turns are pre-rendered by the data engine's
  `processors/hermes.py`, which cannot call into Rust or into this package.

Rendering the second shape again would nest one block inside another, and
leaving the first unrendered would train the model on a result whose call it
cannot see. Both are invisible until a fine-tuned model meets the real thing.
"""

from llm.chat import tools
from llm.chat.template import Role
from llm.config import ChatTurn
from llm.corpus.sft import _conversation, _turns
from llm.orchestrate.chat import _history

COMMAND = "grep -rn ANSWER_2ee433 ."
BODY = "<|codeblock|>notes.txt:3:ANSWER_2ee433<|/codeblock|>"


def messages(tool_message: dict) -> list[dict]:
    """A minimal trainable conversation around one tool turn."""
    return [
        {"role": "user", "content": "Which file contains ANSWER_2ee433?"},
        {"role": "assistant", "content": "<|par|>Looking.<|/par|>"},
        tool_message,
        {"role": "assistant", "content": "notes.txt"},
    ]


def named() -> dict:
    return {"role": "tool", "tool": "bash", "argument": COMMAND, "content": BODY}


def test_a_tool_turn_that_names_its_call_is_rendered_into_the_block():
    turns = _turns(_conversation(messages(named())))
    assert turns is not None
    result = next(turn for turn in turns if turn.role is Role.TOOL)
    assert result.text == tools.render_result("bash", COMMAND, BODY)
    assert result.text.startswith("<|tool-call|>")


def test_the_sft_render_is_the_one_the_chat_path_produces():
    """The whole point, stated against the other renderer rather than against a
    literal: `orchestrate.chat._history` is what serving does with the same turn,
    and a corpus built any other way teaches a layout serving never shows."""
    served = _history([ChatTurn(role="tool", text=BODY, tool="bash", argument=COMMAND)])
    trained = _turns(_conversation([named(), {"role": "assistant", "content": "notes.txt"}]))
    assert trained is not None
    assert trained[0].text == served[0].text


def test_an_already_rendered_tool_turn_is_left_alone():
    """The Hermes shape. Wrapping it again would nest a `<|tool-call|>` block
    inside another one, which is a corpus nothing would notice was wrong."""
    rendered = tools.render_result("bash", COMMAND, BODY)
    turns = _turns(_conversation(messages({"role": "tool", "content": rendered})))
    assert turns is not None
    result = next(turn for turn in turns if turn.role is Role.TOOL)
    assert result.text == rendered
    assert result.text.count("<|tool-call|>") == 1


def test_a_turn_naming_only_half_a_call_is_treated_as_already_rendered():
    """`ChatTurn` refuses this pairing outright; a corpus row cannot raise at a
    caller, so the safe reading is the one that changes no bytes."""
    half = {"role": "tool", "tool": "bash", "content": BODY}
    turns = _turns(_conversation(messages(half)))
    assert turns is not None
    assert next(turn for turn in turns if turn.role is Role.TOOL).text == BODY


def test_an_unknown_tool_drops_the_conversation_rather_than_the_build():
    """The chat path raises here, because a live caller can be told it was
    wrong. A build reading hundreds of thousands of rows cannot: one malformed
    row must cost that row, the bargain `read_conversations` already strikes
    with a document carrying no messages."""
    unknown = {"role": "tool", "tool": "patch", "argument": "a.diff", "content": BODY}
    assert _turns(_conversation(messages(unknown))) is None


# ── the whole path, not just the turn ────────────────────────────────────────


def test_a_demonstration_builds_a_corpus_that_trains_the_answer(tokenizer, data_client):
    """The end the whole chain exists for.

    A gym demonstration is stored unrendered by the data engine, rendered here at
    build time, and the mask has to land on the *answer* — the last assistant
    turn — and not on the shell's output. Cloning teaches the shape of ending on
    an answer, and a mask that covered the tool result would teach the model to
    predict what the shell says instead.
    """
    import numpy as np

    from llm.chat.template import Reasoning
    from llm.corpus import sft as sft_corpus
    from tests.fixtures.tiny_env import TINY_GYM_ID

    selection = sft_corpus.resolve(data_client, [TINY_GYM_ID])
    stats = sft_corpus.build(data_client, tokenizer, selection, Reasoning.KEEP)

    assert stats.conversations == 24, "every demonstration should be trainable"
    assert stats.skipped == 0
    assert 0 < stats.trainable_tokens < stats.total_tokens

    corpus, mask = sft_corpus.load(sft_corpus.dataset_key(selection, Reasoning.KEEP))
    text = tokenizer.decode(
        [int(t) for t in np.asarray(corpus)[: min(len(corpus), 4000)]],
        skip_special_tokens=False,
    )
    # The call header is in the corpus, which is only true because `_turns`
    # rendered it: the data engine stored the body alone.
    assert "<|tool-call|>" in text


def _trained_text(tokenizer, data_client) -> str:
    """Only the tokens the mask actually trains on, decoded."""
    import numpy as np

    from llm.chat.template import Reasoning
    from llm.corpus import sft as sft_corpus
    from tests.fixtures.tiny_env import TINY_GYM_ID

    selection = sft_corpus.resolve(data_client, [TINY_GYM_ID])
    sft_corpus.build(data_client, tokenizer, selection, Reasoning.KEEP)
    corpus, mask = sft_corpus.load(sft_corpus.dataset_key(selection, Reasoning.KEEP))
    return tokenizer.decode(
        [int(t) for t, m in zip(np.asarray(corpus), np.asarray(mask)) if m],
        skip_special_tokens=False,
    )


def test_the_shells_answer_is_never_trained_on(tokenizer, data_client):
    """A tool result is context, not a target. It is also the one span in a gym
    demonstration a model could learn to hallucinate — it is the part that comes
    from outside — so this is the mask assertion that matters most here."""
    assert "<|codeblock|>" not in _trained_text(tokenizer, data_client)


def test_the_command_the_model_types_is_trained_on(tokenizer, data_client):
    """The other half, and the one this file used to assert the opposite of.

    A demonstration exists to teach a model to reach for the shell. SFT trains
    assistant turns and masks tool turns, so that only happens if the call is
    stored on the assistant turn that makes it. It used to be stored on the tool
    turn alone — wholly inside the masked region — and this test asserted
    `"<|tool-call|>" not in trained`, which read as a mask working correctly and
    was really the bug written down: the trained sequences were the narration
    and then `<|/assistant|>`, teaching a model to talk about searching and then
    end its turn without ever calling anything.

    Both halves have to hold at once. The call is trained; the output it
    returned is not.
    """
    trained = _trained_text(tokenizer, data_client)

    assert "<|tool-call|>" in trained
    assert "grep -rn ANSWER_000000 ." in trained
    assert "<|codeblock|>" not in trained


# ── the assistant turn that makes a call ─────────────────────────────────────


def calling() -> dict:
    """An assistant turn naming the call it makes, as the backend stores it."""
    return {
        "role": "assistant",
        "content": "<|par|>Looking.<|/par|>",
        "tool": "bash",
        "argument": COMMAND,
    }


def test_an_assistant_turn_that_names_a_call_renders_the_marker_after_its_text():
    """Where the model emits it: the prose, then the call, then the turn ends.

    Not `render_result`, which is the shape for a turn *answering* a call —
    using it here would wrap the model's own command in the block that replies
    to it, and the marker would land in the masked region either way.
    """
    turns = _turns(_conversation([calling(), {"role": "assistant", "content": "notes.txt"}]))
    assert turns is not None
    assert turns[0].text == f"<|par|>Looking.<|/par|>{tools.call('bash', COMMAND)}"
    assert not turns[0].text.startswith("<|tool-call|>")
    assert "<|tool-result|>" not in turns[0].text


def test_the_live_path_cannot_receive_a_calling_assistant_turn_at_all():
    """Why `_history` has no role branch where `_turns` does.

    A corpus row is a dict and nothing refuses it, so `_turns` has to decide by
    role. A `ChatTurn` is validated, and it refuses this pairing outright — so
    the shape is unreachable on the live path rather than handled there, and a
    defensive branch in `_history` would be code no caller can run.
    """
    import pytest

    with pytest.raises(ValueError, match="only a tool turn"):
        ChatTurn(role="assistant", text="x", tool="bash", argument=COMMAND)


def test_an_assistant_call_lands_in_the_trained_region():
    """The mask, asserted directly on the segments rather than through a build.

    `training_segments` is the thing that decides this, and it decides it by
    role: everything inside an assistant turn is a target, everything inside a
    tool turn is not. So *which turn the call is stored on* is the whole
    question, and this is it in three lines.
    """
    from llm.chat.template import Reasoning, training_segments

    conversation = [
        {"role": "user", "content": "Which file contains ANSWER_2ee433?"},
        calling(),
        named(),
        {"role": "assistant", "content": "notes.txt"},
    ]
    turns = _turns(_conversation(conversation))
    assert turns is not None
    trained = "".join(
        text for text, is_target in training_segments(None, turns, None, Reasoning.KEEP) if is_target
    )
    assert tools.call("bash", COMMAND) in trained
    assert BODY not in trained


def test_an_assistant_turn_naming_an_unknown_tool_drops_the_conversation():
    unknown = {"role": "assistant", "content": "x", "tool": "patch", "argument": "a.diff"}
    assert _turns(_conversation([unknown, {"role": "assistant", "content": "notes.txt"}])) is None
