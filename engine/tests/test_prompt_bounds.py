"""`build_prompt` never returns more tokens than the model can be given.

Dropping whole turns cannot save a conversation whose *latest* turn alone busts
the window — that turn is the one that must be kept — so the docstring's
promise of "a hard-cut latest message" is the only thing standing between a
small model and `Transformer.forward` raising
`position N exceeds context window M`. It went unkept until a 64-token model was
asked a two-line question.
"""

from llm.chat.template import ASSISTANT_OPEN, Role, Turn, build_prompt
from llm.corpus import bpe

LONG = "the quick brown fox jumps over the lazy dog and then does it again " * 12


def test_one_oversized_turn_is_cut_to_the_allowance(tokenizer_module):
    ids = build_prompt(None, [Turn(Role.USER, LONG)], tokenizer_module,
                       ctx_window=64, reserve_for_reply=32)
    assert len(ids) <= 32


def test_a_prompt_that_already_fits_is_untouched(tokenizer_module):
    history = [Turn(Role.USER, "hi")]
    ids = build_prompt("be brief", history, tokenizer_module,
                       ctx_window=4096, reserve_for_reply=256)
    assert ids == build_prompt("be brief", history, tokenizer_module,
                               ctx_window=4096, reserve_for_reply=256)
    assert bpe.decode(tokenizer_module, ids).endswith(ASSISTANT_OPEN)


def test_the_cut_keeps_the_marker_that_asks_the_model_to_speak(tokenizer_module):
    """A prompt missing its trailing `<|assistant|>` is not a short prompt, it
    is a different one: nothing has asked the model to reply."""
    ids = build_prompt(None, [Turn(Role.USER, LONG)], tokenizer_module,
                       ctx_window=64, reserve_for_reply=32)
    assert bpe.decode(tokenizer_module, ids).endswith(ASSISTANT_OPEN)


def test_the_cut_keeps_the_freshest_end_of_the_turn(tokenizer_module):
    """The end is nearest the question, and it is the direction every other
    eviction here already runs."""
    text = "alpha " * 200 + "and finally omega"
    ids = build_prompt(None, [Turn(Role.USER, text)], tokenizer_module,
                       ctx_window=128, reserve_for_reply=64)
    rendered = bpe.decode(tokenizer_module, ids)
    assert "omega" in rendered


def test_a_system_block_alone_larger_than_the_window_still_fits(tokenizer_module):
    """The degenerate end of the same problem: nothing is left for the body, so
    the scaffold itself is what gets cut — tail first kept, head trimmed."""
    ids = build_prompt(LONG, [Turn(Role.USER, "hi")], tokenizer_module,
                       ctx_window=32, reserve_for_reply=16)
    assert len(ids) <= 16
    assert bpe.decode(tokenizer_module, ids).endswith(ASSISTANT_OPEN)
