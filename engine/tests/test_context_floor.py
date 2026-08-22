"""`llm.chat.template.context_floor`: where `/prune` would cut a conversation,
without rendering or storing anything."""

from llm.chat.template import Role, Turn, _fit_history, context_floor


def test_nothing_to_prune_when_history_already_fits(tokenizer_module):
    history = [Turn(Role.USER, "hi"), Turn(Role.ASSISTANT, "hello")]
    assert context_floor(history, tokenizer_module, ctx_window=4096, reserve_for_reply=256) == 0


def test_prunes_the_oldest_whole_turns_past_ctx_window(tokenizer_module):
    filler = "the quick brown fox jumps over the lazy dog and then does it again"
    history = [
        Turn(Role.USER if i % 2 == 0 else Role.ASSISTANT, filler) for i in range(20)
    ]

    # Room for roughly a third of the turns, newest first.
    start = context_floor(history, tokenizer_module, ctx_window=300, reserve_for_reply=0)

    assert 0 < start < len(history) - 1
    # Agrees with the walk it wraps once the tool-turn snap has nothing to do —
    # every turn here is user/assistant, so there is nothing to snap past.
    assert start == _fit_history(history, tokenizer_module, 0, 300)


def test_the_latest_turn_always_survives_even_alone(tokenizer_module):
    history = [Turn(Role.USER, "the quick brown fox jumps over the lazy dog" * 10)]
    start = context_floor(history, tokenizer_module, ctx_window=1, reserve_for_reply=0)
    assert start == 0


def test_never_lands_on_an_orphaned_tool_result(tokenizer_module):
    """A naive cut landing exactly on a `Role.TOOL` turn would keep an answer
    with nothing before it naming what it answers — `context_floor` has to move
    past it instead of returning it as the floor."""
    history = [
        Turn(Role.USER, "where did the cat sit"),
        Turn(Role.ASSISTANT, "<|web-search|>where the cat sat<|/web-search|>"),
        Turn(Role.TOOL, "<|web-search|>where the cat sat<|/web-search|><|par|>on the mat<|/par|>"),
        Turn(Role.ASSISTANT, "on the mat"),
    ]

    # Find the allowance that makes the *unsnapped* walk land exactly on the
    # tool turn, so the snap this test is checking actually has something to do.
    naive_at_tool = None
    for allowance in range(1, 200):
        naive = _fit_history(history, tokenizer_module, 0, allowance)
        if history[naive].role is Role.TOOL:
            naive_at_tool = allowance
            break
    assert naive_at_tool is not None, "no allowance in range lands the naive walk on the tool turn"

    start = context_floor(history, tokenizer_module, ctx_window=naive_at_tool, reserve_for_reply=0)
    assert history[start].role is not Role.TOOL
    assert start == history.index(next(t for t in history if t.role is Role.TOOL)) + 1
