"""A batch of rows at different positions decodes to what each row decodes alone.

That equivalence is the whole correctness claim of `positions` and `key_mask` on
`Transformer.forward`. A rollout batch is rows part-way through their own
episodes, padded to a common length; if padding leaks — into RoPE, into the keys
a real query attends — the batch still produces plausible tokens, just not the
ones the model would have produced. Nothing downstream can notice that, so it
has to be pinned here.
"""

import pytest
import torch

from llm.model import sample
from llm.model.transformer import Transformer

VOCAB = 64
SEED = 20260904

#: Deliberately uneven, and one row far shorter than the rest: the padding a
#: short row carries is where a leak would show.
PROMPTS = [
    [3, 9, 27, 4, 11, 2, 8],
    [5, 1],
    [7, 7, 7, 12, 30],
    [2, 19, 6, 41, 5, 8],
]
STEPS = 6


@pytest.fixture
def model():
    torch.manual_seed(SEED)
    built = Transformer(
        vocab=VOCAB, dimensions=32, heads=4, layers=2, ff_multiplier=2, ctx_window=64
    )
    return built.eval()


def _alone(model, prompt: list[int]) -> list[int]:
    """Greedy decode of one sequence through the path serving already uses."""
    ids = list(prompt)
    logits, caches = model(torch.tensor([ids]), caches=model.empty_caches(), pos_offset=0)
    drawn = []
    for _ in range(STEPS):
        nxt = int(logits[0, -1].argmax())
        drawn.append(nxt)
        logits, caches = model(
            torch.tensor([[nxt]]), caches=caches, pos_offset=len(ids) + len(drawn) - 1
        )
    return drawn


def _lockstep(model, prompts: list[list[int]]) -> list[list[int]]:
    """Greedy decode of every sequence at once, left-padded to a common length."""
    width = max(len(p) for p in prompts)
    ids = torch.zeros(len(prompts), width, dtype=torch.long)
    live = torch.zeros(len(prompts), width, dtype=torch.bool)
    positions = torch.zeros(len(prompts), width, dtype=torch.long)
    for row, prompt in enumerate(prompts):
        pad = width - len(prompt)
        ids[row, pad:] = torch.tensor(prompt)
        live[row, pad:] = True
        positions[row, pad:] = torch.arange(len(prompt))

    drawn = [[] for _ in prompts]
    for _ in range(STEPS):
        logits, _ = model(ids, positions=positions, key_mask=live)
        nxt = sample.pick_rows(logits[:, -1], temperature=0.0)
        for row, token in enumerate(nxt.tolist()):
            drawn[row].append(token)
        ids = torch.cat((ids, nxt[:, None]), dim=1)
        live = torch.cat((live, torch.ones(len(prompts), 1, dtype=torch.bool)), dim=1)
        positions = torch.cat((positions, positions[:, -1:] + 1), dim=1)
    return drawn


def test_lockstep_matches_one_at_a_time(model):
    assert _lockstep(model, PROMPTS) == [_alone(model, p) for p in PROMPTS]


def test_padding_cannot_reach_a_real_query(model):
    """The same row decodes the same whatever it is batched beside.

    The sharper half of the claim above: `test_lockstep_matches_one_at_a_time`
    would still pass if every row leaked identically.
    """
    alone = _lockstep(model, [PROMPTS[1]])[0]
    beside = _lockstep(model, PROMPTS)[1]
    assert alone == beside


def test_a_fully_padded_query_stays_finite(model):
    """A row masked everywhere softmaxes to NaN, and the NaN would reach every
    later query through the cache. The self-attention exemption is what stops it,
    and nothing else in the batch would show its absence."""
    ids = torch.tensor([[0, 0, 5, 1]])
    live = torch.tensor([[False, False, True, True]])
    positions = torch.tensor([[0, 0, 0, 1]])
    logits, _ = model(ids, positions=positions, key_mask=live)
    assert torch.isfinite(logits[:, -1]).all()


def test_pick_rows_matches_pick_row_by_row():
    torch.manual_seed(SEED)
    logits = torch.randn(5, VOCAB)
    for temperature, top_k, top_p in ((0.0, 0, 1.0), (0.0, 8, 0.9)):
        rows = sample.pick_rows(logits, temperature, top_k, top_p).tolist()
        assert rows == [sample.pick(row, temperature, top_k, top_p) for row in logits]


def test_pick_rows_top_p_keeps_the_crossing_token():
    """A top_p below the top token's own probability must still keep it — the
    off-by-one `pick` shifts the cumsum for, checked on the batched twin."""
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    assert sample.pick_rows(logits, temperature=1.0, top_p=0.1).tolist() == [0]
