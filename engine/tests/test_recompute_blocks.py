"""`Transformer.recompute_blocks`: same gradients, fewer activations.

The flag exists to buy VRAM, and it is only worth having if it buys it for
nothing — so the test that matters is the one that says the gradients did not
move. It runs on CPU in fp32, where a difference would be arithmetic rather than
bf16 rounding. The saving itself is a number only a card can report, so what is
pinned here is the mechanism behind it: each block is run a second time, from
the backward pass.

The other half is what must *not* recompute. A decode step passes caches and an
evaluation runs under `no_grad`, and in both a recomputed forward is either
wrong or pure cost — `forward` refuses on both counts and these pin that.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from llm.model.transformer import Transformer

# Small enough to run a real backward on CPU in a test, deep enough that the
# recomputed path is exercised more than once.
ARCH = dict(vocab=64, dimensions=32, heads=4, layers=3, ff_multiplier=2, ctx_window=32)


def _model(recompute: bool) -> Transformer:
    torch.manual_seed(0)
    model = Transformer(**ARCH)
    model.recompute_blocks = recompute
    model.train()
    return model


def _one_step(model: Transformer) -> tuple[float, dict]:
    torch.manual_seed(1)
    x = torch.randint(0, ARCH["vocab"], (2, 16))
    y = torch.randint(0, ARCH["vocab"], (2, 16))
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, ARCH["vocab"]), y.reshape(-1))
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters()}


def test_recomputing_leaves_the_gradients_alone():
    """The whole justification for the flag: it is a memory trade, not a
    numerical one, so every gradient must survive it."""
    plain_loss, plain = _one_step(_model(False))
    recomputed_loss, recomputed = _one_step(_model(True))

    assert plain_loss == pytest.approx(recomputed_loss, rel=1e-9)
    assert plain.keys() == recomputed.keys()
    for name, grad in plain.items():
        torch.testing.assert_close(grad, recomputed[name], rtol=1e-6, atol=1e-7,
                                   msg=lambda m, n=name: f"{n} moved:\n{m}")


def test_the_backward_pass_runs_each_block_again():
    """The mechanism the memory saving *is*, counted rather than weighed.

    Peak VRAM is the number this flag exists for, but it can only be read on a
    device that reports its own allocations and this suite runs on CPU. What
    holds everywhere is why the memory is saved: a stored block is run once and a
    recomputed one twice — the second time from the backward pass, to rebuild
    what was not kept. Counting the calls tests that directly, and a regression
    that silently stopped recomputing would fail here rather than only on a card.
    """
    def block_calls(recompute: bool) -> int:
        model = _model(recompute)
        calls = 0
        original = type(model.blocks[0]).forward

        def counting(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return original(self, *args, **kwargs)

        with patch.object(type(model.blocks[0]), "forward", counting):
            _one_step(model)
        return calls

    assert block_calls(False) == ARCH["layers"]
    assert block_calls(True) == 2 * ARCH["layers"]


def test_a_decode_step_never_recomputes():
    """Recomputing a cached step would rebuild it without its cache. The flag is
    on; the caches are what must win."""
    model = _model(True)
    x = torch.randint(0, ARCH["vocab"], (1, 4))
    logits, caches = model(x, caches=model.empty_caches())

    assert logits.shape == (1, 4, ARCH["vocab"])
    # A recomputed block returns no cache to speak of; a real one returns the
    # keys and values it just computed, for every layer.
    assert len(caches) == ARCH["layers"]
    for key, value in caches:
        assert key is not None and value is not None
        assert key.shape[2] == 4


def test_the_flag_is_not_part_of_the_architecture():
    """It must not reach `arch` or the state_dict: the same weights train with it
    on or off, and a checkpoint that recorded it would make a memory setting into
    a property of the model."""
    from llm.model import checkpoint

    assert "recompute_blocks" not in checkpoint._ARCH_FIELDS
    assert not any("recompute" in key for key in _model(True).state_dict())


@pytest.mark.parametrize("recompute", [True, False])
def test_evaluation_runs_either_way(recompute: bool):
    """Under `no_grad` there is no backward to save anything for, so the flag is
    inert — and must not throw, which is what a recomputed pass with nothing
    requiring grad would do."""
    model = _model(recompute)
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.randint(0, ARCH["vocab"], (1, 8)))
    assert logits.shape == (1, 8, ARCH["vocab"])
