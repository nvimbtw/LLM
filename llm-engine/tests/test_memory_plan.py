"""`llm.train.memory`: what a step costs, and how little of it to trade away.

The planner exists because this card does not refuse an oversized run — the
driver satisfies it out of host RAM and the run neither fails nor progresses.
So the estimate is what stands between a run and that, and these tests pin the
two properties that make it safe: it is monotone in the depth it is choosing,
and it errs *high* against every configuration that has actually been measured.

The second half is the trade itself. Recomputing every block was the only
setting available before, and at the sizes this card trains it is almost always
far more than the run is short of — the point of a count is to stop paying for
headroom nothing asked for.
"""

import pytest
import torch

from llm.model.transformer import Transformer
from llm.train import memory

# Measured on an RTX 3060 through the real `step_fn` and AdamW, bf16 autocast,
# vocab 8192, ff_multiplier 4. See the module docstring in `llm/train/memory.py`
# for how these were taken.
#
# Compiled and eager are both here because they are not the same number and
# `compile_model` defaults to True — inductor fuses cross-entropy's fp32 copy of
# the logits away, which is ~1 GiB at ctx2048. An estimator calibrated on only
# one of the two is wrong by that much on every run of the other kind.
#
# (dimensions, layers, ctx_window, batch_size, recompute_depth, compiled) -> MiB
MEASURED = [
    (512, 16, 1024, 8, 0, False, 4184),
    (512, 16, 1024, 8, 16, False, 1746),
    (512, 18, 2048, 8, 18, False, 2711),
    (512, 18, 2048, 8, 8, False, 5790),
    (512, 18, 2048, 4, 0, False, 4629),
    (512, 18, 2048, 2, 0, False, 2846),
    (512, 24, 2048, 8, 24, False, 3227),
    # Compiled. The first two are `docs/decisions/activations-are-recomputed.md`'s
    # own published pair; the third was measured for this change.
    (512, 16, 1024, 8, 0, True, 3831),
    (512, 16, 1024, 8, 16, True, 1273),
    (512, 18, 2048, 8, 8, True, 4685),
    (512, 18, 2048, 8, 4, True, 6056),
    (512, 18, 2048, 8, 18, True, 1837),
]


def _params(dimensions: int, layers: int, vocab: int = 8192, ff: int = 4) -> int:
    """Parameter count for this shape — embedding tied, no biases, two norms."""
    d = dimensions
    return vocab * d + layers * (4 * d * d + 2 * ff * d * d + 4 * d) + 2 * d


def _estimate(dimensions, layers, ctx, batch, depth, compiled=False) -> float:
    return memory.step_bytes(
        params=_params(dimensions, layers),
        dimensions=dimensions,
        layers=layers,
        ff_multiplier=4,
        vocab=8192,
        ctx_window=ctx,
        batch_size=batch,
        recompute_depth=depth,
        compiled=compiled,
    ) / 2**20


@pytest.mark.parametrize("dimensions,layers,ctx,batch,depth,compiled,measured",
                         MEASURED)
def test_the_estimate_is_close_and_never_under(
    dimensions, layers, ctx, batch, depth, compiled, measured
):
    """Within 15% of measurement, and on the high side of it.

    Being *over* costs a recomputed block the run did not strictly need. Being
    under is what lets a run spill into host RAM and take hours per step, so the
    two directions are not equally wrong and the bound is one-sided on purpose.
    """
    estimate = _estimate(dimensions, layers, ctx, batch, depth, compiled)
    assert estimate >= measured, (
        f"estimate {estimate:.0f} MiB is UNDER the measured {measured} MiB — the "
        f"direction that lets a run spill. Re-fit the constants."
    )
    assert estimate <= measured * 1.15


def test_recomputing_more_blocks_never_costs_more_memory():
    """Monotone in depth, which is what makes the search in `plan` a search."""
    previous = float("inf")
    for depth in range(19):
        estimate = _estimate(512, 18, 2048, 8, depth)
        assert estimate < previous
        previous = estimate


def test_a_run_that_fits_recomputes_nothing():
    """The whole reason the flag is not simply on: speed is not paid for headroom.

    A small model against a card with room to spare must come back at depth 0 —
    every recomputed block is a second forward pass through it.
    """
    model = Transformer(8192, 256, 4, 6, 4, 512)
    plan = memory.plan(model, batch_size=8, device="cuda", budget=7 * 2**30)
    assert plan.recompute_depth == 0


def test_a_run_that_does_not_fit_recomputes_only_what_it_is_short_of():
    """Model 3.0's shape against this card: some blocks, not all of them.

    The measured peaks are 8200 MiB at depth 0 and 2711 MiB at depth 18 against
    a budget near 6.2 GiB, so the answer is strictly between the two ends — which
    is the setting that did not exist before this module.
    """
    model = Transformer(8192, 512, 8, 18, 4, 2048)
    plan = memory.plan(model, batch_size=8, device="cuda", budget=int(6.2 * 2**30))

    assert 0 < plan.recompute_depth < 18, plan.reason
    assert plan.estimated_bytes <= plan.target_bytes
    # One block shallower would not have fitted, or the search overshot.
    assert _estimate(512, 18, 2048, 8, plan.recompute_depth - 1, compiled=True) > plan.target_bytes / 2**20


def test_an_explicit_setting_wins_over_the_plan():
    """A run that has been measured is not second-guessed by an estimate."""
    model = Transformer(8192, 512, 8, 18, 4, 2048)
    tiny = 512 * 2**20

    assert memory.plan(model, batch_size=8, device="cuda", requested=False,
                       budget=tiny).recompute_depth == 0
    assert memory.plan(model, batch_size=8, device="cuda", requested=True,
                       budget=tiny).recompute_depth == 18
    assert memory.plan(model, batch_size=8, device="cuda", requested=6,
                       budget=tiny).recompute_depth == 6


def test_true_means_every_block_and_one_means_one_block():
    """`True` is not `1`, though Python says it is.

    A stored run config written before this was a count says `true` and means
    every block; a new one saying `1` means a single block. `plan` checks `bool`
    before `int` so those two cannot collapse into each other.
    """
    model = Transformer(8192, 256, 4, 6, 4, 256)
    everything = memory.plan(model, batch_size=8, device="cuda", requested=True,
                             budget=2**20)
    one = memory.plan(model, batch_size=8, device="cuda", requested=1, budget=2**20)

    assert everything.recompute_depth == 6
    assert one.recompute_depth == 1


def test_a_shape_that_cannot_fit_says_so_rather_than_pretending():
    """Nothing here can make a 2048-batch fit, and the plan must not claim it can."""
    model = Transformer(8192, 512, 8, 18, 4, 2048)
    plan = memory.plan(model, batch_size=512, device="cuda", budget=2 * 2**30)

    assert plan.recompute_depth == plan.layers
    assert plan.estimated_bytes > plan.budget_bytes
    assert "batch_size" in plan.reason


def test_no_device_budget_stores_everything():
    """CPU, or a card that will not say. The LLM engine's behaviour before planning."""
    model = Transformer(8192, 256, 4, 6, 4, 256)
    assert memory.plan(model, batch_size=8, device="cpu").recompute_depth == 0


@pytest.mark.parametrize("reserved_gib", [0, 0.5, 2])
def test_loading_weights_or_caching_blocks_does_not_shrink_the_cap(monkeypatch, reserved_gib):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (int((7 - reserved_gib) * 2**30), 8 * 2**30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: int(reserved_gib * 2**30))
    assert memory.budget_bytes("cuda") == int(7 * 2**30 * memory.MARGIN_FRACTION)


def test_reference_weights_count_toward_the_same_cap():
    model = Transformer(8192, 512, 8, 18, 4, 2048)
    args = dict(batch_size=8, device="cuda", budget=int(6.2 * 2**30))
    plain = memory.plan(model, **args)
    resident = 700 * 2**20
    anchored = memory.plan(model, **args, resident_bytes=resident)
    assert anchored.recompute_depth > plain.recompute_depth
    assert anchored.estimated_bytes <= anchored.target_bytes
    assert anchored.budget_bytes == plain.budget_bytes


def test_near_limit_plan_leaves_usable_allocator_headroom():
    model = Transformer(8192, 512, 8, 18, 4, 2048)
    plan = memory.plan(model, batch_size=8, device="cuda", budget=int(5.59 * 2**30))
    assert 0 < plan.recompute_depth < plan.layers
    assert plan.budget_bytes - plan.estimated_bytes >= 512 * 2**20
