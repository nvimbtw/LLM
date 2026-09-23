"""What a training step will cost in VRAM, and how much of it to trade away.

This module exists because the card has two limits and only one of them is
enforced. Torch's allocator refuses an allocation past the fraction it is given;
the *driver* on this machine does not refuse one past the physical card. Asked
for 11 GiB on an 8 GiB RTX 3060 it hands back 11 GiB, satisfying the overflow
out of host RAM across the PCIe bus. Nothing fails, nothing warns, and the run
neither crashes nor progresses — it simply runs every step at bus speed instead
of memory speed. Measured on that card: the eighth GiB took 50x longer to
allocate than the seventh, and a 512d/18L/ctx2048/bs8 step never completed at
all while `nvidia-smi` read a comfortable 100% utilization.

So a run cannot be allowed to find out whether it fits by trying. Two things
follow, and this module is both of them:

- **Decide the trade before the first step.** `plan` estimates the step's peak
  and picks the smallest number of blocks to recompute that brings it under
  budget — see `Transformer.recompute_depth`. Recomputing *every* block is the
  setting that was available before and it is almost always too much: at
  512d/18L/ctx2048/bs8 it takes the peak to 2711 MiB of a ~6300 MiB budget and
  charges 1430 ms a step, where the 8 blocks that actually fit cost 1292 ms.
- **Make a wrong estimate loud.** `guard` caps the allocator at the same budget,
  so a step that would have spilled raises `OutOfMemoryError` in seconds instead
  of running the whole run at bus speed. An estimate that is wrong the safe way
  costs a little speed; one that is wrong the other way used to cost the run.

## Where the numbers come from

The estimate is fitted, not derived, and biased high on purpose — see
`ESTIMATE_SAFETY`. Measured on an RTX 3060 through the real `step_fn` and AdamW,
eager, bf16 autocast:

```
eager
512d 8h 16L ctx1024 bs8   depth  0    4184 MiB   depth 16    1746 MiB
512d 8h 18L ctx2048 bs8   depth  8    5790 MiB   depth 18    2711 MiB
512d 8h 18L ctx2048 bs4   depth  0    4629 MiB
512d 8h 18L ctx2048 bs2   depth  0    2846 MiB
512d 8h 24L ctx2048 bs8   depth  0    driver fell over
                                      depth 24    3227 MiB
compiled
512d 8h 16L ctx1024 bs8   depth  0    3831 MiB   depth 16    1273 MiB
512d 8h 18L ctx2048 bs8   depth  4    6056 MiB   depth  8    4685 MiB
                                      depth 18    1837 MiB
```

`512d/18L/ctx2048/bs8 depth 0` is absent from both because it is the run that
started all this: it does not fit, and on this card that does not raise — it
spills and never finishes. The fitted model puts it at ~8600 MiB eager, which
is consistent with the card sitting at 7984 MiB making no progress.

Two coefficients carry it. `ACT_BYTES_PER_TOKEN_PER_DIM` is what one block holds
for the backward pass, per token, per unit of model width: the 16L/ctx1024 pair
above differ by 2438 MiB over 16 blocks and 8192 tokens, which is 38.1 bytes, or
19.05 bf16 elements of width `d`. That matches what the block actually stores —
its input, two normalized copies, q/k/v before and after RoPE, the attention and
projection outputs, and the `ff_multiplier`-wide pair either side of the GELU —
so the constant is scaled by `ff_multiplier` rather than pinned to 4.

Everything that does *not* scale with depth is `_fixed_bytes`: the parameters
with their gradients and AdamW's two moments, the block inputs a recomputed
stack still keeps, one block's worth of rebuild, and the logits.

With `ESTIMATE_SAFETY` applied the predictions land 1-6% *above* every row
above, which is the direction that costs a recomputed block rather than the run.
The driver margin covers memory outside the allocator. A separate planning
target below the allocator cap leaves room for cached fragments and temporary
tensors; that headroom has to be inside the cap to be usable.

**Re-fit these against measurement rather than reasoning about them.** Every
frame carries the peak the run actually reached (`Progress.peak_mib`), so the
error is visible on a real run and not only in a benchmark.
"""

from dataclasses import dataclass

import torch

#: Bytes one block holds for the backward pass, per token, per unit of width, at
#: `ff_multiplier` 4 — see the module docstring for the measurement. Scaled by
#: the run's own multiplier in `_act_bytes_per_block`, since the pair of tensors
#: either side of the GELU is `ff_multiplier`-wide and is most of this number.
ACT_BYTES_PER_TOKEN_PER_DIM = 38.1
_FITTED_FF_MULTIPLIER = 4

#: Bytes per parameter held between steps: an fp32 master weight, an fp32
#: gradient, and AdamW's two fp32 moments, plus the bf16 copy autocast makes of
#: every weight it touches. 16 of the 18 are the first four.
PARAM_BYTES = 18

#: Bytes per logit, eager and compiled. The head is [batch x ctx, vocab] in bf16
#: and cross-entropy is on autocast's fp32 list, so eager materializes an fp32
#: copy beside it and autograd holds a gradient the same size; compiled,
#: inductor fuses the cast into the softmax's reduction and neither ever exists
#: — which is the whole of what `loop.step_fn` is for.
#:
#: **This is where nearly all of compile's memory saving lives**, and it is worth
#: knowing because `compile_model` defaults to True, so it is the common case.
#: Two independent configs agree on ~7.4-8.4 bytes a logit saved:
#:
#:   512d/16L/ctx1024/bs8   67.1M logits   1746 -> 1273 MiB   (7.4 B/logit)
#:   512d/18L/ctx2048/bs8  134.2M logits   5758 -> 4685 MiB   (8.4 B/logit)
#:
#: Per-block activations barely move — 152.4 MiB eager against 159.9 compiled at
#: the first of those — so compile is modelled here and nowhere else. Estimating
#: a compiled run at the eager cost over-recomputes it by ~29%, which is safe and
#: wasteful; it also lowers the largest model the planner believes will fit.
LOGIT_BYTES = 8
LOGIT_BYTES_COMPILED = 1

#: Applied to the finished estimate, because the fit above lands *under*
#: measurement on all six rows — between 0.4% and 4.5%, worst at the deepest
#: model, which is a per-layer term the two coefficients do not quite capture.
#: An under-estimate is the expensive direction (it is what lets a run spill),
#: so the residual is paid back here rather than hidden by widening `MARGIN`:
#: this way the estimate is honest about a step's cost on its own, and the
#: margin stays what it says it is.
ESTIMATE_SAFETY = 1.06

#: How much of the available card to leave outside PyTorch: the CUDA context
#: and external workspaces, which `step_bytes` does not model. The
#: fit's own residual is not in here; `ESTIMATE_SAFETY` pays that. A fraction
#: *and* a floor, because on a card this size the fixed costs do not scale with
#: what happens to be free.
MARGIN_FRACTION = 0.88
MARGIN_FLOOR_BYTES = 384 * 2**20

# Space inside the allocator cap, where cached fragments and temporary tensors
# actually compete with the estimate. The driver margin above cannot pay this.
ALLOCATOR_HEADROOM_FRACTION = 0.10
ALLOCATOR_HEADROOM_BYTES = 512 * 2**20


@dataclass(frozen=True)
class Plan:
    """What a run decided to spend, and why. Recorded in the run's provenance."""

    recompute_depth: int
    layers: int
    estimated_bytes: int
    budget_bytes: int
    #: One line for the run log and the model row — a person reading a slow run
    #: should not have to infer the trade from a number.
    reason: str

    @property
    def estimated_mib(self) -> float:
        return self.estimated_bytes / 2**20

    @property
    def budget_mib(self) -> float:
        return self.budget_bytes / 2**20

    @property
    def target_bytes(self) -> int:
        return planning_target(self.budget_bytes)

    def as_dict(self) -> dict:
        return {
            "recompute_depth": self.recompute_depth,
            "layers": self.layers,
            "estimated_mib": round(self.estimated_mib, 1),
            "budget_mib": round(self.budget_mib, 1),
            "target_mib": round(self.target_bytes / 2**20, 1),
            "reason": self.reason,
        }


def budget_bytes(device: str) -> int:
    """What this card can actually honour right now, less a margin.

    The cap covers the whole allocator, including already-resident weights and
    reusable cached blocks. Counting only driver-free memory charges SFT for its
    loaded base weights twice and makes the answer depend on cache history.
    Other processes and allocations outside PyTorch remain excluded.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info()
    available = free + torch.cuda.memory_reserved()
    return max(0, min(int(available * MARGIN_FRACTION), available - MARGIN_FLOOR_BYTES))


def planning_target(budget: int) -> int:
    headroom = max(ALLOCATOR_HEADROOM_BYTES, int(budget * ALLOCATOR_HEADROOM_FRACTION))
    return max(0, budget - headroom)


def other_allocated_bytes(model, device: str) -> int:
    """Live allocator memory not represented by the training parameter estimate.

    In particular, an optional SFT reference model must fit under the same cap.
    The training weights themselves may already be on CUDA or still on CPU.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return 0
    weights = sum(
        p.numel() * p.element_size() for p in model.parameters()
        if p.device.type == "cuda"
    )
    return max(0, torch.cuda.memory_allocated() - weights)


def _act_bytes_per_block(dimensions: int, ff_multiplier: int, tokens: int) -> float:
    """What one *stored* block costs for the backward pass."""
    # 12 of the fitted 19.05 elements are the ff_multiplier-independent ones (the
    # block input, two norms, q/k/v twice, the attention and projection outputs);
    # the rest is the pair either side of the GELU, which is what scales.
    fixed, scaling = 12.0, ACT_BYTES_PER_TOKEN_PER_DIM / 2 - 12.0
    elements = fixed + scaling * (ff_multiplier / _FITTED_FF_MULTIPLIER)
    return elements * 2 * dimensions * tokens


def _fixed_bytes(
    params: int, dimensions: int, layers: int, ff_multiplier: int, vocab: int,
    tokens: int, compiled: bool,
) -> float:
    """What a step costs with every block recomputed — the floor the trade has.

    Nothing here can be traded away by recomputing: the optimizer state, the one
    block input a recomputed block still keeps, the single block rebuilt at a
    time during the backward, and the logits.
    """
    optimizer = PARAM_BYTES * params
    stored_inputs = layers * tokens * dimensions * 2
    one_rebuild = _act_bytes_per_block(dimensions, ff_multiplier, tokens)
    logits = tokens * vocab * (LOGIT_BYTES_COMPILED if compiled else LOGIT_BYTES)
    return optimizer + stored_inputs + one_rebuild + logits


def step_bytes(
    *,
    params: int,
    dimensions: int,
    layers: int,
    ff_multiplier: int,
    vocab: int,
    ctx_window: int,
    batch_size: int,
    recompute_depth: int,
    compiled: bool = True,
) -> int:
    """Estimated peak VRAM for one training step at this shape and this trade.

    `compiled` defaults to True because `TrainConfig.compile_model` does, and
    getting it wrong is worth ~1 GiB at ctx2048 — see `LOGIT_BYTES`.
    """
    tokens = ctx_window * batch_size
    stored = max(0, layers - recompute_depth)
    total = _fixed_bytes(params, dimensions, layers, ff_multiplier, vocab, tokens,
                         compiled)
    total += stored * _act_bytes_per_block(dimensions, ff_multiplier, tokens)
    return int(total * ESTIMATE_SAFETY)


def plan(
    model,
    *,
    batch_size: int,
    device: str,
    requested: bool | int | None = None,
    budget: int | None = None,
    compiled: bool = True,
    resident_bytes: int = 0,
) -> Plan:
    """How many of `model`'s blocks should recompute for this run to fit.

    `requested` is the run's own say, and it wins: `True` recomputes every block
    and `False` none of them, which are the two settings that existed before this
    module and are still the right ones for somebody who has measured their own
    run. An `int` names the count directly. `None` — the default — is what asks
    for a plan, and is the only value that reads the card.

    Choose the smallest depth that fits below the planning target, leaving
    allocator headroom without paying for full recomputation unnecessarily.
    `resident_bytes` includes other live tensors, such as SFT's reference model;
    both the reported estimate and the cap describe the whole allocator.
    """
    layers = len(model.blocks)
    shape = dict(
        params=sum(p.numel() for p in model.parameters()),
        dimensions=model.dimensions,
        layers=layers,
        ff_multiplier=model.ff_multiplier,
        vocab=model.vocab,
        ctx_window=model.ctx_window,
        batch_size=batch_size,
        compiled=compiled,
    )

    cap = budget_bytes(device) if budget is None else budget
    target = planning_target(cap)

    def estimated(depth: int) -> int:
        return resident_bytes + step_bytes(**shape, recompute_depth=depth)

    if requested is not None:
        # `bool` before `int`, because in Python every bool *is* an int and
        # `True` here means "all of them", not "one of them".
        if isinstance(requested, bool):
            depth = layers if requested else 0
            asked = "every block" if requested else "no block"
        else:
            depth = max(0, min(int(requested), layers))
            asked = f"{depth} of {layers} blocks"
        return Plan(depth, layers, estimated(depth), cap,
                    f"asked for: {asked} recomputed")

    if cap <= 0:
        # No card to plan against — CPU, or a device that will not say. Storing
        # everything is what this engine did before there was a planner.
        return Plan(0, layers, estimated(0), 0,
                    "no device budget to plan against")

    for depth in range(layers + 1):
        estimate = estimated(depth)
        if estimate <= target:
            if depth == 0:
                why = f"fits in {target / 2**20:.0f} MiB with nothing recomputed"
            else:
                why = (
                    f"{depth} of {layers} blocks recomputed to fit "
                    f"{estimate / 2**20:.0f} MiB into {target / 2**20:.0f} MiB"
                )
            why += f"; allocator cap {cap / 2**20:.0f} MiB"
            return Plan(depth, layers, estimate, cap, why)

    # Even recomputing everything does not fit. Say so plainly and let the run
    # start anyway: the estimate is fitted and could be pessimistic, and `guard`
    # turns a genuine overflow into an immediate OOM rather than a silent spill.
    estimate = estimated(layers)
    return Plan(
        layers, layers, estimate, cap,
        f"every block recomputed and still {estimate / 2**20:.0f} MiB against a "
        f"{target / 2**20:.0f} MiB target (cap {cap / 2**20:.0f} MiB) "
        "— lower batch_size or ctx_window",
    )


def guard(device: str, budget: int) -> None:
    """Cap the allocator at `budget`, so overflowing the card fails immediately.

    This is the half of the module that does not depend on the estimate being
    right. Without it the driver satisfies an oversized run out of host RAM and
    the run reads as healthy — full card, 100% utilization — while taking hours
    per step. With it, the same run raises `OutOfMemoryError` on the step that
    would have spilled, which is a thing a person can act on.

    A fraction of the *total* card, because that is the only unit
    `set_per_process_memory_fraction` accepts; `budget` includes both driver-free
    memory and the allocator's existing reservations, less the driver margin.

    **The cap bounds the allocator, not the process.** The CUDA context, cuDNN's
    workspaces and inductor's live outside it, so a cap set at the physical card
    still lets the total overrun and spill — measured: an allocator capped at
    0.86 of total sat at 7951 MiB of an 8192 MiB card and crawled, because the
    cap was honoured and the context was not in it. That is what the margin in
    `budget_bytes` is for. Driver-free plus allocator-reserved memory excludes
    existing external allocations, unlike total device capacity.
    """
    if device != "cuda" or not torch.cuda.is_available() or budget <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total))
