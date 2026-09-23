# A Run Plans Its Own Memory, Because This Card Will Not Refuse One

- **Date**: 2026-09-08
- **Status**: Implemented

## Context

[`activations-are-recomputed.md`](activations-are-recomputed.md), earlier the
same day, gave a run a way to trade time for VRAM and left it off by default:
"a run that already fits should not pay for headroom it is not using." The next
run started after it was 512d / 8 heads / 18 layers / ctx 2048 — 60.9M
parameters — and it did not fit, did not fail, and did not train. Eleven minutes
in, the card read 7.7 of 8 GB at 100% utilization and no frame had reached the
frontend.

Three separate things were wrong, and each one hid the next.

### 1. The flag was unreachable

`recompute_blocks` was added to `TrainConfig` and `SftConfig` and to nothing
else. The Rust backend forwards `/runs/train` as an untyped `serde_json::Value`,
so it was never the obstacle — but the frontend builds the request body from a
fixed field list (`frontend/lib/training.ts`, `components/training/run-config.ts`)
and that list has no `recompute_blocks` in it, and no `batch_size` either. Every
run started from the UI therefore took `recompute_blocks=False` and
`batch_size=8`, which are exactly the two settings that decide whether a large
model fits.

The lever shipped, was measured, was documented, and could not be pulled.

### 2. The card does not refuse an oversized run

This is the part that turned a mistake into a mystery. Asked for 11 GiB on an
8 GiB RTX 3060 under WSL2, the driver returns 11 GiB — satisfying the overflow
out of host RAM across PCIe rather than failing. Measured, one GiB at a time:

```
  +1 GiB (total 7) in  0.009s
  +1 GiB (total 8) in  0.480s   <- crosses into host memory
  +1 GiB (total 9) in  0.085s
  ... allocated 11 GiB on an 8 GiB card
```

So a run that does not fit does not raise `OutOfMemoryError`. It runs every step
at bus speed instead of memory speed, while `nvidia-smi` reports a full card at
100% utilization — the exact picture of a healthy run. Reproduced directly: the
config above never completed a single step in ten minutes.

`torch.cuda.set_per_process_memory_fraction` bounds the *allocator*, not the
driver, and the allocator was under its own limit the whole time. Nothing in the
process could tell that half its memory was on the wrong side of the bus.

### 3. Nothing said the run was alive

`report_every` is `total_steps // TARGET_REPORTS`, and the loop reported on
`(step + 1) % interval == 0` — so the first frame of a 23,840-step run landed on
step 158. At a healthy step that is three minutes of silence; at a spilling one
it is never. Phase 1 publishes frames every second while it packs, so the run
went quiet exactly when training began, which reads as a crash and was not one.

### And underneath, a fourth

Fixing the first three exposed one more. `Collector.observe` records an explicit
attention softmax — `[rows, heads, seq, seq]` in fp32, per layer — and those
matrices lived on the blocks until `after_step` walked them. `after_step` runs
after the optimizer, so every layer's matrix was alive across the whole training
step it was measuring. That is quadratic in sequence length and linear in depth:
invisible at ctx 512, decisive at ctx 2048.

```
18 layers, DIAGNOSTIC_ROWS = 2, recompute depth 6
ctx  512    ordinary 2419 MiB    diagnosed 2743 MiB   1.13x
ctx 1024    ordinary 3743 MiB    diagnosed 4967 MiB   1.33x
ctx 2048    ordinary 6385 MiB    diagnosed   OOM
```

[`gpu-memory-is-handed-back.md`](gpu-memory-is-handed-back.md) took the
diagnosed step from 2.1x to 1.1x at ctx 1024 by moving the recording off the
training path. It did not move the *retention*, because at the contexts measured
then the retention was a few hundred MiB. At 2k it is 4.8 GiB.

## Decision

**A run decides what it can afford before its first step, and the card is made
to refuse what it cannot.** Four changes, in `llm/train/memory.py` and the three
places it touches.

### `recompute_blocks` becomes a count, planned by default

`Transformer.recompute_depth` is how many blocks recompute, spread evenly
through the stack; `recompute_blocks` stays as the all-or-nothing property over
it, so a stored config saying `true` still means every block. The config field
is now `bool | int | None` and defaults to `None` — which is not "off" but "plan
it", the only value that reads the card.

The two ends of a flag were the wrong two settings to choose between. Every
recomputed block saves the same activations and costs the same second forward,
so the trade is linear in the count — measured at 512d/18L/ctx2048/bs8, the run
that started this, eager, against an allocator capped at 7045 MiB:

```
depth   peak MiB   ms/step
    0      OOM      (estimated 8638 MiB)
    2      OOM      (estimated 7991 MiB)
    4      OOM      (estimated 7345 MiB)
    6      6367.6    1240.0
    8      5758.1    1292.1
   10      5148.6    1304.4
   12      4539.1    1375.2
   14      3929.6    1385.6
   18      2710.6    1430.0
```

Both columns are straight lines: **305 MiB and ~16 ms per recomputed block.**
That is what makes the count worth having. The three OOM rows are the guard
below doing its job — before it, each of them was a run that spilled and never
finished.

Compiled — which is the default, and what a real run does — the same shape is
cheaper and the shape of the trade is unchanged:

```
depth   peak MiB   ms/step
    0      OOM
    4      6055.6     804.7
    8      4685.3     855.4
   18      1837.2     915.8
```

301 MiB and 7.9 ms a block — the same memory slope as eager, at half the time
cost, which is what says compile changes the fixed term and not the per-block
one.

**So Model 3.0 fits, compiled, with 4 or 5 of its 18 blocks recomputed, at
~805 ms a step.** `plan` picks 5 against the ~6300 MiB an idle card offers — one
block more conservative than the measured boundary, which is the estimate's
safety doing its job. Taking all 18 instead measures 915.8 ms for 4.2 GiB of
headroom the run has no use for: **14% of every step, forever, to solve a
problem the run did not have.** That difference is the whole argument for a
count.

The same budget now also fits **24 layers at ctx 2048**, at depth 12 of 24 —
the size that made the driver fall over before this change.

### The allocator is capped at what the card can physically honour

`memory.guard` sets the allocator fraction to the budget computed from
`mem_get_info` — free memory, less a margin. Past that the allocator raises
instead of letting the driver reach into host RAM. This is the half that does
not depend on the estimate being right: an estimate that is wrong the safe way
costs a recomputed block, and one that is wrong the other way now costs seconds
rather than the run. `loop.release_device` lifts the cap at the same boundary it
releases the pool, so one run's budget does not bind the next one or the chat
that follows it.

### The first step always reports

`reporting` is now `(step + 1) % interval == 0 or step == resumed_at`, and the
frame carries `peak_mib` and `budget_mib`. Diagnostics sit the opening frame out
— it exists to say the run is alive, not to be the most expensive step in it. A
run can no longer be silent for minutes about whether it is training.

### Attention matrices are reduced as the forward walks past them

`Collector.__enter__` installs a forward hook per attention module that turns
each matrix into its per-head entropy and drops it, so one is alive instead of
`layers` of them. The reduction stays in `Collector` — the model still hands
back raw weights and still knows nothing about what is measured off them, which
is the boundary [`gpu-memory-is-handed-back.md`](gpu-memory-is-handed-back.md)
drew and this deliberately does not cross.

```
              before      after
ctx  512      1.13x       1.015x
ctx 1024      1.33x       1.019x
ctx 2048       OOM        1.025x
```

## What this costs

**The estimate is fitted, not derived.** Two coefficients over six measured
configurations, biased ~6% high so it errs toward a recomputed block rather than
a spill. It will be wrong on hardware it was not fitted on, and on a shape far
from those six. Two things make that survivable: `guard` turns a bad
under-estimate into an immediate error, and every frame carries the peak the run
actually reached, so the error is visible on a real run rather than only in a
benchmark. `tests/test_memory_plan.py` pins the fit against the measured table
and fails if a change makes the estimate land *under* any of them.

**A planned run is not reproducible across cards, or across a desktop.** The
depth is chosen from whatever was free when the run started, so the same config
can train at depth 4 today and depth 7 tomorrow with a browser open. The
arithmetic is identical either way — recomputing does not change gradients — so
what varies is wall-clock and nothing else. The chosen depth is recorded in the
run summary under `memory`, which is what makes two different wall-clocks for
the same architecture explicable after the fact.

**Only the two pipelines that go through `loop.run` are planned** — pretrain
(`orchestrate/train.py`) and SFT (`orchestrate/sft.py`). `orchestrate/rl.py`
drives its own loop and gets neither the plan nor the guard, so an RL run can
still spill the way a pretrain used to. Its memory shape is different enough
(generation under `no_grad` with caches, then a policy-gradient step) that
extending the fit to it without measuring it would be guessing, and none of the
measurements here cover it. It is a real gap and a separate one.

**`guard` is process-wide.** It is lifted at the run boundary rather than scoped
to the run, because the allocator has no per-thread notion of a fraction. A
chat served on another thread of the same uvicorn *during* a run is held to the
run's budget. That was already true of the memory itself.

## What was rejected

**Recomputing every block by default.** It is what the run needed and 3.7 GiB
more besides, at 1.21x every step forever. That is the yo-yo this record exists
to avoid: the old default was too little and its opposite is too much, and the
setting worth having is neither end.

**Measuring instead of estimating** — running the first step at full recompute,
then relaxing. It self-calibrates to compile, `ff_multiplier`, vocabulary and
hardware, which the fit does not. It also changes the graph, and `torch.compile`
recompiles when it does: two extra compiles at ctx 2048 is minutes, paid by
every run to save a coefficient. Worth revisiting if the fit turns out to travel
badly.

**Shrinking `batch_size` instead.** It is the cheaper trade per byte — bs4 at
depth 0 measured 4629 MiB at 66.3 ms per 1k tokens against depth 18's 80.3 —
but batch size is an optimization hyperparameter and this LLM engine has no gradient
accumulation, so a smaller batch is a different run and not merely a slower one.
Recompute depth is the lever that leaves the model unchanged. Accumulation
remains the next thing worth building, and would make batch size a memory knob
too.

**Exposing the depth in the composer.** The plan is automatic and the number it
picks depends on what the card has free at the moment the run starts, which is
not something a person can usefully set in a form an hour earlier. What the
frontend gained instead is the *reading* — `peak_mib` against `budget_mib` on
every frame. `batch_size` not being in the composer is a real gap and a separate
one; it is noted here and not fixed here.

**Re-recording `tests/fixtures/` to make the suite green.** Three tests fail on
this branch — `test_dataflow_trace.py::test_stream_matches_trace` and the two
chat-route tests that read the same fixture. They fail identically at `d9b671d`,
before the memory work began, so they are older than this change and are not
evidence of what it moved. Re-recording them here would destroy exactly the
evidence the trace exists to preserve.
