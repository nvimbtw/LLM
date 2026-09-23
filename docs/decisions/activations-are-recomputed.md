# Activations Are Recomputed, And The Ceiling Moves To The Optimizer

- **Date**: 2026-09-08
- **Status**: Implemented

## Context

[`gpu-memory-is-handed-back.md`](gpu-memory-is-handed-back.md) stopped a run's
ceiling being set by the step that *measures* it. This is the next question,
which is what sets it now: a 54.6M-parameter model — 512d, 8 heads, 16 layers,
ctx 1024 — was using 4.5 GB of an 8 GB card, and a card that also has a desktop
on it has about 7 GiB to give.

Where that goes, measured at batch 8 / ctx 1024 in bf16:

```
params                   54.6 M
weights                    212 MiB
weights + grads + AdamW   1006 MiB   (steady state between steps)
activations               3214 MiB   (held for the backward pass)
PEAK                      4253 MiB
```

**The model is 1 GB and the activations are 3.2 GB.** Three quarters of the
peak is intermediate tensors that exist only because the backward pass will
need them — every block storing its own normalized input, its q/k/v, its
attention output, and the 4x-wide feed-forward tensor either side of the GELU.
That total scales with depth and with batch x context, and *not* with how good
the model is. It is the whole reason a bigger model would not fit: the
parameters are cheap and what surrounds them is not.

The trade against it is standard and this LLM engine simply had not taken it. Keep
each block's *input* only, and when the backward pass needs the insides, run the
block forward a second time to rebuild them.

## Decision

**`Transformer.recompute_blocks`, off by default, set from
`TrainConfig`/`SftConfig` by `llm.train.loop.run`.**

A runtime flag on the module, beside `record_activations`, and deliberately not
a constructor argument: it does not enter `checkpoint._ARCH_FIELDS`, the
state_dict, or a saved checkpoint. The same weights train identically with it on
or off, so it is a property of a *run*, not of a model — a checkpoint that
recorded it would turn a memory setting into part of what the model is.

`forward` recomputes only a pass that is cache-free and requires grad. Both
halves are load-bearing: a decode step passes caches and would be rebuilt
without them, and an evaluation under `no_grad` has no backward to save anything
for, so recomputing there is pure cost.

Measured on an RTX 3060, the same 54.6M config at batch 8 / ctx 1024:

```
                          peak VRAM    ms/step
stock, eager                4220 MiB     446.6
stock + compile             3831 MiB     300.7   <- what shipped before this
recompute, eager            1767 MiB     565.1
recompute + compile         1273 MiB     391.2
```

Against what actually shipped, **3831 -> 1273 MiB for 1.30x the step time**. The
same flag through the real `loop.run` with compile off: 4222 -> 1767 MiB, a run
of 12 steps going from 5.9 s to 7.0 s.

The arithmetic does not move, which is the only reason this is worth having.
One step at 256d/4L, gradients compared parameter by parameter:

```
fp32   loss 9.01274204 vs 9.01274204   max relative gradient delta   4.33e-07
bf16   loss 9.01269531 vs 9.01269531   max relative gradient delta   0.00e+00
```

The fp32 residue is float reassociation, not a different answer; in bf16 the
gradients are bitwise identical. `tests/test_recompute_blocks.py` pins the
parity, the fact that each block runs a second time, and the two paths that must
never recompute.

## What this costs

**Every step is ~1.3x slower, so this is off by default.** A model that already
fits should not pay for headroom it is not using. It is one field on the run,
which is the right granularity: the same card trains a small model fast and a
large model at all, and those are different runs.

**It is not free at small sizes either.** At 54.6M the flag turns a 300 ms step
into a 391 ms one for memory the run did not need — which is exactly why the
default is off rather than "on when it looks big".

## Where the ceiling is now

With activations no longer dominating, what binds is the optimizer. AdamW at
fp32 holds, per parameter, a master weight, a gradient, and two moments — 16
bytes, before a single activation. On ~7 GiB of usable card that is a hard
**~470M parameters with zero room to compute in**, and comfortably under 400M in
practice.

Measured at batch 2 / ctx 1024 with recompute on:

```
 512d x 16L     54.6M params    1186 MiB
1024d x 24L    310.6M params    5680 MiB
```

So the reachable range moved from ~50M to ~300M — about 6x — and the next lever
is the optimizer state, not the activations. What would move it again, in
descending order of what it buys:

- **8-bit optimizer moments** (`bitsandbytes.AdamW8bit`): 16 -> 10 bytes per
  parameter, so ~700M. Well established as quality-neutral but *not* bitwise
  identical, which is why it is not folded into this change.
- **Gradient accumulation.** Fitting a larger model means a smaller resident
  batch, and batch size is an optimization hyperparameter — accumulating
  restores the effective batch that the learning rate and warmup were tuned
  against. The LLM engine has no accumulation today, so a bigger model is currently
  also a smaller-batch run, and that *is* a quality difference.
- **Recomputing every other block** rather than all of them, which buys back
  roughly half the time for roughly half the memory. Worth having only once
  something needs a point between the two settings.

None of these is needed to train ~300M, which is where this leaves the card.

## What was rejected

**On by default.** It costs 30% of every step, forever, to solve a problem only
the largest runs on this card have. The measured numbers above are what a person
needs to decide, and they are in the field description.

**A constructor argument on `Transformer`, or a field in `arch`.** It would ride
into every checkpoint and make two models trained with different memory settings
look like different architectures — and `checkpoint.load` would then have to
decide what to do with a flag that describes a machine rather than a model.

**Recomputing inside `Block.forward` instead of around it.** Finer-grained
control — attention and feed-forward separately — for a much larger blast
radius: the block would have to know whether it is being recomputed, and the
decode path runs through the same method. The block boundary is where the
residual stream already is, which makes it the one place a single stored tensor
is enough to rebuild everything after it.

**`torch.utils.checkpoint` with `use_reentrant=True`.** The reentrant version
does not compose with autocast, requires an input that already requires grad,
and does not tolerate a block returning its `(x, cache)` tuple. All three are
wrong here and none of them fail loudly.

**Waiting for a 13B model to fit.** It cannot, and no amount of activation
saving changes that: 13B parameters is 26 GB of bf16 weights before a gradient
or a moment exists. Numbers in that range come from 4-bit *inference*, where a
13B model is ~6.5 GB of frozen weights and there is no backward pass, no
gradient and no optimizer state at all. Training and serving have different
memory shapes, and this card trains ~300M.
