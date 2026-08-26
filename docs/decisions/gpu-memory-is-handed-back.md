# The GPU's Memory Is Handed Back, And Diagnostics Stop Setting The Ceiling

- **Date**: 2026-08-25
- **Status**: Implemented

## Context

Two symptoms, reported as separate problems, that turned out to be one
mechanism seen at two moments:

1. After a training run finished, the card still read as full — 7.4 of 8 GB on
   an RTX 3060 — until the engine process was restarted.
2. During a run, memory climbed in discrete jumps rather than settling, and
   never came back down. In practice that capped a model at roughly
   512 dimensions / 512 context / 8 heads / 6 layers on 8 GB.

The mechanism is that **nothing hands a CUDA pool back**. Torch's caching
allocator keeps a freed block reserved so the next allocation of that size does
not have to ask the driver again — the right trade inside a run, the wrong one
between them. `nvidia-smi` and Task Manager report that reserved pool, not what
is live. Measured across two runs in one process:

```
run 1: inside, after 3 steps     alloc   379 MiB   reserved  1022 MiB
after run 1 returned (+gc)       alloc    16 MiB   reserved  1022 MiB
run 2: inside, after 3 steps     alloc   387 MiB   reserved  1086 MiB
after run 2 returned (+gc)       alloc    16 MiB   reserved  1086 MiB
after empty_cache()              alloc    16 MiB   reserved    40 MiB
```

So there was **no leak**: allocated memory returned to 16 MiB and did not grow
run over run. The pool was simply never released, in a process that never exits
— `api/runner.py` trains on a thread of the same uvicorn that serves chat.

That explains symptom 1 on its own. Symptom 2 needed the second half: what set
the high-water mark the pool was holding. It was the **diagnostics step**.

`Collector` flipped `record_attention` on and the training forward ran with it,
which meant three costs at once, all landing on the same step:

- The explicit softmax materializes `[batch, heads, seq, seq]` where the fused
  kernel materializes nothing, in fp32 because softmax is on autocast's fp32
  list — and autograd **retained every one of them** for a backward pass that
  had no use for them.
- It ran over the whole batch, for a metric that is a mean over rows.
- It forced the uncompiled callable, putting cross-entropy back in eager and
  with it the second fp32 copy of the logits that `loop.step_fn` exists to
  avoid.

Measured, against an ordinary step at the same config:

```
512d/8h/6L/ctx256    normal   810 MiB   diagnosing   918 MiB  (1.1x)
512d/8h/6L/ctx512    normal  1209 MiB   diagnosing  1714 MiB  (1.4x)
512d/8h/6L/ctx1024   normal  2006 MiB   diagnosing  4172 MiB  (2.1x)
768d/12h/8L/ctx512   normal  2407 MiB   diagnosing  3421 MiB  (1.4x)
```

After the change, at `DIAGNOSTIC_ROWS = 2`, the same diagnosed step costs
1110 MiB at ctx512 and 2020 MiB at ctx1024 — against 1704 and 4169 before.

The cost is quadratic in sequence length, which is why context was the binding
constraint rather than width or depth. And because the pool is never released,
the *first* diagnosed step — a few hundred steps in — raised the run's floor
permanently. The run's real ceiling was set by the step that measures it, not by
the step that trains.

## Decision

**Release the pool at run boundaries, and record diagnostics off the training
path.**

`loop.release_device()` is `torch.cuda.empty_cache()` behind a name that says
why, called from `Runner._execute`'s `finally` in `api/runner.py`. The `finally`
is deliberate: a run that failed is the one that most needs it, because an OOM
leaves the largest pool of all behind and the next run would meet a card still
holding it. It covers every run kind, and it is a no-op off CUDA so no caller
has to know which device the run it just finished was on.

`Collector.observe` replaces the fused recording. It is a `no_grad` forward over
the first `DIAGNOSTIC_ROWS` rows of the batch, run immediately before the
training step it describes — same weights, same batch. The training step is then
untouched by the fact that it is being measured, which also means `loop.run` no
longer keeps an eager callable beside the compiled one and every step trains
through the same graph. `Collector.after_step` additionally releases each
layer's attention matrix as it consumes it, rather than gathering all of them
into a list first: `_entropy` allocates two more tensors the size of the one it
is reading, so holding every layer's alive made the peak scale with depth on top
of the seq² each one already costs.

Two supporting changes:

- **`chat.evict()`**, called by the runner before a run starts. A model cached
  by `orchestrate/chat.py` was never asked to leave — it sat on the card until
  some later chat happened to name a different one. There is one device and one
  process, so that is memory a run cannot have.
- **`PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True`** on the engine service.
  It lets the allocator grow and shrink a segment instead of stranding blocks it
  cannot merge. Not a substitute for either change above; it reduces what a
  single run needs rather than what it holds afterwards.

End to end, on the real `loop.run` at 512d/8h/6L/ctx512/bs8:

```
run with diagnostics, finished       reserved  1182 MiB   peak  1150 MiB
after loop.release_device()          reserved    40 MiB
run with NO diagnostics, finished    reserved  1034 MiB   peak  1005 MiB
```

Diagnostics now add 145 MiB to a run's peak where they added 700, and the pool
comes back at the boundary.

## What this costs

**The attention readout is sampled from two batch rows instead of eight.** This
is a real change in what is measured, not only in what it costs. Against the
eight-row reading, the largest per-head deviation was 0.019 at ctx512 and 0.032
at ctx1024, on a 0–1 scale — a wobble in a trend line rather than a different
answer, and one row already averages `ctx_window` query rows per head, so the
batch dimension was buying precision the chart cannot show. `DIAGNOSTIC_ROWS` is
where to change it if a future reading needs to be tighter; the cost is linear in
it and small now that the pass is `no_grad`.

**A diagnosed step costs one extra forward pass.** It is no-grad, over two rows,
and taken roughly thirty times in a run (`TARGET_DIAG_SAMPLES`). Against what it
replaced — a whole-batch forward *and* backward through the slower attention
path — it is cheaper in time as well as in memory.

**The first chat after any run pays a model reload.** Under a second, against a
run measured in minutes to hours. The trade only makes sense in this direction,
which is why eviction is the runner's call and not something `chat.load` does to
itself.

## What was rejected

**Calling `empty_cache()` per step, or per report.** It is a synchronizing call
that returns memory the very next step will ask for again. The pool exists
because that churn is expensive; the problem was never that it existed, only
that it outlived the run.

**Computing the entropy inside `Attention` and storing the reduced vector.**
It would drop the retained matrix to nothing, which is strictly better on
memory. It also puts a diagnostics metric inside the model, where every future
reading has to be anticipated by the module rather than derived from what it
exposes. `record_attention` handing back raw weights is what lets `explore/`
ask questions the training loop never will.

**Keeping the fused recording and simply shrinking the batch for it.** That
addresses the row count but not the retention, which is the larger of the two —
the fp32 matrices being held for a backward pass that discards them.

**A separate process per run.** It would solve the pool question absolutely, and
process teardown is the only thing that reliably returns *everything*. It also
means shipping the corpus mapping, the progress stream and the cancel flag
across a process boundary, which is a great deal of machinery to buy something
one call already buys. Worth revisiting only if a leak that survives
`release_device` ever turns up — as of this record, none does: allocated memory
returns to 16 MiB, and that residue is the inductor cache and the CUDA context.
