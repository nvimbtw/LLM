# Training memory

[Modular training support](modular-training-support.md) documents the separate
custom-model tensor/export planner, parameter ownership, occurrence recomputation
and bounded diagnostics. The measured legacy planner below retains its existing
Transformer calibration.

Modular pretraining selects occurrence replay from exact ordinary strict-capture
partitions, retaining unselected work, all named output backing storage and the
largest replay region. `null` chooses the fewest fitting prefix occurrences;
explicit true/false/count policies must fit their own estimate. Unknown work or
nonfitting candidates refuse before optimization. Packages retain the profile,
storage bounds and selected/candidate policies; exact recovery preserves its
saved selection even when the allocator ceiling changes. This supported-operation
estimate is separate from hardware calibration. Selected CUDA replay requires
eager execution (`compile_model=false`) until checkpoint compilation is validated.

Pretraining and SFT share `llm/train/loop.py` and its memory planner. RL has a
separate loop and is not covered by this policy. The default keeps bf16,
compilation, the requested batch size and the full context window; it trades
activation storage for recomputation only where the estimate needs it.

## Two limits, with different jobs

`memory.budget_bytes` computes an allocator cap from **driver-free memory plus
PyTorch-reserved memory**, leaving 12% or at least 384 MiB outside the allocator.
Reserved memory includes both live tensors and cached blocks. Including it makes
the cap independent of whether SFT has already loaded its base weights or a
previous run left reusable blocks cached. Other processes and existing CUDA
allocations outside PyTorch remain excluded.

The planner targets a lower number: the cap minus **10% or at least 512 MiB**.
Cached fragments and temporary tensors need room *inside* the cap; the external
driver margin cannot satisfy an allocation that PyTorch's limiter refuses.
The planner still chooses the fewest recomputed blocks that fit that target.
It includes other live PyTorch allocations, such as an opt-in SFT reference
model, without counting the training weights twice.

`recompute_blocks: null` selects the automatic depth. Explicit `true`, `false`
or integer settings still override it. The run summary's `memory` records
`estimated_mib`, `target_mib`, `budget_mib` and the chosen depth; progress frames
continue to report `peak_mib` against the allocator cap in `budget_mib`.

The cap prevents the WSL driver from satisfying an oversized run through slow
system-memory fallback. It is not a promise that every architecture fits: the
estimate is fitted to bf16 measurements, outside workloads can change memory
availability, and full recomputation cannot shrink weights or AdamW state.
An eager fallback while constructing the compiled callable is planned at eager
cost. Compilation failures during its first invocation still propagate.

## Keep reporting from setting the peak

Old gradients are cleared **before** diagnostics and the next training forward,
after the previous update and its report have consumed them. This avoids keeping
two steps' allocations alive together without changing the batch or schedule.

Diagnostics use a separate no-grad forward, reduce attention matrices layer by
layer, and release logits before running additional probes. Weight snapshots
for update ratios live on CPU, so they do not compete with AdamW for GPU memory.
This costs host RAM proportional to block weights and transfers at the diagnostic
cadence, not on ordinary training steps. The ratio is still measured from the
actual before/after weights.

If observing, snapshotting or assembling diagnostics raises a CUDA OOM, the
collector clears its captures, logs the reason and disables further diagnostic
samples for that run. Training, loss reporting and checkpointing continue;
`diag` is absent rather than populated with fabricated zeros. Other exceptions
still propagate, and training/optimizer OOMs are not swallowed. See
[`telemetry.md`](telemetry.md) for the diagnostic contract.

## A failed run must release its weights

`Runner._execute` clears failed worker traceback frames before releasing the
device. A worker future can retain the exception after the catch, and its
traceback otherwise retains the model, optimizer and gradients in those frames.
`loop.release_device` collects reference cycles, resets the allocator cap and
empties the cache at the run boundary. Emptying the cache alone cannot release
live tensors. It is intentionally not done every step, where reuse is faster.

The historical measurements and WSL fallback investigation are recorded in
[`a-run-plans-its-own-memory`](../decisions/a-run-plans-its-own-memory.md).

## Validation on the 8 GiB card

On 2026-09-16, a bounded masked-loss training check used the RTX 3060,
PyTorch 2.13.0+cu130, bf16 and compilation at 512 dimensions, 18 layers,
8 heads, ff multiplier 4, vocabulary 8192, context 2048 and batch 8. The weights
were already on CUDA when the loop started, as in SFT.

The previous loop and diagnostic implementation with its previous budgeting
failed on the first diagnostic step after completing step 1. The revised loop
completed eight steps, seven diagnosed, at 5196 MiB peak allocated against a
6306 MiB cap, recomputing 7 of 18 blocks. A separate run with fewer diagnostics
measured ordinary warmed steps at 0.76–0.89 seconds and diagnosed steps at
1.75–2.24 seconds. Initial compilation took about 48 seconds without the cached
graph. These are short synthetic checks, not a full-dataset completion estimate.
