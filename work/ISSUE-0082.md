---
id: ISSUE-0082
type: issue
title: "Extract the Adam optimizer into training/optimizer.rs"
scope: backend/llm
priority: P2
status: verification
rule: R20
location: backend/src/llm/src/training/transformer/mod.rs:13-15 backend/src/llm/src/training/transformer/mod.rs:1498-1592
blocked-by: [ISSUE-0081]
blocks: [ISSUE-0083]
---

## Description
### What
Move the optimizer — which operates on the **`GpuWorkspace`**, not the model —
out of `transformer/mod.rs` into a new **`training/optimizer.rs`** (one level up
from `transformer/`, per the epic's placement rule).

### What moves
- Adam constants `ADAM_BETA1`, `ADAM_BETA2`, `ADAM_EPS` (~13–15) and the per-step
  grad-clip constant `MAX_GRAD_NORM` (find it near the other consts / clip site —
  it's the `1.0` clip threshold; name it as a `const` if it isn't already, R8).
- `grad_buffers` (~1498), `gpu_clip_gradients` (~1528), `gpu_adam_update` (~1554)
  and its inner `adam!` macro (~1569).

### Shape of the change
- New file `training/optimizer.rs`; add `pub mod optimizer;` to `training/mod.rs`.
- These are already free functions taking `&GpuWorkspace`, so no visibility gymnastics
  — keep `gpu_adam_update` / `gpu_clip_gradients` `pub` (called from `train.rs`),
  make the consts and `grad_buffers` private to the module.
- Update `train.rs` / any caller imports from `training::transformer::gpu_adam_update`
  to `training::optimizer::gpu_adam_update` (and same for `gpu_clip_gradients`).
  Grep for the old paths and fix every call site — this is the one child that
  changes a public path, so the caller updates are part of it.

### Verification
- `cargo build` + `cargo clippy` clean; grep confirms no remaining references to the
  optimizer fns under the old `transformer::` path.
- Full GPU suite passes `--test-threads=1`; the tests that call `gpu_adam_update`
  directly (`test_overfits_single_sequence`, `test_master_accumulates_sub_ulp_updates`,
  `test_raised_lr_trains_stably`, `test_gpu_sgd_does_not_corrupt_weights`) still pass
  after their imports are updated.

## Comments
### Auditor · 2026-07-14
The optimizer lives at `training/` level, not under `transformer/`, because it
steps the workspace — this is the epic's central placement decision. This child
also owns the call-site import updates since it's the one that moves a public path.
