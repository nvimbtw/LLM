---
id: ISSUE-0083
title: "Extract the GPU→CPU weight readback into training/sync.rs"
scope: backend/llm
priority: P2
status: verification
rule: R20
location: backend/src/llm/src/training/transformer/mod.rs:1593-1668
blocked-by: [ISSUE-0082]
blocks: []
---

## Description
### What
Move `sync_weights_from_gpu` — the `GpuWorkspace` → CPU `Model` readback — out of
`transformer/mod.rs` into a new **`training/sync.rs`**. Like the optimizer, it is
training-loop machinery over the workspace, not part of the model, so it sits at
`training/` level. This is the final child; after it, `transformer/mod.rs` holds
only the model struct, `init`, `StepDims`, the step orchestrator, and module wiring.

### What moves
- `sync_weights_from_gpu(ws: &GpuWorkspace, t: &mut Model)` (~1593–1668).

### Shape of the change
- New file `training/sync.rs`; add `pub mod sync;` to `training/mod.rs`.
- Keep `sync_weights_from_gpu` `pub` (called from `train.rs` and the tests); update
  callers from `training::transformer::sync_weights_from_gpu` to
  `training::sync::sync_weights_from_gpu`. Grep every call site and fix it.
- Pure move; no logic change (it already reads the f32 masters after ISSUE-0056).

### Verification
- `cargo build` + `cargo clippy` clean; grep confirms no lingering old-path refs.
- Full GPU suite passes `--test-threads=1` — in particular the tests that call
  `sync_weights_from_gpu` to read weights back
  (`test_gpu_sgd_does_not_corrupt_weights`, `test_master_accumulates_sub_ulp_updates`).
- Confirm `transformer/mod.rs` is now down to the model + step orchestration + `mod`
  declarations, closing the epic's DoD item 2.

## Comments
### Auditor · 2026-07-14
Last child. Placed at `training/` level for the same reason as the optimizer (it
operates on the workspace). Its completion is what lets ISSUE-0077's DoD assert the
monolith is fully decomposed.
