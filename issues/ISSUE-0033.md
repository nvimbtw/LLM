---
id: ISSUE-0033
title: Thread the loss mask through train_step (a masked SFT training step)
scope: backend/llm
priority: P2
status: todo
rule: none
location: backend/src/llm/src/training/transformer.rs:422-454 backend/src/llm/src/training/gpu_workspace.rs:196-213
blocked-by: [ISSUE-0032]
blocks: [ISSUE-0035]
---

## Description
### What
Give the training step a way to accept a per-position loss mask for the current
chunk, so SFT can train on assistant-only positions. Concretely add a masked
entry point alongside the existing one:

```rust
// existing hot path, unchanged:
pub async fn train_step(&self, ws, chunk: &[u32], layers) -> f32

// new, for SFT:
pub async fn train_step_masked(&self, ws, chunk: &[u32], mask: &[u8], layers) -> f32
```

### Why
`train_step` (`transformer.rs:426`) derives its targets internally
(`chunk[1..]`, predict `i+1`) and calls `compute_loss`, which after ISSUE-0032
can honour a mask. But nothing currently *carries* a mask from the caller down to
`compute_loss`. This issue is the plumbing: allocate the device mask buffer, and
add the masked variant that uploads the chunk's mask and computes the active
count before delegating to the (now mask-aware) loss.

### Shape of the fix
- **`GpuWorkspace` (`gpu_workspace.rs`):** add a `d_mask: DeviceBuffer` field
  (f32, length `max_s`), allocated next to `d_targets` in `GpuWorkspace::new`, and
  add its `cudaMemset` to `zero_accumulators` if appropriate. This is the buffer
  ISSUE-0032's `compute_loss` change writes into.
- **`train_step_masked`:** identical to `train_step` except it accepts
  `mask: &[u8]` (length `chunk.len()`, aligned to the chunk's tokens), and passes
  the mask through to a masked `compute_loss`. The mask the caller supplies is
  over the chunk's tokens; inside, the loss is over positions `0..s-1` predicting
  `1..s`, so map the token-mask to the prediction-mask exactly as ISSUE-0031
  defines it (`mask_at_position[i]` reflects whether *target* token `i+1` is
  trainable). Document this index shift in a comment — it's the easy off-by-one to
  get wrong.
- Keep `train_step` byte-for-byte behaviourally identical (pretraining path must
  not regress). The cleanest structure: have `train_step` call
  `train_step_masked` with an implicit all-ones mask, OR factor the shared
  forward/backward body so both entry points reuse it and only `compute_loss`
  differs. Prefer whichever keeps the pretraining numerics identical — verify with
  the existing `test_overfits_single_sequence`.
- Reuse the existing stage helpers (`embed_forward`, `attn_sublayer_forward`, …
  from ISSUE-0004) — do not duplicate the pipeline; only the loss call changes.

### Verification
- `cargo build -p llm` clean; all existing training tests still pass (pretraining
  unchanged).
- New test: a chunk with a mask that zeros every position except one produces the
  same gradient direction as training on that single position alone.

## Comments
### Auditor · 2026-07-10
The plumbing layer between the masked kernel (ISSUE-0032) and the SFT loop
(ISSUE-0035). Split out because it touches `GpuWorkspace` allocation + the
`train_step` API surface — a different concern from the kernel math — and because
the "pretraining path is byte-identical" guarantee deserves its own verification
round. Blocked-by 0032 since the masked variant has nothing to call until the
kernels accept a mask. Flag the token-mask → prediction-mask index shift for the
Verifier; it's the most likely bug.

### VCS · 2026-07-10
Committed as `d5b9a57` on branch `dev`.
