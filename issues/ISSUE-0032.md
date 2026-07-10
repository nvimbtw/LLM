---
id: ISSUE-0032
title: Add a loss mask to the cross-entropy loss and backward CUDA kernels
scope: backend/llm
priority: P2
status: todo
rule: none
location: backend/src/llm/src/training/shaders/shaders.cu:213-298 backend/src/llm/src/training/transformer.rs:695-732
blocked-by: []
blocks: [ISSUE-0033]
---

## Description
### What
Teach the cross-entropy kernels to **ignore masked positions**: a masked position
contributes `0` to the loss and `0` to `dlogits`, and the loss is normalized by
the count of **active** (unmasked) positions rather than by `seq_len`.

### Why
SFT trains only on assistant-reply tokens (ISSUE-0030/0031). Today
`cross_entropy_loss_kernel` and `cross_entropy_bw_kernel`
(`shaders.cu:248` / `:213`) run over every row `0..seq_len` and divide by
`seq_len`. With a mask, roughly 70–90% of a chat example's positions are context
(system/user/scaffold) that must not produce a learning signal — otherwise the
model is trained to reproduce the *user's* words, which is wrong and wastes the
tiny model's capacity.

### Shape of the fix
- Add a `const float *mask` parameter (length `seq_len`, `1.0`=train, `0.0`=ignore)
  to both `cuda_cross_entropy_loss` and `cuda_cross_entropy_backward` and thread it
  into the kernels.
  - **loss kernel:** if `mask[row] == 0`, set `contrib = 0`. Change the normalizer
    from `/ seq_len` to `/ active`, where `active` is the number of unmasked rows.
    Simplest: pass `active` (an `int`) in from the host (the caller already knows
    the mask), so the kernel just does `atomicAdd(loss_out, shared[0] / active)`.
  - **backward kernel:** if `mask[row] == 0`, write zeros across that row's
    `dlogits` and return. Change the gradient scale from `/ seq_len` to `/ active`
    to match the loss.
- **Alternative considered — ignore-index:** instead of a separate mask buffer,
  overload `targets[row] = -1` to mean "ignore" (the final no-target row is
  already zeroed in `compute_loss` via `cudaMemset`). This avoids a new buffer but
  still needs an `active` count for normalization and makes the target array carry
  two meanings. Prefer the **explicit mask buffer** for clarity (matches this
  repo's "direct over clever" style in `backend/CLAUDE.md`); note the alternative
  in a comment.
- **`transformer.rs::compute_loss` (`:695`):** add an `active` count + a device
  mask. Upload the mask to a new `ws.d_mask` buffer (allocated in ISSUE-0033's
  `GpuWorkspace` change — until then, this issue can gate the masked path behind
  a helper that takes the mask slice and computes `active` on the host). Pass
  `active` and `ws.d_mask.ptr` to both FFI calls. Keep the existing unmasked
  behaviour available (pretraining still needs it) — e.g. an all-ones mask with
  `active == s-1` must produce byte-identical results to today.

### Verification
- A CUDA build (`cargo build -p llm`) must stay warning-free; the `.cu` is
  compiled via `build.rs`/nvcc.
- Add/extend a training test asserting: (a) an all-ones mask reproduces the
  current loss for a fixed input (regression guard), and (b) masking out all but
  one position makes the loss equal the single-position cross-entropy of that row.

## Comments
### Auditor · 2026-07-10
The core learning-signal change of the SFT epic (ISSUE-0030). Split from the
`train_step` plumbing (ISSUE-0033) because the kernel/FFI-signature change is a
distinct, self-contained unit in `shaders.cu` + the `compute_loss` call site, and
because the "all-ones mask == current behaviour" regression guard is worth
verifying on its own before the buffer plumbing lands. The `active`-count
normalizer is the subtle part — flag it clearly so the Verifier checks loss and
gradient use the *same* divisor.
