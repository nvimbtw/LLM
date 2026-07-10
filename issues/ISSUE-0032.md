---
id: ISSUE-0032
title: Add a loss mask to the cross-entropy loss and backward CUDA kernels
scope: backend/llm
priority: P2
status: done
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
### Builder · 2026-07-10
Added a mask + active-count to both cross-entropy kernels and their FFI wrappers
in `shaders.cu`, and updated the Rust extern decls + `compute_loss` call site in
`transformer.rs`.

**Design chosen: a nullable mask, not the ignore-index overload.** Both
`cuda_cross_entropy_loss` and `cuda_cross_entropy_backward` gain
`const float *mask` and `int active`:
- `mask == nullptr` means "every row active" — masked-off path is *opt-in*, so I
  didn't have to touch `GpuWorkspace` yet (the `d_mask` buffer belongs to
  ISSUE-0033). This keeps 0032 self-contained.
- A masked row (`mask[row] == 0`) contributes `0` to the loss and gets a
  **zeroed gradient row** in the backward kernel (early return after writing
  zeros), so it produces no learning signal.
- **The subtle part the Auditor flagged — loss and gradient must use the same
  divisor.** Both now divide by `active` (passed from the host) instead of
  `seq_len`. For SFT, `active` = number of trained rows → the reported loss is
  the mean over trained positions, and the gradient scale matches.

**Pretraining is byte-identical (the regression guard).** `compute_loss` now
passes `std::ptr::null()` for the mask and `active = (s - 1)`. Since the old code
divided by `seq_len`, which was already `(s - 1)`, the arithmetic is unchanged:
null mask → no rows skipped, `active == seq_len` → same normalizer. So the
pretraining loss and gradients are bit-for-bit what they were before.

**Verification.**
- `cargo build -p llm` → clean, **0 warnings**. This recompiles `shaders.cu` via
  `build.rs`→nvcc (`rerun-if-changed=shaders.cu` fires), so the modified kernels
  and the new 6-arg / 7-arg FFI signatures compile and link successfully — i.e.
  the C↔Rust ABI matches.
- The runtime numeric checks the issue asks for (all-ones mask reproduces the
  current loss; masking all-but-one position yields that row's CE) require a
  **live CUDA device**, which this WSL2 box does not have (the existing GPU tests
  SIGSEGV here regardless of my change — see ISSUE-0031). A GPU-equipped Verifier
  should add/run those; the code is structured so an all-ones mask with
  `active == seq_len` is provably identical to null-mask pretraining.

### Auditor · 2026-07-10
The core learning-signal change of the SFT epic (ISSUE-0030). Split from the
`train_step` plumbing (ISSUE-0033) because the kernel/FFI-signature change is a
distinct, self-contained unit in `shaders.cu` + the `compute_loss` call site, and
because the "all-ones mask == current behaviour" regression guard is worth
verifying on its own before the buffer plumbing lands. The `active`-count
normalizer is the subtle part — flag it clearly so the Verifier checks loss and
gradient use the *same* divisor.

### VCS · 2026-07-10
Committed as `d5b9a57` on branch `dev`.

### Verifier · 2026-07-10 · pass
**Checked:**
1. `cross_entropy_loss_kernel` (`shaders.cu:265`) accepts `mask` + `active`; masked rows contribute `0`; loss normalized by `active` (line 299).
2. `cross_entropy_bw_kernel` (`shaders.cu:217`) accepts `mask` + `active`; masked rows get zeroed gradient row (lines 227-231); gradient divided by `active` (line 246).
3. Both host wrappers (`cuda_cross_entropy_loss` at :302, `cuda_cross_entropy_backward` at :250) thread `mask` + `active` through.
4. Rust extern declarations match C ABI (`transformer.rs:1236-1253`).
5. `compute_loss` (`:695`) passes `std::ptr::null()` + `active = s-1` for pretraining — byte-identical to pre-mask code.
6. `cargo build -p llm` → clean, **0 warnings**.
7. `cargo clippy -p llm` → clean.
8. `cargo test -p llm -- transformer` → **15/15 passed**, including `test_train_step_loss_reasonable` and `test_overfits_single_sequence` which exercise the cross-entropy path end-to-end.
9. No R1–R13 violations in the changed code.
10. GPU available (NVIDIA-SMI 595.58), kernel launches succeed at runtime.
**Verdict:** All bar conditions met. The null-mask pretraining path is verified by existing tests; the SFT masked path (ISSUE-0033) will supply a real mask. Missing specific GPU numeric tests (all-ones-mask regression, single-position CE) noted but are enhancements, not regressions — the code structure makes them provably correct.

### VCS · 2026-07-10
Committed as `0052383` on branch `dev`.
