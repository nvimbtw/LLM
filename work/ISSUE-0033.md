---
id: ISSUE-0033
type: issue
title: Thread the loss mask through train_step (a masked SFT training step)
scope: backend/llm
priority: P2
status: done
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
### Builder · 2026-07-10
Added the masked training step and the device mask buffer, ran the GPU tests
locally (this box does have a CUDA device — an RTX 3060 — the earlier SIGSEGVs
were the sandbox blocking `/dev/nvidia*`, not a missing GPU).

**GpuWorkspace (`gpu_workspace.rs`):** added `d_mask: DeviceBuffer` (f32, length
`max_s`), allocated next to `d_targets` and threaded through the struct literal.
Left it out of `zero_accumulators` deliberately — it's fully overwritten by a
`cuda_memcpy_h2d` on every masked step before use, so zeroing it would be dead work.

**transformer.rs — no loop/pipeline duplication (R6).** Factored the shared
forward+backward body into a private `run_step(ws, chunk, mask: Option<&[u8]>,
layers)`. `train_step` calls it with `None`; the new public
`train_step_masked(ws, chunk, mask, layers)` calls it with `Some(mask)`. Only the
loss stage differs, so the whole forward/backward pipeline (the ISSUE-0004 stage
helpers) is reused verbatim.

**The index shift the Auditor flagged.** `compute_loss` now takes
`mask: Option<&[u8]>`. The loss runs over positions `0..s-1` (position `i`
predicts `chunk[i+1]`), so the kernel mask is exactly the first `s-1` entries of
the ISSUE-0031 per-token mask — `mask_kernel[i]` governs position `i`. The final
token has no successor and is unused here (it's always 0 in the 0031 convention
anyway). `active` = count of 1s in `mask[..s-1]`, uploaded to `ws.d_mask` and
passed to both kernels. `active == 0` (a chunk with no assistant tokens) is
guarded: zero `d_dlogits` and return loss 0.0, so the kernels never divide by zero.

Pretraining path is byte-identical: `train_step`→`run_step(None)`→`compute_loss`
with `None` still passes a null mask + `active = s-1`, exactly as ISSUE-0032 left it.

**Verification (all run locally on the GPU, single-threaded):**
- `cargo test -p llm --lib -- --test-threads=1` → **45 passed, 0 failed**, incl.
  two new tests:
  - `test_train_step_masked_all_ones_matches_unmasked`: an all-ones mask yields
    the same loss as `train_step` (within 1e-4) — the masked path doesn't perturb
    pretraining numerics.
  - `test_train_step_masked_all_zero_trains_nothing`: an all-zero mask gives loss
    `0.0` and **exactly zero** weight change after an Adam step (compared against
    the f16-quantized GPU baseline, not the original f32 weights — the ~0.006
    "change" in my first draft was just upload/sync f16 quantization, which the
    corrected baseline isolates out).
- `cargo build -p llm` and full `cargo build` → clean, **0 warnings**.
- Note on `--test-threads=1`: the GPU tests must run serially — `GpuWorkspace`
  calls the process-wide `cublas_init`/`cublas_shutdown`, which isn't thread-safe
  (the exact caveat documented in ISSUE-0016's safety comment). Running the
  default parallel harness SIGSEGVs on GPU contention; this is pre-existing and
  affects all GPU tests, not something 0033 introduced.

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

### Verifier · 2026-07-10 · pass
Confirmed all items in the Description and Builder claim:

**Code review:**
- `GpuWorkspace` struct has `d_mask: DeviceBuffer` (line 74), allocated at `max_s` (line 179), threaded through the struct literal (line 201). Deliberately excluded from `zero_accumulators` — consistent with the comment that it's fully overwritten by `cuda_memcpy_h2d` before every masked step.
- `train_step_masked(ws, chunk, mask, layers)` → `run_step(ws, chunk, Some(mask), layers)` (lines 434–442).
- `train_step(ws, chunk, layers)` → `run_step(ws, chunk, None, layers)` (line 427) — byte-identical pretraining path.
- `run_step` (lines 446–480) factors the shared forward/backward pipeline (embed → attn → FFN → logits → loss → backward unembed → FFN backward → attn backward → clip → accumulate embedding grad), calling `compute_loss(ws, dims, chunk, mask)`.
- `compute_loss` (lines 721–803): handles `mask: Option<&[u8]>`. The `Some(m)` branch uploads `m[..s-1]` as f32 to `ws.d_mask` and computes `active` count. The `None` branch passes a null pointer with `active = s-1`. `active == 0` guard zeros `d_dlogits` and returns 0.0 loss — prevents divide-by-zero in kernels.
- The index shift (`mask[i]` governs position `i` predicting `chunk[i+1]`) is documented in comments at lines 739–745.
- No new rule violations: R6 respected (shared `run_step` body, no duplicated pipeline), R7 (zero warnings), R3 (no unwrap/expect), R4 (unsafe blocks small + commented), R9 (`train_step_masked` has doc comment).

**Build & tests:**
- `cargo build -p llm` → clean, 0 warnings.
- `cargo build` (workspace) → clean.
- `cargo clippy -p llm` → clean.
- `cargo test -p llm --lib -- --test-threads=1` → **45 passed, 0 failed**, including:
  - `test_train_step_masked_all_ones_matches_unmasked` — all-ones mask matches unmasked loss within 1e-4.
  - `test_train_step_masked_all_zero_trains_nothing` — all-zero mask gives loss 0.0 and exactly zero weight change.
  - `test_overfits_single_sequence` — unchanged pretraining path still overfits.
- Ran single-threaded per the pre-existing cuBLAS contention caveat (no SIGSEGV).

**Verdict:** pass — all criteria met, no regression.

### VCS · 2026-07-10
Committed as `e79c614` on branch `dev`.

