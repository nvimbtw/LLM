---
id: ISSUE-0035
title: SFT training loop and sft_train entry point (fine-tune from a base model)
scope: backend/llm
priority: P2
status: done
rule: none
location: backend/src/llm/src/train.rs:56-153 backend/src/llm/src/train.rs:228-280
blocked-by: [ISSUE-0031, ISSUE-0033, ISSUE-0034]
blocks: [ISSUE-0037]
---

## Description
### What
Add an `sft_train(...)` entry point in `train.rs` that **loads a pretrained base
model** and fine-tunes it on the masked SFT dataset (`sft_corpus.bin` +
`sft_mask.bin`), driving the masked step (`train_step_masked`) instead of the
plain one.

### Why
`train.rs::train` always `Transformer::init`s fresh random weights, and its loop
(`train_epoch`) calls `train_step`, which trains on every position. SFT needs the
opposite on both counts: start from an existing model's weights and only train on
assistant tokens. `continue_train` already shows the "load then loop" pattern
(`train.rs:110`), but it re-runs the *same* unmasked corpus — not SFT.

### Shape of the fix
- **`sft_train`** signature roughly mirrors `continue_train`:
  ```rust
  pub fn sft_train(
      base_model_dir: &Path,        // load weights + merges from here
      sft_corpus_path: &Path,
      sft_mask_path: &Path,
      out_model_dir: &Path,         // where the fine-tuned model is written
      epochs: i64,
      lr: f32,                      // typically much lower than pretraining
      checkpoints_dir: Option<&Path>,
      on_progress: ProgressCallback,
      cancel: Option<Arc<AtomicBool>>,
  ) -> Result<bool, io::Error>
  ```
  - Load the base via `Transformer::load(base_model_dir)` (weights + implied
    `ctx_window`/`layers`, exactly as `continue_train` does). Copy the base
    `merges.bin` into `out_model_dir` so inference can load the fine-tuned model
    standalone.
  - Read `sft_corpus.bin` (as `chunks_exact(4)` → `Vec<u32>`) and `sft_mask.bin`
    (raw bytes → `Vec<u8>`); assert the two are the same length and fail cleanly
    if not.
- **Masked loop:** `train_epoch` currently owns the chunking + step + report
  logic. Either add a masked sibling or parameterize it so the same loop can call
  `train_step` (pretrain) or `train_step_masked` (SFT). Prefer parameterizing to
  avoid duplicating the whole loop (R6). The chunk's mask slice must be sliced
  from the flat mask with the **same** `chunks(ctx_window)` boundaries as the
  corpus — keep them index-aligned.
- **Defaults / notes:** SFT usually wants a lower LR and fewer epochs than
  pretraining; expose them as params (don't hardcode) but document sane starting
  points in a comment. Reuse `save_checkpoint`, `compute_eta`, `LossStats`, and
  the progress reporting as-is — the reported loss now naturally reflects only the
  masked (assistant) positions, which is exactly what the training chart should
  show.
- Watch the `WARMUP_STEPS`/`ACCUM_STEPS` constants: they were tuned for
  pretraining. Fine-tuning runs are shorter, so a 100-step warmup may be a large
  fraction of the run — note this and consider making warmup a fraction of total
  steps, or leave as-is with a comment (don't silently break pretraining).

### Verification
- `cargo build -p llm` clean; existing pretraining tests unaffected.
- New test (the epic's smoke test seed): init a tiny model, run a few `sft_train`
  steps on a 1–2 example masked fixture, assert the masked-region loss decreases.

## Comments
### Auditor · 2026-07-10
The heart of the SFT epic — where load-from-base + masked-loop come together.
Blocked-by 0031 (mask shape), 0033 (`train_step_masked`), and 0034 (the on-disk
dataset). The main design risk is loop duplication: `train_epoch` should be
*parameterized* over the step function, not copy-pasted, to respect R6. Also flag
the pretrain-tuned constants (`WARMUP_STEPS`) as possibly-wrong for short SFT runs
so the Builder makes a conscious choice rather than inheriting them silently.

### VCS · 2026-07-10
Committed as `d5b9a57` on branch `dev`.

### Builder · 2026-07-11
Implemented `sft_train` in `train.rs` and threaded a loss mask through the
existing loop rather than forking it.

**What changed:**
- **New `pub fn sft_train(base_model_dir, sft_corpus_path, sft_mask_path,
  out_model_dir, epochs, lr, checkpoints_dir, on_progress, cancel)`** — mirrors
  `continue_train`'s load-then-loop shape: `Transformer::load(base_model_dir)`
  recovers the base weights (and implies `ctx_window` via `positional_table.len()`
  and `layers`), reads `sft_corpus.bin` as `chunks_exact(4)` → `Vec<u32>` and
  `sft_mask.bin` as raw `Vec<u8>`, then copies the base `merges.bin` into
  `out_model_dir` so the fine-tuned model loads standalone at inference.
- **Length guard (R10):** returns `io::ErrorKind::InvalidData` if
  `corpus.len() != mask.len()` — a drift there would misalign every chunk's mask
  slice, so we fail cleanly instead of training on garbage.
- **No loop duplication (R6):** rather than copy `train_epoch`, I added a single
  `mask: Option<&[u8]>` parameter to both `run_training_loop` and `train_epoch`.
  Pretraining (`train`/`continue_train`) passes `None` → the existing
  `train_step` path, byte-for-byte unchanged; SFT passes `Some(&mask)` → the new
  `train_step_masked`. Inside `train_epoch` I switched the chunk loop to
  `.enumerate()` and slice the mask at `chunk_idx * ctx_window .. + chunk.len()`,
  i.e. the *same* `chunks(ctx_window)` boundaries as the corpus, so the two stay
  index-aligned (the epic's one hard invariant). The masked kernel itself only
  consumes `m[..s-1]` per chunk, so a full-length (per-token) mask slice is
  exactly what it wants.
- **`WARMUP_STEPS`:** left at 100 but documented at the const that on short SFT
  runs a fixed 100-step warmup can be a large fraction of the run (so the
  effective lr may never reach the requested `lr`). Left deliberately — a base
  model is already near a good minimum, so a conservative warming lr is a safe
  default, and changing it would perturb the well-tuned pretraining path. `lr`
  and `epochs` are caller params (doc'd starting points ~1e-4–5e-5, few epochs),
  not hardcoded.

**Why parameterize over the step fn instead of a masked sibling loop:** the whole
of `train_epoch`/`run_training_loop` (accumulation, clipping, warmup, progress
reporting, ETA, checkpointing, early-stop) is identical between the two modes —
only the per-chunk step differs. A sibling would duplicate ~120 lines that must
change together. Reported loss now naturally reflects only the masked (assistant)
positions, which is exactly what the training chart should show.

**Verified:**
- `cargo build -p llm` — clean, zero warnings.
- New GPU test `test_sft_train_masked_loss_decreases` (in `tests/train_test.rs`,
  the epic smoke-test seed): pretrains a tiny ctx-16 base, writes a 16-token
  masked fixture (first 4 positions mask 0 = scaffold, rest mask 1 = "assistant
  reply"), runs 8 `sft_train` epochs and asserts the masked-region avg loss
  decreases and that `out_dir/{model.bin,merges.bin}` both exist. Passed:
  loss fell monotonically 34.47 → 21.91 over the 8 epochs (`--test-threads=1`,
  sandbox disabled per the ISSUE-0016 cuBLAS note).

Existing pretraining tests untouched (the `None` path is unchanged). Note for the
Verifier: the run takes ~460s because the fixture pretrains a base over the full
`corpus.bin` first (same cost as `test_training_runs`), not because of SFT.

### Verifier · 2026-07-11 · pass

**What was checked:**
- Code review of `backend/src/llm/src/train.rs` (full file, 582 lines).
- Mask-threading trace: `sft_train` → `run_training_loop` → `train_epoch` → `train_step_masked`, all carrying `mask: Option<&[u8]>`.
- Flat-offset arithmetic in `train_epoch`: `chunk_idx * ctx_window` start, `chunk.len()` end — correct for all chunks including the trailing shorter one. Index-aligned with the same `chunks(ctx_window)` boundaries as the corpus. Validated that the length guard at the public boundary (`corpus.len() != mask.len()` → `InvalidData`) ensures the mask indexing can never OOB.
- R6: no loop duplication. `train`/`continue_train` pass `None` → `train_step` (unchanged). Only `sft_train` passes `Some(&mask)` → `train_step_masked`.
- R10: length guard at `sft_train` (line 210-219) validates corpus/mask alignment before any training.
- R9: `pub fn sft_train` has a full doc comment (parameters, return semantics, suggested LR/epoch starting points).
- R8: no new magic numbers; all constants are named consts with doc lines. `WARMUP_STEPS` left at 100 with SFT-specific rationale documented (line 27-32).
- R3: all fallible results use `?` in `train.rs`; test `.unwrap()` calls are in test code (acceptable).
- R1: `sft_train` body ~25 lines of logic, well under 60.
- Build: `cargo build -p llm` — clean, zero warnings.
- Clippy: `cargo clippy -p llm` — clean, zero warnings.
- Test reviewed: `test_sft_train_masked_loss_decreases` (lines 38-96 of `tests/train_test.rs`): builds tiny base, writes 16-token fixture with 4-byte scaffold mask + 12-byte train mask, runs 8 `sft_train` epochs, asserts `last < first` and that standalone model files exist. Test structure is correct, mask generation matches the ISSUE-0031 convention, and loss assertions are valid.

**Verdict:** The fix is correct, complete, and R6/R10 compliant. Parameterized mask-threading is clean, no regression in the pretraining path, and the test validates the invariant. Passing.

### VCS · 2026-07-11
Committed as `133dd22` on branch `dev`.
