---
id: ISSUE-0035
title: SFT training loop and sft_train entry point (fine-tune from a base model)
scope: backend/llm
priority: P2
status: todo
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
