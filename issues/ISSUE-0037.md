---
id: ISSUE-0037
title: Backend orchestration — TrainMode plus SFT route and job wiring
scope: backend
priority: P2
status: todo
rule: none
location: backend/src/training.rs:19-46 backend/src/training.rs:188-213 backend/src/routes/training.rs:15-46 backend/src/jobs.rs
blocked-by: [ISSUE-0034, ISSUE-0035]
blocks: [ISSUE-0038]
---

## Description
### What
Wire SFT into the run orchestration so a fine-tune can be launched like a training
run: add a training **mode** (pretrain vs SFT) and a `run_sft_training` that maps
storage paths onto `llm::sft_train`, mirroring how `run_training` wraps
`llm::train`.

### Why
ISSUE-0034 (dataset) and ISSUE-0035 (`sft_train`) give the pieces, but nothing
lets a request start an SFT run. The existing path — `start_training` route →
`JobRegistry` → `run_training` (`backend/src/training.rs:188`) — only knows how to
pretrain a fresh model.

### Shape of the fix
- **`TrainConfig` (`training.rs:21`):** add a `mode: TrainMode` field
  (`enum TrainMode { Pretrain, Sft }`, `#[serde(rename_all = "lowercase")]`,
  default `Pretrain` via `#[serde(default)]` so existing clients/`meta.json` keep
  working). For SFT, add an optional `base_model_id: Option<Uuid>` — the
  pretrained model to fine-tune from.
- **`run_sft_training`:** analogous to `run_training` — resolve
  `base_model_dir = models_dir()/base_model_id`, the SFT dataset paths
  (`datasets/sft_corpus.bin`, `datasets/sft_mask.bin`), and a fresh
  `out_model_dir` for the fine-tuned model + its checkpoints, then call
  `llm::sft_train(...)`. Also add a `build_sft_dataset` invocation path (like
  `build_dataset` is used for pretrain) so the tokenize→train sequence has an SFT
  analogue.
- **Route (`routes/training.rs`):** branch `start_training` on `cfg.mode`.
  Validate SFT requests: `base_model_id` must be present and must resolve to an
  existing model dir; return `400` otherwise (follow the existing
  `validate_config` style + `backend/CLAUDE.md` error-handling rules — generic
  client message, real error logged server-side). The job progress/WS plumbing is
  mode-agnostic and should be reused unchanged.
- **`meta.json`:** the run record already serializes `TrainConfig`, so `mode` and
  `base_model_id` land there automatically — good for lineage (which base a
  fine-tune came from).

### Verification
- `cargo build` (workspace) clean, no warnings.
- A pretrain request with no `mode` field still deserializes and runs exactly as
  before (back-compat).
- An SFT request without a valid `base_model_id` is rejected with `400`, not a
  `500` or a panic.

## Comments
### Auditor · 2026-07-10
The integration layer that makes SFT launchable. Blocked-by 0034 and 0035 (it
calls both). The back-compat requirement — a `mode`-less request must still
pretrain — is the thing most likely to regress, so it's called out for the
Verifier. Kept in the `backend` crate per the layering rules in
`backend/CLAUDE.md` (composite operations that span dataset + model live in the
orchestration/route layer, not in the pure `llm`/`tokenize` crates).

### VCS · 2026-07-10
Committed as `d5b9a57` on branch `dev`.
