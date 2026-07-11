---
id: ISSUE-0037
title: Backend orchestration — TrainMode plus SFT route and job wiring
scope: backend
priority: P2
status: done
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

### Builder · 2026-07-11
Wired SFT into the run orchestration across `training.rs`, `jobs.rs`, and
`routes/training.rs`.

**What changed:**
- **`TrainConfig` (`training.rs`):** added `enum TrainMode { Pretrain, Sft }`
  (`#[serde(rename_all = "lowercase")]`, `#[derive(Default)]` → `Pretrain`) plus
  two `#[serde(default)]` fields, `mode: TrainMode` and
  `base_model_id: Option<Uuid>`. `serde(default)` on both is the back-compat
  hinge — a pre-existing body/`meta.json` with neither key deserializes as a
  pretrain run unchanged.
- **`run_sft_training` (`training.rs`):** mirrors `run_training`, but resolves
  `base_model_dir` from `base_model_id`, the SFT dataset paths
  (`datasets/sft_corpus.bin` + `sft_mask.bin`), and a fresh `out_model_dir` from
  `storage_id`, then calls `llm::train::sft_train`. Guards `base_model_id`
  presence with an `InvalidInput` error rather than trusting the route invariant
  (the crate is also reachable from tests/future callers).
- **`jobs.rs`:** split `run_job` into a dispatch (`match config.mode`) over
  `run_pretrain_job` (the old body, unchanged) and a new `run_sft_job`. The SFT
  worker runs phase 1 = `build_sft_dataset(&base_model_dir)` (emits a `tokenize`
  frame carrying conversation/token/trainable counts, flips status to `Training`),
  phase 2 = `run_sft_training`. Both modes share progress reporting and a new
  `finish_outcome` helper that maps the `io::Result<bool>` onto `finish` (R6 — the
  cancelled/done/failed mapping now lives in one place instead of being duplicated
  per mode).
- **Meta generalization (R6):** rather than a second meta path, `write_meta` and
  `RunMeta.dataset` now take an already-serialized `serde_json::Value`, so one
  `meta.json` shape serves both `DatasetStats` (pretrain) and `SftDatasetStats`
  (SFT). `finish` takes `&serde_json::Value`; each worker does `json!(stats)` at
  its call site. Chose a `Value` over a generic `<D: Serialize>` deliberately —
  `backend/CLAUDE.md` prefers the direct option over trait/generic machinery.
- **Route (`routes/training.rs`):** added `validate_sft` (a no-op for pretrain, so
  the call site stays a single unconditional check — avoids the collapsible-`if`
  clippy lint). For SFT it requires `base_model_id` present and resolving to an
  on-disk `model.bin`, returning `400` (client error) — never a `500`/panic — per
  the `backend/CLAUDE.md` error rules. Documented that the model hyperparameters
  are ignored for SFT (loaded from the base) but still pass `validate_config`; the
  frontend (0038) mirrors the base's values into the request.
- Removed the five `#[allow(dead_code)]` markers on the 0034 SFT items
  (`SftTurn`, `SftConversation`, `SftDatasetStats`, `parse_role`,
  `build_sft_dataset`) — now genuinely reachable, so the allows would themselves
  become stale (and R7 wants no unjustified allows).

**Verified:**
- `cargo build` (workspace) — clean, zero warnings.
- `cargo clippy` — clean (fixed one collapsible-`if` it flagged).
- `cargo test -p backend training::sft_tests` — 4/4 pass, including two new
  deserialization tests: `test_trainconfig_defaults_to_pretrain_without_mode`
  (a `mode`-less body → `Pretrain`, `base_model_id: None` — the exact back-compat
  regression the Auditor flagged) and `test_trainconfig_parses_sft_mode`
  (`"mode":"sft"` + a base id round-trip). The existing dataset test still passes.
- SFT rejection path (`validate_sft`) is pure fs/param logic returning `400`; not
  worth a full HTTP integration test here, but exercised by inspection: missing
  `base_model_id` → "required", non-resolving id → "does not resolve".

The GPU end-to-end SFT run itself is exercised by ISSUE-0035's
`test_sft_train_masked_loss_decreases`; this issue is the wiring above it.

### Verifier · 2026-07-11 · pass

**Verdict:** All checks pass — wiring is correct, complete, and clean.

**Verified:**

1. **`cargo build` (workspace):** clean, zero warnings.
2. **`cargo clippy`:** clean, no issues.
3. **`cargo test -p backend training::sft_tests`:** 4/4 pass
   (`test_build_sft_dataset_aligns_and_counts`,
   `test_trainconfig_defaults_to_pretrain_without_mode`,
   `test_trainconfig_parses_sft_mode`,
   `test_build_sft_dataset_rejects_unknown_role`).

**Code review — changes introduced by this issue:**

| File | What was checked | Verdict |
|---|---|---|
| `training.rs:19-28` | `TrainMode` enum — `Pretrain` is `#[default]`, `#[serde(rename_all = "lowercase")]` | ✅ |
| `training.rs:30-66` | `TrainConfig` — `mode` + `base_model_id` with `#[serde(default)]` for back-compat | ✅ |
| `training.rs:241-268` | `run_sft_training` — guards `base_model_id`, maps storage paths, calls `sft_train` | ✅ |
| `jobs.rs:185-189` | `run_job` dispatch on `config.mode` | ✅ |
| `jobs.rs:253-304` | `run_sft_job` — phase 1 `build_sft_dataset`, phase 2 `run_sft_training`, shares `finish_outcome` | ✅ |
| `jobs.rs:308-327` | `finish_outcome` helper (R6 dedup of terminal-status mapping) | ✅ |
| `routes/training.rs:288-298` | `validate_sft` — no-op for pretrain, 400 for missing/invalid `base_model_id` | ✅ |
| `routes/training.rs:17-46` | `start_training` calls both `validate_config` and `validate_sft` | ✅ |

**Back-compat confirmed:** A `mode`-less body defaults to `Pretrain` with `base_model_id: None` via `#[serde(default)]`. Test `test_trainconfig_defaults_to_pretrain_without_mode` passes (R10/R11 clean — no panics on external input).

**Rule check:** All new/changed code respects R1–R13. `#[allow(dead_code)]` correctly removed from the five ISSUE-0034 items now that they are reachable; the removed allows would themselves violate R7 if left.

**Observation (no bounce needed):** `build_sft_dataset` (~90 lines body) exceeds R1's 60-line guideline. This is pre-existing from ISSUE-0034 — this issue only removed its `#[allow(dead_code)]` and did not modify its structure. A separate Auditor finding may be warranted if the length is deemed actionable.
