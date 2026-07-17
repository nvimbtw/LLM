---
id: ISSUE-0072
type: issue
title: "R19 phase-marker comments in jobs.rs and routes/training.rs signal extraction need"
scope: backend
priority: P2
status: todo
rule: R19
location: backend/src/jobs.rs:193 backend/src/jobs.rs:211 backend/src/jobs.rs:242 backend/src/jobs.rs:291 backend/src/jobs.rs:316 backend/src/jobs.rs:508 backend/src/routes/training.rs:103 backend/src/routes/training.rs:129 backend/src/routes/training.rs:157
blocked-by: []
blocks: []
---

## Description

Phase-marker comments in `jobs.rs` and `routes/training.rs` narrate *what* the next block of code does — violating **R19 (comments earn their place)**. These comments are symptoms of functions that do more than one thing (R13) — they phase-comment a multi-step function instead of extracting each step.

### Locations

#### `jobs.rs`
| Line | Comment | Context |
|------|---------|---------|
| 193 | `// ── worker ──` | Section banner before the worker dispatch functions. The functions below are already named (`run_job`, `run_pretrain_job`, `run_sft_job`). |
| 211 | `// ---- Phase 1: tokenize / build the dataset ----` | Inside `run_pretrain_job`: marks the tokenization phase. The function already has a name indicating what it does. |
| 242 | `// ---- Phase 2: train ----` | Inside `run_pretrain_job`: marks the training phase. |
| 291 | `// ---- Phase 1: render conversations → sft_corpus.bin + sft_mask.bin ----` | Inside `run_sft_job`: marks the dataset-building phase. |
| 316 | `// ---- Phase 2: fine-tune from the base ----` | Inside `run_sft_job`: marks the fine-tuning phase. |
| 508 | `// ── small helpers ──` | Section banner before small utility functions at the bottom of the file. |

#### `routes/training.rs`
| Line | Comment | Context |
|------|---------|---------|
| 103 | `// ---- authenticate on the first text frame ----` | Inside `training_session`: marks the authentication loop. |
| 129 | `// ---- attach frame: { "job_id": "<uuid>" } ----` | Inside `training_session`: marks the attach loop. |
| 157 | `// ---- catch-up snapshot, then live frames ----` | Inside `training_session`: marks the snapshot/frame-sending section. |

### Why this matters

Each `training_session` function in `routes/training.rs` and each phase in `jobs.rs` is doing multiple distinct things separated by a comment instead of by function boundaries. Per R13 ("if you can only describe what a function does with an 'and', it should be two functions"), these should be extracted.

### Shape of the fix

1. **`jobs.rs`**: Extract the two phases of `run_pretrain_job` and `run_sft_job` into named helper functions (e.g., `build_dataset_for_job` and `run_training_for_job`). The section banners at 193 and 508 can be deleted once extraction is done.

2. **`routes/training.rs`**: Extract the three stages of `training_session` into named functions (e.g., `authenticate_session`, `attach_to_job`, `stream_frames`). Delete the phase-marker comments.

### Verification
- The extracted functions have self-documenting names that make the phase markers unnecessary
- No `// ---- Phase ... ----` comments remain
- `cargo build` compiles clean
- All existing tests pass

## Comments
_No comments yet._
