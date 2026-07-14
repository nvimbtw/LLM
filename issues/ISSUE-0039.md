---
id: ISSUE-0039
title: Split build_sft_dataset — ~90 lines exceeds R1 ~60-line guideline
scope: backend
priority: P3
status: todo
rule: R1
location: backend/src/training.rs:369-460
blocked-by: []
blocks: []
---

## Description

`build_sft_dataset` (lines 369–460) has a body of ~90 lines (80 non-whitespace lines), exceeding the R1 ~60-line guideline by ~50%. The function does three distinct things in one body:

1. **Load/prepare the base tokenizer** — load merges, encode the `<|endoftext|>` boundary token (lines 370–376).
2. **Collect and validate SFT files** — list `.jsonl` files in the SFT dir, sort them, reject empty dirs (lines 378–394).
3. **Process each conversation** — parse JSON, validate roles, render through `build_training_example`, accumulate `corpus` + `mask` vectors (lines 400–440).
4. **Guard empty output, write files, compute stats** — corpus-empty check, write `sft_corpus.bin` and `sft_mask.bin`, count trainable tokens (lines 442–459).

### Shape of the fix

Extract step 2 into a small named helper and step 3 into a second helper, keeping `build_sft_dataset` as a short orchestrator:

```rust
/// Collect and sort `.jsonl` files from the SFT dir. Returns an error if
/// the dir is missing or contains no `.jsonl` files.
fn collect_sft_files() -> io::Result<Vec<PathBuf>> { ... }

/// Parse every conversation in `files` through the base tokenizer and
/// render it into aligned `(corpus, mask)` vectors, inserting
/// `<|endoftext|>` boundary tokens between conversations.
fn render_conversations(
    files: &[PathBuf],
    merges: &[((u32, u32), u32)],
    eot: &[u32],
) -> io::Result<(Vec<u32>, Vec<u8>, usize)> { ... }
```

`build_sft_dataset` would then become ~15 lines — load merges, encode EOT, call `collect_sft_files`, call `render_conversations`, guard empty, write output, return stats. Each extracted helper is well under 60 lines.

Alternatively, if the three-line `render_conversations` extraction seems too small a boundary, a single `collect_and_render_sft_files` helper (combining steps 2+3) would still cut the parent by half. The key point is that the file-listing, the JSON parsing/role-validation, and the corpus-building loop should not share a function with the merges-loading and output-writing.

### Why not a blocker

P3 — readability only, no correctness risk. All existing code paths, tests, and the SFT training pipeline are unaffected by the refactor.

## Comments
### Auditor · 2026-07-11

Flagged by the Verifier in ISSUE-0037 (observation) as pre-existing from ISSUE-0034: `build_sft_dataset` was introduced in that issue and never refactored. Not filing against ISSUE-0034 since it's already `done`; this is a standalone cleanup issue.
