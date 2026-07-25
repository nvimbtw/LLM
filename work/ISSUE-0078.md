---
id: ISSUE-0078
type: issue
title: "transformer.rs → transformer/mod.rs (directory module form, no logic change)"
scope: backend/llm
priority: P2
status: verification
rule: R20
location: backend/src/llm/src/training/transformer.rs backend/src/llm/src/training/mod.rs:4
blocked-by: [EPIC-0060]
blocks: [ISSUE-0079]
---

## Description
### What
Convert the single file `training/transformer.rs` into a **directory module**
`training/transformer/mod.rs` with its contents byte-identical. This is pure
scaffolding: it creates the module folder the later children drop sibling files
into, and it must leave behaviour and the public path (`training::transformer::*`)
completely unchanged.

### Shape of the change
- `git mv backend/src/llm/src/training/transformer.rs
  backend/src/llm/src/training/transformer/mod.rs` (move, don't retype — preserve
  history and content).
- `training/mod.rs` already declares `pub mod transformer;` (line 4) — a directory
  module resolves through `transformer/mod.rs`, so **no change is needed there**;
  confirm it still compiles without edits.
- Do not move, rename, or re-split any item in this child. The `Transformer`/`Model`
  struct, the free functions, and the `#[cfg(test)] mod tests` all stay in
  `mod.rs` for now; 0079–0083 carve them out.

### Out of scope
- No extraction, no visibility changes, no logic changes — those are 0079+.

### Verification
- `cargo build` + `cargo clippy` clean from `backend/`.
- The full GPU suite passes unchanged (`--test-threads=1`).
- `git status` shows a rename (R) of `transformer.rs` → `transformer/mod.rs`, no
  content diff.

## Comments
### Auditor · 2026-07-14
First child of EPIC-0077 — the inert directory conversion, so every later
extraction is a sibling-file add rather than also inventing the module. Gated on
EPIC-0060 so the rename lands before the file is carved up.
