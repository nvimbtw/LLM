---
id: ISSUE-0061
title: "Rename training::transformer::Transformer to training::transformer::Model"
scope: backend/llm
priority: P2
status: verification
rule: R21
location: backend/src/llm/src/training/transformer.rs:114
blocked-by: []
blocks: [ISSUE-0060]
---

## Description

`training::transformer::Transformer` stutters — the module segment `transformer` adds nothing beyond the type name. Per **R21**, rename the struct to `Model` so the path reads `training::transformer::Model`, which is self-documenting: this module's `Model` is the decoder-only transformer model.

### Scope of change
A mechanical rename across these files that reference `Transformer`:

| File | Usage |
|------|-------|
| `backend/src/llm/src/training/transformer.rs` | `pub struct Transformer` definition, all `impl Transformer` blocks, helper `fn write_model(t: &Transformer, ...)` |
| `backend/src/llm/src/training/gpu_workspace.rs` | `use crate::training::transformer::Transformer;` — used in `GpuWorkspace::new(t: &Transformer)` and `upload_weights` |
| `backend/src/llm/src/train.rs` | All entry points create/accept `Transformer` |
| `backend/src/llm/src/infer.rs` | `use crate::training::transformer::Transformer;` — `CachedModel` holds a `Transformer` |
| `backend/src/llm/src/training/mod.rs` | Re-exports (if any — check after rename) |
| `backend/src/...` | Any other `use` referencing `transformer::Transformer` |

### Shape of the fix
1. `s/Transformer/Model/g` in `transformer.rs` (struct def, impl blocks, `from_bytes` return, `read_model` return, etc.)
2. Update `use` declarations in `gpu_workspace.rs`, `train.rs`, `infer.rs` to import `transformer::Model` instead of `transformer::Transformer`
3. Where the struct is referenced by name (e.g. `let t = Transformer::init(...)`) rename to `let model = Model::init(...)` — or keep the variable name `t` and just change the type

### Verification
- `cargo build -p llm` passes with zero warnings
- All existing tests still pass (`cargo test -p llm`)
- No references to `transformer::Transformer` remain in any source file

## Comments
### Builder · 2026-07-14
Renamed `Transformer` -> `Model` (word-boundary, whole crate incl. tests/train_test.rs). No residual `transformer::Transformer` refs; `ModelHeader`/`ModelFileHeader`/`CachedModel` untouched (distinct symbols). `cargo build`+`clippy` clean, 45/45 lib GPU tests pass (`--test-threads=1`).
