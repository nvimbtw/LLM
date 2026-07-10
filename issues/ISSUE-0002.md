---
id: ISSUE-0002
title: Remove dead cuda_sgd_momentum FFI (or wire it up)
scope: backend/llm
priority: P3
status: done
rule: R7
location: backend/src/llm/src/training/transformer.rs:1240
blocked-by: []
blocks: []
---

## Description
`cuda_sgd_momentum` is declared in the `extern "C"` block but never called (the
optimizer moved to Adam), producing a standing `dead_code` warning. R7 requires a
clean build. Either delete the declaration (and the kernel if unused elsewhere) or
add a justified `#[allow(dead_code)]` with a note if it's kept intentionally.

## Comments
### Verifier · migrated · pass
Confirmed `cuda_sgd_momentum` removed from the `extern "C"` block in
transformer.rs. `cargo build` shows no related warning.
