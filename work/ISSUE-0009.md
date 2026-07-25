---
id: ISSUE-0009
type: issue
title: Unsafe blocks in train_step lack safety comments (R4)
scope: backend/llm
priority: P2
status: todo
rule: R4
location: backend/src/llm/src/training/transformer.rs:409-849
blocked-by: []
blocks: []
---

## Description
Every `unsafe { ... }` block in `train_step` (~40 instances across 440 lines)
invokes a CUDA FFI function with no comment explaining why the call is sound —
which invariants the caller guarantees (buffer sizes, alignment, GPU state). R4
requires each unsafe block to carry a short justification. E.g. `// safe: all d_*
buffers are allocated at max_s*d*f16 and s ≤ max_s`.

## Comments
_No comments yet._
