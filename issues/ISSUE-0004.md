---
id: ISSUE-0004
title: Split train_step — it is ~440 lines (R1)
scope: backend/llm
priority: P1
status: todo
rule: R1
location: backend/src/llm/src/training/transformer.rs:409-849
blocked-by: []
blocks: []
---

## Description
`train_step` spans ~440 lines doing embedding forward → per-layer forward pass
through all layers → logits + loss → backward pass (layers in reverse) → gradient
clipping → embedding gradient accumulation in one body. Each stage block currently
prefixed with `// ── N. … ──` should be a named helper (e.g. `embedding_forward`,
`forward_pass`, `compute_loss`, `backward_pass`, `clip_gradients`).

## Comments
_No comments yet._
