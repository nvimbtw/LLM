---
id: ISSUE-0081
type: issue
title: "Extract the backward pass into transformer/backward.rs"
scope: backend/llm
priority: P2
status: verification
rule: R20
location: backend/src/llm/src/training/transformer/mod.rs:807-1028
blocked-by: [ISSUE-0080]
blocks: [ISSUE-0082]
---

## Description
### What
Move the backward-pass stage helpers into a new `transformer/backward.rs`,
mirroring 0080. The `train_step_masked` orchestrator stays in `mod.rs` and calls
these across the boundary.

### What moves (current line ranges)
- `backward_unembed` (~807), `ffn_sublayer_backward` (~835), `attn_sublayer_backward`
  (~905), `clip_gradients` (~990, the per-step model-level clip method), and
  `accumulate_embedding_grad` (~1013–1028).
- Note: this `Model::clip_gradients` method is distinct from the free function
  `gpu_clip_gradients` (~1528), which moves to `optimizer.rs` in 0082 — do not
  conflate them.

### Shape of the change
- New file `transformer/backward.rs` with an `impl Model { … }` block; add
  `mod backward;` to `transformer/mod.rs`.
- **Visibility:** moved helpers become `pub(super)` (called by `train_step_masked`
  in `mod.rs`). No `pub`. Import shared items via `use super::…`.
- Pure move + visibility; no gradient-math change.

### Verification
- `cargo build` + `cargo clippy` clean; full GPU suite passes `--test-threads=1`
  (`test_embedding_backward_gpu`, `test_gpu_sgd_does_not_corrupt_weights`,
  `test_overfits_single_sequence`, masked-gradient tests) — these are the guard
  that the backward move preserved behaviour.

## Comments
### Auditor · 2026-07-14
Symmetric with 0080. Called out the `clip_gradients` (method, stays adjacent to
backward) vs `gpu_clip_gradients` (free fn, goes to optimizer) distinction so the
Builder doesn't move the wrong one.
