---
id: ISSUE-0080
type: issue
title: "Extract the forward pass into transformer/forward.rs"
scope: backend/llm
priority: P2
status: verification
rule: R20
location: backend/src/llm/src/training/transformer/mod.rs:494-806
blocked-by: [ISSUE-0079]
blocks: [ISSUE-0081]
---

## Description
### What
Move the forward-pass stage helpers into a new `transformer/forward.rs`, leaving
the `train_step` / `train_step_masked` **orchestrator** in `mod.rs` (it drives
forward → loss → backward → clip → accumulate, so it belongs with the model, not
inside one pass).

### What moves (current line ranges)
- `embed_forward` (~494), `attn_sublayer_forward` (~521), `ffn_sublayer_forward`
  (~642), `project_logits` (~702), `compute_loss` (~721–806).

### What stays in mod.rs
- `train_step` (~426), `train_step_masked` (~434), the `StepDims` struct (~20), and
  the `// ── train_step stage helpers ──` banner region header (delete the banner
  per R19 once the helpers move — the file boundary now carries that meaning).

### Shape of the change
- New file `transformer/forward.rs` with an `impl Model { … }` block holding the
  moved helpers; add `mod forward;` to `transformer/mod.rs`.
- **Visibility:** each moved helper becomes `pub(super)` so `train_step_masked` in
  `mod.rs` can call it across the file boundary. Do not make them `pub`. Any free
  item they need from `mod.rs` (e.g. `StepDims`, constants) gets `pub(super)` or a
  `use super::…` import as appropriate.
- No arithmetic change — this is a move + visibility adjustment only.

### Verification
- `cargo build` + `cargo clippy` clean; full GPU suite passes `--test-threads=1`
  (`test_train_step_loss_reasonable`, `test_embedding_forward_cpu_vs_gpu`,
  `test_overfits_single_sequence`, the masked-forward tests).

## Comments
### Auditor · 2026-07-14
Kept the step orchestrator in `mod.rs` on purpose — it spans forward+backward, so
it is the model's step driver, not part of the forward file. The `pub(super)`
contract is the crux of this child.
