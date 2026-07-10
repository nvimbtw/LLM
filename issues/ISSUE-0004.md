---
id: ISSUE-0004
title: Split train_step — it is ~440 lines (R1)
scope: backend/llm
priority: P1
status: done
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
### Builder · 2026-07-10
The monolithic ~440-line `train_step` is already gone in the current tree — the
split landed as part of commit `124f56c` (the Transformer code-quality batch) but
this issue was never moved off `todo`, so I'm closing it out here rather than
re-editing code that already satisfies R1.

What the code looks like now (`backend/src/llm/src/training/transformer.rs`):
`train_step` (lines 426–454) is ~28 lines — it computes `StepDims` once, then
drives the pipeline purely by delegation:

- `embed_forward` — embedding lookup + positional add → initial hidden state.
- per-layer forward loop: `attn_sublayer_forward` then `ffn_sublayer_forward`.
- `project_logits` → `compute_loss` (cross-entropy + its logit gradient).
- `backward_unembed`, then a reverse per-layer loop of `ffn_sublayer_backward`
  then `attn_sublayer_backward`.
- `clip_gradients` and `accumulate_embedding_grad` close out the backward pass.

Each stage the issue called out (`embedding_forward`, `forward_pass`,
`compute_loss`, `backward_pass`, `clip_gradients`) now has a named helper; the
split is actually finer-grained than requested — the forward/backward passes are
broken at the attention-vs-feed-forward sublayer boundary, which is the real
semantic seam.

On helper length vs R1: `attn_sublayer_forward` (~117 lines) and
`attn_sublayer_backward` (~81) run over 60 lines, but each is a single cohesive
sublayer stage dominated by one-arg-per-line `unsafe extern "C"` CUDA calls. Per
the R1 *FFI caveat* these are judged by logical-step count, not raw lines, and
split at real semantic boundaries (attention sublayer vs. feed-forward sublayer)
— so they satisfy R1 as-is. The shared safety contract for the `unsafe` blocks is
documented once in the comment block above `embed_forward` (lines 456–464).

Ran: `cargo build -p llm` — finished clean, zero warnings.

### Verifier · 2026-07-10 · pass

Confirmed `train_step` is now ~28 lines (426–454) and delegates to named helpers:
`embed_forward`, `attn_sublayer_forward`, `ffn_sublayer_forward`, `project_logits`,
`compute_loss`, `backward_unembed`, `ffn_sublayer_backward`, `attn_sublayer_backward`,
`clip_gradients`, `accumulate_embedding_grad`.  Each stage requested in the description
has a matching named helper; helpers >60 lines (`attn_sublayer_forward`, `attn_sublayer_backward`)
are justified by the R1 FFI caveat (cohesive CUDA-call stages split at real semantic
boundaries).  Checked: safety contract at lines 456–464 covers all `unsafe` blocks (R4);
no new unwraps/expects in the split code (R3); constants named (`LAYER_NORM_EPS`,
`TRAIN_STEP_CLIP_NORM`, etc. — R8); all 19 training tests pass including the critical
`test_overfits_single_sequence` (learning works end-to-end); `cargo build -p llm`
and `cargo clippy -p llm` clean (pre-existing `cuda_sgd_momentum` warning is a known
separate issue, ISSUE-0002).  No regression introduced by the refactor.
