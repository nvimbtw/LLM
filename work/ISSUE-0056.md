---
id: ISSUE-0056
type: issue
title: "Seed f32 masters on load/init from f32; save from the master (lossless round-trip)"
scope: backend/llm
priority: P1
status: todo
rule: none
location: backend/src/llm/src/training/gpu_workspace.rs:225-264 backend/src/llm/src/training/transformer.rs:1591
blocked-by: [ISSUE-0053, ISSUE-0055]
blocks: [ISSUE-0057]
---

## Description
### What
Make the f32 master the source of truth end-to-end: **seed** it from the f32 CPU
weights when weights are uploaded, and **read it back** (not the f16 view) when
weights are synced to CPU for save. This is what turns 0053–0055 from inert
plumbing into a working, lossless mixed-precision path.

### Seed on upload (`GpuWorkspace::upload_weights`, gpu_workspace.rs:225)
`upload_weights` already flattens each f32 CPU weight and calls
`d_*.upload_f32_slice(...)` (which quantizes to f16). For every **trained** weight
add a parallel `m_*.from_slice(&flat)` / `m_*.upload(&flat)` that stores the
**full-precision f32** into the master — so the master is seeded from the original
f32 values, *not* from the f16 view (strictly better fidelity than the
"seed masters from f16" phrasing in ISSUE-0050). Keep uploading the f16 view too
(the first forward pass before any Adam step reads the view).
- This covers both entry paths for free, since both pretrain init and the SFT base
  load funnel through `upload_weights` (`train.rs` calls it after
  `Transformer::load`/`init`). Confirm that is true for the SFT base path and note
  it for the Verifier; if the SFT base uploads by a different route, seed there
  too.
- `d_positional` keeps no master (unchanged, still f16-only, never stepped).

### Save from the master (`sync_weights_from_gpu`, transformer.rs:1591)
Switch each trained weight's read-back from `ws.d_*.to_f32_vec()` (widens the f16
view) to `ws.m_*.to_vec()` (the exact f32 master). This removes the f16
re-quantize round-trip on save that EPIC-0030's closure comment flagged
(SFT re-quantized even a zero-gradient base). The on-disk format is already f32,
so **no format change and no back-compat break** — older saved models still load
(they seed the master via `upload_weights` on next train). `d_positional` (not
updated) still reads from the f16 view as today.

### Verification
- `cargo build` + `cargo clippy` clean; GPU tests pass (`--test-threads=1`).
- Targeted assertion (extend `transformer::tests`): after `upload_weights` then
  `gpu_adam_update` with an all-zero gradient, `sync_weights_from_gpu` returns
  weights **bit-identical** to the uploaded f32 input (no f16 round-trip) — the
  property EPIC-0030 could not assert because the master didn't exist. (Bias
  correction makes step-1 `m_hat` nonzero only if the gradient is nonzero; with a
  true zero gradient the update is exactly 0, so the master is untouched.)

## Comments
### Auditor · 2026-07-13
The child that makes the master canonical on both ends. Deliberately seeds from
the f32 CPU weights (better than seeding from the f16 view) and reads the master
on save, which as a bonus closes the f16-round-trip gap EPIC-0030 documented.
Flagged the SFT base-load path for the Verifier to confirm it funnels through
`upload_weights`.
