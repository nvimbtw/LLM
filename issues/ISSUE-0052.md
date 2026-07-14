---
id: ISSUE-0052
title: "Epic: Mixed-precision training — f32 master weights (f16 compute) to lift the LR cap"
scope: backend/llm
priority: P1
status: todo
rule: none
location: backend/src/llm/src/training/gpu_workspace.rs backend/src/llm/src/training/shaders/shaders.cu backend/src/llm/src/training/transformer.rs backend/src/llm/src/train.rs
blocked-by: [ISSUE-0053, ISSUE-0054, ISSUE-0055, ISSUE-0056, ISSUE-0057, ISSUE-0058]
blocks: []
---

## Description
**Tracking/epic issue — no code lives here directly.** It is `blocked-by` the
child chain 0053–0058; the Builder picks up the children in dependency order and
closes THIS issue by running the end-to-end LR-cap verification (see "Definition
of done"). This is the designed output of the meta-issue **ISSUE-0050** — read
that issue first for the full root-cause writeup; this epic is the plan.

### The goal
Introduce a **f32 master copy of every trained weight** while keeping **all
compute in f16**. Adam updates the f32 master (`w_f32 += Δw`); a cast refreshes
the disposable f16 "view" the matmuls consume. This removes the swamping that
today caps the usable learning rate at ~1e-5 and leaves from-scratch pretrains
badly undertrained.

### Why now / the root cause (settled, from ISSUE-0050)
Every trained weight is a `DeviceBufferF16`; `adam_kernel` reads `__half *w`,
converts to f32, applies the update, and writes back `__half` — so the **master
value lives in f16**. Near `w ≈ 1.0` the f16 gap is ~1e-3, so an Adam step of
order `Δw ≈ 1e-4` rounds away entirely (`w + Δw == w`). At small LR the update
vanishes; at larger LR the weight lurches in coarse f16 quanta and diverges
("kaboom" at 3e-4). Gradients and Adam moments (`m`, `v`) are **already f32**;
the stabilizers (grad-norm clip `MAX_GRAD_NORM=1.0`, 100-step LR warmup,
max-subtracted softmax) are present and are **not** the cause. The one missing
piece is the f32 master.

### The design (child issues)
| # | Issue | Layer | Depends on |
| --- | --- | --- | --- |
| 0053 | f32 master buffer per trained weight in `GpuWorkspace` | backend/llm | — |
| 0054 | `adam_kernel`: update the f32 master + refresh the f16 view | backend/llm | — |
| 0055 | Thread master pointers through `gpu_adam_update` | backend/llm | 0053, 0054 |
| 0056 | Seed masters on load/init from f32; save from the master | backend/llm | 0053, 0055 |
| 0057 | Verification: an LR that previously diverged now trains | backend/llm | 0056 |
| 0058 | Mode-aware LR defaults (pretrain ~1e-4, SFT ~1e-5) | backend | 0057 |

(ISSUE-0059 — smaller default vocab for small corpora — is a related cheap win
split out as *independent* of this epic; it is not in the `blocked-by` chain
because it neither depends on nor gates the master-weight fix.)

### The one invariant every child shares
The **f32 master is the source of truth** for a trained parameter; the
`DeviceBufferF16` weight is a **derived, disposable view** regenerated from the
master every optimizer step. No code path may treat the f16 weight as canonical
after 0056 lands: uploads seed the master (not just the view), the optimizer
writes the master (and refreshes the view in the same step), and save reads the
master. Positional embeddings and any other *non-updated* buffers keep no master
— only the 2 shared + 10×layers weights that `gpu_adam_update` actually steps.

### Memory cost (from ISSUE-0050)
Per trained parameter: f16 `w` (2B) + f32 `m` + f32 `v` (8B) = 10B today; the
f32 master adds +4B → **14B/param** (~+40 MB on a 10M-param model). Compute
throughput is unchanged (still f16 matmuls).

### Open questions the children resolve
- **Fused vs. separate cast** (0054): fuse the f16 write into the Adam kernel's
  tail (one launch, one read/write of the view) rather than a second global
  `f32_to_f16` pass. Decision: **fuse** — `f32_to_f16_kernel` already exists for
  any path that needs a standalone cast, but the per-step refresh rides the Adam
  kernel to avoid an extra launch per weight.
- **Loss scaling** (0057 confirms): grads + moments are already f32 and grad-norm
  is clipped, so no loss-scaling is expected to be needed. The verification child
  confirms this empirically rather than assuming it.
- **Save format** (0056): keep persisting **f32** weights (the on-disk format is
  already f32 — `sync_weights_from_gpu` widens f16→f32 today). Reading the master
  instead of the f16 view makes save *lossless* and removes the f16 re-quantize
  round-trip that the SFT epic (ISSUE-0030) flagged. No on-disk format change, so
  existing saved models load unchanged.

### Non-goals
- No change to the wire protocol, job model, or frontend.
- Not rewriting the trainer back to full f32 — **compute stays f16 by design**.
- Not touching non-updated buffers (positional table, activation scratch).

### Definition of done (what the Builder verifies to close THIS issue)
1. All children 0053–0058 are `done`.
2. The 0057 verification run demonstrates the LR cap is lifted: a short run at an
   LR that provably diverged on `dev` before the fix (target 1e-4, ideally 3e-4)
   now shows a monotone-ish decreasing loss instead of NaN/blow-up, and the
   existing GPU trainer tests (`train_test.rs`, `transformer::tests`) still pass
   (`--test-threads=1`, sandbox disabled — the cuBLAS single-thread constraint
   from ISSUE-0016 still applies).
3. `backend/CLAUDE.md` gains a short "Mixed-precision (f32 master weights)" note:
   the master-is-truth invariant, the per-step cast, and the 14B/param budget.

## Comments
### Auditor · 2026-07-13 · epic design
Filed as the designed epic requested by ISSUE-0050. Decomposed to mirror
ISSUE-0030/0040: a storage child, a kernel child, a wiring child, an
init/load/save child, a verification child, and a defaults child, each with a
single location and its own DoD. The chain is strictly ordered
(storage+kernel → wiring → seed/save → verify → defaults) so each child is one
Builder→Verifier round. The vocab-default cheap win is deliberately **not** in
the chain (ISSUE-0059, independent) so it can't gate the correctness fix.
Root cause is settled in ISSUE-0050, so the children start from a known cause.
