---
id: ISSUE-0050
type: issue
title: "Design the epic: mixed-precision (f32 master weights) CUDA training"
scope: backend/llm
priority: P1
status: done
rule: none
location: backend/src/llm/src/training/gpu_workspace.rs backend/src/llm/src/training/shaders/shaders.cu backend/src/llm/src/train.rs
blocked-by: []
blocks: [EPIC-0052]
---

## Description

**Meta / design issue.** The deliverable of THIS issue is *not* code — it is a
designed epic (a parent tracking issue + a `blocked-by` chain of small,
independently-verifiable children), filed in the same shape as EPIC-0030 and
EPIC-0040. Do not implement the CUDA change under this id; produce the plan the
Builder will execute.

### The problem this epic must solve

Training is numerically fragile: the learning rate can only be pushed to ~1e-5
before the loss diverges ("kaboom" at 3e-4). A from-scratch pretrain at a survivable
LR is badly undertrained (word-salad output). The root cause is understood:

- The model was originally f32 everywhere, then switched to **f16 everywhere** for
  speed/memory. Compute in f16 is fine.
- But the **weights themselves are stored and updated in f16** — every weight is a
  `DeviceBufferF16`, and `adam_kernel` writes back into `__half *w`
  (`training/shaders/shaders.cu`, `training/gpu_workspace.rs`). There is **no f32
  master copy**.
- The gradient accumulators and Adam moments (`m`, `v`) are already **f32**.
- The loop already has the usual stabilizers — Adam (f32 moments), global
  gradient-norm clipping (`MAX_GRAD_NORM=1.0`), and a 100-step linear LR warmup
  (`WARMUP_STEPS`) — and the softmax kernels are numerically stable
  (max-subtraction present). So those are **not** the cause.

**Why f16-only weights cap the LR (swamping).** f16 has ~3–4 significant digits;
near `w ≈ 1.0` the smallest representable gap is ~1e-3. An Adam update of order
`Δw ≈ 1e-4` therefore rounds away in `w + Δw` — the update vanishes at small LR, and
at larger LR the weights move in coarse f16 quanta with no f32 value accumulating
the fine changes, so the step is jerky and diverges. This is the classic
mixed-precision failure mode of a hand-rolled f16 trainer.

### The fix the epic should design (standard mixed precision)

Keep **f16 for all compute** (unchanged throughput), but keep an **f32 master copy
of the weights**:

- Adam updates the **f32 master** (`w_f32 += Δw`) — no swamping.
- Cast f32→f16 once per step to produce the f16 weights the matmuls consume (a
  disposable "view", regenerated each step).
- Initialise the master from the f16 weights on load (and, for SFT, from the base
  model's f16 weights); save from the f16 view as today (or from the master — the
  epic decides).

Cost is modest: per parameter today is f16 `w` (2B) + f32 `m` + f32 `v` (8B) = 10B;
adding an f32 master is +4B → **14B/param** (~40 MB on a 10M-param model). Compute
stays f16.

### What the epic design must cover (checklist for the child chain)

- **Storage:** add an f32 master buffer alongside each `DeviceBufferF16` weight in
  `gpu_workspace.rs` (embeddings in/out, positions, every per-layer weight).
- **Optimizer kernel:** change `adam_kernel` to read/write the f32 master, plus a
  cast-to-f16 step (separate kernel or fused tail) that refreshes the f16 view.
- **Init/load/save:** seed masters from f16 on `Transformer::load` (pretrain and the
  SFT base path, `train.rs`); confirm checkpoint/save semantics stay correct.
- **Correctness check:** a way to confirm the change actually raises the usable LR
  (e.g. a short run at 1e-4 / 3e-4 that previously diverged now trains).
- **Cheap wins to fold in (or split out):** mode-aware LR defaults (pretrain ~1e-4,
  SFT ~1e-5) and a smaller default vocab (32768→~8192) for small corpora. Decide
  whether these belong in this epic or are separate.

### Open design questions for the epic to resolve

- One extra global cast kernel per step vs. fusing the f16 write into the Adam
  kernel's tail — perf vs. simplicity.
- Whether gradients/activations stay f16 (they should) and whether any accumulation
  needs loss-scaling (likely not, since grads/moments are already f32 and clipping
  is present — but the epic should confirm, not assume).
- Save format: keep persisting f16 weights (smaller artifacts, matches inference) vs.
  persisting f32 masters. Backwards-compat with existing saved models.

### Non-goals

- Not touching the wire protocol, job model, or the frontend.
- Not rewriting the whole trainer back to f32 — compute stays f16 by design.

### Definition of done (for THIS meta issue)

A parent epic issue exists with a `blocked-by` chain of small children covering
storage, kernel, init/load/save, verification, and the LR/vocab defaults, each with
a single location and its own DoD — mirroring how EPIC-0040 decomposes its work.
No production CUDA code is changed under this id.

## Comments
### Auditor · 2026-07-12
Filed as a meta/design issue at the user's request: capture *what* the
mixed-precision fix must achieve and the known root cause now, while the epic
itself is designed later. The diagnosis (f16-only master weights → swamping → LR
cap) was worked out with the user from the actual buffers/kernels
(`gpu_workspace.rs`, `shaders.cu`), so the child chain starts from a settled cause,
not a guess. P1: this is a latent correctness/quality bug — models train far worse
than they should at any safe LR.

### Auditor · 2026-07-13 · design delivered
Epic designed and filed as **EPIC-0052** with the `blocked-by` chain
**0053–0058** plus the independent cheap-win **ISSUE-0059**:

| # | Child | Covers |
| --- | --- | --- |
| 0053 | f32 master buffers in `GpuWorkspace` | storage |
| 0054 | `adam_kernel` updates the master + refreshes the f16 view (fused) | kernel |
| 0055 | thread master pointers through `gpu_adam_update` | wiring |
| 0056 | seed masters from f32 on load; save from the master | init/load/save |
| 0057 | verification — a previously-divergent LR now trains | correctness check |
| 0058 | mode-aware LR defaults (pretrain ~1e-4, SFT ~1e-5) | defaults |
| 0059 | smaller default vocab (32768→~8192) — *independent, not in chain* | cheap win |

Resolved ISSUE-0050's open questions in the epic: **fuse** the f16 refresh into
the Adam kernel tail (no extra launch); **no on-disk format change** — keep
persisting f32 and read the master on save (lossless, closes the f16 round-trip
EPIC-0030 flagged); **loss-scaling confirmed empirically** in 0057 rather than
assumed. DoD of this meta-issue (a parent epic + small children covering storage,
kernel, init/load/save, verification, and the LR/vocab defaults) is met; no
production CUDA code was changed under this id. Marking `done`.

