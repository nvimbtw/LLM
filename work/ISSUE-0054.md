---
id: ISSUE-0054
type: issue
title: "adam_kernel: update the f32 master weight and refresh the f16 view in one pass"
scope: backend/llm
priority: P1
status: todo
rule: none
location: backend/src/llm/src/training/shaders/shaders.cu:166-183 backend/src/llm/src/training/transformer.rs:1476
blocked-by: []
blocks: [ISSUE-0055]
---

## Description
### What
Change the CUDA Adam update so the **f32 master is the canonical weight** and the
`__half` weight is a derived view refreshed in the same kernel. Today
(`shaders.cu:166`):

```c
float grad = g[i] * grad_scale;
... m/v update (f32, correct) ...
w[i] = __float2half(__half2float(w[i]) - lr * m_hat / (sqrtf(v_hat)+eps));  // f16 is the master — the bug
```

becomes (new signature carries both pointers):

```c
// w_master: float*  (canonical)   w_view: __half*  (derived)
float grad = g[i] * grad_scale;
... m/v update (unchanged) ...
float updated = w_master[i] - lr * m_hat / (sqrtf(v_hat)+eps);  // accumulate in f32
w_master[i] = updated;
w_view[i]   = __float2half(updated);                             // refresh the view, fused
```

### Shape of the change
- **`adam_kernel`** (`shaders.cu:166`): add a `float *w_master` parameter; read
  the previous weight from `w_master` (not `__half2float(w)`); write both
  `w_master` and the `__half` view. Keep the m/v math byte-identical.
- **`extern "C" cuda_adam`** launcher (`shaders.cu:1418`): add the `float
  *w_master` param and forward it.
- **Rust `extern` decl** for `cuda_adam` (`transformer.rs:1476`): add the matching
  `*const f32` / `*mut f32` master arg. Do **not** change `gpu_adam_update`'s call
  sites here beyond what compiles — the pointer threading is 0055's job; if
  keeping this child self-contained is awkward, land 0054+0055 as one Verifier
  round but keep them as separate commits. (Auditor's preference: 0054 changes
  only CUDA + the extern signature; 0055 fills in the argument at every call.)

### Design decision (from the epic): fuse, don't add a launch
The f16 refresh is **fused into the Adam kernel tail** (one write of the view per
param, no second global pass). `f32_to_f16_kernel` (`shaders.cu:205`) already
exists for any standalone cast need (e.g. seeding), but the per-step refresh must
not add a kernel launch per weight.

### Correctness note
With a **zero gradient**, `m_hat`→0 over steps but is not exactly 0 on step 1 due
to bias correction; that is unchanged from today. What *is* fixed: a nonzero
`Δw ≈ 1e-4` now accumulates in `w_master` (f32) across steps instead of rounding
away in f16 — that is the whole point.

### Verification
- `.cu` recompiles (the build's `nvcc` step) and `cargo build` links clean.
- Unit intuition to assert in 0057, not here: feeding a small constant gradient
  for N steps moves `w_master` by ≈ N·step even when each step is below the f16
  ULP of the starting weight.

## Comments
### Auditor · 2026-07-13
The heart of the epic — the actual numerical fix. Isolated to the kernel + the
FFI signature so the Verifier reviews the arithmetic in one place. Fused-refresh
decision is inherited from EPIC-0052's resolved open questions.
