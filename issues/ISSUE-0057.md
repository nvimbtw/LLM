---
id: ISSUE-0057
title: "Verification — an LR that previously diverged now trains (LR cap lifted)"
scope: backend/llm
priority: P1
status: todo
rule: none
location: backend/src/llm/tests/train_test.rs
blocked-by: [ISSUE-0056]
blocks: [ISSUE-0058]
---

## Description
### What
Prove the epic's payoff: with f32 masters in place, a learning rate that provably
blew up before the fix now trains stably. This is the child that closes the
numerical loop and lets ISSUE-0058 raise the default LR with evidence.

### The test(s) (extend `train_test.rs`, GPU, `--test-threads=1`)
1. **Master accumulates sub-ULP updates (unit-level, deterministic).** Build a
   tiny `Transformer::init`, `upload_weights`, then feed a small *constant*
   gradient (e.g. via the existing grad buffers) and call `gpu_adam_update` for N
   steps at an LR whose per-step `Δw` is below the f16 ULP near the starting
   weight. Assert `sync_weights_from_gpu` shows the weight moved by ≈ N·Δw — i.e.
   the change accumulated in f32 instead of rounding away. On `dev` (f16 master)
   this weight would not move; here it must.
2. **A previously-divergent LR now decreases loss.** Overfit a small fixed
   sequence (mirror `test_overfits_single_sequence`) at the target LR (**1e-4**;
   attempt **3e-4** and, if it's still legitimately unstable at that LR for this
   toy size, record the honest ceiling in the comment rather than forcing a pass).
   Assert the loss is finite throughout and trends **down**, not NaN/blow-up.

### What "done" looks like
- Both assertions pass on the GPU box.
- The existing trainer tests (`train_test.rs`, `transformer::tests`) still pass —
  the fix didn't regress the working small-LR path.
- A one-paragraph comment records the **before/after LR ceiling actually
  observed** (e.g. "diverged at 1e-4 on `dev`; stable and decreasing at 1e-4
  post-fix, 3e-4 marginal"), because ISSUE-0052's DoD item 2 and ISSUE-0058's
  default both cite this number.

### Note on loss scaling (resolves an ISSUE-0052 open question)
If test 2 is stable without any loss-scaling, that empirically confirms none is
needed (grads + moments already f32, grad-norm clipped). If it is *not* stable,
do **not** silently add loss-scaling here — file a follow-up child; loss-scaling
is out of this epic's declared scope unless this test forces it.

## Comments
### Auditor · 2026-07-13
The empirical proof of the whole epic and the source of the LR number 0058 will
bake into the defaults. Written to demand an *honest* observed ceiling rather than
a forced green, and to route any surprise loss-scaling need to a new issue instead
of scope-creeping this one.
