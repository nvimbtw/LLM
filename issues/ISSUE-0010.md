---
id: ISSUE-0010
title: Inline magic hyperparameter values should be named constants (R8)
scope: backend/llm
priority: P1
status: done
rule: R8
location: backend/src/llm/src/training/transformer.rs:248,459,561,715,797,818,1327-1329
blocked-by: []
blocks: []
---

## Description
The following tunable values are inlined as literals with no named `const`:
- `1e-5f32` — layer-norm epsilon, 5 places (lines 248, 459, 561, 715, 797)
- `1.0f32` — gradient clip max norm (line 818), duplicated in train.rs:162
- `0.9`, `0.999`, `1e-8` — Adam beta1, beta2, epsilon (lines 1327-1329)
- `3` — early-stopping patience (train.rs:154)
- `4` — gradient accumulation steps (train.rs:158)
- `100` — linear LR warmup steps (train.rs:166)

Each should be a documented `const` at the module top.

## Comments
### Verifier · migrated · pass
Confirmed `LAYER_NORM_EPS`, `TRAIN_STEP_CLIP_NORM`, `ADAM_BETA1`, `ADAM_BETA2`,
`ADAM_EPS` defined in transformer.rs; `EARLY_STOP_PATIENCE`, `ACCUM_STEPS`,
`MAX_GRAD_NORM`, `WARMUP_STEPS` defined in train.rs. All literals replaced.
