---
id: ISSUE-0001
type: issue
title: Derive inference d_ff from weights, not a hardcoded ×4
scope: backend/llm
priority: P1
status: done
rule: R10
location: backend/src/llm/src/training/transformer.rs:345
blocked-by: []
blocks: []
---

## Description
`feed_forward` hardcodes `let d_ff = d_model * 4;`, but the FF width is
`dimensions * ff_multiplier` and `ff_multiplier` is a user-configurable training
field. Training uses the real width; only the inference path assumes 4. The
frontend default is 4, which masks it — but the moment a model is trained with
`ff_multiplier != 4`, inference reads `w1` with the wrong stride and generates
garbage regardless of training quality. Derive `d_ff` from `w1[0].len()` (the
actual weight shape) instead of assuming ×4.

## Comments
### Verifier · migrated · pass
Confirmed `let d_ff = w1[0].len();` at line 345 with explanatory comment.
`cargo build` clean.
