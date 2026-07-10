---
id: ISSUE-0022
title: sample_token panics on top_k=0 or empty logits (R10, R11)
scope: backend/llm
priority: P2
status: todo
rule: R10, R11
location: backend/src/llm/src/infer.rs:233-257
blocked-by: []
blocks: []
---

## Description
`sample_token` does not validate that `top_k > 0` or that `logits` is non-empty. If
`top_k` is 0, `candidates.truncate(0)` empties the vec and `candidates[0]` panics.
If logits is empty, the initial collect produces an empty vec and indexing panics.
Add early returns or guards with documented behaviour.

## Comments
_No comments yet._
