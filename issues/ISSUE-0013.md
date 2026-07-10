---
id: ISSUE-0013
title: Forward pass logic duplicated between forward and forward_seq (R6)
scope: backend/llm
priority: P2
status: done
rule: R6
location: backend/src/llm/src/training/transformer.rs:118-229
blocked-by: []
blocks: []
---

## Description
`forward` (lines 118-177) and `forward_seq` (lines 179-229) have a near-identical
inner loop: embed → positional → per-layer { ln → attn → residual → ln → ff →
residual }. The only difference is that `forward` also produces training targets.
Factor the shared forward-pass body into a private helper (e.g.
`compute_hidden_states`) parameterized over whether targets are needed.

## Comments
### Verifier · migrated · pass
Confirmed `pub async fn forward` has been removed entirely. Only `forward_seq`
remains — the duplication is eliminated.
